#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include <vector>

#include "mgpu/MGPUDialect.h"
#include "mgpu/MGPUOps.h"
#include "mgpu/MGPUPasses.h"

using namespace mlir;

namespace {

class BufferMap {
public:
  static BufferMap build(func::FuncOp func) {
    BufferMap map;
    bool sawLayout = false;
    func.walk([&](mgpu::MemoryLayoutOp layout) {
      sawLayout = true;
      for (auto [value, offset] :
           llvm::zip(layout.getBuffers(), layout.getOffsets())) {
        map.offsets[value] = offset;
      }
    });
    if (!sawLayout) {
      int64_t cursor = 0;
      for (Value arg : func.getArguments()) {
        if (auto type = dyn_cast<MemRefType>(arg.getType())) {
          map.offsets[arg] = cursor;
          cursor += byteSize(type);
        }
      }
      func.walk([&](memref::AllocOp alloc) {
        if (auto type = dyn_cast<MemRefType>(alloc.getType())) {
          map.offsets[alloc.getResult()] = cursor;
          cursor += byteSize(type);
        }
      });
      // Quantization/dequantization ops produce memref SSA values directly
      // rather than writing into an explicit allocation.  Assign those result
      // values a stable fallback address so ``mgpu-to-isa`` can lower a chain
      // such as quant -> dequant without an injected memory layout.
      func.walk([&](Operation *op) {
        for (Value result : op->getResults()) {
          if (map.offsets.count(result))
            continue;
          if (auto type = dyn_cast<MemRefType>(result.getType())) {
            map.offsets[result] = cursor;
            cursor += byteSize(type);
          }
        }
      });
    }
    return map;
  }

  std::optional<int64_t> lookup(Value value) const {
    auto it = offsets.find(value);
    if (it == offsets.end())
      return std::nullopt;
    return it->second;
  }

  int64_t base(Value value) const {
    if (auto offset = lookup(value))
      return *offset;
    llvm::report_fatal_error(
        "mgpu-to-isa: missing buffer offset in mgpu.memory_layout");
  }

private:
  static int64_t byteSize(MemRefType type) {
    int64_t elements = 1;
    for (int64_t dim : type.getShape())
      elements *= dim;
    Type elementType = type.getElementType();
    if (auto intType = dyn_cast<IntegerType>(elementType);
        intType && intType.getWidth() == 4) {
      // INT4/UINT4 values are packed two per byte.
      return (elements + 1) / 2;
    }
    return elements * elementByteSize(elementType);
  }

  static int64_t elementByteSize(Type elementType) {
    if (auto intType = dyn_cast<IntegerType>(elementType))
      return std::max<int64_t>(1, intType.getWidth() / 8);
    if (isa<Float8E4M3FNType>(elementType) || isa<Float8E5M2Type>(elementType))
      return 1;
    if (elementType.isF16() || elementType.isBF16())
      return 2;
    if (elementType.isF64())
      return 8;
    return 4;
  }

  DenseMap<Value, int64_t> offsets;
};

class VirtualRegisterAllocator {
public:
  int acquire() {
    if (!freeRegisters.empty()) {
      int reg = freeRegisters.back();
      freeRegisters.pop_back();
      return reg;
    }
    return nextRegister++;
  }

  void release(int reg) {
    freeRegisters.push_back(reg);
  }

  std::string name(int reg) const {
    return "R" + std::to_string(reg);
  }

private:
  int nextRegister = 0;
  std::vector<int> freeRegisters;
};

class LinearScanRegisterAllocator {
public:
  LinearScanRegisterAllocator(int physicalRegisters, int spillBase)
      : physicalRegisters(physicalRegisters), spillBase(spillBase) {}

  int acquire(llvm::raw_ostream &os) {
    if (!freeRegisters.empty()) {
      int reg = freeRegisters.back();
      freeRegisters.pop_back();
      return reg;
    }
    if (nextRegister < physicalRegisters)
      return nextRegister++;

    // No physical register is free.  Choose the oldest allocated register as
    // a round-robin spill victim and emit a stack-store before reusing it.
    int victim = nextRegister++ % physicalRegisters;
    int spillSlot = nextSpillSlot++;
    os << "STG [4096 + " << (spillSlot * 4) << "], " << name(victim) << "\n";
    return victim;
  }

  void release(int reg) {
    freeRegisters.push_back(reg);
  }

  std::string name(int reg) const {
    return "R" + std::to_string(reg);
  }

private:
  int physicalRegisters;
  int spillBase;
  int nextRegister = 0;
  int nextSpillSlot = 0;
  std::vector<int> freeRegisters;
};

struct MGpuToIsaPass
    : public PassWrapper<MGpuToIsaPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MGpuToIsaPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    math::MathDialect, memref::MemRefDialect,
                    mgpu::MGPUDialect>();
  }

  StringRef getArgument() const final { return "mgpu-to-isa"; }
  StringRef getDescription() const final {
    return "Emit simulator ISA for supported mgpu launches";
  }

  void runOnOperation() override {
    bufferMap = BufferMap::build(getOperation());
    SmallVector<mgpu::LaunchOp> launches;
    getOperation().walk(
        [&](mgpu::LaunchOp launch) { launches.push_back(launch); });

    IRRewriter rewriter(&getContext());
    for (mgpu::LaunchOp launch : launches) {
      if (auto isa = emitElementwiseGenericIsa(launch)) {
        rewriter.setInsertionPoint(launch);
        rewriter.create<mgpu::IsaOp>(
            launch.getLoc(),
            StringAttr::get(&getContext(), isa->code),
            StringAttr::get(&getContext(), isa->grid),
            StringAttr::get(&getContext(), isa->block));
        rewriter.eraseOp(launch);
      } else if (auto isa = emitVectorAddIsa(launch)) {
        rewriter.setInsertionPoint(launch);
        rewriter.create<mgpu::IsaOp>(
            launch.getLoc(),
            StringAttr::get(&getContext(), isa->code),
            StringAttr::get(&getContext(), isa->grid),
            StringAttr::get(&getContext(), isa->block));
        rewriter.eraseOp(launch);
      } else if (auto isa = emitLinalgGenericIsa(launch)) {
        rewriter.setInsertionPoint(launch);
        rewriter.create<mgpu::IsaOp>(
            launch.getLoc(),
            StringAttr::get(&getContext(), isa->code),
            StringAttr::get(&getContext(), isa->grid),
            StringAttr::get(&getContext(), isa->block));
        rewriter.eraseOp(launch);
      } else if (auto isa = emitGlobalAvgPoolIsa(launch)) {
        rewriter.setInsertionPoint(launch);
        rewriter.create<mgpu::IsaOp>(
            launch.getLoc(),
            StringAttr::get(&getContext(), isa->code),
            StringAttr::get(&getContext(), isa->grid),
            StringAttr::get(&getContext(), isa->block));
        rewriter.eraseOp(launch);
      } else if (auto isa = emitUnaryGenericIsa(launch)) {
        rewriter.setInsertionPoint(launch);
        rewriter.create<mgpu::IsaOp>(
            launch.getLoc(),
            StringAttr::get(&getContext(), isa->code),
            StringAttr::get(&getContext(), isa->grid),
            StringAttr::get(&getContext(), isa->block));
        rewriter.eraseOp(launch);
      } else if (auto isa = emitActivationIsa(launch)) {
        rewriter.setInsertionPoint(launch);
        rewriter.create<mgpu::IsaOp>(
            launch.getLoc(),
            StringAttr::get(&getContext(), isa->code),
            StringAttr::get(&getContext(), isa->grid),
            StringAttr::get(&getContext(), isa->block));
        rewriter.eraseOp(launch);
      } else if (auto isa = emitMatmulIsa(launch)) {
        rewriter.setInsertionPoint(launch);
        rewriter.create<mgpu::IsaOp>(
            launch.getLoc(),
            StringAttr::get(&getContext(), isa->code),
            StringAttr::get(&getContext(), isa->grid),
            StringAttr::get(&getContext(), isa->block));
        rewriter.eraseOp(launch);
      } else if (auto isa = emitScalarMatmulIsa(launch)) {
        rewriter.setInsertionPoint(launch);
        rewriter.create<mgpu::IsaOp>(
            launch.getLoc(),
            StringAttr::get(&getContext(), isa->code),
            StringAttr::get(&getContext(), isa->grid),
            StringAttr::get(&getContext(), isa->block));
        rewriter.eraseOp(launch);
      } else if (auto isa = emitLinearIsa(launch)) {
        rewriter.setInsertionPoint(launch);
        rewriter.create<mgpu::IsaOp>(
            launch.getLoc(),
            StringAttr::get(&getContext(), isa->code),
            StringAttr::get(&getContext(), isa->grid),
            StringAttr::get(&getContext(), isa->block));
        rewriter.eraseOp(launch);
      } else if (auto isa = emitConvIsa(launch)) {
        rewriter.setInsertionPoint(launch);
        rewriter.create<mgpu::IsaOp>(
            launch.getLoc(),
            StringAttr::get(&getContext(), isa->code),
            StringAttr::get(&getContext(), isa->grid),
            StringAttr::get(&getContext(), isa->block));
        rewriter.eraseOp(launch);
      } else if (auto isa = emitDepthwiseConvIsa(launch)) {
        rewriter.setInsertionPoint(launch);
        rewriter.create<mgpu::IsaOp>(
            launch.getLoc(),
            StringAttr::get(&getContext(), isa->code),
            StringAttr::get(&getContext(), isa->grid),
            StringAttr::get(&getContext(), isa->block));
        rewriter.eraseOp(launch);
      } else if (auto isa = emitPoolingIsa(launch, /*maxPool=*/true)) {
        rewriter.setInsertionPoint(launch);
        rewriter.create<mgpu::IsaOp>(
            launch.getLoc(),
            StringAttr::get(&getContext(), isa->code),
            StringAttr::get(&getContext(), isa->grid),
            StringAttr::get(&getContext(), isa->block));
        rewriter.eraseOp(launch);
      } else if (auto isa = emitPoolingIsa(launch, /*maxPool=*/false)) {
        rewriter.setInsertionPoint(launch);
        rewriter.create<mgpu::IsaOp>(
            launch.getLoc(),
            StringAttr::get(&getContext(), isa->code),
            StringAttr::get(&getContext(), isa->grid),
            StringAttr::get(&getContext(), isa->block));
        rewriter.eraseOp(launch);
      } else if (auto isa = emitRopeIsa(launch)) {
        rewriter.setInsertionPoint(launch);
        rewriter.create<mgpu::IsaOp>(
            launch.getLoc(),
            StringAttr::get(&getContext(), isa->code),
            StringAttr::get(&getContext(), isa->grid),
            StringAttr::get(&getContext(), isa->block));
        rewriter.eraseOp(launch);
      } else if (auto isa = emitCausalMaskIsa(launch)) {
        rewriter.setInsertionPoint(launch);
        rewriter.create<mgpu::IsaOp>(
            launch.getLoc(),
            StringAttr::get(&getContext(), isa->code),
            StringAttr::get(&getContext(), isa->grid),
            StringAttr::get(&getContext(), isa->block));
        rewriter.eraseOp(launch);
      } else if (auto isa = emitKVCacheAppendIsa(launch)) {
        rewriter.setInsertionPoint(launch);
        rewriter.create<mgpu::IsaOp>(
            launch.getLoc(),
            StringAttr::get(&getContext(), isa->code),
            StringAttr::get(&getContext(), isa->grid),
            StringAttr::get(&getContext(), isa->block));
        rewriter.eraseOp(launch);
      } else if (auto isa = emitKVCacheGatherIsa(launch)) {
        rewriter.setInsertionPoint(launch);
        rewriter.create<mgpu::IsaOp>(
            launch.getLoc(),
            StringAttr::get(&getContext(), isa->code),
            StringAttr::get(&getContext(), isa->grid),
            StringAttr::get(&getContext(), isa->block));
        rewriter.eraseOp(launch);
      } else if (auto isa = emitSoftmaxIsa(launch)) {
        rewriter.setInsertionPoint(launch);
        rewriter.create<mgpu::IsaOp>(
            launch.getLoc(),
            StringAttr::get(&getContext(), isa->code),
            StringAttr::get(&getContext(), isa->grid),
            StringAttr::get(&getContext(), isa->block));
        rewriter.eraseOp(launch);
      } else if (auto isa = emitMaskedSoftmaxIsa(launch)) {
        rewriter.setInsertionPoint(launch);
        rewriter.create<mgpu::IsaOp>(
            launch.getLoc(),
            StringAttr::get(&getContext(), isa->code),
            StringAttr::get(&getContext(), isa->grid),
            StringAttr::get(&getContext(), isa->block));
        rewriter.eraseOp(launch);
      } else if (auto isa = emitLayerNormIsa(launch)) {
        rewriter.setInsertionPoint(launch);
        rewriter.create<mgpu::IsaOp>(
            launch.getLoc(),
            StringAttr::get(&getContext(), isa->code),
            StringAttr::get(&getContext(), isa->grid),
            StringAttr::get(&getContext(), isa->block));
        rewriter.eraseOp(launch);
      } else if (auto isa = emitRMSNormIsa(launch)) {
        rewriter.setInsertionPoint(launch);
        rewriter.create<mgpu::IsaOp>(
            launch.getLoc(),
            StringAttr::get(&getContext(), isa->code),
            StringAttr::get(&getContext(), isa->grid),
            StringAttr::get(&getContext(), isa->block));
        rewriter.eraseOp(launch);
      } else if (auto isa = emitReshapeIsa(launch)) {
        rewriter.setInsertionPoint(launch);
        rewriter.create<mgpu::IsaOp>(
            launch.getLoc(),
            StringAttr::get(&getContext(), isa->code),
            StringAttr::get(&getContext(), isa->grid),
            StringAttr::get(&getContext(), isa->block));
        rewriter.eraseOp(launch);
      } else if (auto isa = emitTransposeIsa(launch)) {
        rewriter.setInsertionPoint(launch);
        rewriter.create<mgpu::IsaOp>(
            launch.getLoc(),
            StringAttr::get(&getContext(), isa->code),
            StringAttr::get(&getContext(), isa->grid),
            StringAttr::get(&getContext(), isa->block));
        rewriter.eraseOp(launch);
      } else if (auto isa = emitConcatIsa(launch)) {
        rewriter.setInsertionPoint(launch);
        rewriter.create<mgpu::IsaOp>(
            launch.getLoc(),
            StringAttr::get(&getContext(), isa->code),
            StringAttr::get(&getContext(), isa->grid),
            StringAttr::get(&getContext(), isa->block));
        rewriter.eraseOp(launch);
      } else if (auto isa = emitSliceIsa(launch)) {
        rewriter.setInsertionPoint(launch);
        rewriter.create<mgpu::IsaOp>(
            launch.getLoc(),
            StringAttr::get(&getContext(), isa->code),
            StringAttr::get(&getContext(), isa->grid),
            StringAttr::get(&getContext(), isa->block));
        rewriter.eraseOp(launch);
      } else if (auto isa = emitPadIsa(launch)) {
        rewriter.setInsertionPoint(launch);
        rewriter.create<mgpu::IsaOp>(
            launch.getLoc(),
            StringAttr::get(&getContext(), isa->code),
            StringAttr::get(&getContext(), isa->grid),
            StringAttr::get(&getContext(), isa->block));
        rewriter.eraseOp(launch);
      } else if (auto isa = emitResizeIsa(launch)) {
        rewriter.setInsertionPoint(launch);
        rewriter.create<mgpu::IsaOp>(
            launch.getLoc(),
            StringAttr::get(&getContext(), isa->code),
            StringAttr::get(&getContext(), isa->grid),
            StringAttr::get(&getContext(), isa->block));
        rewriter.eraseOp(launch);
      } else if (auto isa = emitBatchNormIsa(launch)) {
        rewriter.setInsertionPoint(launch);
        rewriter.create<mgpu::IsaOp>(
            launch.getLoc(),
            StringAttr::get(&getContext(), isa->code),
            StringAttr::get(&getContext(), isa->grid),
            StringAttr::get(&getContext(), isa->block));
        rewriter.eraseOp(launch);
      } else if (auto isa = emitGroupNormIsa(launch)) {
        rewriter.setInsertionPoint(launch);
        rewriter.create<mgpu::IsaOp>(
            launch.getLoc(),
            StringAttr::get(&getContext(), isa->code),
            StringAttr::get(&getContext(), isa->grid),
            StringAttr::get(&getContext(), isa->block));
        rewriter.eraseOp(launch);
      } else if (auto isa = emitBoxDecodeIsa(launch)) {
        rewriter.setInsertionPoint(launch);
        rewriter.create<mgpu::IsaOp>(
            launch.getLoc(),
            StringAttr::get(&getContext(), isa->code),
            StringAttr::get(&getContext(), isa->grid),
            StringAttr::get(&getContext(), isa->block));
        rewriter.eraseOp(launch);
      } else if (auto isa = emitNmsIsa(launch)) {
        rewriter.setInsertionPoint(launch);
        rewriter.create<mgpu::IsaOp>(
            launch.getLoc(),
            StringAttr::get(&getContext(), isa->code),
            StringAttr::get(&getContext(), isa->grid),
            StringAttr::get(&getContext(), isa->block));
        rewriter.eraseOp(launch);
      } else if (auto isa = emitAttentionIsa(launch)) {
        rewriter.setInsertionPoint(launch);
        rewriter.create<mgpu::IsaOp>(
            launch.getLoc(),
            StringAttr::get(&getContext(), isa->code),
            StringAttr::get(&getContext(), isa->grid),
            StringAttr::get(&getContext(), isa->block));
        rewriter.eraseOp(launch);
      } else if (auto isa = emitQuantIsa(launch)) {
        rewriter.setInsertionPoint(launch);
        rewriter.create<mgpu::IsaOp>(
            launch.getLoc(),
            StringAttr::get(&getContext(), isa->code),
            StringAttr::get(&getContext(), isa->grid),
            StringAttr::get(&getContext(), isa->block));
        rewriter.eraseOp(launch);
      } else if (auto isa = emitGenericMGpuIsa(launch)) {
        rewriter.setInsertionPoint(launch);
        rewriter.create<mgpu::IsaOp>(
            launch.getLoc(),
            StringAttr::get(&getContext(), isa->code),
            StringAttr::get(&getContext(), isa->grid),
            StringAttr::get(&getContext(), isa->block));
        rewriter.eraseOp(launch);
      } else {
        // Every launch must be replaced by an ``mgpu.isa`` op.  Failing with
        // a diagnostic here is intentional: silently leaving an
        // ``mgpu.launch`` in the module would make the Python side think the
        // model lowered successfully while the simulator would have no
        // instruction stream for this launch.
        diagnoseUnsupportedLaunch(launch);
        signalPassFailure();
        return;
      }
    }
  }

  struct IsaText {
    std::string code;
    std::string grid;
    std::string block;
  };

  BufferMap bufferMap;

  std::string materializeAffineExpr(llvm::raw_ostream &os, AffineExpr expr,
                                    ArrayRef<int> indexRegs,
                                    VirtualRegisterAllocator &regs) const {
    if (auto constant = dyn_cast<AffineConstantExpr>(expr)) {
      int reg = regs.acquire();
      os << "MOV " << regs.name(reg) << ", " << constant.getValue() << "\n";
      return regs.name(reg);
    }
    if (auto dim = dyn_cast<AffineDimExpr>(expr)) {
      unsigned position = dim.getPosition();
      if (position >= indexRegs.size())
        return "";
      return regs.name(indexRegs[position]);
    }

    auto binary = dyn_cast<AffineBinaryOpExpr>(expr);
    if (!binary)
      return "";

    std::string lhs =
        materializeAffineExpr(os, binary.getLHS(), indexRegs, regs);
    std::string rhs =
        materializeAffineExpr(os, binary.getRHS(), indexRegs, regs);
    if (lhs.empty() || rhs.empty())
      return "";

    int result = regs.acquire();
    if (binary.getKind() == AffineExprKind::Add) {
      os << "ADD " << regs.name(result) << ", " << lhs << ", " << rhs << "\n";
    } else if (binary.getKind() == AffineExprKind::Mul) {
      os << "MUL " << regs.name(result) << ", " << lhs << ", " << rhs << "\n";
    } else {
      return "";
    }
    return regs.name(result);
  }

  std::optional<IsaText>
  emitElementwiseGenericIsa(mgpu::LaunchOp launch) {
    SmallVector<linalg::GenericOp> generics;
    launch.getBody().front().walk(
        [&](linalg::GenericOp generic) { generics.push_back(generic); });

    for (linalg::GenericOp generic : generics) {
      auto dstOp = cast<DestinationStyleOpInterface>(generic.getOperation());
      if (dstOp.getDpsInputs().size() != 2 || dstOp.getDpsInits().empty())
        continue;

      auto opcode = binaryArithOpcode(generic);
      if (!opcode)
        continue;

      auto outputType = dyn_cast<MemRefType>(dstOp.getDpsInits()[0].getType());
      if (!outputType || outputType.getRank() < 1 || outputType.getRank() > 4)
        continue;
      if (!outputType.getElementType().isF32())
        continue;
      if (llvm::any_of(outputType.getShape(),
                       [](int64_t dim) { return ShapedType::isDynamic(dim); }))
        continue;

      SmallVector<MemRefType> inputTypes;
      inputTypes.reserve(2);
      for (Value input : dstOp.getDpsInputs()) {
        auto type = dyn_cast<MemRefType>(input.getType());
        if (!type || type.getRank() < 1 || type.getRank() > 4 ||
            !type.getElementType().isF32())
          continue;
        if (llvm::any_of(type.getShape(),
                         [](int64_t dim) { return ShapedType::isDynamic(dim); }))
          continue;
        inputTypes.push_back(type);
      }
      if (inputTypes.size() != 2)
        continue;

      auto maps = generic.getIndexingMapsArray();
      if (maps.size() != 3)
        continue;

      int64_t rank = outputType.getRank();
      SmallVector<int64_t> shape(rank);
      for (int64_t i = 0; i < rank; ++i)
        shape[i] = outputType.getShape()[i];

      int64_t total = 1;
      for (int64_t dim : shape)
        total *= dim;

      VirtualRegisterAllocator regs;
      int tid = regs.acquire();
      int limit = regs.acquire();
      int pred = regs.acquire();

      SmallVector<int> indexRegs;
      for (int64_t i = 0; i < rank; ++i)
        indexRegs.push_back(regs.acquire());

      std::string code;
      llvm::raw_string_ostream os(code);
      os << "MOV " << regs.name(tid) << ", ctaid.x\n"
         << "MUL " << regs.name(tid) << ", " << regs.name(tid) << ", ntid.x\n"
         << "ADD " << regs.name(tid) << ", " << regs.name(tid) << ", tid.x\n"
         << "MOV " << regs.name(limit) << ", " << total << "\n"
         << "CMP.LT " << regs.name(pred) << ", " << regs.name(tid) << ", "
         << regs.name(limit) << "\n"
         << "BRA body, " << regs.name(pred) << ", end\n"
         << "BRA end\n"
         << "body:\n";

      if (rank == 1) {
        os << "MOV " << regs.name(indexRegs[0]) << ", " << regs.name(tid)
           << "\n";
      } else if (rank == 2) {
        os << "IDIV " << regs.name(indexRegs[0]) << ", " << regs.name(tid)
           << ", " << shape[1] << "\n"
           << "MUL R31, " << regs.name(indexRegs[0]) << ", " << shape[1]
           << "\n"
           << "SUB " << regs.name(indexRegs[1]) << ", " << regs.name(tid)
           << ", R31\n";
      } else if (rank == 3) {
        int64_t outer = shape[1] * shape[2];
        os << "IDIV " << regs.name(indexRegs[0]) << ", " << regs.name(tid)
           << ", " << outer << "\n"
           << "MUL R31, " << regs.name(indexRegs[0]) << ", " << outer << "\n"
           << "SUB R30, " << regs.name(tid) << ", R31\n"
           << "IDIV " << regs.name(indexRegs[1]) << ", R30, " << shape[2]
           << "\n"
           << "MUL R31, " << regs.name(indexRegs[1]) << ", " << shape[2]
           << "\n"
           << "SUB " << regs.name(indexRegs[2]) << ", R30, R31\n";
      } else {
        int64_t outer = shape[1] * shape[2] * shape[3];
        int64_t middle = shape[2] * shape[3];
        os << "IDIV " << regs.name(indexRegs[0]) << ", " << regs.name(tid)
           << ", " << outer << "\n"
           << "MUL R31, " << regs.name(indexRegs[0]) << ", " << outer << "\n"
           << "SUB R30, " << regs.name(tid) << ", R31\n"
           << "IDIV " << regs.name(indexRegs[1]) << ", R30, " << middle << "\n"
           << "MUL R31, " << regs.name(indexRegs[1]) << ", " << middle << "\n"
           << "SUB R29, R30, R31\n"
           << "IDIV " << regs.name(indexRegs[2]) << ", R29, " << shape[3]
           << "\n"
           << "MUL R31, " << regs.name(indexRegs[2]) << ", " << shape[3]
           << "\n"
           << "SUB " << regs.name(indexRegs[3]) << ", R29, R31\n";
      }

      SmallVector<std::string> loadedValues;
      for (unsigned i = 0; i < inputTypes.size(); ++i) {
        MemRefType inputType = inputTypes[i];
        AffineMap map = maps[i];
        int64_t inputRank = inputType.getRank();
        int linearReg = regs.acquire();
        os << "MOV " << regs.name(linearReg) << ", 0\n";

        for (int64_t j = 0; j < inputRank; ++j) {
          std::string coord = materializeAffineExpr(
              os, map.getResult(j), indexRegs, regs);
          if (coord.empty())
            return std::nullopt;

          int64_t stride = 1;
          for (int64_t k = j + 1; k < inputRank; ++k)
            stride *= inputType.getShape()[k];

          std::string term = coord;
          if (stride != 1) {
            int termReg = regs.acquire();
            os << "MUL " << regs.name(termReg) << ", " << coord << ", "
               << stride << "\n";
            term = regs.name(termReg);
          }
          os << "ADD " << regs.name(linearReg) << ", " << regs.name(linearReg)
             << ", " << term << "\n";
        }

        int addrReg = regs.acquire();
        os << "MUL " << regs.name(addrReg) << ", " << regs.name(linearReg)
           << ", 4\n"
           << "ADD " << regs.name(addrReg) << ", " << regs.name(addrReg)
           << ", " << bufferMap.base(dstOp.getDpsInputs()[i]) << "\n";
        int valueReg = regs.acquire();
        os << "LDG " << regs.name(valueReg) << ", [" << regs.name(addrReg)
           << "]\n";
        loadedValues.push_back(regs.name(valueReg));
      }

      int resultReg = regs.acquire();
      os << *opcode << " " << regs.name(resultReg) << ", " << loadedValues[0]
         << ", " << loadedValues[1] << "\n";

      int outputAddrReg = regs.acquire();
      os << "MUL " << regs.name(outputAddrReg) << ", " << regs.name(tid)
         << ", 4\n"
         << "ADD " << regs.name(outputAddrReg) << ", " << regs.name(outputAddrReg)
         << ", " << bufferMap.base(dstOp.getDpsInits()[0]) << "\n"
         << "STG [" << regs.name(outputAddrReg) << "], " << regs.name(resultReg)
         << "\n"
         << "BRA end\n"
         << "end:\n"
         << "EXIT\n";
      os.flush();

      IsaText isa;
      isa.code = code;
      isa.grid = std::to_string((total + 31) / 32) + "x1x1";
      isa.block = "32x1x1";
      return isa;
    }
    return std::nullopt;
  }

  static double getFloatConstant(Value value) {
    auto constantOp = value.getDefiningOp<arith::ConstantOp>();
    if (!constantOp)
      return 0.0;
    if (auto floatAttr = dyn_cast<FloatAttr>(constantOp.getValue()))
      return floatAttr.getValueAsDouble();
    return 0.0;
  }

  std::optional<IsaText> emitUnaryGenericIsa(mgpu::LaunchOp launch) {
    SmallVector<linalg::GenericOp> generics;
    launch.getBody().front().walk(
        [&](linalg::GenericOp generic) { generics.push_back(generic); });

    for (linalg::GenericOp generic : generics) {
      auto dstOp = cast<DestinationStyleOpInterface>(generic.getOperation());
      if (dstOp.getDpsInputs().size() != 1 || dstOp.getDpsInits().empty())
        continue;

      auto inputType = dyn_cast<MemRefType>(dstOp.getDpsInputs()[0].getType());
      auto outputType = dyn_cast<MemRefType>(dstOp.getDpsInits()[0].getType());
      if (!inputType || !outputType)
        continue;
      if (!inputType.getElementType().isF32() ||
          !outputType.getElementType().isF32())
        continue;
      if (llvm::any_of(inputType.getShape(),
                       [](int64_t dim) { return ShapedType::isDynamic(dim); }) ||
          llvm::any_of(outputType.getShape(),
                       [](int64_t dim) { return ShapedType::isDynamic(dim); }))
        continue;

      int64_t inputElements = 1;
      int64_t outputElements = 1;
      for (int64_t dim : inputType.getShape())
        inputElements *= dim;
      for (int64_t dim : outputType.getShape())
        outputElements *= dim;
      if (inputElements != outputElements)
        continue;

      bool hasRsqrt = false;
      bool hasDiv = false;
      bool hasAdd = false;
      bool hasTrunc = false;
      generic.getRegion().walk([&](Operation *op) {
        if (isa<math::RsqrtOp>(op))
          hasRsqrt = true;
        else if (isa<arith::DivFOp>(op))
          hasDiv = true;
        else if (isa<arith::AddFOp>(op))
          hasAdd = true;
        else if (isa<arith::TruncFOp>(op))
          hasTrunc = true;
      });

      if (!hasRsqrt && !hasDiv && !(hasAdd && hasTrunc))
        continue;

      Value inputBlockArg = generic.getRegion().front().getArgument(0);
      std::string opcode;
      double constant = 0.0;
      if (hasRsqrt) {
        opcode = "RSQRT";
      } else if (hasDiv) {
        opcode = "DIV";
        std::optional<arith::DivFOp> divOp;
        generic.getRegion().walk([&](arith::DivFOp op) {
          if (!divOp)
            divOp = op;
        });
        if (!divOp)
          continue;
        Value lhs = divOp->getLhs();
        Value rhs = divOp->getRhs();
        if (lhs == inputBlockArg) {
          constant = getFloatConstant(rhs);
          if (!rhs.getDefiningOp<arith::ConstantOp>())
            continue;
        } else if (rhs == inputBlockArg) {
          continue;
        } else {
          continue;
        }
      } else if (hasAdd && hasTrunc) {
        opcode = "ADD";
        std::optional<arith::TruncFOp> truncOp;
        generic.getRegion().walk([&](arith::TruncFOp op) {
          if (!truncOp)
            truncOp = op;
        });
        if (!truncOp)
          continue;
        Value constantOperand = truncOp->getIn();
        if (!constantOperand.getDefiningOp<arith::ConstantOp>())
          continue;
        constant = getFloatConstant(constantOperand);
      }

      int64_t total = outputElements;
      int64_t inputBase = bufferMap.base(dstOp.getDpsInputs()[0]);
      int64_t outputBase = bufferMap.base(dstOp.getDpsInits()[0]);

      std::string code;
      llvm::raw_string_ostream os(code);
      emitCnnPrologue(os, total);
      os << "MUL R3, R0, 4\n"
         << "ADD R3, R3, " << inputBase << "\n"
         << "LDG R5, [R3]\n";
      if (opcode == "RSQRT") {
        os << "RSQRT R6, R5\n";
      } else {
        os << opcode << " R6, R5, " << constant << "\n";
      }
      os << "MUL R4, R0, 4\n"
         << "ADD R4, R4, " << outputBase << "\n"
         << "STG [R4], R6\n";
      emitCnnFinish(os);
      os.flush();

      IsaText isa;
      isa.code = code;
      isa.grid = std::to_string((total + 31) / 32) + "x1x1";
      isa.block = "32x1x1";
      return isa;
    }
    return std::nullopt;
  }

  void diagnoseUnsupportedLaunch(mgpu::LaunchOp launch) {
    Operation *firstUnsupported = nullptr;
    launch.getBody().front().walk([&](Operation *op) {
      if (!firstUnsupported && !isa<mgpu::ReturnOp, mgpu::ThreadIdOp,
                                    mgpu::BlockIdOp, mgpu::GridDimOp,
                                    mgpu::BlockDimOp>(op)) {
        firstUnsupported = op;
      }
    });

    InFlightDiagnostic diag =
        launch.emitOpError("cannot lower mgpu.launch to simulator ISA");
    if (!firstUnsupported) {
      diag << "; launch body contains no computable operation";
      return;
    }

    diag << "; unsupported operation: " << firstUnsupported->getName();
    for (OpOperand &operand : firstUnsupported->getOpOperands()) {
      Type type = operand.get().getType();
      if (auto memref = dyn_cast<MemRefType>(type)) {
        diag << " operand=" << memref;
      } else if (type) {
        diag << " dtype=" << type;
      }
    }
    for (OpResult result : firstUnsupported->getResults()) {
      if (auto memref = dyn_cast<MemRefType>(result.getType()))
        diag << " result=" << memref;
    }
  }

  std::optional<IsaText> emitVectorAddIsa(mgpu::LaunchOp launch) {
    SmallVector<linalg::GenericOp> generics;
    launch.getBody().front().walk(
        [&](linalg::GenericOp generic) { generics.push_back(generic); });

    linalg::GenericOp addOp;
    for (linalg::GenericOp generic : generics) {
      if (containsAdd(generic)) {
        addOp = generic;
        break;
      }
    }
    if (!addOp)
      return std::nullopt;

    auto dstOp = cast<DestinationStyleOpInterface>(addOp.getOperation());
    if (dstOp.getDpsInputs().size() != 2 || dstOp.getDpsInits().empty())
      return std::nullopt;

    auto outputType = dyn_cast<MemRefType>(dstOp.getDpsInits()[0].getType());
    if (!outputType)
      return std::nullopt;

    int64_t n = 1;
    for (int64_t dim : outputType.getShape())
      n *= dim;
    int64_t blockSize = 32;
    int64_t gridX = (n + blockSize - 1) / blockSize;
    int64_t leftBase = bufferMap.base(dstOp.getDpsInputs()[0]);
    int64_t rightBase = bufferMap.base(dstOp.getDpsInputs()[1]);
    int64_t outputBase = bufferMap.base(dstOp.getDpsInits()[0]);
    int64_t leftDelta = rightBase - leftBase;
    int64_t outputDelta = outputBase - leftBase;

    VirtualRegisterAllocator regs;
    int tid = regs.acquire();
    int limit = regs.acquire();
    int pred = regs.acquire();
    int addr = regs.acquire();
    int base = regs.acquire();
    int left = regs.acquire();
    int right = regs.acquire();
    int sum = regs.acquire();

    std::string code;
    llvm::raw_string_ostream os(code);
    os << "MOV " << regs.name(tid) << ", ctaid.x\n"
       << "MUL " << regs.name(tid) << ", " << regs.name(tid) << ", ntid.x\n"
       << "ADD " << regs.name(tid) << ", " << regs.name(tid) << ", tid.x\n"
       << "MOV " << regs.name(limit) << ", " << n << "\n"
       << "CMP.LT " << regs.name(pred) << ", " << regs.name(tid) << ", "
       << regs.name(limit) << "\n"
       << "BRA body, " << regs.name(pred) << ", end\n"
       << "BRA end\n"
       << "body:\n"
       << "MUL " << regs.name(addr) << ", " << regs.name(tid) << ", 4\n"
       << "MOV " << regs.name(base) << ", " << leftBase << "\n"
       << "ADD " << regs.name(addr) << ", " << regs.name(addr) << ", "
       << regs.name(base) << "\n"
       << "LDG " << regs.name(left) << ", [" << regs.name(addr) << "]\n"
       << "LDG " << regs.name(right) << ", [" << regs.name(addr) << " + "
       << leftDelta << "]\n"
       << "ADD " << regs.name(sum) << ", " << regs.name(left) << ", "
       << regs.name(right) << "\n"
       << "STG [" << regs.name(addr) << " + " << outputDelta << "], "
       << regs.name(sum) << "\n"
       << "BRA end\n"
       << "end:\n"
       << "EXIT\n";
    os.flush();

    regs.release(sum);
    regs.release(right);
    regs.release(left);
    regs.release(base);
    regs.release(addr);
    regs.release(pred);
    regs.release(limit);
    regs.release(tid);

    IsaText result;
    result.code = code;
    result.grid = std::to_string(gridX) + "x1x1";
    result.block = std::to_string(blockSize) + "x1x1";
    return result;
  }

  static bool containsAdd(linalg::GenericOp generic) {
    bool found = false;
    generic.getRegion().walk(
        [&](arith::AddFOp) { found = true; });
    return found;
  }

  static std::optional<std::string> binaryArithOpcode(linalg::GenericOp generic) {
    std::string opcode;
    generic.getRegion().walk([&](Operation *op) {
      if (isa<arith::AddFOp>(op))
        opcode = "ADD";
      else if (isa<arith::SubFOp>(op))
        opcode = "SUB";
      else if (isa<arith::MulFOp>(op))
        opcode = "MUL";
      else if (isa<arith::DivFOp>(op))
        opcode = "DIV";
    });
    if (opcode.empty())
      return std::nullopt;
    return opcode;
  }

  static std::optional<std::string> reductionArithOpcode(
      linalg::GenericOp generic, bool &divideByCount) {
    std::string opcode;
    divideByCount = false;
    generic.getRegion().walk([&](Operation *op) {
      if (isa<arith::AddFOp>(op))
        opcode = "ADD";
      else if (isa<arith::MaximumFOp>(op))
        opcode = "MAX";
      else if (isa<arith::DivFOp>(op))
        divideByCount = true;
    });
    if (opcode.empty())
      return std::nullopt;
    return opcode;
  }

  std::optional<IsaText> emitLinalgGenericIsa(mgpu::LaunchOp launch) {
    SmallVector<linalg::GenericOp> generics;
    launch.getBody().front().walk(
        [&](linalg::GenericOp generic) { generics.push_back(generic); });

    for (linalg::GenericOp generic : generics) {
      if (auto isa = emitBinaryGenericIsa(launch, generic))
        return isa;
      if (auto isa = emitReductionGenericIsa(launch, generic))
        return isa;
    }
    return std::nullopt;
  }

  std::optional<IsaText> emitBinaryGenericIsa(mgpu::LaunchOp launch,
                                               linalg::GenericOp generic) {
    auto dstOp = cast<DestinationStyleOpInterface>(generic.getOperation());
    if (dstOp.getDpsInputs().size() != 2 || dstOp.getDpsInits().empty())
      return std::nullopt;

    auto opcode = binaryArithOpcode(generic);
    if (!opcode)
      return std::nullopt;

    auto outputType = dyn_cast<MemRefType>(dstOp.getDpsInits()[0].getType());
    if (!outputType || outputType.getRank() != 1)
      return std::nullopt;

    int64_t n = outputType.getShape()[0];
    int64_t blockSize = 32;
    int64_t gridX = (n + blockSize - 1) / blockSize;
    int64_t leftBase = bufferMap.base(dstOp.getDpsInputs()[0]);
    int64_t rightBase = bufferMap.base(dstOp.getDpsInputs()[1]);
    int64_t outputBase = bufferMap.base(dstOp.getDpsInits()[0]);
    int64_t leftDelta = rightBase - leftBase;
    int64_t outputDelta = outputBase - leftBase;

    VirtualRegisterAllocator regs;
    int tid = regs.acquire();
    int limit = regs.acquire();
    int pred = regs.acquire();
    int addr = regs.acquire();
    int left = regs.acquire();
    int right = regs.acquire();
    int result = regs.acquire();

    std::string code;
    llvm::raw_string_ostream os(code);
    os << "MOV " << regs.name(tid) << ", ctaid.x\n"
       << "MUL " << regs.name(tid) << ", " << regs.name(tid) << ", ntid.x\n"
       << "ADD " << regs.name(tid) << ", " << regs.name(tid) << ", tid.x\n"
       << "MOV " << regs.name(limit) << ", " << n << "\n"
       << "CMP.LT " << regs.name(pred) << ", " << regs.name(tid) << ", "
       << regs.name(limit) << "\n"
       << "BRA body, " << regs.name(pred) << ", end\n"
       << "BRA end\n"
       << "body:\n"
       << "MUL " << regs.name(addr) << ", " << regs.name(tid) << ", 4\n"
       << "ADD " << regs.name(addr) << ", " << regs.name(addr) << ", "
       << leftBase << "\n"
       << "LDG " << regs.name(left) << ", [" << regs.name(addr) << "]\n"
       << "LDG " << regs.name(right) << ", [" << regs.name(addr) << " + "
       << leftDelta << "]\n"
       << *opcode << " " << regs.name(result) << ", " << regs.name(left) << ", "
       << regs.name(right) << "\n"
       << "STG [" << regs.name(addr) << " + " << outputDelta << "], "
       << regs.name(result) << "\n"
       << "BRA end\n"
       << "end:\n"
       << "EXIT\n";
    os.flush();

    IsaText isa;
    isa.code = code;
    isa.grid = std::to_string(gridX) + "x1x1";
    isa.block = std::to_string(blockSize) + "x1x1";
    return isa;
  }

  std::optional<IsaText> emitReductionGenericIsa(mgpu::LaunchOp launch,
                                                  linalg::GenericOp generic) {
    auto dstOp = cast<DestinationStyleOpInterface>(generic.getOperation());
    if (dstOp.getDpsInputs().size() != 1 || dstOp.getDpsInits().size() != 1)
      return std::nullopt;

    bool divideByCount = false;
    auto opcode = reductionArithOpcode(generic, divideByCount);
    if (!opcode)
      return std::nullopt;

    auto inputType = dyn_cast<MemRefType>(dstOp.getDpsInputs()[0].getType());
    auto outputType = dyn_cast<MemRefType>(dstOp.getDpsInits()[0].getType());
    if (!inputType || !outputType || inputType.getRank() != 1 ||
        outputType.getRank() != 0)
      return std::nullopt;

    int64_t n = inputType.getShape()[0];
    int64_t inputBase = bufferMap.base(dstOp.getDpsInputs()[0]);
    int64_t outputBase = bufferMap.base(dstOp.getDpsInits()[0]);
    std::string identity = (*opcode == "MAX") ? "-1.0e30" : "0";

    // Single-block shared-memory tree reduction.  The block size is the next
    // power of two large enough for ``n`` (and at least one full warp), so all
    // inactive lanes can store the reduction identity and keep every BAR.SYNC
    // convergent.  This mirrors CUDA's common intra-block reduction strategy.
    int64_t blockSize = 32;
    while (blockSize < n)
      blockSize *= 2;

    std::string code;
    llvm::raw_string_ostream os(code);

    // R0 = tid.x, R1 = bound/stride, R2 = predicate, R3 = byte offset,
    // R4 = value/address, R5..R9 = load/reduce temporaries.
    os << "MOV R0, tid.x\n"
       << "MUL R3, R0, 4\n"
       << "MOV R1, " << n << "\n"
       << "CMP.LT R2, R0, R1\n"
       << "BRA load, R2, load_end\n"
       << "BRA load_end\n"
       << "load:\n"
       << "MOV R4, " << inputBase << "\n"
       << "ADD R7, R3, R4\n"
       << "LDG R5, [R7]\n"
       << "MOV R4, R5\n"
       << "BRA load_end\n"
       << "load_skip:\n"
       << "MOV R4, " << identity << "\n"
       << "BRA load_end\n"
       << "load_end:\n"
       << "STS [R3], R4\n"
       << "BAR.SYNC\n";

    for (int64_t stride = blockSize / 2; stride > 0; stride /= 2) {
      std::string body = "tree_" + std::to_string(stride) + "_body";
      std::string end = "tree_" + std::to_string(stride) + "_end";
      os << "MOV R1, " << stride << "\n"
         << "CMP.LT R2, R0, R1\n"
         << "BRA " << body << ", R2, " << end << "\n"
         << "BRA " << end << "\n"
         << body << ":\n"
         << "MUL R5, R0, 4\n"
         << "MOV R6, R0\n"
         << "ADD R6, R6, R1\n"
         << "MUL R6, R6, 4\n"
         << "LDS R7, [R5]\n"
         << "LDS R8, [R6]\n"
         << *opcode << " R9, R7, R8\n"
         << "STS [R5], R9\n"
         << "BRA " << end << "\n"
         << end << ":\n"
         << "BAR.SYNC\n";
    }

    os << "CMP.EQ R2, R0, 0\n"
       << "BRA store, R2, store_end\n"
       << "BRA store_end\n"
       << "store:\n"
       << "LDS R4, [R3]\n";
    if (divideByCount)
      os << "DIV R4, R4, " << n << "\n";
    os << "MOV R5, " << outputBase << "\n"
       << "STG [R5], R4\n"
       << "BRA store_end\n"
       << "store_end:\n"
       << "EXIT\n";
    os.flush();

    IsaText isa;
    isa.code = code;
    isa.grid = "1x1x1";
    isa.block = std::to_string(blockSize) + "x1x1";
    return isa;
  }

  enum class ActivationKind { Relu, LeakyRelu, Sigmoid, Silu, Gelu };

  static std::optional<ActivationKind>
  detectActivationKind(linalg::GenericOp generic) {
    bool hasTanh = false;
    bool hasSelect = false;
    bool hasExp = false;
    bool hasDiv = false;
    bool hasMaximum = false;
    bool hasMul = false;
    generic.getRegion().walk([&](Operation *op) {
      if (isa<math::TanhOp>(op))
        hasTanh = true;
      else if (isa<arith::SelectOp>(op))
        hasSelect = true;
      else if (isa<math::ExpOp>(op))
        hasExp = true;
      else if (isa<arith::DivFOp>(op))
        hasDiv = true;
      else if (isa<arith::MaximumFOp>(op))
        hasMaximum = true;
      else if (isa<arith::MulFOp>(op))
        hasMul = true;
    });

    if (hasTanh)
      return ActivationKind::Gelu;
    if (hasSelect)
      return ActivationKind::LeakyRelu;
    if (hasExp)
      return hasMul ? ActivationKind::Silu : ActivationKind::Sigmoid;
    if (hasMaximum)
      return ActivationKind::Relu;
    return std::nullopt;
  }

  static void emitActivationBody(llvm::raw_ostream &os, ActivationKind kind) {
    if (kind == ActivationKind::Relu) {
      os << "MAX R6, R5, 0\n";
      return;
    }
    if (kind == ActivationKind::LeakyRelu) {
      os << "CMP.LT R7, R5, 0\n"
         << "MUL R8, R5, 0.01\n"
         << "SEL R6, R7, R8, R5\n";
      return;
    }
    if (kind == ActivationKind::Sigmoid || kind == ActivationKind::Silu) {
      os << "MUL R7, R5, -1.4426950408889634\n"
         << "EXP2 R8, R7\n"
         << "MOV R9, 1\n"
         << "ADD R10, R8, R9\n"
         << "RCP R6, R10\n";
      if (kind == ActivationKind::Silu)
        os << "MUL R6, R5, R6\n";
      return;
    }
    if (kind == ActivationKind::Gelu) {
      os << "MUL R7, R5, R5\n"
         << "MUL R8, R7, R5\n"
         << "MUL R8, R8, 0.044715\n"
         << "ADD R8, R8, R5\n"
         << "MUL R8, R8, 0.7978845608028654\n"
         << "MUL R9, R8, 2\n"
         << "MUL R9, R9, -1.4426950408889634\n"
         << "EXP2 R10, R9\n"
         << "MOV R11, 1\n"
         << "ADD R12, R10, R11\n"
         << "RCP R13, R12\n"
         << "MUL R13, R13, 2\n"
         << "MOV R14, 1\n"
         << "SUB R13, R13, R14\n"
         << "MOV R14, 1\n"
         << "ADD R14, R14, R13\n"
         << "MUL R6, R5, R14\n"
         << "MUL R6, R6, 0.5\n";
      return;
    }
  }

  std::optional<IsaText> emitActivationIsa(mgpu::LaunchOp launch) {
    SmallVector<linalg::GenericOp> generics;
    launch.getBody().front().walk(
        [&](linalg::GenericOp generic) { generics.push_back(generic); });

    for (linalg::GenericOp generic : generics) {
      auto kind = detectActivationKind(generic);
      if (!kind)
        continue;

      auto dstOp = cast<DestinationStyleOpInterface>(generic.getOperation());
      if (dstOp.getDpsInputs().size() != 1 || dstOp.getDpsInits().empty())
        continue;

      auto inputType = dyn_cast<MemRefType>(dstOp.getDpsInputs()[0].getType());
      auto outputType = dyn_cast<MemRefType>(dstOp.getDpsInits()[0].getType());
      if (!inputType || !outputType)
        continue;

      int64_t inputElements = 1;
      int64_t outputElements = 1;
      for (int64_t dim : inputType.getShape())
        inputElements *= dim;
      for (int64_t dim : outputType.getShape())
        outputElements *= dim;
      if (inputElements != outputElements)
        continue;

      int64_t n = outputElements;
      int64_t inputBase = bufferMap.base(dstOp.getDpsInputs()[0]);
      int64_t outputBase = bufferMap.base(dstOp.getDpsInits()[0]);
      int64_t blockSize = 32;
      int64_t gridX = (n + blockSize - 1) / blockSize;

      std::string code;
      llvm::raw_string_ostream os(code);
      os << "MOV R0, ctaid.x\n"
         << "MUL R0, R0, ntid.x\n"
         << "ADD R0, R0, tid.x\n"
         << "MOV R1, " << n << "\n"
         << "CMP.LT R2, R0, R1\n"
         << "BRA body, R2, end\n"
         << "BRA end\n"
       << "body:\n"
       << "MUL R3, R0, 4\n"
       << "ADD R3, R3, " << inputBase << "\n"
       << "LDG R5, [R3]\n";
      emitActivationBody(os, *kind);
      os << "MUL R4, R0, 4\n"
         << "ADD R4, R4, " << outputBase << "\n"
         << "STG [R4], R6\n"
         << "BRA end\n"
         << "end:\n"
         << "EXIT\n";
      os.flush();

      IsaText isa;
      isa.code = code;
      isa.grid = std::to_string(gridX) + "x1x1";
      isa.block = std::to_string(blockSize) + "x1x1";
      return isa;
    }
    return std::nullopt;
  }

  static void emitCnnPrologue(llvm::raw_ostream &os, int64_t total) {
    os << "MOV R0, ctaid.x\n"
       << "MUL R0, R0, ntid.x\n"
       << "ADD R0, R0, tid.x\n"
       << "MOV R1, " << total << "\n"
       << "CMP.LT R2, R0, R1\n"
       << "BRA body, R2, end\n"
       << "BRA end\n"
       << "body:\n";
  }

  static void emitCnnFinish(llvm::raw_ostream &os) {
    os << "BRA end\n"
       << "end:\n"
       << "EXIT\n";
  }

  std::optional<IsaText> emitDepthwiseConvIsa(mgpu::LaunchOp launch) {
    SmallVector<linalg::DepthwiseConv2DNchwChwOp> ops;
    launch.getBody().front().walk(
        [&](linalg::DepthwiseConv2DNchwChwOp op) { ops.push_back(op); });
    if (ops.empty())
      return std::nullopt;

    auto op = ops.front();
    if (op.getInputs().size() < 2 || op.getOutputs().empty())
      return std::nullopt;

    auto imageType = dyn_cast<MemRefType>(op.getInputs()[0].getType());
    auto weightType = dyn_cast<MemRefType>(op.getInputs()[1].getType());
    auto outputType = dyn_cast<MemRefType>(op.getOutputs()[0].getType());
    if (!imageType || !weightType || !outputType)
      return std::nullopt;
    if (imageType.getRank() != 4 || weightType.getRank() != 3 ||
        outputType.getRank() != 4)
      return std::nullopt;

    int64_t n = imageType.getShape()[0];
    int64_t c = imageType.getShape()[1];
    int64_t h = imageType.getShape()[2];
    int64_t w = imageType.getShape()[3];
    int64_t kh = weightType.getShape()[1];
    int64_t kw = weightType.getShape()[2];
    int64_t oh = outputType.getShape()[2];
    int64_t ow = outputType.getShape()[3];
    if (n != 1)
      return std::nullopt;

    int64_t total = c * oh * ow;
    int64_t imageBase = bufferMap.base(op.getInputs()[0]);
    int64_t weightBase = bufferMap.base(op.getInputs()[1]);
    int64_t outputBase = bufferMap.base(op.getOutputs()[0]);
    int64_t strideY = 1;
    int64_t strideX = 1;

    std::string code;
    llvm::raw_string_ostream os(code);
    emitCnnPrologue(os, total);
    os << "IDIV R1, R0, " << (oh * ow) << "\n"
       << "MUL R4, R1, " << (oh * ow) << "\n"
       << "SUB R5, R0, R4\n"
       << "IDIV R2, R5, " << ow << "\n"
       << "MUL R6, R2, " << ow << "\n"
       << "SUB R3, R5, R6\n"
       << "MOV R20, 0\n";

    for (int64_t ky = 0; ky < kh; ++ky) {
      for (int64_t kx = 0; kx < kw; ++kx) {
        int64_t ihConst = ky;
        int64_t iwConst = kx;
        os << "MOV R4, " << strideY << "\n"
           << "MUL R4, R2, R4\n"
           << "ADD R4, R4, " << ihConst << "\n"
           << "CMP.GE R5, R4, 0\n"
           << "MOV R6, " << h << "\n"
           << "CMP.LT R7, R4, R6\n"
           << "MUL R8, R5, R7\n"
           << "MOV R9, 0\n"
           << "MAX R10, R4, R9\n"
           << "MOV R9, " << (h - 1) << "\n"
           << "MIN R10, R10, R9\n"
           << "MOV R11, " << strideX << "\n"
           << "MUL R11, R3, R11\n"
           << "ADD R11, R11, " << iwConst << "\n"
           << "CMP.GE R12, R11, 0\n"
           << "MOV R13, " << w << "\n"
           << "CMP.LT R14, R11, R13\n"
           << "MUL R15, R12, R14\n"
           << "MUL R18, R8, R15\n"
           << "MOV R16, 0\n"
           << "MAX R17, R11, R16\n"
           << "MOV R16, " << (w - 1) << "\n"
           << "MIN R17, R17, R16\n"
           << "MOV R19, " << h << "\n"
           << "MUL R19, R1, R19\n"
           << "ADD R19, R19, R10\n"
           << "MUL R19, R19, " << w << "\n"
           << "ADD R19, R19, R17\n"
           << "MUL R19, R19, 4\n"
           << "ADD R19, R19, " << imageBase << "\n"
           << "LDG R6, [R19]\n"
           << "MOV R7, 0\n"
           << "SEL R8, R18, R6, R7\n"
           << "MOV R9, " << kh << "\n"
           << "MUL R9, R1, R9\n"
           << "ADD R9, R9, " << ky << "\n"
           << "MUL R9, R9, " << kw << "\n"
           << "ADD R9, R9, " << kx << "\n"
           << "MUL R9, R9, 4\n"
           << "ADD R9, R9, " << weightBase << "\n"
           << "LDG R11, [R9]\n"
           << "FMA R20, R8, R11, R20\n";
      }
    }

    os << "MUL R4, R1, " << (oh * ow) << "\n"
       << "MUL R5, R2, " << ow << "\n"
       << "ADD R4, R4, R5\n"
       << "ADD R4, R4, R3\n"
       << "MUL R4, R4, 4\n"
       << "ADD R4, R4, " << outputBase << "\n"
       << "STG [R4], R20\n";
    emitCnnFinish(os);
    os.flush();

    IsaText isa;
    isa.code = code;
    isa.grid = std::to_string((total + 31) / 32) + "x1x1";
    isa.block = "32x1x1";
    return isa;
  }

  std::optional<IsaText> emitPoolingIsa(mgpu::LaunchOp launch, bool maxPool) {
    SmallVector<Operation *> pools;
    launch.getBody().front().walk([&](Operation *op) {
      if (maxPool) {
        if (isa<linalg::PoolingNchwMaxOp>(op))
          pools.push_back(op);
      } else if (isa<linalg::PoolingNchwSumOp>(op)) {
        pools.push_back(op);
      }
    });
    if (pools.empty())
      return std::nullopt;

    Operation *op = pools.front();
    auto dstOp = cast<DestinationStyleOpInterface>(op);
    if (dstOp.getDpsInputs().empty() || dstOp.getDpsInits().empty())
      return std::nullopt;
    auto inputType = dyn_cast<MemRefType>(dstOp.getDpsInputs()[0].getType());
    auto outputType = dyn_cast<MemRefType>(dstOp.getDpsInits()[0].getType());
    if (!inputType || !outputType || inputType.getRank() != 4 ||
        outputType.getRank() != 4)
      return std::nullopt;
    if (dstOp.getDpsInputs().size() < 2)
      return std::nullopt;
    auto kernelType =
        dyn_cast<MemRefType>(dstOp.getDpsInputs()[1].getType());
    if (!kernelType || kernelType.getRank() != 2)
      return std::nullopt;

    int64_t n = inputType.getShape()[0];
    int64_t c = inputType.getShape()[1];
    int64_t h = inputType.getShape()[2];
    int64_t w = inputType.getShape()[3];
    int64_t oh = outputType.getShape()[2];
    int64_t ow = outputType.getShape()[3];
    if (n != 1)
      return std::nullopt;

    int64_t kernelY = kernelType.getShape()[0];
    int64_t kernelX = kernelType.getShape()[1];
    int64_t strideY = 1;
    int64_t strideX = 1;
    int64_t dilY = 1;
    int64_t dilX = 1;
    if (auto strides =
            op->getAttrOfType<DenseIntElementsAttr>("strides")) {
      SmallVector<int64_t, 2> values;
      for (int64_t value : strides.getValues<int64_t>())
        values.push_back(value);
      if (values.size() == 2) {
        strideY = values[0];
        strideX = values[1];
      }
    }
    if (auto dilations =
            op->getAttrOfType<DenseIntElementsAttr>("dilations")) {
      SmallVector<int64_t, 2> values;
      for (int64_t value : dilations.getValues<int64_t>())
        values.push_back(value);
      if (values.size() == 2) {
        dilY = values[0];
        dilX = values[1];
      }
    }
    int64_t padTotalY =
        (oh - 1) * strideY + (kernelY - 1) * dilY + 1 - h;
    int64_t padTotalX =
        (ow - 1) * strideX + (kernelX - 1) * dilX + 1 - w;
    int64_t padY = padTotalY > 0 ? (padTotalY + 1) / 2 : 0;
    int64_t padX = padTotalX > 0 ? (padTotalX + 1) / 2 : 0;
    int64_t total = c * oh * ow;
    int64_t inputBase = bufferMap.base(dstOp.getDpsInputs()[0]);
    int64_t outputBase = bufferMap.base(dstOp.getDpsInits()[0]);
    std::string arith = maxPool ? "MAX" : "ADD";
    std::string identity = maxPool ? "-1.0e30" : "0";

    std::string code;
    llvm::raw_string_ostream os(code);
    emitCnnPrologue(os, total);
    os << "IDIV R1, R0, " << (oh * ow) << "\n"
       << "MUL R4, R1, " << (oh * ow) << "\n"
       << "SUB R5, R0, R4\n"
       << "IDIV R2, R5, " << ow << "\n"
       << "MUL R6, R2, " << ow << "\n"
       << "SUB R3, R5, R6\n"
       << "MOV R20, " << identity << "\n"
       << "MOV R21, 0\n"
       << "MOV R22, " << (kernelY * kernelX) << "\n"
       << "pool_loop:\n"
       << "CMP.LT R23, R21, R22\n"
       << "BRA pool_body, R23, pool_end\n"
       << "BRA pool_end\n"
       << "pool_body:\n"
       << "IDIV R24, R21, " << kernelX << "\n"
       << "MUL R25, R24, " << kernelX << "\n"
       << "SUB R26, R21, R25\n"
       << "MOV R30, " << strideY << "\n"
       << "MUL R30, R2, R30\n"
       << "MOV R31, " << dilY << "\n"
       << "MUL R31, R24, R31\n"
       << "ADD R30, R30, R31\n"
       << "MOV R31, " << strideX << "\n"
       << "MUL R31, R3, R31\n"
       << "MOV R32, " << dilX << "\n"
       << "MUL R32, R26, R32\n"
       << "ADD R31, R31, R32\n"
       << "SUB R30, R30, " << padY << "\n"
       << "SUB R31, R31, " << padX << "\n"
       << "CMP.GE R33, R30, 0\n"
       << "MOV R34, " << h << "\n"
       << "CMP.LT R35, R30, R34\n"
       << "MUL R36, R33, R35\n"
       << "CMP.GE R37, R31, 0\n"
       << "MOV R38, " << w << "\n"
       << "CMP.LT R39, R31, R38\n"
       << "MUL R40, R37, R39\n"
       << "MUL R41, R36, R40\n"
       << "MOV R42, 0\n"
       << "MAX R30, R30, R42\n"
       << "MOV R42, " << (h - 1) << "\n"
       << "MIN R30, R30, R42\n"
       << "MOV R42, 0\n"
       << "MAX R31, R31, R42\n"
       << "MOV R42, " << (w - 1) << "\n"
       << "MIN R31, R31, R42\n"
       << "MOV R6, " << h << "\n"
       << "MUL R6, R1, R6\n"
       << "ADD R6, R6, R30\n"
       << "MUL R6, R6, " << w << "\n"
       << "ADD R6, R6, R31\n"
       << "MUL R6, R6, 4\n"
       << "ADD R6, R6, " << inputBase << "\n"
       << "LDG R8, [R6]\n"
       << "MOV R42, " << identity << "\n"
       << "SEL R8, R41, R8, R42\n"
       << arith << " R20, R20, R8\n"
       << "ADD R21, R21, 1\n"
       << "BRA pool_loop\n"
       << "pool_end:\n";

    if (!maxPool)
      os << "DIV R20, R20, " << (kernelY * kernelX) << "\n";
    os << "MUL R4, R1, " << (oh * ow) << "\n"
       << "MUL R5, R2, " << ow << "\n"
       << "ADD R4, R4, R5\n"
       << "ADD R4, R4, R3\n"
       << "MUL R4, R4, 4\n"
       << "ADD R4, R4, " << outputBase << "\n"
       << "STG [R4], R20\n";
    emitCnnFinish(os);
    os.flush();

    IsaText isa;
    isa.code = code;
    isa.grid = std::to_string((total + 31) / 32) + "x1x1";
    isa.block = "32x1x1";
    return isa;
  }

  std::optional<IsaText> emitGlobalAvgPoolIsa(mgpu::LaunchOp launch) {
    SmallVector<linalg::GenericOp> generics;
    launch.getBody().front().walk(
        [&](linalg::GenericOp generic) { generics.push_back(generic); });
    Value finalOutput;
    for (linalg::GenericOp generic : generics) {
      bool hasDiv = false;
      generic.getRegion().walk([&](arith::DivFOp) { hasDiv = true; });
      if (hasDiv) {
        auto dstOp = cast<DestinationStyleOpInterface>(generic.getOperation());
        if (!dstOp.getDpsInits().empty())
          finalOutput = dstOp.getDpsInits()[0];
      }
      auto dstOp = cast<DestinationStyleOpInterface>(generic.getOperation());
      if (dstOp.getDpsInputs().size() != 1 || dstOp.getDpsInits().empty())
        continue;
      auto inputType = dyn_cast<MemRefType>(dstOp.getDpsInputs()[0].getType());
      auto outputType = dyn_cast<MemRefType>(dstOp.getDpsInits()[0].getType());
      if (!inputType || !outputType || inputType.getRank() != 4 ||
          outputType.getRank() != 4 || outputType.getShape()[2] != 1 ||
          outputType.getShape()[3] != 1)
        continue;

      bool hasAdd = false;
      generic.getRegion().walk([&](arith::AddFOp) { hasAdd = true; });
      if (!hasAdd)
        continue;

      int64_t c = inputType.getShape()[1];
      int64_t h = inputType.getShape()[2];
      int64_t w = inputType.getShape()[3];
      int64_t total = c;
      int64_t inputBase = bufferMap.base(dstOp.getDpsInputs()[0]);
      int64_t outputBase =
          finalOutput ? bufferMap.base(finalOutput)
                      : inputBase + memrefByteSize(inputType);

      std::string code;
      llvm::raw_string_ostream os(code);
      emitCnnPrologue(os, total);
      os << "MOV R1, R0\n"
         << "MOV R20, 0\n"
         << "MOV R21, 0\n"
         << "MOV R22, " << (h * w) << "\n"
         << "gap_loop:\n"
         << "CMP.LT R23, R21, R22\n"
         << "BRA gap_body, R23, gap_end\n"
         << "BRA gap_end\n"
         << "gap_body:\n"
         << "IDIV R24, R21, " << w << "\n"
         << "MUL R25, R24, " << w << "\n"
         << "SUB R26, R21, R25\n"
         << "MOV R4, " << h << "\n"
         << "MUL R4, R1, R4\n"
         << "ADD R4, R4, R24\n"
         << "MUL R4, R4, " << w << "\n"
         << "ADD R4, R4, R26\n"
         << "MUL R4, R4, 4\n"
         << "ADD R4, R4, " << inputBase << "\n"
         << "LDG R6, [R4]\n"
         << "ADD R20, R20, R6\n"
         << "ADD R21, R21, 1\n"
         << "BRA gap_loop\n"
         << "gap_end:\n";
      os << "DIV R20, R20, " << (h * w) << "\n"
         << "MUL R4, R1, 4\n"
         << "ADD R4, R4, " << outputBase << "\n"
         << "STG [R4], R20\n";
      emitCnnFinish(os);
      os.flush();

      IsaText isa;
      isa.code = code;
      isa.grid = std::to_string((total + 31) / 32) + "x1x1";
      isa.block = "32x1x1";
      return isa;
    }
    return std::nullopt;
  }

  static int64_t memrefByteSize(MemRefType type) {
    int64_t elements = 1;
    for (int64_t dim : type.getShape())
      elements *= dim;
    return elements * (type.getElementTypeBitWidth() / 8);
  }

  std::optional<IsaText> emitRopeIsa(mgpu::LaunchOp launch) {
    SmallVector<mgpu::RopeOp> ops;
    launch.getBody().front().walk([&](mgpu::RopeOp op) { ops.push_back(op); });
    if (ops.empty())
      return std::nullopt;

    auto op = ops.front();
    auto xType = dyn_cast<MemRefType>(op.getX().getType());
    auto cosType = dyn_cast<MemRefType>(op.getCos().getType());
    auto sinType = dyn_cast<MemRefType>(op.getSin().getType());
    auto outType = dyn_cast<MemRefType>(op.getOutput().getType());
    if (!xType || !cosType || !sinType || !outType || xType.getRank() < 1 ||
        xType.getRank() > 4)
      return std::nullopt;

    int64_t total = 1;
    for (int64_t dim : xType.getShape())
      total *= dim;
    int64_t dim = xType.getShape().back();
    if (dim % 2 != 0)
      return std::nullopt;
    int64_t leading = total / dim;
    int64_t cosDim = cosType.getNumElements() / leading;
    int64_t sinDim = sinType.getNumElements() / leading;
    if ((cosDim != dim && cosDim != dim / 2) ||
        (sinDim != dim && sinDim != dim / 2))
      return std::nullopt;
    int64_t cosPairStride = cosDim == dim ? 2 : 1;
    int64_t sinPairStride = sinDim == dim ? 2 : 1;
    int64_t xBase = bufferMap.base(op.getX());
    int64_t cosBase = bufferMap.base(op.getCos());
    int64_t sinBase = bufferMap.base(op.getSin());
    int64_t outBase = bufferMap.base(op.getOutput());

    std::string code;
    llvm::raw_string_ostream os(code);
    emitCnnPrologue(os, total);
    os << "IDIV R1, R0, " << dim << "\n"
       << "MUL R2, R1, " << dim << "\n"
       << "SUB R3, R0, R2\n"
       << "IDIV R4, R3, 2\n"
       << "MUL R5, R4, 2\n"
       << "SUB R6, R3, R5\n"
       << "MUL R7, R1, " << dim << "\n"
       << "ADD R7, R7, R5\n"
       << "MUL R7, R7, 4\n"
       << "ADD R7, R7, " << xBase << "\n"
       << "LDG R9, [R7]\n"
       << "ADD R10, R7, 4\n"
       << "LDG R11, [R10]\n"
       << "MUL R12, R1, " << cosDim << "\n"
       << "MUL R13, R4, " << cosPairStride << "\n"
       << "ADD R12, R12, R13\n"
       << "MUL R12, R12, 4\n"
       << "ADD R12, R12, " << cosBase << "\n"
       << "LDG R14, [R12]\n"
       << "MUL R15, R1, " << sinDim << "\n"
       << "MUL R16, R4, " << sinPairStride << "\n"
       << "ADD R15, R15, R16\n"
       << "MUL R15, R15, 4\n"
       << "ADD R15, R15, " << sinBase << "\n"
       << "LDG R16, [R15]\n"
       << "MUL R17, R9, R14\n"
       << "MUL R18, R11, R16\n"
       << "SUB R19, R17, R18\n"
       << "MUL R20, R9, R16\n"
       << "MUL R21, R11, R14\n"
       << "ADD R22, R20, R21\n"
       << "CMP.EQ R23, R6, 0\n"
       << "SEL R24, R23, R19, R22\n"
       << "MUL R25, R0, 4\n"
       << "ADD R25, R25, " << outBase << "\n"
       << "STG [R25], R24\n";
    emitCnnFinish(os);
    os.flush();

    IsaText isa;
    isa.code = code;
    isa.grid = std::to_string((total + 31) / 32) + "x1x1";
    isa.block = "32x1x1";
    return isa;
  }

  std::optional<IsaText> emitCausalMaskIsa(mgpu::LaunchOp launch) {
    SmallVector<mgpu::CausalMaskOp> ops;
    launch.getBody().front().walk(
        [&](mgpu::CausalMaskOp op) { ops.push_back(op); });
    if (ops.empty())
      return std::nullopt;

    auto outType = dyn_cast<MemRefType>(ops.front().getOutput().getType());
    if (!outType || outType.getRank() != 4 || outType.getShape()[0] != 1 ||
        outType.getShape()[1] != 1)
      return std::nullopt;
    int64_t qLen = outType.getShape()[2];
    int64_t kLen = outType.getShape()[3];
    int64_t total = qLen * kLen;
    int64_t outputBase = bufferMap.base(ops.front().getOutput());

    std::string code;
    llvm::raw_string_ostream os(code);
    emitCnnPrologue(os, total);
    os << "IDIV R1, R0, " << kLen << "\n"
       << "MUL R2, R1, " << kLen << "\n"
       << "SUB R3, R0, R2\n"
       << "CMP.GE R4, R1, R3\n"
       << "MUL R5, R0, 4\n"
       << "ADD R5, R5, " << outputBase << "\n"
       << "STG [R5], R4\n";
    emitCnnFinish(os);
    os.flush();

    IsaText isa;
    isa.code = code;
    isa.grid = std::to_string((total + 31) / 32) + "x1x1";
    isa.block = "32x1x1";
    return isa;
  }

  std::optional<IsaText> emitKVCacheAppendIsa(mgpu::LaunchOp launch) {
    SmallVector<mgpu::KVCacheAppendOp> ops;
    launch.getBody().front().walk(
        [&](mgpu::KVCacheAppendOp op) { ops.push_back(op); });
    if (ops.empty())
      return std::nullopt;

    auto op = ops.front();
    auto cacheType = dyn_cast<MemRefType>(op.getCache().getType());
    auto valueType = dyn_cast<MemRefType>(op.getValue().getType());
    auto outType = dyn_cast<MemRefType>(op.getOutput().getType());
    if (!cacheType || !valueType || !outType)
      return std::nullopt;
    int64_t cacheBase = bufferMap.base(op.getCache());
    int64_t valueBase = bufferMap.base(op.getValue());
    int64_t outBase = bufferMap.base(op.getOutput());
    int64_t total = 1;
    for (int64_t dim : outType.getShape())
      total *= dim;

    std::string code;
    llvm::raw_string_ostream os(code);
    emitCnnPrologue(os, total);
    os << "MOV R3, " << valueBase << "\n"
       << "MUL R2, R0, 4\n"
       << "ADD R3, R3, R2\n"
       << "LDG R5, [R3]\n"
       << "MOV R3, " << outBase << "\n"
       << "ADD R3, R3, R2\n"
       << "STG [R3], R5\n";
    emitCnnFinish(os);
    os.flush();

    IsaText isa;
    isa.code = code;
    isa.grid = std::to_string((total + 31) / 32) + "x1x1";
    isa.block = "32x1x1";
    return isa;
  }

  std::optional<IsaText> emitKVCacheGatherIsa(mgpu::LaunchOp launch) {
    SmallVector<mgpu::KVCacheGatherOp> ops;
    launch.getBody().front().walk(
        [&](mgpu::KVCacheGatherOp op) { ops.push_back(op); });
    if (ops.empty())
      return std::nullopt;

    auto op = ops.front();
    auto cacheType = dyn_cast<MemRefType>(op.getCache().getType());
    auto positionsType = dyn_cast<MemRefType>(op.getPositions().getType());
    auto outType = dyn_cast<MemRefType>(op.getOutput().getType());
    if (!cacheType || !positionsType || !outType || cacheType.getRank() != 4 ||
        outType.getRank() != 4 || outType.getShape()[0] != 1 ||
        outType.getShape()[1] != 1)
      return std::nullopt;

    int64_t cacheBase = bufferMap.base(op.getCache());
    int64_t positionsBase = bufferMap.base(op.getPositions());
    int64_t outBase = bufferMap.base(op.getOutput());
    int64_t dim = cacheType.getShape()[3];
    int64_t total = outType.getShape()[2] * dim;

    std::string code;
    llvm::raw_string_ostream os(code);
    emitCnnPrologue(os, total);
    os << "IDIV R1, R0, " << dim << "\n"
       << "MUL R2, R1, " << dim << "\n"
       << "SUB R3, R0, R2\n"
       << "MOV R4, 0\n"
       << "MUL R4, R4, 4\n"
       << "ADD R4, R4, " << positionsBase << "\n"
       << "LDG R6, [R4]\n"
       << "MOV R7, " << dim << "\n"
       << "MUL R7, R6, R7\n"
       << "ADD R7, R7, R3\n"
       << "MUL R7, R7, 4\n"
       << "ADD R7, R7, " << cacheBase << "\n"
       << "LDG R9, [R7]\n"
       << "MUL R10, R0, 4\n"
       << "ADD R10, R10, " << outBase << "\n"
       << "STG [R10], R9\n";
    emitCnnFinish(os);
    os.flush();

    IsaText isa;
    isa.code = code;
    isa.grid = std::to_string((total + 31) / 32) + "x1x1";
    isa.block = "32x1x1";
    return isa;
  }

  std::optional<IsaText> emitSoftmaxIsa(mgpu::LaunchOp launch) {
    SmallVector<mgpu::SoftmaxOp> ops;
    launch.getBody().front().walk([&](mgpu::SoftmaxOp op) { ops.push_back(op); });
    if (ops.empty())
      return std::nullopt;

    auto xType = dyn_cast<MemRefType>(ops.front().getX().getType());
    auto outType = dyn_cast<MemRefType>(ops.front().getOutput().getType());
    if (!xType || !outType || xType.getRank() != 2 || outType.getRank() != 2)
      return std::nullopt;
    int64_t rows = xType.getShape()[0];
    int64_t dim = xType.getShape()[1];
    int64_t xBase = bufferMap.base(ops.front().getX());
    int64_t outBase = bufferMap.base(ops.front().getOutput());

    std::string code;
    llvm::raw_string_ostream os(code);
    emitCnnPrologue(os, rows);
    os << "MOV R20, -1000000000.0\n";
    for (int64_t d = 0; d < dim; ++d) {
      os << "MOV R2, " << dim << "\n"
         << "MUL R2, R0, R2\n"
         << "ADD R2, R2, " << d << "\n"
         << "MUL R2, R2, 4\n"
         << "ADD R2, R2, " << xBase << "\n"
         << "LDG R4, [R2]\n"
         << "MAX R20, R20, R4\n";
    }
    os << "MOV R21, 0\n";
    for (int64_t d = 0; d < dim; ++d) {
      os << "MOV R2, " << dim << "\n"
         << "MUL R2, R0, R2\n"
         << "ADD R2, R2, " << d << "\n"
         << "MUL R2, R2, 4\n"
         << "ADD R2, R2, " << xBase << "\n"
         << "LDG R4, [R2]\n"
         << "SUB R4, R4, R20\n"
         << "MUL R4, R4, 1.4426950408889634\n"
         << "EXP2 R5, R4\n"
         << "ADD R21, R21, R5\n";
    }
    for (int64_t d = 0; d < dim; ++d) {
      os << "MOV R2, " << dim << "\n"
         << "MUL R2, R0, R2\n"
         << "ADD R2, R2, " << d << "\n"
         << "MUL R2, R2, 4\n"
         << "ADD R2, R2, " << xBase << "\n"
         << "LDG R4, [R2]\n"
         << "SUB R4, R4, R20\n"
         << "MUL R4, R4, 1.4426950408889634\n"
         << "EXP2 R5, R4\n"
         << "RCP R6, R21\n"
         << "MUL R5, R5, R6\n"
         << "MOV R2, " << dim << "\n"
         << "MUL R2, R0, R2\n"
         << "ADD R2, R2, " << d << "\n"
         << "MUL R2, R2, 4\n"
         << "ADD R2, R2, " << outBase << "\n"
         << "STG [R2], R5\n";
    }
    emitCnnFinish(os);
    os.flush();

    IsaText isa;
    isa.code = code;
    isa.grid = std::to_string((rows + 31) / 32) + "x1x1";
    isa.block = "32x1x1";
    return isa;
  }

  std::optional<IsaText> emitMaskedSoftmaxIsa(mgpu::LaunchOp launch) {
    SmallVector<mgpu::MaskedSoftmaxOp> ops;
    launch.getBody().front().walk(
        [&](mgpu::MaskedSoftmaxOp op) { ops.push_back(op); });
    if (ops.empty())
      return std::nullopt;

    auto op = ops.front();
    auto xType = dyn_cast<MemRefType>(op.getX().getType());
    auto maskType = dyn_cast<MemRefType>(op.getMask().getType());
    auto outType = dyn_cast<MemRefType>(op.getOutput().getType());
    if (!xType || !maskType || !outType || xType.getRank() != 2 ||
        maskType.getRank() != 2 || outType.getRank() != 2)
      return std::nullopt;
    if (xType.getShape() != maskType.getShape())
      return std::nullopt;
    int64_t rows = xType.getShape()[0];
    int64_t dim = xType.getShape()[1];
    int64_t xBase = bufferMap.base(op.getX());
    int64_t maskBase = bufferMap.base(op.getMask());
    int64_t outBase = bufferMap.base(op.getOutput());

    std::string code;
    llvm::raw_string_ostream os(code);
    emitCnnPrologue(os, rows);

    os << "MOV R20, -1000000000.0\n";
    for (int64_t d = 0; d < dim; ++d) {
      os << "MOV R2, " << dim << "\n"
         << "MUL R2, R0, R2\n"
         << "ADD R2, R2, " << d << "\n"
         << "MUL R2, R2, 4\n"
         << "ADD R2, R2, " << xBase << "\n"
         << typedLoadOpcode(xType) << " R4, [R2]\n"
         << "MOV R2, " << dim << "\n"
         << "MUL R2, R0, R2\n"
         << "ADD R2, R2, " << d << "\n"
         << "MUL R2, R2, 4\n"
         << "ADD R2, R2, " << maskBase << "\n"
         << typedLoadOpcode(maskType) << " R5, [R2]\n"
         << "CMP.EQ R6, R5, 0\n"
         << "MOV R7, -1000000000.0\n"
         << "SEL R4, R6, R7, R4\n"
         << "MAX R20, R20, R4\n";
    }

    os << "MOV R21, 0\n";
    for (int64_t d = 0; d < dim; ++d) {
      os << "MOV R2, " << dim << "\n"
         << "MUL R2, R0, R2\n"
         << "ADD R2, R2, " << d << "\n"
         << "MUL R2, R2, 4\n"
         << "ADD R2, R2, " << xBase << "\n"
         << typedLoadOpcode(xType) << " R4, [R2]\n"
         << "MOV R2, " << dim << "\n"
         << "MUL R2, R0, R2\n"
         << "ADD R2, R2, " << d << "\n"
         << "MUL R2, R2, 4\n"
         << "ADD R2, R2, " << maskBase << "\n"
         << typedLoadOpcode(maskType) << " R5, [R2]\n"
         << "CMP.EQ R6, R5, 0\n"
         << "MOV R7, -1000000000.0\n"
         << "SEL R4, R6, R7, R4\n"
         << "SUB R4, R4, R20\n"
         << "MUL R4, R4, 1.4426950408889634\n"
         << "EXP2 R5, R4\n"
         << "ADD R21, R21, R5\n";
    }

    for (int64_t d = 0; d < dim; ++d) {
      os << "MOV R2, " << dim << "\n"
         << "MUL R2, R0, R2\n"
         << "ADD R2, R2, " << d << "\n"
         << "MUL R2, R2, 4\n"
         << "ADD R2, R2, " << xBase << "\n"
         << typedLoadOpcode(xType) << " R4, [R2]\n"
         << "MOV R2, " << dim << "\n"
         << "MUL R2, R0, R2\n"
         << "ADD R2, R2, " << d << "\n"
         << "MUL R2, R2, 4\n"
         << "ADD R2, R2, " << maskBase << "\n"
         << typedLoadOpcode(maskType) << " R5, [R2]\n"
         << "CMP.EQ R6, R5, 0\n"
         << "MOV R7, -1000000000.0\n"
         << "SEL R4, R6, R7, R4\n"
         << "SUB R4, R4, R20\n"
         << "MUL R4, R4, 1.4426950408889634\n"
         << "EXP2 R5, R4\n"
         << "RCP R6, R21\n"
         << "MUL R5, R5, R6\n"
         << "MOV R2, " << dim << "\n"
         << "MUL R2, R0, R2\n"
         << "ADD R2, R2, " << d << "\n"
         << "MUL R2, R2, 4\n"
         << "ADD R2, R2, " << outBase << "\n"
         << typedStoreOpcode(outType) << " [R2], R5\n";
    }

    emitCnnFinish(os);
    os.flush();

    IsaText isa;
    isa.code = code;
    isa.grid = std::to_string((rows + 31) / 32) + "x1x1";
    isa.block = "32x1x1";
    return isa;
  }

  std::optional<IsaText> emitLayerNormIsa(mgpu::LaunchOp launch) {
    SmallVector<mgpu::LayerNormOp> ops;
    launch.getBody().front().walk(
        [&](mgpu::LayerNormOp op) { ops.push_back(op); });
    if (ops.empty())
      return std::nullopt;

    auto op = ops.front();
    auto xType = dyn_cast<MemRefType>(op.getX().getType());
    auto weightType = dyn_cast<MemRefType>(op.getWeight().getType());
    auto biasType = dyn_cast<MemRefType>(op.getBias().getType());
    auto outType = dyn_cast<MemRefType>(op.getOutput().getType());
    if (!xType || !weightType || !biasType || !outType || xType.getRank() != 2)
      return std::nullopt;
    int64_t rows = xType.getShape()[0];
    int64_t dim = xType.getShape()[1];
    int64_t xBase = bufferMap.base(op.getX());
    int64_t weightBase = bufferMap.base(op.getWeight());
    int64_t biasBase = bufferMap.base(op.getBias());
    int64_t outBase = bufferMap.base(op.getOutput());
    double eps = 1e-5;

    std::string code;
    llvm::raw_string_ostream os(code);
    emitCnnPrologue(os, rows);
    os << "MOV R20, 0\n"
       << "MOV R21, 0\n";
    for (int64_t d = 0; d < dim; ++d) {
      os << "MOV R2, " << dim << "\n"
         << "MUL R2, R0, R2\n"
         << "ADD R2, R2, " << d << "\n"
         << "MUL R2, R2, 4\n"
         << "ADD R2, R2, " << xBase << "\n"
         << "LDG R4, [R2]\n"
         << "ADD R20, R20, R4\n"
         << "MUL R5, R4, R4\n"
         << "ADD R21, R21, R5\n";
    }
    os << "MOV R2, " << dim << "\n"
       << "DIV R22, R20, R2\n"
       << "DIV R23, R21, R2\n"
       << "MUL R24, R22, R22\n"
       << "SUB R23, R23, R24\n"
       << "ADD R23, R23, " << eps << "\n"
       << "RSQRT R24, R23\n";
    for (int64_t d = 0; d < dim; ++d) {
      os << "MOV R2, " << dim << "\n"
         << "MUL R2, R0, R2\n"
         << "ADD R2, R2, " << d << "\n"
         << "MUL R2, R2, 4\n"
         << "ADD R2, R2, " << xBase << "\n"
         << "LDG R4, [R2]\n"
         << "SUB R4, R4, R22\n"
         << "MUL R4, R4, R24\n"
         << "MOV R5, " << d << "\n"
         << "MUL R5, R5, 4\n"
         << "ADD R5, R5, " << weightBase << "\n"
         << "LDG R7, [R5]\n"
         << "MUL R4, R4, R7\n"
         << "MOV R5, " << d << "\n"
         << "MUL R5, R5, 4\n"
         << "ADD R5, R5, " << biasBase << "\n"
         << "LDG R7, [R5]\n"
         << "ADD R4, R4, R7\n"
         << "MOV R2, " << dim << "\n"
         << "MUL R2, R0, R2\n"
         << "ADD R2, R2, " << d << "\n"
         << "MUL R2, R2, 4\n"
         << "ADD R2, R2, " << outBase << "\n"
         << "STG [R2], R4\n";
    }
    emitCnnFinish(os);
    os.flush();

    IsaText isa;
    isa.code = code;
    isa.grid = std::to_string((rows + 31) / 32) + "x1x1";
    isa.block = "32x1x1";
    return isa;
  }

  std::optional<IsaText> emitRMSNormIsa(mgpu::LaunchOp launch) {
    SmallVector<mgpu::RMSNormOp> ops;
    launch.getBody().front().walk(
        [&](mgpu::RMSNormOp op) { ops.push_back(op); });
    if (ops.empty())
      return std::nullopt;

    auto op = ops.front();
    auto xType = dyn_cast<MemRefType>(op.getX().getType());
    auto weightType = dyn_cast<MemRefType>(op.getWeight().getType());
    auto outType = dyn_cast<MemRefType>(op.getOutput().getType());
    if (!xType || !weightType || !outType || xType.getRank() != 2)
      return std::nullopt;
    int64_t rows = xType.getShape()[0];
    int64_t dim = xType.getShape()[1];
    int64_t xBase = bufferMap.base(op.getX());
    int64_t weightBase = bufferMap.base(op.getWeight());
    int64_t outBase = bufferMap.base(op.getOutput());
    double eps = 1e-5;

    std::string code;
    llvm::raw_string_ostream os(code);
    emitCnnPrologue(os, rows);
    os << "MOV R20, 0\n";
    for (int64_t d = 0; d < dim; ++d) {
      os << "MOV R2, " << dim << "\n"
         << "MUL R2, R0, R2\n"
         << "ADD R2, R2, " << d << "\n"
         << "MUL R2, R2, 4\n"
         << "ADD R2, R2, " << xBase << "\n"
         << "LDG R4, [R2]\n"
         << "MUL R4, R4, R4\n"
         << "ADD R20, R20, R4\n";
    }
    os << "MOV R2, " << dim << "\n"
       << "DIV R21, R20, R2\n"
       << "ADD R21, R21, " << eps << "\n"
       << "RSQRT R22, R21\n";
    for (int64_t d = 0; d < dim; ++d) {
      os << "MOV R2, " << dim << "\n"
         << "MUL R2, R0, R2\n"
         << "ADD R2, R2, " << d << "\n"
         << "MUL R2, R2, 4\n"
         << "ADD R2, R2, " << xBase << "\n"
         << "LDG R4, [R2]\n"
         << "MUL R4, R4, R22\n"
         << "MOV R5, " << d << "\n"
         << "MUL R5, R5, 4\n"
         << "ADD R5, R5, " << weightBase << "\n"
         << "LDG R7, [R5]\n"
         << "MUL R4, R4, R7\n"
         << "MOV R2, " << dim << "\n"
         << "MUL R2, R0, R2\n"
         << "ADD R2, R2, " << d << "\n"
         << "MUL R2, R2, 4\n"
         << "ADD R2, R2, " << outBase << "\n"
         << "STG [R2], R4\n";
    }
    emitCnnFinish(os);
    os.flush();

    IsaText isa;
    isa.code = code;
    isa.grid = std::to_string((rows + 31) / 32) + "x1x1";
    isa.block = "32x1x1";
    return isa;
  }

  std::optional<IsaText> emitReshapeIsa(mgpu::LaunchOp launch) {
    SmallVector<mgpu::ReshapeOp> ops;
    launch.getBody().front().walk([&](mgpu::ReshapeOp op) { ops.push_back(op); });
    if (ops.empty())
      return std::nullopt;

    auto inputType = dyn_cast<MemRefType>(ops.front().getInput().getType());
    auto outType = dyn_cast<MemRefType>(ops.front().getOutput().getType());
    if (!inputType || !outType)
      return std::nullopt;
    int64_t n = 1;
    for (int64_t dim : outType.getShape())
      n *= dim;
    int64_t inputBase = bufferMap.base(ops.front().getInput());
    int64_t outBase = bufferMap.base(ops.front().getOutput());

    std::string code;
    llvm::raw_string_ostream os(code);
    emitCnnPrologue(os, n);
    os << "MUL R3, R0, 4\n"
       << "ADD R3, R3, " << inputBase << "\n"
       << "LDG R5, [R3]\n"
       << "MUL R4, R0, 4\n"
       << "ADD R4, R4, " << outBase << "\n"
       << "STG [R4], R5\n";
    emitCnnFinish(os);
    os.flush();

    IsaText isa;
    isa.code = code;
    isa.grid = std::to_string((n + 31) / 32) + "x1x1";
    isa.block = "32x1x1";
    return isa;
  }

  std::optional<IsaText> emitTransposeIsa(mgpu::LaunchOp launch) {
    SmallVector<mgpu::TransposeOp> ops;
    launch.getBody().front().walk(
        [&](mgpu::TransposeOp op) { ops.push_back(op); });
    if (ops.empty())
      return std::nullopt;

    auto inputType = dyn_cast<MemRefType>(ops.front().getInput().getType());
    auto outType = dyn_cast<MemRefType>(ops.front().getOutput().getType());
    if (!inputType || !outType || inputType.getRank() != outType.getRank() ||
        inputType.getRank() < 2 || inputType.getRank() > 3)
      return std::nullopt;
    int64_t rank = inputType.getRank();
    SmallVector<int64_t> outputAxisForInput;
    for (int64_t inAxis = 0; inAxis < rank; ++inAxis) {
      int64_t found = -1;
      for (int64_t outAxis = 0; outAxis < rank; ++outAxis) {
        if (inputType.getShape()[inAxis] == outType.getShape()[outAxis]) {
          found = outAxis;
          break;
        }
      }
      if (found == -1)
        return std::nullopt;
      outputAxisForInput.push_back(found);
    }
    int64_t inputBase = bufferMap.base(ops.front().getInput());
    int64_t outBase = bufferMap.base(ops.front().getOutput());
    int64_t total = 1;
    for (int64_t dim : outType.getShape())
      total *= dim;

    std::string code;
    llvm::raw_string_ostream os(code);
    emitCnnPrologue(os, total);
    if (rank == 2) {
      os << "IDIV R1, R0, " << outType.getShape()[1] << "\n"
         << "MUL R2, R1, " << outType.getShape()[1] << "\n"
         << "SUB R3, R0, R2\n";
    } else {
      int64_t mid = outType.getShape()[1] * outType.getShape()[2];
      os << "IDIV R1, R0, " << mid << "\n"
         << "MUL R2, R1, " << mid << "\n"
         << "SUB R4, R0, R2\n"
         << "IDIV R3, R4, " << outType.getShape()[2] << "\n"
         << "MUL R5, R3, " << outType.getShape()[2] << "\n"
         << "SUB R6, R4, R5\n";
    }
    os << "MOV R20, 0\n";
    for (int64_t inAxis = 0; inAxis < rank; ++inAxis) {
      int64_t outAxis = outputAxisForInput[inAxis];
      int64_t stride = 1;
      for (int64_t next = inAxis + 1; next < rank; ++next)
        stride *= inputType.getShape()[next];
      std::string outReg = rank == 2 ? (outAxis == 0 ? "R1" : "R3")
                                     : (outAxis == 0 ? "R1"
                                                     : outAxis == 1 ? "R3"
                                                                    : "R6");
      if (stride != 1)
        os << "MUL R21, " << outReg << ", " << stride << "\n"
           << "ADD R20, R20, R21\n";
      else
        os << "ADD R20, R20, " << outReg << "\n";
    }
    os << "MUL R20, R20, 4\n"
       << "ADD R20, R20, " << inputBase << "\n"
       << "LDG R21, [R20]\n"
       << "MUL R22, R0, 4\n"
       << "ADD R22, R22, " << outBase << "\n"
       << "STG [R22], R21\n";
    emitCnnFinish(os);
    os.flush();

    IsaText isa;
    isa.code = code;
    isa.grid = std::to_string((total + 31) / 32) + "x1x1";
    isa.block = "32x1x1";
    return isa;
  }

  std::optional<IsaText> emitConcatIsa(mgpu::LaunchOp launch) {
    SmallVector<mgpu::ConcatOp> ops;
    launch.getBody().front().walk([&](mgpu::ConcatOp op) { ops.push_back(op); });
    if (ops.empty())
      return std::nullopt;

    auto op = ops.front();
    auto firstType = dyn_cast<MemRefType>(op.getFirst().getType());
    auto secondType = dyn_cast<MemRefType>(op.getSecond().getType());
    auto outType = dyn_cast<MemRefType>(op.getOutput().getType());
    if (!firstType || !secondType || !outType ||
        firstType.getRank() != secondType.getRank() ||
        firstType.getRank() != outType.getRank() ||
        firstType.getRank() < 1 || firstType.getRank() > 3)
      return std::nullopt;
    int64_t rank = firstType.getRank();
    int64_t axis = -1;
    for (int64_t a = 0; a < rank; ++a) {
      bool matches = true;
      for (int64_t d = 0; d < rank; ++d) {
        int64_t expected = d == a
                               ? firstType.getShape()[d] +
                                     secondType.getShape()[d]
                               : firstType.getShape()[d];
        if (expected != outType.getShape()[d]) {
          matches = false;
          break;
        }
      }
      if (matches) {
        axis = a;
        break;
      }
    }
    if (axis == -1)
      return std::nullopt;
    int64_t firstLen = firstType.getShape()[axis];
    int64_t n = outType.getNumElements();
    int64_t firstBase = bufferMap.base(op.getFirst());
    int64_t secondBase = bufferMap.base(op.getSecond());
    int64_t outBase = bufferMap.base(op.getOutput());

    std::string code;
    llvm::raw_string_ostream os(code);
    emitCnnPrologue(os, n);
    if (rank == 1) {
      os << "MOV R2, " << firstLen << "\n"
         << "CMP.LT R3, R0, R2\n"
         << "BRA first, R3, end\n"
         << "BRA second\n"
         << "first:\n"
         << "MUL R4, R0, 4\n"
         << "ADD R4, R4, " << firstBase << "\n"
         << "LDG R6, [R4]\n"
         << "BRA store\n"
         << "second:\n"
         << "MOV R4, R0\n"
         << "MOV R5, " << firstLen << "\n"
         << "SUB R4, R4, R5\n"
         << "MUL R4, R4, 4\n"
         << "ADD R4, R4, " << secondBase << "\n"
         << "LDG R6, [R4]\n"
         << "BRA store\n"
         << "store:\n"
         << "MUL R7, R0, 4\n"
         << "ADD R7, R7, " << outBase << "\n"
         << "STG [R7], R6\n";
    } else if (rank == 2) {
      int64_t cols = outType.getShape()[1];
      os << "IDIV R1, R0, " << cols << "\n"
         << "MUL R2, R1, " << cols << "\n"
         << "SUB R3, R0, R2\n";
      std::string outAxisReg = axis == 0 ? "R1" : "R3";
      os << "MOV R20, " << firstLen << "\n"
         << "CMP.LT R21, " << outAxisReg << ", R20\n"
         << "MOV R22, " << firstBase << "\n"
         << "MOV R23, " << secondBase << "\n"
         << "SEL R24, R21, R22, R23\n"
         << "MOV R25, " << outAxisReg << "\n"
         << "SUB R26, R25, R20\n"
         << "SEL R27, R21, R25, R26\n";
      if (axis == 0)
        os << "MUL R23, R27, " << cols << "\n"
           << "ADD R23, R23, R3\n";
      else
        os << "MUL R23, R1, " << cols << "\n"
           << "ADD R23, R23, R27\n";
      os << "MUL R23, R23, 4\n"
         << "ADD R23, R23, R24\n"
         << "LDG R24, [R23]\n"
         << "MUL R25, R0, 4\n"
         << "ADD R25, R25, " << outBase << "\n"
         << "STG [R25], R24\n";
    } else {
      return std::nullopt;
    }
    emitCnnFinish(os);
    os.flush();

    IsaText isa;
    isa.code = code;
    isa.grid = std::to_string((n + 31) / 32) + "x1x1";
    isa.block = "32x1x1";
    return isa;
  }

  std::optional<IsaText> emitSliceIsa(mgpu::LaunchOp launch) {
    SmallVector<mgpu::SliceOp> ops;
    launch.getBody().front().walk([&](mgpu::SliceOp op) { ops.push_back(op); });
    if (ops.empty())
      return std::nullopt;

    auto op = ops.front();
    auto inputType = dyn_cast<MemRefType>(op.getInput().getType());
    auto outType = dyn_cast<MemRefType>(op.getOutput().getType());
    if (!inputType || !outType || inputType.getRank() != outType.getRank() ||
        inputType.getRank() < 1 || inputType.getRank() > 3)
      return std::nullopt;
    int64_t rank = inputType.getRank();
    auto starts = op.getStarts();
    auto ends = op.getEnds();
    auto axes = op.getAxes();
    auto steps = op.getSteps();
    SmallVector<int64_t> startByAxis(rank, 0);
    SmallVector<int64_t> stepByAxis(rank, 1);
    for (unsigned i = 0; i < axes.size(); ++i)
      startByAxis[axes[i]] = starts[i];
    for (unsigned i = 0; i < axes.size(); ++i)
      stepByAxis[axes[i]] = steps[i];
    int64_t n = outType.getNumElements();
    int64_t inputBase = bufferMap.base(op.getInput());
    int64_t outBase = bufferMap.base(op.getOutput());

    std::string code;
    llvm::raw_string_ostream os(code);
    emitCnnPrologue(os, n);
    if (rank == 1) {
      os << "MUL R2, R0, " << stepByAxis[0] << "\n"
         << "ADD R2, R2, " << startByAxis[0] << "\n";
    } else if (rank == 2) {
      os << "IDIV R1, R0, " << outType.getShape()[1] << "\n"
         << "MUL R2, R1, " << outType.getShape()[1] << "\n"
         << "SUB R3, R0, R2\n"
         << "MUL R1, R1, " << stepByAxis[0] << "\n"
         << "ADD R1, R1, " << startByAxis[0] << "\n"
         << "MUL R3, R3, " << stepByAxis[1] << "\n"
         << "ADD R3, R3, " << startByAxis[1] << "\n"
         << "MUL R2, R1, " << inputType.getShape()[1] << "\n"
         << "ADD R2, R2, R3\n";
    } else {
      int64_t mid = outType.getShape()[1] * outType.getShape()[2];
      os << "IDIV R1, R0, " << mid << "\n"
         << "MUL R2, R1, " << mid << "\n"
         << "SUB R4, R0, R2\n"
         << "IDIV R3, R4, " << outType.getShape()[2] << "\n"
         << "MUL R5, R3, " << outType.getShape()[2] << "\n"
         << "SUB R6, R4, R5\n"
         << "MUL R1, R1, " << stepByAxis[0] << "\n"
         << "ADD R1, R1, " << startByAxis[0] << "\n"
         << "MUL R3, R3, " << stepByAxis[1] << "\n"
         << "ADD R3, R3, " << startByAxis[1] << "\n"
         << "MUL R6, R6, " << stepByAxis[2] << "\n"
         << "ADD R6, R6, " << startByAxis[2] << "\n"
         << "MUL R2, R1, " << inputType.getShape()[1] << "\n"
         << "ADD R2, R2, R3\n"
         << "MUL R2, R2, " << inputType.getShape()[2] << "\n"
         << "ADD R2, R2, R6\n";
    }
    os << "MUL R2, R2, 4\n"
       << "ADD R2, R2, " << inputBase << "\n"
       << "LDG R4, [R2]\n"
       << "MUL R5, R0, 4\n"
       << "ADD R5, R5, " << outBase << "\n"
       << "STG [R5], R4\n";
    emitCnnFinish(os);
    os.flush();

    IsaText isa;
    isa.code = code;
    isa.grid = std::to_string((n + 31) / 32) + "x1x1";
    isa.block = "32x1x1";
    return isa;
  }

  std::optional<IsaText> emitPadIsa(mgpu::LaunchOp launch) {
    SmallVector<mgpu::PadOp> ops;
    launch.getBody().front().walk([&](mgpu::PadOp op) { ops.push_back(op); });
    if (ops.empty())
      return std::nullopt;

    auto op = ops.front();
    auto inputType = dyn_cast<MemRefType>(op.getInput().getType());
    auto outType = dyn_cast<MemRefType>(op.getOutput().getType());
    if (inputType && outType && inputType.getRank() == 4 &&
        outType.getRank() == 4) {
      int64_t c = inputType.getShape()[1];
      int64_t h = inputType.getShape()[2];
      int64_t w = inputType.getShape()[3];
      int64_t cOut = outType.getShape()[1];
      int64_t oh = outType.getShape()[2];
      int64_t ow = outType.getShape()[3];
      int64_t beforeY = op.getBeforeY();
      int64_t afterY = op.getAfterY();
      int64_t beforeX = op.getBeforeX();
      int64_t afterX = op.getAfterX();
      if (c != cOut || oh != h + beforeY + afterY ||
          ow != w + beforeX + afterX)
        return std::nullopt;

      int64_t total = c * oh * ow;
      int64_t inputBase = bufferMap.base(op.getInput());
      int64_t outBase = bufferMap.base(op.getOutput());
      double value = op.getValue().convertToDouble();

      std::string code;
      llvm::raw_string_ostream os(code);
      emitCnnPrologue(os, total);
      os << "IDIV R1, R0, " << (oh * ow) << "\n"
         << "MUL R4, R1, " << (oh * ow) << "\n"
         << "SUB R5, R0, R4\n"
         << "IDIV R2, R5, " << ow << "\n"
         << "MUL R6, R2, " << ow << "\n"
         << "SUB R3, R5, R6\n"
         << "SUB R30, R2, " << beforeY << "\n"
         << "SUB R31, R3, " << beforeX << "\n"
         << "CMP.GE R4, R30, 0\n"
         << "MOV R6, " << h << "\n"
         << "CMP.LT R7, R30, R6\n"
         << "MUL R8, R4, R7\n"
         << "CMP.GE R9, R31, 0\n"
         << "MOV R10, " << w << "\n"
         << "CMP.LT R11, R31, R10\n"
         << "MUL R12, R9, R11\n"
         << "MUL R13, R8, R12\n"
         << "MOV R14, 0\n"
         << "MAX R15, R30, R14\n"
         << "MOV R14, " << (h - 1) << "\n"
         << "MIN R15, R15, R14\n"
         << "MOV R16, 0\n"
         << "MAX R17, R31, R16\n"
         << "MOV R16, " << (w - 1) << "\n"
         << "MIN R17, R17, R16\n"
         << "MOV R18, " << h << "\n"
         << "MUL R18, R1, R18\n"
         << "ADD R18, R18, R15\n"
         << "MUL R18, R18, " << w << "\n"
         << "ADD R18, R18, R17\n"
         << "MUL R18, R18, 4\n"
         << "ADD R18, R18, " << inputBase << "\n"
         << "LDG R20, [R18]\n"
         << "MOV R21, " << value << "\n"
         << "SEL R22, R13, R20, R21\n"
         << "MUL R18, R0, 4\n"
         << "ADD R18, R18, " << outBase << "\n"
         << "STG [R18], R22\n";
      emitCnnFinish(os);
      os.flush();

      IsaText isa;
      isa.code = code;
      isa.grid = std::to_string((total + 31) / 32) + "x1x1";
      isa.block = "32x1x1";
      return isa;
    }

    if (!inputType || !outType || inputType.getRank() != 1 ||
        outType.getRank() != 1)
      return std::nullopt;
    int64_t before = op.getBefore();
    int64_t inputLen = inputType.getShape()[0];
    int64_t n = outType.getShape()[0];
    int64_t inputBase = bufferMap.base(op.getInput());
    int64_t outBase = bufferMap.base(op.getOutput());
    double value = op.getValue().convertToDouble();

    std::string code;
    llvm::raw_string_ostream os(code);
    emitCnnPrologue(os, n);
    os << "MOV R3, " << before << "\n"
       << "SUB R4, R0, R3\n"
       << "CMP.GE R5, R4, 0\n"
       << "MOV R6, " << inputLen << "\n"
       << "CMP.LT R7, R4, R6\n"
       << "MUL R8, R5, R7\n"
       << "MOV R9, 0\n"
       << "MAX R10, R4, R9\n"
       << "MOV R9, " << (inputLen - 1) << "\n"
       << "MIN R10, R10, R9\n"
       << "MUL R10, R10, 4\n"
       << "ADD R10, R10, " << inputBase << "\n"
       << "LDG R12, [R10]\n"
       << "MOV R13, " << value << "\n"
       << "SEL R14, R8, R12, R13\n"
       << "MUL R10, R0, 4\n"
       << "ADD R10, R10, " << outBase << "\n"
       << "STG [R10], R14\n";
    emitCnnFinish(os);
    os.flush();

    IsaText isa;
    isa.code = code;
    isa.grid = std::to_string((n + 31) / 32) + "x1x1";
    isa.block = "32x1x1";
    return isa;
  }

  std::optional<IsaText> emitResizeIsa(mgpu::LaunchOp launch) {
    SmallVector<mgpu::ResizeOp> ops;
    launch.getBody().front().walk(
        [&](mgpu::ResizeOp op) { ops.push_back(op); });
    if (ops.empty())
      return std::nullopt;

    auto inputType = dyn_cast<MemRefType>(ops.front().getInput().getType());
    auto outType = dyn_cast<MemRefType>(ops.front().getOutput().getType());
    if (inputType && outType && inputType.getRank() == 4 &&
        outType.getRank() == 4) {
      int64_t n = inputType.getShape()[0];
      int64_t c = inputType.getShape()[1];
      int64_t h = inputType.getShape()[2];
      int64_t w = inputType.getShape()[3];
      int64_t cOut = outType.getShape()[1];
      int64_t oh = outType.getShape()[2];
      int64_t ow = outType.getShape()[3];
      if (n != 1 || c != cOut)
        return std::nullopt;
      int64_t total = c * oh * ow;
      int64_t inputBase = bufferMap.base(ops.front().getInput());
      int64_t outBase = bufferMap.base(ops.front().getOutput());

      std::string code;
      llvm::raw_string_ostream os(code);
      emitCnnPrologue(os, total);
      os << "IDIV R1, R0, " << (oh * ow) << "\n"
         << "MUL R2, R1, " << (oh * ow) << "\n"
         << "SUB R3, R0, R2\n"
         << "IDIV R2, R3, " << ow << "\n"
         << "MUL R4, R2, " << ow << "\n"
         << "SUB R5, R3, R4\n"
         << "MUL R6, R2, " << h << "\n"
         << "IDIV R6, R6, " << oh << "\n"
         << "MUL R7, R5, " << w << "\n"
         << "IDIV R7, R7, " << ow << "\n"
         << "MOV R8, R1\n"
         << "MUL R8, R8, " << h << "\n"
         << "ADD R8, R8, R6\n"
         << "MUL R8, R8, " << w << "\n"
         << "ADD R8, R8, R7\n"
         << "MUL R8, R8, 4\n"
         << "ADD R8, R8, " << inputBase << "\n"
         << "LDG R10, [R8]\n"
         << "MUL R11, R0, 4\n"
         << "ADD R11, R11, " << outBase << "\n"
         << "STG [R11], R10\n";
      emitCnnFinish(os);
      os.flush();

      IsaText isa;
      isa.code = code;
      isa.grid = std::to_string((total + 31) / 32) + "x1x1";
      isa.block = "32x1x1";
      return isa;
    }

    if (!inputType || !outType || inputType.getRank() != 1 ||
        outType.getRank() != 1)
      return std::nullopt;
    int64_t inputLen = inputType.getShape()[0];
    int64_t outputLen = outType.getShape()[0];
    int64_t inputBase = bufferMap.base(ops.front().getInput());
    int64_t outBase = bufferMap.base(ops.front().getOutput());

    std::string code;
    llvm::raw_string_ostream os(code);
    emitCnnPrologue(os, outputLen);
    os << "MOV R2, R0\n"
       << "MUL R2, R2, " << inputLen << "\n"
       << "MOV R3, " << outputLen << "\n"
       << "IDIV R4, R2, R3\n"
       << "MUL R4, R4, 4\n"
       << "ADD R4, R4, " << inputBase << "\n"
       << "LDG R6, [R4]\n"
       << "MUL R7, R0, 4\n"
       << "ADD R7, R7, " << outBase << "\n"
       << "STG [R7], R6\n";
    emitCnnFinish(os);
    os.flush();

    IsaText isa;
    isa.code = code;
    isa.grid = std::to_string((outputLen + 31) / 32) + "x1x1";
    isa.block = "32x1x1";
    return isa;
  }

  std::optional<IsaText> emitBatchNormIsa(mgpu::LaunchOp launch) {
    SmallVector<mgpu::BatchNormOp> ops;
    launch.getBody().front().walk(
        [&](mgpu::BatchNormOp op) { ops.push_back(op); });
    if (ops.empty())
      return std::nullopt;

    auto op = ops.front();
    auto xType = dyn_cast<MemRefType>(op.getX().getType());
    auto scaleType = dyn_cast<MemRefType>(op.getScale().getType());
    auto biasType = dyn_cast<MemRefType>(op.getBias().getType());
    auto meanType = dyn_cast<MemRefType>(op.getMean().getType());
    auto varType = dyn_cast<MemRefType>(op.getVar().getType());
    auto outType = dyn_cast<MemRefType>(op.getOutput().getType());
    if (!xType || !scaleType || !biasType || !meanType || !varType ||
        !outType || xType.getRank() != 4)
      return std::nullopt;

    int64_t c = xType.getShape()[1];
    int64_t h = xType.getShape()[2];
    int64_t w = xType.getShape()[3];
    int64_t hw = h * w;
    int64_t total = c * hw;
    int64_t xBase = bufferMap.base(op.getX());
    int64_t scaleBase = bufferMap.base(op.getScale());
    int64_t biasBase = bufferMap.base(op.getBias());
    int64_t meanBase = bufferMap.base(op.getMean());
    int64_t varBase = bufferMap.base(op.getVar());
    int64_t outBase = bufferMap.base(op.getOutput());
    double eps = 1e-5;

    std::string code;
    llvm::raw_string_ostream os(code);
    emitCnnPrologue(os, total);
    os << "IDIV R1, R0, " << hw << "\n"
       << "MUL R3, R1, " << hw << "\n"
       << "SUB R2, R0, R3\n"
       << "MUL R3, R1, 4\n"
       << "ADD R3, R3, " << meanBase << "\n"
       << "LDG R5, [R3]\n"
       << "MUL R6, R1, 4\n"
       << "ADD R6, R6, " << varBase << "\n"
       << "LDG R8, [R6]\n"
       << "ADD R8, R8, " << eps << "\n"
       << "RSQRT R8, R8\n"
       << "MUL R6, R0, 4\n"
       << "ADD R6, R6, " << xBase << "\n"
       << "LDG R9, [R6]\n"
       << "SUB R9, R9, R5\n"
       << "MUL R9, R9, R8\n"
       << "MUL R10, R1, 4\n"
       << "ADD R10, R10, " << scaleBase << "\n"
       << "LDG R12, [R10]\n"
       << "MUL R9, R9, R12\n"
       << "MUL R10, R1, 4\n"
       << "ADD R10, R10, " << biasBase << "\n"
       << "LDG R12, [R10]\n"
       << "ADD R9, R9, R12\n"
       << "MUL R10, R0, 4\n"
       << "ADD R10, R10, " << outBase << "\n"
       << "STG [R10], R9\n";
    emitCnnFinish(os);
    os.flush();

    IsaText isa;
    isa.code = code;
    isa.grid = std::to_string((total + 31) / 32) + "x1x1";
    isa.block = "32x1x1";
    return isa;
  }

  std::optional<IsaText> emitGroupNormIsa(mgpu::LaunchOp launch) {
    SmallVector<mgpu::GroupNormOp> ops;
    launch.getBody().front().walk(
        [&](mgpu::GroupNormOp op) { ops.push_back(op); });
    if (ops.empty())
      return std::nullopt;

    auto op = ops.front();
    auto xType = dyn_cast<MemRefType>(op.getX().getType());
    auto weightType = dyn_cast<MemRefType>(op.getWeight().getType());
    auto biasType = dyn_cast<MemRefType>(op.getBias().getType());
    auto outType = dyn_cast<MemRefType>(op.getOutput().getType());
    if (!xType || !weightType || !biasType || !outType || xType.getRank() != 4)
      return std::nullopt;

    int64_t c = xType.getShape()[1];
    int64_t h = xType.getShape()[2];
    int64_t w = xType.getShape()[3];
    int64_t groups = op.getNumGroups();
    if (c % groups != 0)
      return std::nullopt;
    int64_t channelsPerGroup = c / groups;
    int64_t hw = h * w;
    int64_t count = channelsPerGroup * hw;
    int64_t total = c * hw;
    int64_t xBase = bufferMap.base(op.getX());
    int64_t weightBase = bufferMap.base(op.getWeight());
    int64_t biasBase = bufferMap.base(op.getBias());
    int64_t outBase = bufferMap.base(op.getOutput());
    double eps = 1e-5;

    std::string code;
    llvm::raw_string_ostream os(code);
    emitCnnPrologue(os, total);
    os << "IDIV R1, R0, " << hw << "\n"
       << "MUL R3, R1, " << hw << "\n"
       << "SUB R2, R0, R3\n"
       << "IDIV R26, R1, " << channelsPerGroup << "\n"
       << "MOV R21, 0\n"
       << "MOV R22, 0\n";
    for (int64_t channel = 0; channel < c; ++channel) {
      int64_t group = channel / channelsPerGroup;
      for (int64_t ih = 0; ih < h; ++ih) {
        for (int64_t iw = 0; iw < w; ++iw) {
          int64_t idx = channel * hw + ih * w + iw;
          os << "MOV R3, " << group << "\n"
             << "CMP.EQ R4, R26, R3\n"
             << "MOV R3, " << idx << "\n"
             << "MUL R3, R3, 4\n"
             << "ADD R3, R3, " << xBase << "\n"
             << "LDG R6, [R3]\n"
             << "MUL R7, R6, R4\n"
             << "ADD R21, R21, R7\n"
             << "MUL R6, R6, R6\n"
             << "MUL R6, R6, R4\n"
             << "ADD R22, R22, R6\n";
        }
      }
    }
    os << "MOV R3, " << count << "\n"
       << "DIV R23, R21, R3\n"
       << "DIV R24, R22, R3\n"
       << "MUL R25, R23, R23\n"
       << "SUB R24, R24, R25\n"
       << "ADD R24, R24, " << eps << "\n"
       << "RSQRT R24, R24\n"
       << "IDIV R3, R2, " << w << "\n"
       << "MUL R4, R3, " << w << "\n"
       << "SUB R5, R2, R4\n"
       << "MOV R6, " << h << "\n"
       << "MUL R6, R1, R6\n"
       << "ADD R6, R6, R3\n"
       << "MUL R6, R6, " << w << "\n"
       << "ADD R6, R6, R5\n"
       << "MUL R6, R6, 4\n"
       << "ADD R6, R6, " << xBase << "\n"
       << "LDG R8, [R6]\n"
       << "SUB R8, R8, R23\n"
       << "MUL R8, R8, R24\n"
       << "MUL R9, R1, 4\n"
       << "ADD R9, R9, " << weightBase << "\n"
       << "LDG R11, [R9]\n"
       << "MUL R8, R8, R11\n"
       << "MUL R9, R1, 4\n"
       << "ADD R9, R9, " << biasBase << "\n"
       << "LDG R11, [R9]\n"
       << "ADD R8, R8, R11\n"
       << "MUL R9, R0, 4\n"
       << "ADD R9, R9, " << outBase << "\n"
       << "STG [R9], R8\n";
    emitCnnFinish(os);
    os.flush();

    IsaText isa;
    isa.code = code;
    isa.grid = std::to_string((total + 31) / 32) + "x1x1";
    isa.block = "32x1x1";
    return isa;
  }

  std::optional<IsaText> emitBoxDecodeIsa(mgpu::LaunchOp launch) {
    SmallVector<mgpu::BoxDecodeOp> ops;
    launch.getBody().front().walk(
        [&](mgpu::BoxDecodeOp op) { ops.push_back(op); });
    if (ops.empty())
      return std::nullopt;

    auto op = ops.front();
    auto boxesType = dyn_cast<MemRefType>(op.getBoxes().getType());
    auto deltasType = dyn_cast<MemRefType>(op.getDeltas().getType());
    auto outType = dyn_cast<MemRefType>(op.getOutput().getType());
    if (!boxesType || !deltasType || !outType || boxesType.getRank() != 2 ||
        boxesType.getShape()[1] != 4)
      return std::nullopt;
    int64_t n = boxesType.getShape()[0];
    int64_t total = n * 4;
    int64_t boxesBase = bufferMap.base(op.getBoxes());
    int64_t deltasBase = bufferMap.base(op.getDeltas());
    int64_t outBase = bufferMap.base(op.getOutput());

    std::string code;
    llvm::raw_string_ostream os(code);
    emitCnnPrologue(os, total);
    os << "IDIV R1, R0, 4\n"
       << "MUL R2, R1, 4\n"
       << "SUB R3, R0, R2\n";
    // Load box components and deltas into R20..R27.
    for (int i = 0; i < 4; ++i) {
      os << "MUL R4, R1, 4\n"
         << "ADD R4, R4, " << i << "\n"
         << "MUL R4, R4, 4\n"
         << "ADD R4, R4, " << boxesBase << "\n"
         << "LDG R" << (20 + i) << ", [R4]\n"
         << "MUL R4, R1, 4\n"
         << "ADD R4, R4, " << i << "\n"
         << "MUL R4, R4, 4\n"
         << "ADD R4, R4, " << deltasBase << "\n"
         << "LDG R" << (24 + i) << ", [R4]\n";
    }
    os << "MUL R4, R24, R22\n"
       << "ADD R4, R20, R4\n"
       << "MUL R5, R25, R23\n"
       << "ADD R5, R21, R5\n"
       << "MUL R6, R26, 1.4426950408889634\n"
       << "EXP2 R6, R6\n"
       << "MUL R7, R22, R6\n"
       << "MUL R8, R27, 1.4426950408889634\n"
       << "EXP2 R8, R8\n"
       << "MUL R9, R23, R8\n"
       << "MUL R10, R7, 0.5\n"
       << "SUB R28, R4, R10\n"
       << "ADD R29, R4, R10\n"
       << "MUL R11, R9, 0.5\n"
       << "SUB R30, R5, R11\n"
       << "ADD R31, R5, R11\n"
       << "MOV R27, 0\n"
       << "CMP.EQ R12, R3, 0\n"
       << "MUL R13, R12, R28\n"
       << "ADD R27, R27, R13\n"
       << "CMP.EQ R14, R3, 1\n"
       << "MUL R15, R14, R30\n"
       << "ADD R27, R27, R15\n"
       << "CMP.EQ R16, R3, 2\n"
       << "MUL R17, R16, R29\n"
       << "ADD R27, R27, R17\n"
       << "CMP.EQ R18, R3, 3\n"
       << "MUL R19, R18, R31\n"
       << "ADD R27, R27, R19\n"
       << "MUL R20, R0, 4\n"
       << "ADD R20, R20, " << outBase << "\n"
       << "STG [R20], R27\n";
    emitCnnFinish(os);
    os.flush();

    IsaText isa;
    isa.code = code;
    isa.grid = std::to_string((total + 31) / 32) + "x1x1";
    isa.block = "32x1x1";
    return isa;
  }

  std::optional<IsaText> emitNmsIsa(mgpu::LaunchOp launch) {
    SmallVector<mgpu::NMSOp> ops;
    launch.getBody().front().walk([&](mgpu::NMSOp op) { ops.push_back(op); });
    if (ops.empty())
      return std::nullopt;

    auto op = ops.front();
    auto boxesType = dyn_cast<MemRefType>(op.getBoxes().getType());
    auto scoresType = dyn_cast<MemRefType>(op.getScores().getType());
    auto outType = dyn_cast<MemRefType>(op.getOutput().getType());
    if (!boxesType || !scoresType || !outType || boxesType.getRank() != 2 ||
        boxesType.getShape()[1] != 4 || scoresType.getRank() != 1)
      return std::nullopt;
    int64_t numBoxes = boxesType.getShape()[0];
    int64_t boxesBase = bufferMap.base(op.getBoxes());
    int64_t scoresBase = bufferMap.base(op.getScores());
    int64_t outBase = bufferMap.base(op.getOutput());
    double threshold = op.getIouThreshold().convertToDouble();

    std::string code;
    llvm::raw_string_ostream os(code);
    emitCnnPrologue(os, numBoxes);
    os << "MOV R31, R0\n"
       << "MUL R4, R0, 4\n"
       << "ADD R4, R4, " << scoresBase << "\n"
       << "LDG R20, [R4]\n";
    for (int i = 0; i < 4; ++i) {
      os << "MUL R4, R0, 4\n"
         << "ADD R4, R4, " << i << "\n"
         << "MUL R4, R4, 4\n"
         << "ADD R4, R4, " << boxesBase << "\n"
         << "LDG R" << (21 + i) << ", [R4]\n";
    }
    os << "MOV R30, 1\n";
    for (int64_t j = 0; j < numBoxes; ++j) {
      os << "MOV R4, " << j << "\n"
         << "MUL R4, R4, 4\n"
         << "ADD R4, R4, " << scoresBase << "\n"
         << "LDG R6, [R4]\n"
         << "CMP.GT R7, R6, R20\n";
      for (int i = 0; i < 4; ++i) {
        os << "MOV R4, " << (j * 4 + i) << "\n"
           << "MUL R4, R4, 4\n"
           << "ADD R4, R4, " << boxesBase << "\n"
           << "LDG R" << (8 + i) << ", [R4]\n";
      }
      os << "MAX R12, R21, R8\n"
         << "MAX R13, R22, R9\n"
         << "MIN R14, R23, R10\n"
         << "MIN R15, R24, R11\n"
         << "SUB R16, R14, R12\n"
         << "MOV R17, 0\n"
         << "MAX R16, R16, R17\n"
         << "SUB R18, R15, R13\n"
         << "MAX R18, R18, R17\n"
         << "MUL R16, R16, R18\n"
         << "SUB R4, R23, R21\n"
         << "SUB R5, R24, R22\n"
         << "MUL R6, R4, R5\n"
         << "SUB R4, R10, R8\n"
         << "SUB R5, R11, R9\n"
         << "MUL R4, R4, R5\n"
         << "ADD R5, R6, R4\n"
         << "SUB R5, R5, R16\n"
         << "MOV R8, 1\n"
         << "MAX R5, R5, R8\n"
         << "DIV R9, R16, R5\n"
         << "CMP.GT R10, R9, " << threshold << "\n"
         << "MUL R11, R7, R10\n"
         << "MOV R12, 1\n"
         << "SUB R12, R12, R11\n"
         << "MUL R30, R30, R12\n";
    }
    os << "MOV R4, R31\n"
       << "MOV R5, -1\n"
       << "SEL R6, R30, R4, R5\n"
       << "MUL R7, R0, 4\n"
       << "ADD R7, R7, " << outBase << "\n"
       << "STG [R7], R6\n";
    emitCnnFinish(os);
    os.flush();

    IsaText isa;
    isa.code = code;
    isa.grid = std::to_string((numBoxes + 31) / 32) + "x1x1";
    isa.block = "32x1x1";
    return isa;
  }

  static void emitFragmentLoad(llvm::raw_ostream &os, int dest, int rowGroup,
                               int colGroup, int64_t base, int64_t stride) {
    os << "MOV R26, " << (rowGroup * 8) << "\n"
       << "ADD R26, R26, R24\n"
       << "MUL R26, R26, " << stride << "\n"
       << "MOV R27, " << (colGroup * 4) << "\n"
       << "ADD R27, R27, R25\n"
       << "ADD R26, R26, R27\n"
       << "MUL R26, R26, 4\n"
       << "ADD R26, R26, " << base << "\n"
       << "LDG R" << dest << ", [R26]\n";
  }

  static void emitFragmentStore(llvm::raw_ostream &os, int source, int rowGroup,
                                int colGroup, int64_t base, int64_t stride) {
    os << "MOV R26, " << (rowGroup * 8) << "\n"
       << "ADD R26, R26, R24\n"
       << "MUL R26, R26, " << stride << "\n"
       << "MOV R27, " << (colGroup * 4) << "\n"
       << "ADD R27, R27, R25\n"
       << "ADD R26, R26, R27\n"
       << "MUL R26, R26, 4\n"
       << "ADD R26, R26, " << base << "\n"
       << "STG [R26], R" << source << "\n";
  }

  static void emitInt8FragmentLoad(llvm::raw_ostream &os, int dest,
                                   int rowGroup, int colGroup, int64_t base,
                                   int64_t stride) {
    os << "MOV R26, " << (rowGroup * 8) << "\n"
       << "ADD R26, R26, R24\n"
       << "MUL R26, R26, " << stride << "\n"
       << "MOV R27, " << (colGroup * 4) << "\n"
       << "ADD R27, R27, R25\n"
       << "ADD R26, R26, R27\n"
       << "ADD R26, R26, " << base << "\n"
       << "LDG.I8 R" << dest << ", [R26]\n";
  }

  static void emitInt8FragmentStore(llvm::raw_ostream &os, int source,
                                    int rowGroup, int colGroup, int64_t base,
                                    int64_t stride) {
    os << "MOV R26, " << (rowGroup * 8) << "\n"
       << "ADD R26, R26, R24\n"
       << "MUL R26, R26, " << stride << "\n"
       << "MOV R27, " << (colGroup * 4) << "\n"
       << "ADD R27, R27, R25\n"
       << "ADD R26, R26, R27\n"
       << "MUL R26, R26, 4\n"
       << "ADD R26, R26, " << base << "\n"
       << "STG.I32 [R26], R" << source << "\n";
  }

  static void emitLaneIndex(llvm::raw_ostream &os) {
    os << "IDIV R24, tid.x, 4\n"
       << "MUL R25, R24, 4\n"
       << "SUB R25, tid.x, R25\n";
  }

  static void emitZeroRegisters(llvm::raw_ostream &os, int first, int last) {
    for (int reg = first; reg < last; ++reg)
      os << "MOV R" << reg << ", 0\n";
  }

  static void emitZeroTile(llvm::raw_ostream &os, int baseReg) {
    for (int offset = 0; offset < 8; ++offset)
      os << "MOV R" << (baseReg + offset) << ", 0\n";
  }

  static void emitLoadTileFromShared(llvm::raw_ostream &os, int destBase,
                                     int rowGroup, int colGroup,
                                     int64_t sharedOffset) {
    os << "MOV R26, " << (rowGroup * 8) << "\n"
       << "ADD R26, R26, R24\n"
       << "MOV R27, " << (colGroup * 4) << "\n"
       << "ADD R27, R27, R25\n"
       << "MUL R28, R26, 16\n"
       << "ADD R28, R28, R27\n"
       << "MUL R28, R28, 4\n"
       << "ADD R28, R28, " << sharedOffset << "\n"
       << "LDS R42, [R28]\n"
       << "MOV R" << destBase << ", R42\n";
  }

  static void emitStoreTileToShared(llvm::raw_ostream &os, int sourceBase,
                                    int rowGroup, int colGroup,
                                    int64_t sharedOffset) {
    os << "MOV R26, " << (rowGroup * 8) << "\n"
       << "ADD R26, R26, R24\n"
       << "MOV R27, " << (colGroup * 4) << "\n"
       << "ADD R27, R27, R25\n"
       << "MUL R28, R26, 16\n"
       << "ADD R28, R28, R27\n"
       << "MUL R28, R28, 4\n"
       << "ADD R28, R28, " << sharedOffset << "\n"
       << "STS [R28], R" << sourceBase << "\n";
  }

  static void emitQueryTileLoopStart(llvm::raw_ostream &os, int qTiles,
                                     const std::string &endLabel) {
    os << "MOV R40, 0\n"
       << "MOV R41, " << qTiles << "\n"
       << "query_tile_loop:\n"
       << "CMP.LT R42, R40, R41\n"
       << "BRA query_tile_body, R42, " << endLabel << "\n"
       << "BRA " << endLabel << "\n"
       << "query_tile_body:\n";
  }

  static void emitQueryTileLoopEnd(llvm::raw_ostream &os,
                                   const std::string &endLabel) {
    os << "ADD R40, R40, 1\n"
       << "BRA query_tile_loop\n"
       << endLabel << ":\n";
  }

  static void emitKeyTileLoopStart(llvm::raw_ostream &os, int kTiles,
                                   const std::string &endLabel) {
    os << "MOV R43, 0\n"
       << "MOV R44, " << kTiles << "\n"
       << "key_tile_loop:\n"
       << "CMP.LT R45, R43, R44\n"
       << "BRA key_tile_body, R45, " << endLabel << "\n"
       << "BRA " << endLabel << "\n"
       << "key_tile_body:\n";
  }

  static void emitKeyTileLoopEnd(llvm::raw_ostream &os,
                                 const std::string &endLabel) {
    os << "ADD R43, R43, 1\n"
       << "BRA key_tile_loop\n"
       << endLabel << ":\n";
  }

  static void emitMmaAccumulate(llvm::raw_ostream &os, int dstBase,
                                int aBase, int bBase, int cBase) {
    os << "MMA R" << dstBase << ", R" << aBase << ", R" << bBase << ", R"
       << cBase << "\n";
  }

  static void emitStoreFullTile(llvm::raw_ostream &os, int sourceBase,
                                int64_t sharedOffset) {
    for (int slot = 0; slot < 8; ++slot)
      emitStoreTileToShared(os, sourceBase + slot, slot / 4, slot % 4,
                            sharedOffset);
  }

  static void emitStoreSharedFragment(llvm::raw_ostream &os, int source,
                                      int rowGroup, int colGroup,
                                      int64_t offset) {
    os << "MOV R26, " << (rowGroup * 8) << "\n"
       << "ADD R26, R26, R24\n"
       << "MOV R27, " << (colGroup * 4) << "\n"
       << "ADD R27, R27, R25\n"
       << "MUL R28, R26, 16\n"
       << "ADD R28, R28, R27\n"
       << "MUL R28, R28, 4\n"
       << "ADD R28, R28, " << offset << "\n"
       << "STS [R28], R" << source << "\n";
  }

  std::optional<IsaText> emitMatmulIsa(mgpu::LaunchOp launch) {
    SmallVector<linalg::MatmulOp> matmuls;
    launch.getBody().front().walk(
        [&](linalg::MatmulOp matmul) { matmuls.push_back(matmul); });
    if (matmuls.empty())
      return std::nullopt;

    linalg::MatmulOp matmul = matmuls.front();
    auto dstOp = cast<DestinationStyleOpInterface>(matmul.getOperation());
    if (dstOp.getDpsInputs().size() < 2 || dstOp.getDpsInits().empty())
      return std::nullopt;

    auto aType = dyn_cast<MemRefType>(dstOp.getDpsInputs()[0].getType());
    auto bType = dyn_cast<MemRefType>(dstOp.getDpsInputs()[1].getType());
    auto cType = dyn_cast<MemRefType>(dstOp.getDpsInits()[0].getType());
    if (!aType || !bType || !cType)
      return std::nullopt;
    if (aType.getRank() != 2 || bType.getRank() != 2 || cType.getRank() != 2)
      return std::nullopt;

    int64_t m = aType.getShape()[0];
    int64_t k = aType.getShape()[1];
    int64_t kB = bType.getShape()[0];
    int64_t n = bType.getShape()[1];
    if (m != 16 || k != 16 || kB != 16 || n != 16)
      return std::nullopt;

    Type aElementType = aType.getElementType();
    Type bElementType = bType.getElementType();
    Type cElementType = cType.getElementType();
    bool isF32 = aElementType.isF32() && bElementType.isF32() &&
                 cElementType.isF32();
    auto aIntType = dyn_cast<IntegerType>(aElementType);
    auto bIntType = dyn_cast<IntegerType>(bElementType);
    auto cIntType = dyn_cast<IntegerType>(cElementType);
    bool isInt8 = aIntType && bIntType && cIntType &&
                  aIntType.getWidth() == 8 && bIntType.getWidth() == 8 &&
                  cIntType.getWidth() == 32;
    if (!isF32 && !isInt8)
      return std::nullopt;

    int64_t aBase = bufferMap.base(dstOp.getDpsInputs()[0]);
    int64_t bBase = bufferMap.base(dstOp.getDpsInputs()[1]);
    int64_t cBase = bufferMap.base(dstOp.getDpsInits()[0]);

    std::string code;
    llvm::raw_string_ostream os(code);
    os << "IDIV R24, tid.x, 4\n"
       << "MUL R25, R24, 4\n"
       << "SUB R25, tid.x, R25\n";
    for (int reg = 16; reg < 24; ++reg)
      os << "MOV R" << reg << ", 0\n";

    for (int slot = 0; slot < 8; ++slot) {
      int rowGroup = slot / 4;
      int colGroup = slot % 4;
      if (isInt8) {
        emitInt8FragmentLoad(os, slot, rowGroup, colGroup, aBase, 16);
        emitInt8FragmentLoad(os, 8 + slot, rowGroup, colGroup, bBase, 16);
      } else {
        emitFragmentLoad(os, slot, rowGroup, colGroup, aBase, 16);
        emitFragmentLoad(os, 8 + slot, rowGroup, colGroup, bBase, 16);
      }
    }

    os << (isInt8 ? "MMA.I8" : "MMA") << " R16, R0, R8, R16\n";

    for (int slot = 0; slot < 8; ++slot) {
      int rowGroup = slot / 4;
      int colGroup = slot % 4;
      if (isInt8)
        emitInt8FragmentStore(os, 16 + slot, rowGroup, colGroup, cBase, 16);
      else
        emitFragmentStore(os, 16 + slot, rowGroup, colGroup, cBase, 16);
    }
    os << "EXIT\n";
    os.flush();

    IsaText result;
    result.code = code;
    result.grid = "1x1x1";
    result.block = "32x1x1";
    return result;
  }

  std::optional<IsaText> emitScalarMatmulIsa(mgpu::LaunchOp launch) {
    SmallVector<linalg::MatmulOp> matmuls;
    launch.getBody().front().walk(
        [&](linalg::MatmulOp matmul) { matmuls.push_back(matmul); });
    if (matmuls.empty())
      return std::nullopt;

    linalg::MatmulOp matmul = matmuls.front();
    auto dstOp = cast<DestinationStyleOpInterface>(matmul.getOperation());
    if (dstOp.getDpsInputs().size() < 2 || dstOp.getDpsInits().empty())
      return std::nullopt;

    auto aType = dyn_cast<MemRefType>(dstOp.getDpsInputs()[0].getType());
    auto bType = dyn_cast<MemRefType>(dstOp.getDpsInputs()[1].getType());
    auto cType = dyn_cast<MemRefType>(dstOp.getDpsInits()[0].getType());
    if (!aType || !bType || !cType)
      return std::nullopt;
    if (aType.getRank() != 2 || bType.getRank() != 2 || cType.getRank() != 2)
      return std::nullopt;

    int64_t m = aType.getShape()[0];
    int64_t k = aType.getShape()[1];
    int64_t kB = bType.getShape()[0];
    int64_t n = bType.getShape()[1];
    int64_t total = m * n;
    if (k != kB)
      return std::nullopt;

    Type aElementType = aType.getElementType();
    Type bElementType = bType.getElementType();
    Type cElementType = cType.getElementType();
    if (isNibbleType(aElementType) || isNibbleType(bElementType) ||
        isNibbleType(cElementType))
      return std::nullopt;

    int64_t aStride = elementStride(aElementType);
    int64_t bStride = elementStride(bElementType);
    int64_t cStride = elementStride(cElementType);
    int64_t aBase = bufferMap.base(dstOp.getDpsInputs()[0]);
    int64_t bBase = bufferMap.base(dstOp.getDpsInputs()[1]);
    int64_t cBase = bufferMap.base(dstOp.getDpsInits()[0]);
    int64_t gridX = (total + 31) / 32;

    std::string code;
    llvm::raw_string_ostream os(code);
    os << "MOV R0, ctaid.x\n"
       << "MUL R0, R0, ntid.x\n"
       << "ADD R0, R0, tid.x\n"
       << "MOV R1, " << total << "\n"
       << "CMP.LT R2, R0, R1\n"
       << "BRA body, R2, end\n"
       << "BRA end\n"
       << "body:\n"
       << "MOV R1, " << n << "\n"
       << "IDIV R2, R0, R1\n"
       << "MUL R3, R2, R1\n"
       << "SUB R4, R0, R3\n"
       << "MOV R5, 0\n"
       << "MOV R6, 0\n"
       << "MOV R7, " << k << "\n"
       << "mat_loop:\n"
       << "CMP.LT R8, R6, R7\n"
       << "BRA mat_body, R8, mat_end\n"
       << "BRA mat_end\n"
       << "mat_body:\n"
       << "MUL R9, R2, R7\n"
       << "ADD R9, R9, R6\n"
       << "MUL R9, R9, " << aStride << "\n"
       << "ADD R9, R9, " << aBase << "\n"
       << typedLoadOpcode(aType) << " R10, [R9]\n"
       << "MUL R11, R6, R1\n"
       << "ADD R11, R11, R4\n"
       << "MUL R11, R11, " << bStride << "\n"
       << "ADD R11, R11, " << bBase << "\n"
       << typedLoadOpcode(bType) << " R12, [R11]\n"
       << "MUL R13, R10, R12\n"
       << "ADD R5, R5, R13\n"
       << "ADD R6, R6, 1\n"
       << "BRA mat_loop\n"
       << "mat_end:\n"
       << "MUL R9, R2, R1\n"
       << "ADD R9, R9, R4\n"
       << "MUL R9, R9, " << cStride << "\n"
       << "ADD R9, R9, " << cBase << "\n"
       << typedStoreOpcode(cType) << " [R9], R5\n"
       << "BRA end\n"
       << "end:\n"
       << "EXIT\n";
    os.flush();

    IsaText result;
    result.code = code;
    result.grid = std::to_string(gridX) + "x1x1";
    result.block = "32x1x1";
    return result;
  }

  std::optional<IsaText> emitLinearIsa(mgpu::LaunchOp launch) {
    SmallVector<mgpu::LinearOp> linears;
    launch.getBody().front().walk(
        [&](mgpu::LinearOp linear) { linears.push_back(linear); });
    if (linears.empty())
      return std::nullopt;

    auto linear = linears.front();
    auto inputType = dyn_cast<MemRefType>(linear.getInput().getType());
    auto weightType = dyn_cast<MemRefType>(linear.getWeight().getType());
    auto outputType = dyn_cast<MemRefType>(linear.getOutput().getType());
    if (!inputType || !weightType || !outputType)
      return std::nullopt;
    if (inputType.getRank() != 2 || weightType.getRank() != 2 ||
        outputType.getRank() != 2)
      return std::nullopt;

    int64_t m = inputType.getShape()[0];
    int64_t k = inputType.getShape()[1];
    int64_t n = weightType.getShape()[0];
    int64_t kWeight = weightType.getShape()[1];
    int64_t total = m * n;
    if (k != kWeight)
      return std::nullopt;
    if (isNibbleType(inputType.getElementType()) ||
        isNibbleType(weightType.getElementType()) ||
        isNibbleType(outputType.getElementType()))
      return std::nullopt;

    int64_t inputStride = elementStride(inputType.getElementType());
    int64_t weightStride = elementStride(weightType.getElementType());
    int64_t outputStride = elementStride(outputType.getElementType());
    int64_t inputBase = bufferMap.base(linear.getInput());
    int64_t weightBase = bufferMap.base(linear.getWeight());
    int64_t outputBase = bufferMap.base(linear.getOutput());
    int64_t biasBase =
        linear.getHasBias() ? bufferMap.base(linear.getBias()) : 0;
    auto biasType = linear.getHasBias()
                        ? dyn_cast<MemRefType>(linear.getBias().getType())
                        : nullptr;
    int64_t gridX = (total + 31) / 32;

    std::string code;
    llvm::raw_string_ostream os(code);
    os << "MOV R0, ctaid.x\n"
       << "MUL R0, R0, ntid.x\n"
       << "ADD R0, R0, tid.x\n"
       << "MOV R1, " << total << "\n"
       << "CMP.LT R2, R0, R1\n"
       << "BRA body, R2, end\n"
       << "BRA end\n"
       << "body:\n"
       << "MOV R1, " << n << "\n"
       << "IDIV R2, R0, R1\n"
       << "MUL R3, R2, R1\n"
       << "SUB R4, R0, R3\n"
       << "MOV R5, 0\n"
       << "MOV R6, 0\n"
       << "MOV R7, " << k << "\n"
       << "linear_loop:\n"
       << "CMP.LT R8, R6, R7\n"
       << "BRA linear_body, R8, linear_end\n"
       << "BRA linear_end\n"
       << "linear_body:\n"
       << "MUL R9, R2, R7\n"
       << "ADD R9, R9, R6\n"
       << "MUL R9, R9, " << inputStride << "\n"
       << "ADD R9, R9, " << inputBase << "\n"
       << typedLoadOpcode(inputType) << " R10, [R9]\n"
       << "MUL R11, R4, R7\n"
       << "ADD R11, R11, R6\n"
       << "MUL R11, R11, " << weightStride << "\n"
       << "ADD R11, R11, " << weightBase << "\n"
       << typedLoadOpcode(weightType) << " R12, [R11]\n"
       << "MUL R13, R10, R12\n"
       << "ADD R5, R5, R13\n"
       << "ADD R6, R6, 1\n"
       << "BRA linear_loop\n"
       << "linear_end:\n";

    if (linear.getHasBias()) {
      os << "MUL R9, R4, "
         << (biasType ? elementStride(biasType.getElementType()) : 4) << "\n"
         << "ADD R9, R9, " << biasBase << "\n"
         << (biasType ? typedLoadOpcode(biasType) : "LDG") << " R10, [R9]\n"
         << "ADD R5, R5, R10\n";
    }

    os << "MUL R9, R2, R1\n"
       << "ADD R9, R9, R4\n"
       << "MUL R9, R9, " << outputStride << "\n"
       << "ADD R9, R9, " << outputBase << "\n"
       << typedStoreOpcode(outputType) << " [R9], R5\n"
       << "BRA end\n"
       << "end:\n"
       << "EXIT\n";
    os.flush();

    IsaText result;
    result.code = code;
    result.grid = std::to_string(gridX) + "x1x1";
    result.block = "32x1x1";
    return result;
  }

  static void emitConvAFragment(
      llvm::raw_ostream &os, int dest, int rowGroup, int colGroup,
      int64_t imageBase, int64_t c, int64_t h, int64_t w, int64_t kh,
      int64_t kw, int64_t ow, int64_t spatialSize, int64_t reductionSize,
      int64_t strideY, int64_t strideX, int64_t padY, int64_t padX) {
    int64_t area = kh * kw;
    os << "MOV R26, " << (rowGroup * 8) << "\n"
       << "ADD R26, R26, R24\n"
       << "MOV R27, " << (colGroup * 4) << "\n"
       << "ADD R27, R27, R25\n"
       << "MOV R39, " << spatialSize << "\n"
       << "CMP.LT R28, R26, R39\n"
       << "MOV R39, " << reductionSize << "\n"
       << "CMP.LT R40, R27, R39\n"
       << "MUL R28, R28, R40\n"
       << "MOV R39, " << (spatialSize - 1) << "\n"
       << "MIN R29, R26, R39\n"
       << "MOV R39, " << (reductionSize - 1) << "\n"
       << "MIN R30, R27, R39\n"
       << "IDIV R31, R29, " << ow << "\n"
       << "MUL R32, R31, " << ow << "\n"
       << "SUB R32, R29, R32\n"
       << "IDIV R33, R30, " << area << "\n"
       << "MUL R34, R33, " << area << "\n"
       << "SUB R34, R30, R34\n"
       << "IDIV R35, R34, " << kw << "\n"
       << "MUL R36, R35, " << kw << "\n"
       << "SUB R36, R34, R36\n"
       << "MOV R37, " << strideY << "\n"
       << "MUL R37, R31, R37\n"
       << "ADD R37, R37, R35\n"
       << "SUB R37, R37, " << padY << "\n"
       << "MOV R38, " << strideX << "\n"
       << "MUL R38, R32, R38\n"
       << "ADD R38, R38, R36\n"
       << "SUB R38, R38, " << padX << "\n"
       << "MOV R39, 0\n"
       << "MAX R37, R37, R39\n"
       << "MOV R39, " << (h - 1) << "\n"
       << "MIN R37, R37, R39\n"
       << "MOV R39, 0\n"
       << "MAX R38, R38, R39\n"
       << "MOV R39, " << (w - 1) << "\n"
       << "MIN R38, R38, R39\n"
       << "MOV R41, " << h << "\n"
       << "MUL R41, R33, R41\n"
       << "ADD R41, R41, R37\n"
       << "MUL R41, R41, " << w << "\n"
       << "ADD R41, R41, R38\n"
       << "MUL R41, R41, 4\n"
       << "ADD R41, R41, " << imageBase << "\n"
       << "LDG R42, [R41]\n"
       << "MOV R39, 0\n"
       << "SEL R" << dest << ", R28, R42, R39\n";
  }

  static void emitConvBFragment(
      llvm::raw_ostream &os, int dest, int rowGroup, int colGroup,
      int64_t weightBase, int64_t c, int64_t kh, int64_t kw, int64_t f,
      int64_t reductionSize) {
    int64_t area = kh * kw;
    os << "MOV R26, " << (rowGroup * 8) << "\n"
       << "ADD R26, R26, R24\n"
       << "MOV R27, " << (colGroup * 4) << "\n"
       << "ADD R27, R27, R25\n"
       << "MOV R39, " << reductionSize << "\n"
       << "CMP.LT R28, R26, R39\n"
       << "MOV R39, " << f << "\n"
       << "CMP.LT R40, R27, R39\n"
       << "MUL R28, R28, R40\n"
       << "MOV R39, " << (reductionSize - 1) << "\n"
       << "MIN R29, R26, R39\n"
       << "MOV R39, " << (f - 1) << "\n"
       << "MIN R30, R27, R39\n"
       << "IDIV R33, R29, " << area << "\n"
       << "MUL R34, R33, " << area << "\n"
       << "SUB R34, R29, R34\n"
       << "IDIV R35, R34, " << kw << "\n"
       << "MUL R36, R35, " << kw << "\n"
       << "SUB R36, R34, R36\n"
       << "MOV R37, " << c << "\n"
       << "MUL R37, R30, R37\n"
       << "ADD R37, R37, R33\n"
       << "MUL R37, R37, " << kh << "\n"
       << "ADD R37, R37, R35\n"
       << "MUL R37, R37, " << kw << "\n"
       << "ADD R37, R37, R36\n"
       << "MUL R37, R37, 4\n"
       << "ADD R37, R37, " << weightBase << "\n"
       << "LDG R42, [R37]\n"
       << "MOV R39, 0\n"
       << "SEL R" << dest << ", R28, R42, R39\n";
  }

  std::optional<IsaText> emitConvIsa(mgpu::LaunchOp launch) {
    Value imageValue;
    Value weightValue;
    Value outputValue;
    Value biasValue;
    bool hasBias = false;
    int64_t padY = 0;
    int64_t padX = 0;
    int64_t strideY = 1;
    int64_t strideX = 1;
    int64_t dilY = 1;
    int64_t dilX = 1;

    SmallVector<mgpu::Conv2DOp> customConvs;
    launch.getBody().front().walk(
        [&](mgpu::Conv2DOp conv) { customConvs.push_back(conv); });
    if (!customConvs.empty()) {
      auto conv = customConvs.front();
      imageValue = conv.getImage();
      weightValue = conv.getWeight();
      outputValue = conv.getOutput();
      hasBias = conv.getHasBias();
      biasValue = hasBias ? conv.getBias() : imageValue;
      padY = conv.getPadY();
      padX = conv.getPadX();
      strideY = conv.getStrideY();
      strideX = conv.getStrideX();
      dilY = conv.getDilY();
      dilX = conv.getDilX();
    } else {
      SmallVector<linalg::Conv2DNchwFchwOp> convs;
      launch.getBody().front().walk(
          [&](linalg::Conv2DNchwFchwOp conv) { convs.push_back(conv); });
      if (convs.empty())
        return std::nullopt;
      auto conv = convs.front();
      auto dstOp = cast<DestinationStyleOpInterface>(conv.getOperation());
      if (dstOp.getDpsInputs().size() < 2 || dstOp.getDpsInits().empty())
        return std::nullopt;
      imageValue = dstOp.getDpsInputs()[0];
      weightValue = dstOp.getDpsInputs()[1];
      outputValue = dstOp.getDpsInits()[0];
      if (auto strides = conv.getStrides()) {
        SmallVector<int64_t, 2> values;
        for (int64_t value : strides.getValues<int64_t>())
          values.push_back(value);
        if (values.size() == 2) {
          strideY = values[0];
          strideX = values[1];
        }
      }
      if (auto dilations = conv.getDilations()) {
        SmallVector<int64_t, 2> values;
        for (int64_t value : dilations.getValues<int64_t>())
          values.push_back(value);
        if (values.size() == 2) {
          dilY = values[0];
          dilX = values[1];
        }
      }
    }

    auto imageType = dyn_cast<MemRefType>(imageValue.getType());
    auto weightType = dyn_cast<MemRefType>(weightValue.getType());
    auto outputType = dyn_cast<MemRefType>(outputValue.getType());
    auto biasType =
        hasBias ? dyn_cast<MemRefType>(biasValue.getType()) : nullptr;
    if (!imageType || !weightType || !outputType)
      return std::nullopt;
    if (imageType.getRank() != 4 || weightType.getRank() != 4 ||
        outputType.getRank() != 4)
      return std::nullopt;
    if (isNibbleType(imageType.getElementType()) ||
        isNibbleType(weightType.getElementType()) ||
        isNibbleType(outputType.getElementType()))
      return std::nullopt;

    int64_t n = imageType.getShape()[0];
    int64_t c = imageType.getShape()[1];
    int64_t h = imageType.getShape()[2];
    int64_t w = imageType.getShape()[3];
    int64_t f = weightType.getShape()[0];
    int64_t cWeight = weightType.getShape()[1];
    int64_t kh = weightType.getShape()[2];
    int64_t kw = weightType.getShape()[3];
    int64_t nOut = outputType.getShape()[0];
    int64_t fOut = outputType.getShape()[1];
    int64_t oh = outputType.getShape()[2];
    int64_t ow = outputType.getShape()[3];

    if (n != 1 || nOut != 1 || c != cWeight || f != fOut)
      return std::nullopt;
    int64_t spatialSize = oh * ow;
    int64_t reductionSize = c * kh * kw;
    bool typedConv = !imageType.getElementType().isF32() ||
                     !weightType.getElementType().isF32() ||
                     !outputType.getElementType().isF32();
    if (typedConv || padY != 0 || padX != 0 || strideY != 1 || strideX != 1 ||
        dilY != 1 || dilX != 1 || spatialSize > 16 || reductionSize > 16 ||
        f > 16)
      return emitScalarConvIsa(imageValue, weightValue, outputValue, biasValue,
                               biasType, hasBias, c, h, w, f, kh, kw, oh, ow,
                               padY, padX, strideY, strideX, dilY, dilX);

    int64_t imageBase = bufferMap.base(imageValue);
    int64_t weightBase = bufferMap.base(weightValue);
    int64_t outputBase = bufferMap.base(outputValue);
    int64_t biasBase = hasBias ? bufferMap.base(biasValue) : 0;

    std::string code;
    llvm::raw_string_ostream os(code);
    emitLaneIndex(os);
    emitZeroRegisters(os, 16, 24);
    for (int slot = 0; slot < 8; ++slot) {
      int rowGroup = slot / 4;
      int colGroup = slot % 4;
      emitConvAFragment(os, slot, rowGroup, colGroup, imageBase, c, h, w, kh,
                        kw, ow, spatialSize, reductionSize, 1, 1, 0, 0);
      emitConvBFragment(os, 8 + slot, rowGroup, colGroup, weightBase, c, kh, kw,
                        f, reductionSize);
    }
    os << "MMA R16, R0, R8, R16\n";
    for (int slot = 0; slot < 8; ++slot)
      emitStoreSharedFragment(os, 16 + slot, slot / 4, slot % 4, 0);
    os << "BAR.SYNC\n";

    os << "MOV R0, ctaid.x\n"
       << "MUL R0, R0, ntid.x\n"
       << "ADD R0, R0, tid.x\n"
       << "MOV R1, " << (spatialSize * f) << "\n"
       << "CMP.LT R2, R0, R1\n"
       << "BRA conv_body, R2, conv_end\n"
       << "BRA conv_end\n"
       << "conv_body:\n"
       << "IDIV R1, R0, " << spatialSize << "\n"
       << "MUL R2, R1, " << spatialSize << "\n"
       << "SUB R3, R0, R2\n"
       << "IDIV R4, R3, " << ow << "\n"
       << "MUL R5, R4, " << ow << "\n"
       << "SUB R6, R3, R5\n"
       << "MUL R7, R3, 16\n"
       << "ADD R7, R7, R1\n"
       << "MUL R7, R7, 4\n"
       << "LDS R8, [R7]\n";
    if (hasBias)
      os << "MUL R10, R1, 4\n"
         << "ADD R10, R10, " << biasBase << "\n"
         << "LDG R11, [R10]\n"
         << "ADD R8, R8, R11\n";
    os << "MUL R9, R1, " << oh << "\n"
       << "ADD R9, R9, R4\n"
       << "MUL R9, R9, " << ow << "\n"
       << "ADD R9, R9, R6\n"
       << "MUL R9, R9, 4\n"
       << "ADD R9, R9, " << outputBase << "\n"
       << "STG [R9], R8\n"
       << "BRA conv_end\n"
       << "conv_end:\n"
       << "EXIT\n";
    os.flush();

    IsaText result;
    result.code = code;
    result.grid = "1x1x1";
    result.block = "32x1x1";
    return result;
  }

  std::optional<IsaText> emitScalarConvIsa(
      Value imageValue, Value weightValue, Value outputValue, Value biasValue,
      MemRefType biasType, bool hasBias, int64_t c, int64_t h, int64_t w,
      int64_t f, int64_t kh, int64_t kw, int64_t oh, int64_t ow, int64_t padY,
      int64_t padX, int64_t strideY, int64_t strideX, int64_t dilY,
      int64_t dilX) {
    auto imageType = dyn_cast<MemRefType>(imageValue.getType());
    auto weightType = dyn_cast<MemRefType>(weightValue.getType());
    auto outputType = dyn_cast<MemRefType>(outputValue.getType());
    if (!imageType || !weightType || !outputType)
      return std::nullopt;
    int64_t imageStride = elementStride(imageType.getElementType());
    int64_t weightStride = elementStride(weightType.getElementType());
    int64_t outputStride = elementStride(outputType.getElementType());
    int64_t imageBase = bufferMap.base(imageValue);
    int64_t weightBase = bufferMap.base(weightValue);
    int64_t outputBase = bufferMap.base(outputValue);
    int64_t biasBase = hasBias ? bufferMap.base(biasValue) : 0;
    int64_t total = f * oh * ow;

    std::string code;
    llvm::raw_string_ostream os(code);
    emitCnnPrologue(os, total);
    os << "IDIV R1, R0, " << (oh * ow) << "\n"
       << "MUL R4, R1, " << (oh * ow) << "\n"
       << "SUB R5, R0, R4\n"
       << "IDIV R2, R5, " << ow << "\n"
       << "MUL R6, R2, " << ow << "\n"
       << "SUB R3, R5, R6\n"
       << "MOV R20, 0\n";

    int64_t area = kh * kw;
    int64_t reductionSize = c * area;
    os << "MOV R21, 0\n"
       << "MOV R22, " << reductionSize << "\n"
       << "conv_reduce_loop:\n"
       << "CMP.LT R23, R21, R22\n"
       << "BRA conv_reduce_body, R23, conv_reduce_end\n"
       << "BRA conv_reduce_end\n"
       << "conv_reduce_body:\n"
       << "IDIV R24, R21, " << area << "\n"
       << "MUL R25, R24, " << area << "\n"
       << "SUB R26, R21, R25\n"
       << "IDIV R27, R26, " << kw << "\n"
       << "MUL R28, R27, " << kw << "\n"
       << "SUB R29, R26, R28\n"
       << "MOV R30, " << strideY << "\n"
       << "MUL R30, R2, R30\n"
       << "MOV R32, " << dilY << "\n"
       << "MUL R32, R27, R32\n"
       << "ADD R30, R30, R32\n"
       << "SUB R30, R30, " << padY << "\n"
       << "CMP.GE R4, R30, 0\n"
       << "MOV R5, " << h << "\n"
       << "CMP.LT R6, R30, R5\n"
       << "MUL R7, R4, R6\n"
       << "MOV R8, 0\n"
       << "MAX R9, R30, R8\n"
       << "MOV R8, " << (h - 1) << "\n"
       << "MIN R10, R9, R8\n"
       << "MOV R31, " << strideX << "\n"
       << "MUL R31, R3, R31\n"
       << "MOV R33, " << dilX << "\n"
       << "MUL R33, R29, R33\n"
       << "ADD R31, R31, R33\n"
       << "SUB R31, R31, " << padX << "\n"
       << "CMP.GE R11, R31, 0\n"
       << "MOV R12, " << w << "\n"
       << "CMP.LT R13, R31, R12\n"
       << "MUL R14, R11, R13\n"
       << "MUL R15, R7, R14\n"
       << "MOV R16, 0\n"
       << "MAX R17, R31, R16\n"
       << "MOV R16, " << (w - 1) << "\n"
       << "MIN R18, R17, R16\n"
       << "MOV R19, " << h << "\n"
       << "MUL R19, R24, R19\n"
       << "ADD R19, R19, R10\n"
       << "MUL R19, R19, " << w << "\n"
       << "ADD R19, R19, R18\n"
       << "MUL R19, R19, " << imageStride << "\n"
       << "ADD R19, R19, " << imageBase << "\n"
       << typedLoadOpcode(imageType) << " R4, [R19]\n"
       << "MOV R5, 0\n"
       << "SEL R6, R15, R4, R5\n"
       << "MOV R7, " << c << "\n"
       << "MUL R7, R1, R7\n"
       << "ADD R7, R7, R24\n"
       << "MUL R7, R7, " << kh << "\n"
       << "ADD R7, R7, R27\n"
       << "MUL R7, R7, " << kw << "\n"
       << "ADD R7, R7, R29\n"
       << "MUL R7, R7, " << weightStride << "\n"
       << "ADD R7, R7, " << weightBase << "\n"
       << typedLoadOpcode(weightType) << " R8, [R7]\n"
       << "FMA R20, R6, R8, R20\n"
       << "ADD R21, R21, 1\n"
       << "BRA conv_reduce_loop\n"
       << "conv_reduce_end:\n";

    if (hasBias) {
      os << "MUL R4, R1, "
         << (biasType ? elementStride(biasType.getElementType()) : 4) << "\n"
         << "ADD R4, R4, " << biasBase << "\n"
         << (biasType ? typedLoadOpcode(biasType) : "LDG") << " R6, [R4]\n"
         << "ADD R20, R20, R6\n";
    }

    os << "MUL R4, R1, " << (oh * ow) << "\n"
       << "MUL R5, R2, " << ow << "\n"
       << "ADD R4, R4, R5\n"
       << "ADD R4, R4, R3\n"
       << "MUL R4, R4, " << outputStride << "\n"
       << "ADD R4, R4, " << outputBase << "\n"
       << typedStoreOpcode(outputType) << " [R4], R20\n";
    emitCnnFinish(os);
    os.flush();

    IsaText isa;
    isa.code = code;
    isa.grid = std::to_string((total + 31) / 32) + "x1x1";
    isa.block = "32x1x1";
    return isa;
  }

  static void emitAttentionQFragment(llvm::raw_ostream &os, int dest,
                                     int rowGroup, int colGroup,
                                     int64_t qBase, int64_t seq, int64_t dim) {
    os << "MOV R26, " << (rowGroup * 8) << "\n"
       << "ADD R26, R26, R24\n"
       << "MOV R27, " << (colGroup * 4) << "\n"
       << "ADD R27, R27, R25\n"
       << "MOV R39, " << seq << "\n"
       << "CMP.LT R28, R26, R39\n"
       << "MOV R39, " << dim << "\n"
       << "CMP.LT R40, R27, R39\n"
       << "MUL R28, R28, R40\n"
       << "MOV R39, " << (seq - 1) << "\n"
       << "MIN R29, R26, R39\n"
       << "MOV R39, " << (dim - 1) << "\n"
       << "MIN R30, R27, R39\n"
       << "MOV R37, " << dim << "\n"
       << "MUL R37, R29, R37\n"
       << "ADD R37, R37, R30\n"
       << "MUL R37, R37, 4\n"
       << "ADD R37, R37, " << qBase << "\n"
       << "LDG R42, [R37]\n"
       << "MOV R39, 0\n"
       << "SEL R" << dest << ", R28, R42, R39\n";
  }

  static void emitAttentionQFragmentTiled(
      llvm::raw_ostream &os, int dest, int rowGroup, int colGroup,
      int64_t qBase, int64_t seq, int64_t dim, int64_t rowOffset,
      int64_t colOffset) {
    os << "MOV R26, " << rowOffset << "\n"
       << "ADD R26, R26, " << (rowGroup * 8) << "\n"
       << "ADD R26, R26, R24\n"
       << "MOV R27, " << colOffset << "\n"
       << "ADD R27, R27, " << (colGroup * 4) << "\n"
       << "ADD R27, R27, R25\n"
       << "MOV R39, " << seq << "\n"
       << "CMP.LT R28, R26, R39\n"
       << "MOV R39, " << dim << "\n"
       << "CMP.LT R40, R27, R39\n"
       << "MUL R28, R28, R40\n"
       << "MOV R39, " << (seq - 1) << "\n"
       << "MIN R29, R26, R39\n"
       << "MOV R39, " << (dim - 1) << "\n"
       << "MIN R30, R27, R39\n"
       << "MOV R37, " << dim << "\n"
       << "MUL R37, R29, R37\n"
       << "ADD R37, R37, R30\n"
       << "MUL R37, R37, 4\n"
       << "ADD R37, R37, " << qBase << "\n"
       << "LDG R42, [R37]\n"
       << "MOV R39, 0\n"
       << "SEL R" << dest << ", R28, R42, R39\n";
  }

  static void emitAttentionKTransposeFragment(
      llvm::raw_ostream &os, int dest, int rowGroup, int colGroup,
      int64_t kBase, int64_t seq, int64_t dim) {
    os << "MOV R26, " << (rowGroup * 8) << "\n"
       << "ADD R26, R26, R24\n"
       << "MOV R27, " << (colGroup * 4) << "\n"
       << "ADD R27, R27, R25\n"
       << "MOV R39, " << dim << "\n"
       << "CMP.LT R28, R26, R39\n"
       << "MOV R39, " << seq << "\n"
       << "CMP.LT R40, R27, R39\n"
       << "MUL R28, R28, R40\n"
       << "MOV R39, " << (dim - 1) << "\n"
       << "MIN R29, R26, R39\n"
       << "MOV R39, " << (seq - 1) << "\n"
       << "MIN R30, R27, R39\n"
       << "MOV R37, " << dim << "\n"
       << "MUL R37, R30, R37\n"
       << "ADD R37, R37, R29\n"
       << "MUL R37, R37, 4\n"
       << "ADD R37, R37, " << kBase << "\n"
       << "LDG R42, [R37]\n"
       << "MOV R39, 0\n"
       << "SEL R" << dest << ", R28, R42, R39\n";
  }

  static void emitAttentionKTransposeFragmentTiled(
      llvm::raw_ostream &os, int dest, int rowGroup, int colGroup,
      int64_t kBase, int64_t seq, int64_t dim, int64_t rowOffset,
      int64_t colOffset) {
    os << "MOV R26, " << rowOffset << "\n"
       << "ADD R26, R26, " << (rowGroup * 8) << "\n"
       << "ADD R26, R26, R24\n"
       << "MOV R27, " << colOffset << "\n"
       << "ADD R27, R27, " << (colGroup * 4) << "\n"
       << "ADD R27, R27, R25\n"
       << "MOV R39, " << dim << "\n"
       << "CMP.LT R28, R26, R39\n"
       << "MOV R39, " << seq << "\n"
       << "CMP.LT R40, R27, R39\n"
       << "MUL R28, R28, R40\n"
       << "MOV R39, " << (dim - 1) << "\n"
       << "MIN R29, R26, R39\n"
       << "MOV R39, " << (seq - 1) << "\n"
       << "MIN R30, R27, R39\n"
       << "MOV R37, " << dim << "\n"
       << "MUL R37, R30, R37\n"
       << "ADD R37, R37, R29\n"
       << "MUL R37, R37, 4\n"
       << "ADD R37, R37, " << kBase << "\n"
       << "LDG R42, [R37]\n"
       << "MOV R39, 0\n"
       << "SEL R" << dest << ", R28, R42, R39\n";
  }

  static void emitAttentionSharedAFragment(llvm::raw_ostream &os, int dest,
                                           int rowGroup, int colGroup,
                                           int64_t offset) {
    os << "MOV R26, " << (rowGroup * 8) << "\n"
       << "ADD R26, R26, R24\n"
       << "MOV R27, " << (colGroup * 4) << "\n"
       << "ADD R27, R27, R25\n"
       << "MUL R37, R26, 16\n"
       << "ADD R37, R37, R27\n"
       << "MUL R37, R37, 4\n"
       << "ADD R37, R37, " << offset << "\n"
       << "LDS R42, [R37]\n"
       << "MOV R" << dest << ", R42\n";
  }

  static void emitAttentionVFragment(llvm::raw_ostream &os, int dest,
                                     int rowGroup, int colGroup,
                                     int64_t vBase, int64_t seq, int64_t dim) {
    os << "MOV R26, " << (rowGroup * 8) << "\n"
       << "ADD R26, R26, R24\n"
       << "MOV R27, " << (colGroup * 4) << "\n"
       << "ADD R27, R27, R25\n"
       << "MOV R39, " << seq << "\n"
       << "CMP.LT R28, R26, R39\n"
       << "MOV R39, " << dim << "\n"
       << "CMP.LT R40, R27, R39\n"
       << "MUL R28, R28, R40\n"
       << "MOV R39, " << (seq - 1) << "\n"
       << "MIN R29, R26, R39\n"
       << "MOV R39, " << (dim - 1) << "\n"
       << "MIN R30, R27, R39\n"
       << "MOV R37, " << dim << "\n"
       << "MUL R37, R29, R37\n"
       << "ADD R37, R37, R30\n"
       << "MUL R37, R37, 4\n"
       << "ADD R37, R37, " << vBase << "\n"
       << "LDG R42, [R37]\n"
       << "MOV R39, 0\n"
       << "SEL R" << dest << ", R28, R42, R39\n";
  }

  static void emitAttentionSoftmax(llvm::raw_ostream &os, int64_t seq,
                                   double scale, int64_t maskBase, bool masked,
                                   int64_t offset) {
    os << "MOV R3, -1000000000.0\n";
    for (int64_t j = 0; j < seq; ++j) {
      emitAttentionScoreRead(os, j, seq, scale, maskBase, masked, offset);
      os << "MAX R3, R3, R2\n";
    }
    os << "MOV R4, 0\n";
    for (int64_t j = 0; j < seq; ++j) {
      emitAttentionScoreRead(os, j, seq, scale, maskBase, masked, offset);
      os << "SUB R2, R2, R3\n"
         << "MUL R2, R2, 1.4426950408889634\n"
         << "EXP2 R5, R2\n"
         << "ADD R4, R4, R5\n";
    }
    for (int64_t j = 0; j < seq; ++j) {
      os << "MUL R1, R0, 16\n"
         << "ADD R1, R1, " << j << "\n"
         << "MUL R1, R1, 4\n"
         << "ADD R1, R1, " << offset << "\n"
         << "LDS R2, [R1]\n"
         << "MUL R2, R2, " << scale << "\n";
      if (masked) {
        os << "MOV R6, " << seq << "\n"
           << "MUL R6, R0, R6\n"
           << "ADD R6, R6, " << j << "\n"
           << "MUL R6, R6, 4\n"
           << "ADD R6, R6, " << maskBase << "\n"
           << "LDG R8, [R6]\n"
           << "CMP.EQ R6, R8, 0\n"
           << "MOV R7, -1000000000.0\n"
           << "SEL R2, R6, R7, R2\n";
      }
      os << "SUB R2, R2, R3\n"
         << "MUL R2, R2, 1.4426950408889634\n"
         << "EXP2 R5, R2\n"
         << "DIV R5, R5, R4\n"
         << "STS [R1], R5\n";
    }
  }

  static void emitAttentionScoreRead(llvm::raw_ostream &os, int64_t j,
                                     int64_t seq, double scale,
                                     int64_t maskBase, bool masked,
                                     int64_t offset) {
    os << "MUL R1, R0, 16\n"
       << "ADD R1, R1, " << j << "\n"
       << "MUL R1, R1, 4\n"
       << "ADD R1, R1, " << offset << "\n"
       << "LDS R2, [R1]\n"
       << "MUL R2, R2, " << scale << "\n";
    if (masked) {
      os << "MOV R6, " << seq << "\n"
         << "MUL R6, R0, R6\n"
         << "ADD R6, R6, " << j << "\n"
         << "MUL R6, R6, 4\n"
         << "ADD R6, R6, " << maskBase << "\n"
         << "LDG R8, [R6]\n"
         << "CMP.EQ R6, R8, 0\n"
         << "MOV R7, -1000000000.0\n"
         << "SEL R2, R6, R7, R2\n";
    }
  }

  std::optional<IsaText> emitScalarAttentionIsa(mgpu::AttentionOp attention) {
    auto qType = dyn_cast<MemRefType>(attention.getQ().getType());
    auto kType = dyn_cast<MemRefType>(attention.getK().getType());
    auto vType = dyn_cast<MemRefType>(attention.getV().getType());
    auto outputType = dyn_cast<MemRefType>(attention.getOutput().getType());
    if (!qType || !kType || !vType || !outputType || qType.getRank() != 4 ||
        kType.getRank() != 4 || vType.getRank() != 4 ||
        outputType.getRank() != 4)
      return std::nullopt;

    int64_t batch = qType.getShape()[0];
    int64_t heads = qType.getShape()[1];
    int64_t seq = qType.getShape()[2];
    int64_t dim = qType.getShape()[3];
    if (batch != 1)
      return std::nullopt;

    double scale = attention.getScale().convertToDouble();
    bool masked = attention.getMasked();
    int64_t qBase = bufferMap.base(attention.getQ());
    int64_t kBase = bufferMap.base(attention.getK());
    int64_t vBase = bufferMap.base(attention.getV());
    int64_t maskBase = bufferMap.base(attention.getMask());
    int64_t outBase = bufferMap.base(attention.getOutput());
    int64_t total = heads * seq * dim;

    std::string code;
    llvm::raw_string_ostream os(code);
    emitCnnPrologue(os, total);

    os << "IDIV R1, R0, " << (seq * dim) << "\n"
       << "MUL R4, R1, " << (seq * dim) << "\n"
       << "SUB R5, R0, R4\n"
       << "IDIV R2, R5, " << dim << "\n"
       << "MUL R6, R2, " << dim << "\n"
       << "SUB R3, R5, R6\n";

    // Pass 1: max score.
    os << "MOV R20, -1.0e30\n"
       << "MOV R21, 0\n"
       << "max_loop:\n"
       << "CMP.LT R22, R21, " << seq << "\n"
       << "BRA max_body, R22, max_end\n"
       << "BRA max_end\n"
       << "max_body:\n"
       << "MOV R23, 0\n"
       << "MOV R24, 0\n"
       << "dot1_loop:\n"
       << "CMP.LT R25, R24, " << dim << "\n"
       << "BRA dot1_body, R25, dot1_end\n"
       << "BRA dot1_end\n"
       << "dot1_body:\n"
       << "MOV R26, " << dim << "\n"
       << "MUL R26, R2, R26\n"
       << "ADD R26, R26, R24\n"
       << "MUL R26, R26, 4\n"
       << "ADD R26, R26, " << qBase << "\n"
       << "LDG R27, [R26]\n"
       << "MOV R28, " << dim << "\n"
       << "MUL R28, R21, R28\n"
       << "ADD R28, R28, R24\n"
       << "MUL R28, R28, 4\n"
       << "ADD R28, R28, " << kBase << "\n"
       << "LDG R29, [R28]\n"
       << "FMA R23, R27, R29, R23\n"
       << "ADD R24, R24, 1\n"
       << "BRA dot1_loop\n"
       << "dot1_end:\n";
    if (masked) {
      os << "CMP.LE R26, R21, R2\n"
         << "MOV R27, 0\n"
         << "MOV R28, -1.0e30\n"
         << "SEL R29, R26, R27, R28\n";
    } else {
      os << "MOV R29, 0\n";
    }
    os << "MUL R30, R23, " << scale << "\n"
       << "ADD R30, R30, R29\n"
       << "MAX R20, R20, R30\n"
       << "ADD R21, R21, 1\n"
       << "BRA max_loop\n"
       << "max_end:\n";

    // Pass 2: denominator sum exp.
    os << "MOV R21, 0\n"
       << "MOV R31, 0\n"
       << "sum_loop:\n"
       << "CMP.LT R22, R21, " << seq << "\n"
       << "BRA sum_body, R22, sum_end\n"
       << "BRA sum_end\n"
       << "sum_body:\n"
       << "MOV R23, 0\n"
       << "MOV R24, 0\n"
       << "dot2_loop:\n"
       << "CMP.LT R25, R24, " << dim << "\n"
       << "BRA dot2_body, R25, dot2_end\n"
       << "BRA dot2_end\n"
       << "dot2_body:\n"
       << "MOV R26, " << dim << "\n"
       << "MUL R26, R2, R26\n"
       << "ADD R26, R26, R24\n"
       << "MUL R26, R26, 4\n"
       << "ADD R26, R26, " << qBase << "\n"
       << "LDG R27, [R26]\n"
       << "MOV R28, " << dim << "\n"
       << "MUL R28, R21, R28\n"
       << "ADD R28, R28, R24\n"
       << "MUL R28, R28, 4\n"
       << "ADD R28, R28, " << kBase << "\n"
       << "LDG R29, [R28]\n"
       << "FMA R23, R27, R29, R23\n"
       << "ADD R24, R24, 1\n"
       << "BRA dot2_loop\n"
       << "dot2_end:\n";
    if (masked) {
      os << "CMP.LE R26, R21, R2\n"
         << "MOV R27, 0\n"
         << "MOV R28, -1.0e30\n"
         << "SEL R29, R26, R27, R28\n";
    } else {
      os << "MOV R29, 0\n";
    }
    os << "MUL R30, R23, " << scale << "\n"
       << "ADD R30, R30, R29\n"
       << "SUB R30, R30, R20\n"
       << "MUL R32, R30, 1.4426950408889634\n"
       << "EXP2 R33, R32\n"
       << "ADD R31, R31, R33\n"
       << "ADD R21, R21, 1\n"
       << "BRA sum_loop\n"
       << "sum_end:\n";

    // Output for this thread's dim index.
    os << "MOV R21, 0\n"
       << "MOV R34, 0\n"
       << "out_loop:\n"
       << "CMP.LT R22, R21, " << seq << "\n"
       << "BRA out_body, R22, out_end\n"
       << "BRA out_end\n"
       << "out_body:\n"
       << "MOV R23, 0\n"
       << "MOV R24, 0\n"
       << "dot3_loop:\n"
       << "CMP.LT R25, R24, " << dim << "\n"
       << "BRA dot3_body, R25, dot3_end\n"
       << "BRA dot3_end\n"
       << "dot3_body:\n"
       << "MOV R26, " << dim << "\n"
       << "MUL R26, R2, R26\n"
       << "ADD R26, R26, R24\n"
       << "MUL R26, R26, 4\n"
       << "ADD R26, R26, " << qBase << "\n"
       << "LDG R27, [R26]\n"
       << "MOV R28, " << dim << "\n"
       << "MUL R28, R21, R28\n"
       << "ADD R28, R28, R24\n"
       << "MUL R28, R28, 4\n"
       << "ADD R28, R28, " << kBase << "\n"
       << "LDG R29, [R28]\n"
       << "FMA R23, R27, R29, R23\n"
       << "ADD R24, R24, 1\n"
       << "BRA dot3_loop\n"
       << "dot3_end:\n";
    if (masked) {
      os << "CMP.LE R26, R21, R2\n"
         << "MOV R27, 0\n"
         << "MOV R28, -1.0e30\n"
         << "SEL R29, R26, R27, R28\n";
    } else {
      os << "MOV R29, 0\n";
    }
    os << "MUL R30, R23, " << scale << "\n"
       << "ADD R30, R30, R29\n"
       << "SUB R30, R30, R20\n"
       << "MUL R32, R30, 1.4426950408889634\n"
       << "EXP2 R33, R32\n"
       << "DIV R33, R33, R31\n"
       << "MOV R35, " << dim << "\n"
       << "MUL R35, R21, R35\n"
       << "ADD R35, R35, R3\n"
       << "MUL R35, R35, 4\n"
       << "ADD R35, R35, " << vBase << "\n"
       << "LDG R36, [R35]\n"
       << "FMA R34, R33, R36, R34\n"
       << "ADD R21, R21, 1\n"
       << "BRA out_loop\n"
       << "out_end:\n"
       << "MUL R37, R0, 4\n"
       << "ADD R37, R37, " << outBase << "\n"
       << "STG [R37], R34\n";
    emitCnnFinish(os);
    os.flush();

    IsaText result;
    result.code = code;
    result.grid = std::to_string((total + 31) / 32) + "x1x1";
    result.block = "32x1x1";
    return result;
  }

  std::optional<IsaText> emitAttentionIsa(mgpu::LaunchOp launch) {
    SmallVector<mgpu::AttentionOp> attentions;
    launch.getBody().front().walk(
        [&](mgpu::AttentionOp attention) { attentions.push_back(attention); });
    if (attentions.empty())
      return std::nullopt;

    mgpu::AttentionOp attention = attentions.front();
    auto qType = dyn_cast<MemRefType>(attention.getQ().getType());
    auto kType = dyn_cast<MemRefType>(attention.getK().getType());
    auto vType = dyn_cast<MemRefType>(attention.getV().getType());
    auto outputType = dyn_cast<MemRefType>(attention.getOutput().getType());
    if (!qType || !kType || !vType || !outputType)
      return std::nullopt;
    if (qType.getRank() == 4 && kType.getRank() == 4 &&
        vType.getRank() == 4 && outputType.getRank() == 4) {
      int64_t seq = qType.getShape()[2];
      int64_t dim = qType.getShape()[3];
      if (seq > 16 || dim > 16)
        return emitScalarAttentionIsa(attention);
    }
    if ((qType.getRank() != 2 && qType.getRank() != 4) ||
        kType.getRank() != qType.getRank() ||
        vType.getRank() != qType.getRank() ||
        outputType.getRank() != qType.getRank())
      return std::nullopt;

    bool rank4 = qType.getRank() == 4;
    int64_t seq = rank4 ? qType.getShape()[2] : qType.getShape()[0];
    int64_t dim = rank4 ? qType.getShape()[3] : qType.getShape()[1];
    if (seq > 16 || dim > 16 ||
        (rank4 ? kType.getShape()[2] : kType.getShape()[0]) != seq ||
        (rank4 ? kType.getShape()[3] : kType.getShape()[1]) != dim ||
        (rank4 ? vType.getShape()[2] : vType.getShape()[0]) != seq ||
        (rank4 ? vType.getShape()[3] : vType.getShape()[1]) != dim ||
        (rank4 ? outputType.getShape()[2] : outputType.getShape()[0]) != seq ||
        (rank4 ? outputType.getShape()[3] : outputType.getShape()[1]) != dim)
      return std::nullopt;

    double scale = attention.getScale().convertToDouble();
    bool masked = attention.getMasked();
    int64_t qBase = bufferMap.base(attention.getQ());
    int64_t kBase = bufferMap.base(attention.getK());
    int64_t vBase = bufferMap.base(attention.getV());
    int64_t maskBase = bufferMap.base(attention.getMask());
    int64_t outputBase = bufferMap.base(attention.getOutput());
    int64_t scoreOffset = 0;
    int64_t outputOffset = 16 * 16 * 4;

    std::string code;
    llvm::raw_string_ostream os(code);
    emitLaneIndex(os);
    emitZeroRegisters(os, 16, 24);
    for (int slot = 0; slot < 8; ++slot) {
      int rowGroup = slot / 4;
      int colGroup = slot % 4;
      emitAttentionQFragment(os, slot, rowGroup, colGroup, qBase, seq, dim);
      emitAttentionKTransposeFragment(os, 8 + slot, rowGroup, colGroup, kBase,
                                      seq, dim);
    }
    os << "MMA R16, R0, R8, R16\n";
    for (int slot = 0; slot < 8; ++slot)
      emitStoreSharedFragment(os, 16 + slot, slot / 4, slot % 4, scoreOffset);
    os << "BAR.SYNC\n";

    os << "MOV R0, ctaid.x\n"
       << "MUL R0, R0, ntid.x\n"
       << "ADD R0, R0, tid.x\n"
       << "MOV R1, " << seq << "\n"
       << "CMP.LT R2, R0, R1\n"
       << "BRA attn_softmax_body, R2, attn_softmax_end\n"
       << "BRA attn_softmax_end\n"
       << "attn_softmax_body:\n";
    emitAttentionSoftmax(os, seq, scale, maskBase, masked, scoreOffset);
    os << "BRA attn_softmax_end\n"
       << "attn_softmax_end:\n"
       << "BAR.SYNC\n";

    emitZeroRegisters(os, 16, 24);
    for (int slot = 0; slot < 8; ++slot) {
      int rowGroup = slot / 4;
      int colGroup = slot % 4;
      emitAttentionSharedAFragment(os, slot, rowGroup, colGroup, scoreOffset);
      emitAttentionVFragment(os, 8 + slot, rowGroup, colGroup, vBase, seq, dim);
    }
    os << "MMA R16, R0, R8, R16\n";
    for (int slot = 0; slot < 8; ++slot)
      emitStoreSharedFragment(os, 16 + slot, slot / 4, slot % 4, outputOffset);
    os << "BAR.SYNC\n";

    os << "MOV R0, ctaid.x\n"
       << "MUL R0, R0, ntid.x\n"
       << "ADD R0, R0, tid.x\n"
       << "MOV R1, " << (seq * dim) << "\n"
       << "CMP.LT R2, R0, R1\n"
       << "BRA attn_copy_body, R2, attn_copy_end\n"
       << "BRA attn_copy_end\n"
       << "attn_copy_body:\n"
       << "IDIV R1, R0, " << dim << "\n"
       << "MUL R2, R1, " << dim << "\n"
       << "SUB R3, R0, R2\n"
       << "MUL R4, R1, 16\n"
       << "ADD R4, R4, R3\n"
       << "MUL R4, R4, 4\n"
       << "ADD R4, R4, " << outputOffset << "\n"
       << "LDS R5, [R4]\n"
       << "MUL R4, R1, " << dim << "\n"
       << "ADD R4, R4, R3\n"
       << "MUL R4, R4, 4\n"
       << "ADD R4, R4, " << outputBase << "\n"
       << "STG [R4], R5\n"
       << "BRA attn_copy_end\n"
       << "attn_copy_end:\n"
       << "EXIT\n";
    os.flush();

    IsaText result;
    result.code = code;
    result.grid = "1x1x1";
    result.block = "32x1x1";
    return result;
  }

  static bool isNibbleType(Type type) {
    auto intType = dyn_cast<IntegerType>(type);
    return intType && intType.getWidth() == 4;
  }

  static int64_t elementStride(Type type) {
    if (auto intType = dyn_cast<IntegerType>(type))
      return std::max<int64_t>(1, intType.getWidth() / 8);
    if (isa<Float8E4M3FNType>(type) || isa<Float8E5M2Type>(type))
      return 1;
    if (type.isF16() || type.isBF16())
      return 2;
    if (type.isF64())
      return 8;
    return 4;
  }

  static std::string typedLoadOpcode(MemRefType type) {
    return _typedOpcode("LDG", type.getElementType());
  }

  static std::string typedStoreOpcode(MemRefType type) {
    return _typedOpcode("STG", type.getElementType());
  }

  static std::string _typedOpcode(StringRef prefix, Type elementType) {
    if (elementType.isF32())
      return prefix.str();
    if (elementType.isF16())
      return (prefix + ".F16").str();
    if (elementType.isBF16())
      return (prefix + ".BF16").str();
    if (isa<Float8E4M3FNType>(elementType))
      return (prefix + ".F8E4M3").str();
    if (isa<Float8E5M2Type>(elementType))
      return (prefix + ".F8E5M2").str();
    if (auto intType = dyn_cast<IntegerType>(elementType)) {
      if (intType.isUnsigned())
        return (prefix + (intType.getWidth() == 4 ? ".U4" : ".U8")).str();
      if (intType.getWidth() == 4)
        return (prefix + ".I4").str();
      if (intType.getWidth() == 8)
        return (prefix + ".I8").str();
      if (intType.getWidth() == 32)
        return (prefix + ".I32").str();
    }
    llvm::report_fatal_error("mgpu-to-isa: unsupported quant element type");
  }

  void emitElementAddress(llvm::raw_ostream &os, StringRef dst,
                          StringRef indexReg, int64_t base, Type elementType) {
    if (isNibbleType(elementType)) {
      // Nibble-typed loads/stores use nibble addresses.  The byte offset in
      // ``BufferMap`` must therefore be converted to a nibble offset before
      // adding the linear element index.
      os << "MUL " << dst << ", " << base << ", 2\n"
         << "ADD " << dst << ", " << dst << ", " << indexReg << "\n";
      return;
    }
    os << "MUL " << dst << ", " << indexReg << ", "
       << elementStride(elementType) << "\n"
       << "ADD " << dst << ", " << dst << ", " << base << "\n";
  }

  std::optional<IsaText> emitQuantIsa(mgpu::LaunchOp launch) {
    mgpu::QuantOp quant;
    mgpu::DequantOp dequant;
    mgpu::RequantOp requant;
    int quantOpCount = 0;
    launch.getBody().front().walk([&](Operation *op) {
      if (isa<mgpu::QuantOp>(op)) {
        if (!quant)
          quant = cast<mgpu::QuantOp>(op);
        ++quantOpCount;
      } else if (isa<mgpu::DequantOp>(op)) {
        if (!dequant)
          dequant = cast<mgpu::DequantOp>(op);
        ++quantOpCount;
      } else if (isa<mgpu::RequantOp>(op)) {
        if (!requant)
          requant = cast<mgpu::RequantOp>(op);
        ++quantOpCount;
      }
    });
    if (quantOpCount != 1)
      return std::nullopt;

    Value inputValue;
    Value outputValue;
    double inputScale = 1.0;
    double outputScale = 1.0;
    int64_t inputZeroPoint = 0;
    int64_t outputZeroPoint = 0;
    bool isQuant = false;
    bool isDequant = false;

    if (quant) {
      inputValue = quant.getInput();
      outputValue = quant.getOutput();
      inputScale = quant.getScale().convertToDouble();
      outputZeroPoint = quant.getZeroPoint();
      isQuant = true;
    } else if (dequant) {
      inputValue = dequant.getInput();
      outputValue = dequant.getOutput();
      inputScale = dequant.getScale().convertToDouble();
      inputZeroPoint = dequant.getZeroPoint();
      isDequant = true;
    } else {
      inputValue = requant.getInput();
      outputValue = requant.getOutput();
      inputScale = requant.getInputScale().convertToDouble();
      inputZeroPoint = requant.getInputZeroPoint();
      outputScale = requant.getOutputScale().convertToDouble();
      outputZeroPoint = requant.getOutputZeroPoint();
    }

    auto inputType = dyn_cast<MemRefType>(inputValue.getType());
    auto outputType = dyn_cast<MemRefType>(outputValue.getType());
    if (!inputType || !outputType || inputType.getRank() != 1 ||
        outputType.getRank() != 1)
      return std::nullopt;
    if (inputType.getShape()[0] != outputType.getShape()[0])
      return std::nullopt;
    if (!isa<IntegerType>(inputType.getElementType()) &&
        !isa<IntegerType>(outputType.getElementType()))
      return std::nullopt;

    int64_t n = inputType.getShape()[0];
    int64_t inputBase = bufferMap.base(inputValue);
    int64_t outputBase = bufferMap.base(outputValue);
    int64_t blockSize = 32;
    int64_t gridX = (n + blockSize - 1) / blockSize;

    std::string code;
    llvm::raw_string_ostream os(code);
    os << "MOV R0, ctaid.x\n"
       << "MUL R0, R0, ntid.x\n"
       << "ADD R0, R0, tid.x\n"
       << "MOV R1, " << n << "\n"
       << "CMP.LT R2, R0, R1\n"
       << "BRA body, R2, end\n"
       << "BRA end\n"
       << "body:\n";

    emitElementAddress(os, "R3", "R0", inputBase, inputType.getElementType());
    emitElementAddress(os, "R4", "R0", outputBase, outputType.getElementType());
    os << typedLoadOpcode(inputType) << " R5, [R3]\n";

    if (isQuant) {
      os << "DIV R5, R5, " << inputScale << "\n"
         << "ADD R5, R5, " << outputZeroPoint << "\n"
         << "ROUND R5, R5\n";
    } else if (isDequant) {
      os << "I2F R5, R5\n"
         << "SUB R5, R5, " << inputZeroPoint << "\n"
         << "MUL R5, R5, " << inputScale << "\n";
    } else {
      os << "I2F R5, R5\n"
         << "SUB R5, R5, " << inputZeroPoint << "\n"
         << "MUL R5, R5, " << inputScale << "\n"
         << "DIV R5, R5, " << outputScale << "\n"
         << "ADD R5, R5, " << outputZeroPoint << "\n"
         << "ROUND R5, R5\n";
    }

    os << typedStoreOpcode(outputType) << " [R4], R5\n"
       << "BRA end\n"
       << "end:\n"
       << "EXIT\n";
    os.flush();

    IsaText result;
    result.code = code;
    result.grid = std::to_string(gridX) + "x1x1";
    result.block = std::to_string(blockSize) + "x1x1";
    return result;
  }

  std::optional<IsaText> emitGenericMGpuIsa(mgpu::LaunchOp launch) {
    std::string code;
    llvm::raw_string_ostream os(code);
    bool foundMGpuOp = false;
    bool foundUnsupportedCompute = false;
    LinearScanRegisterAllocator regs(/*physicalRegisters=*/4,
                                     /*spillBase=*/4096);
    std::vector<int> liveRegisters;

    os << "// mgpu.launch generic lowering\n";
    for (Operation &op : launch.getBody().front()) {
      if (isa<mgpu::ThreadIdOp>(op)) {
        int reg = regs.acquire(os);
        os << "MOV " << regs.name(reg) << ", tid.x\n";
        liveRegisters.push_back(reg);
        foundMGpuOp = true;
      } else if (isa<mgpu::BlockIdOp>(op)) {
        int reg = regs.acquire(os);
        os << "MOV " << regs.name(reg) << ", ctaid.x\n";
        liveRegisters.push_back(reg);
        foundMGpuOp = true;
      } else if (isa<mgpu::GridDimOp>(op)) {
        int reg = regs.acquire(os);
        os << "MOV " << regs.name(reg) << ", nctaid.x\n";
        liveRegisters.push_back(reg);
        foundMGpuOp = true;
      } else if (isa<mgpu::BlockDimOp>(op)) {
        int reg = regs.acquire(os);
        os << "MOV " << regs.name(reg) << ", ntid.x\n";
        liveRegisters.push_back(reg);
        foundMGpuOp = true;
      } else if (isa<mgpu::SharedAllocOp>(op)) {
        os << "// mgpu.shared_alloc\n";
        foundMGpuOp = true;
      } else if (isa<mgpu::AsyncCopyOp>(op)) {
        os << "// mgpu.async_copy\n";
        foundMGpuOp = true;
      } else if (isa<mgpu::BarrierOp>(op)) {
        os << "BAR.SYNC\n";
        foundMGpuOp = true;
      } else if (isa<mgpu::MmaOp>(op)) {
        os << "// mgpu.mma reserved\n";
        foundMGpuOp = true;
      } else if (isa<mgpu::QuantOp>(op)) {
        os << "// mgpu.quant\n";
        foundMGpuOp = true;
      } else if (isa<mgpu::DequantOp>(op)) {
        os << "// mgpu.dequant\n";
        foundMGpuOp = true;
      } else if (isa<mgpu::RequantOp>(op)) {
        os << "// mgpu.requant\n";
        foundMGpuOp = true;
      } else if (isa<memref::AllocOp, memref::DeallocOp>(op)) {
        // Buffer declarations are side-effect plumbing for already-specialized
        // mgpu ops (for example quantization).  They are not compute and must
        // not make the generic launch look unsupported.
      } else if (isa<mgpu::ReturnOp>(op)) {
        os << "EXIT\n";
        foundMGpuOp = true;
      } else {
        foundUnsupportedCompute = true;
      }
    }

    for (int reg : liveRegisters)
      regs.release(reg);

    os.flush();

    if (!foundMGpuOp || foundUnsupportedCompute)
      return std::nullopt;

    IsaText result;
    result.code = code;
    result.grid = "1x1x1";
    result.block = "32x1x1";
    return result;
  }
};

} // namespace

std::unique_ptr<Pass> mlir::mgpu::createMGpuToIsaPass() {
  return std::make_unique<MGpuToIsaPass>();
}
