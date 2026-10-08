#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/raw_ostream.h"

#include <vector>

#include "mgpu/MGPUDialect.h"
#include "mgpu/MGPUOps.h"
#include "mgpu/MGPUPasses.h"

using namespace mlir;

namespace {

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
                    memref::MemRefDialect, mgpu::MGPUDialect>();
  }

  StringRef getArgument() const final { return "mgpu-to-isa"; }
  StringRef getDescription() const final {
    return "Emit simulator ISA for supported mgpu launches";
  }

  void runOnOperation() override {
    SmallVector<mgpu::LaunchOp> launches;
    getOperation().walk(
        [&](mgpu::LaunchOp launch) { launches.push_back(launch); });

    IRRewriter rewriter(&getContext());
    for (mgpu::LaunchOp launch : launches) {
      if (auto isa = emitVectorAddIsa(launch)) {
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
      } else if (auto isa = emitGenericMGpuIsa(launch)) {
        rewriter.setInsertionPoint(launch);
        rewriter.create<mgpu::IsaOp>(
            launch.getLoc(),
            StringAttr::get(&getContext(), isa->code),
            StringAttr::get(&getContext(), isa->grid),
            StringAttr::get(&getContext(), isa->block));
        rewriter.eraseOp(launch);
      }
    }
  }

  struct IsaText {
    std::string code;
    std::string grid;
    std::string block;
  };

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
    if (!outputType || outputType.getRank() != 1)
      return std::nullopt;

    int64_t n = outputType.getShape()[0];
    int64_t blockSize = 32;
    int64_t gridX = (n + blockSize - 1) / blockSize;
    int64_t leftDelta = n * 4;
    int64_t outputDelta = 2 * n * 4;

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
       << "MOV " << regs.name(base) << ", 0\n"
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

    int64_t aBase = 0;
    int64_t bBase = m * k * 4;
    int64_t cBase = bBase + k * n * 4;

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
      emitFragmentLoad(os, slot, rowGroup, colGroup, aBase, 16);
      emitFragmentLoad(os, 8 + slot, rowGroup, colGroup, bBase, 16);
    }

    os << "MMA R16, R0, R8, R16\n";

    for (int slot = 0; slot < 8; ++slot) {
      int rowGroup = slot / 4;
      int colGroup = slot % 4;
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

  std::optional<IsaText> emitGenericMGpuIsa(mgpu::LaunchOp launch) {
    std::string code;
    llvm::raw_string_ostream os(code);
    bool foundMGpuOp = false;
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
      } else if (isa<mgpu::ReturnOp>(op)) {
        os << "EXIT\n";
        foundMGpuOp = true;
      }
    }

    for (int reg : liveRegisters)
      regs.release(reg);

    os.flush();

    if (!foundMGpuOp)
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
