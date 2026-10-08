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
      } else if (auto isa = emitConvIsa(launch)) {
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

  static void emitLaneIndex(llvm::raw_ostream &os) {
    os << "IDIV R24, tid.x, 4\n"
       << "MUL R25, R24, 4\n"
       << "SUB R25, tid.x, R25\n";
  }

  static void emitZeroRegisters(llvm::raw_ostream &os, int first, int last) {
    for (int reg = first; reg < last; ++reg)
      os << "MOV R" << reg << ", 0\n";
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
    SmallVector<linalg::Conv2DNchwFchwOp> convs;
    launch.getBody().front().walk(
        [&](linalg::Conv2DNchwFchwOp conv) { convs.push_back(conv); });
    if (convs.empty())
      return std::nullopt;

    linalg::Conv2DNchwFchwOp conv = convs.front();
    auto dstOp = cast<DestinationStyleOpInterface>(conv.getOperation());
    if (dstOp.getDpsInputs().size() < 2 || dstOp.getDpsInits().empty())
      return std::nullopt;

    auto imageType = dyn_cast<MemRefType>(dstOp.getDpsInputs()[0].getType());
    auto weightType = dyn_cast<MemRefType>(dstOp.getDpsInputs()[1].getType());
    auto outputType = dyn_cast<MemRefType>(dstOp.getDpsInits()[0].getType());
    if (!imageType || !weightType || !outputType)
      return std::nullopt;
    if (imageType.getRank() != 4 || weightType.getRank() != 4 ||
        outputType.getRank() != 4)
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
    if (spatialSize > 16 || reductionSize > 16 || f > 16)
      return std::nullopt;

    int64_t imageBase = 0;
    int64_t weightBase = n * c * h * w * 4;
    int64_t outputBase = weightBase + f * c * kh * kw * 4;

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
       << "LDS R8, [R7]\n"
       << "MUL R9, R1, " << oh << "\n"
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
    if (qType.getRank() != 2 || kType.getRank() != 2 ||
        vType.getRank() != 2 || outputType.getRank() != 2)
      return std::nullopt;

    int64_t seq = qType.getShape()[0];
    int64_t dim = qType.getShape()[1];
    if (seq > 16 || dim > 16 || kType.getShape()[0] != seq ||
        kType.getShape()[1] != dim || vType.getShape()[0] != seq ||
        vType.getShape()[1] != dim || outputType.getShape()[0] != seq ||
        outputType.getShape()[1] != dim)
      return std::nullopt;

    double scale = attention.getScale().convertToDouble();
    bool masked = attention.getMasked();
    int64_t qBase = 0;
    int64_t kBase = seq * dim * 4;
    int64_t vBase = kBase + seq * dim * 4;
    int64_t maskBase = vBase + seq * dim * 4;
    int64_t outputBase = maskBase + seq * seq * 4;
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
