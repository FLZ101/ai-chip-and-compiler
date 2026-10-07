//===-- mgpu_dialect.td - Minimal GPU device dialect ---------------------===//
//
// This file defines the small ``mgpu`` dialect used by the compiler.
// It is intentionally kept close to the simulator's execution model:
// grid/block/thread indices, shared memory, barriers, async copies, and the
// reserved tensor-core MMA operation.
//
//===----------------------------------------------------------------------===//

#ifndef MGPU_DIALECT
#define MGPU_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def MGPU_Dialect : Dialect {
  let name = "mgpu";
  let summary = "Minimal GPU device dialect for the simulator target";
  let cppNamespace = "::mlir::mgpu";
}

def MGPU_Dimension : I32EnumAttr<"Dimension",
  "Thread/block dimension component", [
  I32EnumAttrCase<"X", 0, "x">,
  I32EnumAttrCase<"Y", 1, "y">,
  I32EnumAttrCase<"Z", 2, "z">
]> {
  let cppNamespace = "::mlir::mgpu";
}

def MGPU_DimensionAttr : EnumAttr<MGPU_Dialect, MGPU_Dimension, "dimension"> {
  let assemblyFormat = "$value";
}

class MGPU_Op<string mnemonic, list<Trait> traits = []>
    : Op<MGPU_Dialect, mnemonic, traits>;

def MGPU_LaunchOp : MGPU_Op<"launch", [NoMemoryEffect]> {
  let summary = "Launch one thread block";
  let arguments = (ins Index:$blockSizeX,
                       Index:$blockSizeY,
                       Index:$blockSizeZ,
                       Index:$threadSizeX,
                       Index:$threadSizeY,
                       Index:$threadSizeZ);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    attr-dict
    `blocks` `(` $blockSizeX `,` $blockSizeY `,` $blockSizeZ `)`
    `threads` `(` $threadSizeX `,` $threadSizeY `,` $threadSizeZ `)`
    $body
  }];
}

def MGPU_ThreadIdOp : MGPU_Op<"thread_id", [Pure]> {
  let summary = "Return a thread index component";
  let arguments = (ins MGPU_DimensionAttr:$dim);
  let results = (outs Index:$result);
}

def MGPU_BlockIdOp : MGPU_Op<"block_id", [Pure]> {
  let summary = "Return a block index component";
  let arguments = (ins MGPU_DimensionAttr:$dim);
  let results = (outs Index:$result);
}

def MGPU_GridDimOp : MGPU_Op<"grid_dim", [Pure]> {
  let summary = "Return a grid dimension component";
  let arguments = (ins MGPU_DimensionAttr:$dim);
  let results = (outs Index:$result);
}

def MGPU_BlockDimOp : MGPU_Op<"block_dim", [Pure]> {
  let summary = "Return a block dimension component";
  let arguments = (ins MGPU_DimensionAttr:$dim);
  let results = (outs Index:$result);
}

def MGPU_SharedAllocOp : MGPU_Op<"shared_alloc", [MemoryEffects<[MemAlloc]>]> {
  let summary = "Allocate a buffer in block-shared memory";
  let arguments = (ins Index:$byteSize);
  let results = (outs AnyMemRef:$result);
}

def MGPU_BarrierOp : MGPU_Op<"barrier", []> {
  let summary = "Synchronize all warps in the current block";
  let assemblyFormat = "attr-dict";
}

def MGPU_AsyncCopyOp : MGPU_Op<"async_copy", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Copy bytes between address spaces";
  let arguments = (ins AnyMemRef:$source,
                       Index:$sourceOffset,
                       AnyMemRef:$destination,
                       Index:$destinationOffset,
                       Index:$bytes);
}

def MGPU_MmaOp : MGPU_Op<"mma", [Pure]> {
  let summary = "Reserved tensor-core matrix-multiply-accumulate";
  let description = [{
    ``mgpu.mma`` represents a future tensor-core instruction.  It is defined
    now so later tiling and scheduling passes have a stable target op.  The
    simulator currently reserves the ISA ``MMA`` opcode but does not execute it.
  }];
  let arguments = (ins AnyMemRef:$lhs,
                       AnyMemRef:$rhs,
                       AnyType:$accumulator);
  let results = (outs AnyMemRef:$result);
}

def MGPU_ReturnOp : MGPU_Op<"return", [Terminator]> {
  let summary = "Terminator for an mgpu.launch region";
  let assemblyFormat = "attr-dict";
}

#endif // MGPU_DIALECT
