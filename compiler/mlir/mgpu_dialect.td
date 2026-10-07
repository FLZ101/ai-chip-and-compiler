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
include "mlir/Interfaces/SideEffectInterfaces.td"

def MGPU_Dialect : Dialect {
  let name = "mgpu";
  let summary = "Minimal GPU device dialect for the simulator target";
  let cppNamespace = "::mlir::mgpu";
}

class MGPU_Op<string mnemonic, list<Trait> traits = []>
    : Op<MGPU_Dialect, mnemonic, traits>;

def MGPU_LaunchOp : MGPU_Op<"launch", [NoMemoryEffect]> {
  let summary = "Launch one thread block";
  let arguments = (ins Variadic<Index>:$blockSizes,
                       Variadic<Index>:$threadSizes);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    `blocks` `(` $blockSizes `)` `threads` `(` $threadSizes `)`
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

def MGPU_DimensionAttr : EnumAttr<MGPU_Dialect, "x,y,z", "dimension"> {
  let summary = "Thread/block dimension component";
}

#endif // MGPU_DIALECT
