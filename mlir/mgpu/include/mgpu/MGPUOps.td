#ifndef MGPU_OPS
#define MGPU_OPS

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
  let hasVerifier = 1;
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
  let hasVerifier = 1;
  let arguments = (ins I32Attr:$dim);
  let results = (outs Index:$result);
  let assemblyFormat = "$dim attr-dict `:` type($result)";
}

def MGPU_BlockIdOp : MGPU_Op<"block_id", [Pure]> {
  let summary = "Return a block index component";
  let hasVerifier = 1;
  let arguments = (ins I32Attr:$dim);
  let results = (outs Index:$result);
  let assemblyFormat = "$dim attr-dict `:` type($result)";
}

def MGPU_GridDimOp : MGPU_Op<"grid_dim", [Pure]> {
  let summary = "Return a grid dimension component";
  let hasVerifier = 1;
  let arguments = (ins I32Attr:$dim);
  let results = (outs Index:$result);
  let assemblyFormat = "$dim attr-dict `:` type($result)";
}

def MGPU_BlockDimOp : MGPU_Op<"block_dim", [Pure]> {
  let summary = "Return a block dimension component";
  let hasVerifier = 1;
  let arguments = (ins I32Attr:$dim);
  let results = (outs Index:$result);
  let assemblyFormat = "$dim attr-dict `:` type($result)";
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
    ``mgpu.mma`` represents a future tensor-core instruction.
  }];
  let arguments = (ins AnyMemRef:$lhs,
                       AnyMemRef:$rhs,
                       AnyType:$accumulator);
  let results = (outs AnyMemRef:$result);
}

def MGPU_AttentionOp : MGPU_Op<"attention", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Warp-wide scaled dot-product attention target op";
  let description = [{
    ``mgpu.attention`` represents the complete attention computation:
    ``softmax(Q @ K^T * scale) @ V``.

    The ``mask`` operand is present even when ``masked`` is false; the lowering
    ignores it in that case.  Keeping the operand explicit makes the target
    instruction easier to use before mask materialization has been decided.
  }];
  let arguments = (ins AnyMemRef:$q,
                       AnyMemRef:$k,
                       AnyMemRef:$v,
                       AnyMemRef:$mask,
                       AnyMemRef:$output,
                       F32Attr:$scale,
                       UnitAttr:$masked);
  let results = (outs);
  let assemblyFormat = [{
    $q `,` $k `,` $v `,` $mask `,` $output
    attr-dict
    `:` type($q) `,` type($k) `,` type($v) `,` type($mask) `,` type($output)
  }];
}

def MGPU_RopeOp : MGPU_Op<"rope", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Apply rotary position embedding";
  let arguments = (ins AnyMemRef:$x,
                       AnyMemRef:$cos,
                       AnyMemRef:$sin,
                       AnyMemRef:$output);
  let results = (outs);
  let assemblyFormat = [{
    $x `,` $cos `,` $sin `,` $output
    attr-dict
    `:` type($x) `,` type($cos) `,` type($sin) `,` type($output)
  }];
}

def MGPU_CausalMaskOp : MGPU_Op<"causal_mask", [MemoryEffects<[MemWrite]>]> {
  let summary = "Write a lower-triangular causal mask";
  let arguments = (ins AnyMemRef:$output);
  let results = (outs);
  let assemblyFormat = "$output attr-dict `:` type($output)";
}

def MGPU_KVCacheAppendOp : MGPU_Op<"kv_cache_append", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Append a value to a KV cache";
  let arguments = (ins AnyMemRef:$cache,
                       AnyMemRef:$value,
                       AnyMemRef:$output);
  let results = (outs);
  let assemblyFormat = [{
    $cache `,` $value `,` $output
    attr-dict
    `:` type($cache) `,` type($value) `,` type($output)
  }];
}

def MGPU_KVCacheGatherOp : MGPU_Op<"kv_cache_gather", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Gather a position from a KV cache";
  let arguments = (ins AnyMemRef:$cache,
                       AnyMemRef:$positions,
                       AnyMemRef:$output);
  let results = (outs);
  let assemblyFormat = [{
    $cache `,` $positions `,` $output
    attr-dict
    `:` type($cache) `,` type($positions) `,` type($output)
  }];
}

def MGPU_QuantOp : MGPU_Op<"quant", [Pure]> {
  let summary = "Quantize a value using scale and zero point";
  let arguments = (ins AnyType:$input,
                       F32Attr:$scale,
                       I32Attr:$zeroPoint,
                       TypeAttr:$storageType);
  let results = (outs AnyType:$result);
}

def MGPU_DequantOp : MGPU_Op<"dequant", [Pure]> {
  let summary = "Dequantize a packed value using scale and zero point";
  let arguments = (ins AnyType:$input,
                       F32Attr:$scale,
                       I32Attr:$zeroPoint,
                       TypeAttr:$computeType);
  let results = (outs AnyType:$result);
}

def MGPU_RequantOp : MGPU_Op<"requant", [Pure]> {
  let summary = "Requantize between two quantization domains";
  let arguments = (ins AnyType:$input,
                       F32Attr:$inputScale,
                       I32Attr:$inputZeroPoint,
                       F32Attr:$outputScale,
                       I32Attr:$outputZeroPoint,
                       TypeAttr:$storageType);
  let results = (outs AnyType:$result);
}

def MGPU_ReturnOp : MGPU_Op<"return", [Terminator]> {
  let summary = "Terminator for an mgpu.launch region";
  let hasVerifier = 1;
  let assemblyFormat = "attr-dict";
}

def MGPU_IsaOp : MGPU_Op<"isa", []> {
  let summary = "Container for simulator ISA emitted by mgpu-to-isa";
  let arguments = (ins StrAttr:$code,
                       StrAttr:$grid,
                       StrAttr:$block);
}

#endif // MGPU_OPS
