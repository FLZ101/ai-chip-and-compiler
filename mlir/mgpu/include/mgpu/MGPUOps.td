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

def MGPU_SoftmaxOp : MGPU_Op<"softmax", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Row-wise softmax";
  let arguments = (ins AnyMemRef:$x,
                       AnyMemRef:$output);
  let results = (outs);
  let assemblyFormat = "$x `,` $output attr-dict `:` type($x) `,` type($output)";
}

def MGPU_MaskedSoftmaxOp
    : MGPU_Op<"masked_softmax", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Masked row-wise softmax";
  let arguments = (ins AnyMemRef:$x,
                       AnyMemRef:$mask,
                       AnyMemRef:$output);
  let results = (outs);
  let assemblyFormat = [{
    $x `,` $mask `,` $output
    attr-dict
    `:` type($x) `,` type($mask) `,` type($output)
  }];
}

def MGPU_LayerNormOp : MGPU_Op<"layer_norm", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Layer normalization";
  let arguments = (ins AnyMemRef:$x,
                       AnyMemRef:$weight,
                       AnyMemRef:$bias,
                       AnyMemRef:$output);
  let results = (outs);
  let assemblyFormat = [{
    $x `,` $weight `,` $bias `,` $output
    attr-dict
    `:` type($x) `,` type($weight) `,` type($bias) `,` type($output)
  }];
}

def MGPU_RMSNormOp : MGPU_Op<"rms_norm", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "RMS normalization";
  let arguments = (ins AnyMemRef:$x,
                       AnyMemRef:$weight,
                       AnyMemRef:$output);
  let results = (outs);
  let assemblyFormat = [{
    $x `,` $weight `,` $output
    attr-dict
    `:` type($x) `,` type($weight) `,` type($output)
  }];
}

def MGPU_ReshapeOp : MGPU_Op<"reshape", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Reshape a buffer without changing element order";
  let arguments = (ins AnyMemRef:$input,
                       AnyMemRef:$output);
  let results = (outs);
  let assemblyFormat = "$input `,` $output attr-dict `:` type($input) `,` type($output)";
}

def MGPU_TransposeOp : MGPU_Op<"transpose", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Transpose a rank-2 buffer";
  let arguments = (ins AnyMemRef:$input,
                       AnyMemRef:$output);
  let results = (outs);
  let assemblyFormat = "$input `,` $output attr-dict `:` type($input) `,` type($output)";
}

def MGPU_ConcatOp : MGPU_Op<"concat", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Concatenate two rank-1 buffers";
  let arguments = (ins AnyMemRef:$first,
                       AnyMemRef:$second,
                       AnyMemRef:$output);
  let results = (outs);
  let assemblyFormat = [{
    $first `,` $second `,` $output
    attr-dict
    `:` type($first) `,` type($second) `,` type($output)
  }];
}

def MGPU_SliceOp : MGPU_Op<"slice", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Slice one rank-1 buffer";
  let arguments = (ins AnyMemRef:$input,
                       AnyMemRef:$output,
                       I64Attr:$start,
                       I64Attr:$step);
  let results = (outs);
  let assemblyFormat = [{
    $input `,` $output
    `start` `=` $start `,` `step` `=` $step
    attr-dict
    `:` type($input) `,` type($output)
  }];
}

def MGPU_PadOp : MGPU_Op<"pad", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Pad one rank-1 buffer";
  let arguments = (ins AnyMemRef:$input,
                       AnyMemRef:$output,
                       I64Attr:$before,
                       I64Attr:$after,
                       I64Attr:$beforeY,
                       I64Attr:$afterY,
                       I64Attr:$beforeX,
                       I64Attr:$afterX,
                       F32Attr:$value);
  let results = (outs);
  let assemblyFormat = [{
    $input `,` $output
    `before` `=` $before `,` `after` `=` $after `,` `value` `=` $value
    `,` `beforeY` `=` $beforeY `,` `afterY` `=` $afterY
    `,` `beforeX` `=` $beforeX `,` `afterX` `=` $afterX
    attr-dict
    `:` type($input) `,` type($output)
  }];
}

def MGPU_ResizeOp : MGPU_Op<"resize", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Resize one rank-1 buffer with nearest-neighbor sampling";
  let arguments = (ins AnyMemRef:$input,
                       AnyMemRef:$output);
  let results = (outs);
  let assemblyFormat = "$input `,` $output attr-dict `:` type($input) `,` type($output)";
}

def MGPU_BatchNormOp : MGPU_Op<"batch_norm", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Batch normalization";
  let arguments = (ins AnyMemRef:$x,
                       AnyMemRef:$scale,
                       AnyMemRef:$bias,
                       AnyMemRef:$mean,
                       AnyMemRef:$var,
                       AnyMemRef:$output);
  let results = (outs);
  let assemblyFormat = [{
    $x `,` $scale `,` $bias `,` $mean `,` $var `,` $output
    attr-dict
    `:` type($x) `,` type($scale) `,` type($bias) `,` type($mean) `,` type($var) `,` type($output)
  }];
}

def MGPU_GroupNormOp : MGPU_Op<"group_norm", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Group normalization";
  let arguments = (ins AnyMemRef:$x,
                       AnyMemRef:$weight,
                       AnyMemRef:$bias,
                       AnyMemRef:$output,
                       I64Attr:$numGroups);
  let results = (outs);
  let assemblyFormat = [{
    $x `,` $weight `,` $bias `,` $output
    `num_groups` `=` $numGroups
    attr-dict
    `:` type($x) `,` type($weight) `,` type($bias) `,` type($output)
  }];
}

def MGPU_BoxDecodeOp : MGPU_Op<"box_decode", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Decode detection box deltas";
  let arguments = (ins AnyMemRef:$boxes,
                       AnyMemRef:$deltas,
                       AnyMemRef:$output);
  let results = (outs);
  let assemblyFormat = [{
    $boxes `,` $deltas `,` $output
    attr-dict
    `:` type($boxes) `,` type($deltas) `,` type($output)
  }];
}

def MGPU_NMSOp : MGPU_Op<"nms", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Greedy non-maximum suppression";
  let arguments = (ins AnyMemRef:$boxes,
                       AnyMemRef:$scores,
                       AnyMemRef:$output,
                       F32Attr:$iouThreshold);
  let results = (outs);
  let assemblyFormat = [{
    $boxes `,` $scores `,` $output
    `iou_threshold` `=` $iouThreshold
    attr-dict
    `:` type($boxes) `,` type($scores) `,` type($output)
  }];
}

def MGPU_MemoryLayoutOp : MGPU_Op<"memory_layout", [NoMemoryEffect]> {
  let summary = "Associate memref SSA values with simulator byte offsets";
  let description = [{
    ``mgpu.memory_layout`` is consumed by ``mgpu-to-isa``.  It is not a
    runtime operation; it only carries the host-side buffer placement chosen
    by the Python driver.
  }];
  let arguments = (ins Variadic<AnyMemRef>:$buffers,
                       DenseI64ArrayAttr:$offsets);
  let results = (outs);
  let hasVerifier = 1;
  let assemblyFormat = [{
    $buffers `offsets` `=` $offsets attr-dict `:` type($buffers)
  }];
}

def MGPU_LinearOp : MGPU_Op<"linear", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Dense linear layer with optional bias";
  let arguments = (ins AnyMemRef:$input,
                       AnyMemRef:$weight,
                       AnyMemRef:$bias,
                       AnyMemRef:$output,
                       UnitAttr:$hasBias);
  let results = (outs);
  let assemblyFormat = [{
    $input `,` $weight `,` $bias `,` $output
    attr-dict
    `:` type($input) `,` type($weight) `,` type($bias) `,` type($output)
  }];
}

def MGPU_Conv2DOp : MGPU_Op<"conv2d", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "2D convolution with optional bias";
  let arguments = (ins AnyMemRef:$image,
                       AnyMemRef:$weight,
                       AnyMemRef:$bias,
                       AnyMemRef:$output,
                       I64Attr:$padY,
                       I64Attr:$padX,
                       I64Attr:$strideY,
                       I64Attr:$strideX,
                       I64Attr:$dilY,
                       I64Attr:$dilX,
                       UnitAttr:$hasBias);
  let results = (outs);
  let assemblyFormat = [{
    $image `,` $weight `,` $bias `,` $output
    `padY` `=` $padY `,` `padX` `=` $padX `,` `strideY` `=` $strideY `,` `strideX` `=` $strideX
    `,` `dilY` `=` $dilY `,` `dilX` `=` $dilX
    attr-dict
    `:` type($image) `,` type($weight) `,` type($bias) `,` type($output)
  }];
}

def MGPU_QuantOp : MGPU_Op<"quant", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Quantize a memref using scale and zero point";
  let arguments = (ins AnyMemRef:$input,
                       AnyMemRef:$output,
                       F32Attr:$scale,
                       I32Attr:$zeroPoint,
                       TypeAttr:$storageType);
  let results = (outs);
}

def MGPU_DequantOp : MGPU_Op<"dequant", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Dequantize a memref using scale and zero point";
  let arguments = (ins AnyMemRef:$input,
                       AnyMemRef:$output,
                       F32Attr:$scale,
                       I32Attr:$zeroPoint,
                       TypeAttr:$computeType);
  let results = (outs);
}

def MGPU_RequantOp : MGPU_Op<"requant", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Requantize between two quantization domains";
  let arguments = (ins AnyMemRef:$input,
                       AnyMemRef:$output,
                       F32Attr:$inputScale,
                       I32Attr:$inputZeroPoint,
                       F32Attr:$outputScale,
                       I32Attr:$outputZeroPoint,
                       TypeAttr:$storageType);
  let results = (outs);
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
