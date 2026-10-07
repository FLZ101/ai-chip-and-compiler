# Model of a Modern GPU

This document describes the programming and execution model of a modern GPU, with enough detail to guide the minimal GPU simulator and MLIR compiler in this repository.

The goal is not to describe one specific vendor architecture exactly. It is to define the essential model shared by NVIDIA-, AMD-, and mobile-GPU-style designs.

---

## 1. High-level device model

A GPU is a throughput-oriented accelerator. It executes many independent threads at once to hide memory latency and to exploit data-level parallelism.

```text
Host CPU
   |
   | commands, buffers, kernels
   v
Host interface / command processor
   |
   +--> global memory / HBM
   |         |
   |         v
   |      L2 cache
   |
   +--> many streaming multiprocessors (SMs)
              |
              +--> L1 / shared memory
              +--> register files
              +--> warp schedulers
              +--> ALUs / SFUs
              +--> tensor cores
```

Important idea: the GPU is not one giant core. It is a large collection of simpler cores, each with many threads.

---

## 2. Main components

### 2.1 Host interface and command processor

The CPU sends commands to the GPU:

- memory allocation,
- host-to-device and device-to-host copies,
- kernel launches,
- synchronization,
- profiling commands.

The command processor decodes and schedules those commands. In CUDA-like terminology, this is the GPU-side driver and kernel launch path.

### 2.2 Global memory

Global memory is the large device memory pool, typically HBM or GDDR. It is:

- byte-addressable,
- much slower than on-chip storage,
- cached through L2,
- visible to all threads and blocks,
- persistent across the lifetime of a kernel unless explicitly freed.

In the simulator, global memory is modeled as a byte-addressable array.

### 2.3 L2 cache

L2 sits between all SMs and global memory. It:

- reduces repeated global-memory traffic,
- serves coalesced accesses efficiently,
- is shared by the whole device.

The minimal simulator can start with a simple L2 cache or no cache. A simple cache is enough to expose hit/miss behavior and memory bandwidth effects.

### 2.4 Streaming multiprocessor

An SM is the unit of compute. A modern SM contains:

- one or more warp schedulers,
- a large register file,
- many execution lanes,
- shared memory,
- L1 cache,
- load/store units,
- special-function units,
- tensor cores.

The SM executes many warps. When one warp stalls on memory, another warp can issue instructions. This is the main latency-hiding mechanism.

---

## 3. Threads, warps, and SIMT

### 3.1 Thread

A thread is the smallest unit of execution visible to the programmer. Each thread has:

- its own registers,
- a program counter,
- a thread ID,
- private local memory when needed.

### 3.2 Warp

Threads are grouped into warps. In the common model, a warp contains 32 threads.

All threads in a warp execute the same instruction at the same time. This is the SIMT model:

```text
warp:
  lane 0
  lane 1
  ...
  lane 31

instruction stream:
  ADD R0, R1, R2   <- all 32 lanes execute this
  LDG R3, [R4]     <- all 32 lanes issue this
```

Each lane operates on its own register file.

### 3.3 Active mask and divergence

When a warp encounters a branch such as:

```c
if (tid.x % 2 == 0) {
    A();
} else {
    B();
}
```

different lanes may choose different paths. The warp cannot truly execute two instructions at once. Instead:

1. It executes the taken path with an active mask containing only those lanes.
2. It executes the other path with the complementary active mask.
3. It reconverges after both paths complete.

Inactive lanes do not update registers or memory.

Divergence reduces effective throughput because both branches must be serialized.

### 3.4 Reconvergence

Modern GPUs track a stack of active masks and branch targets. When all paths from a divergent branch finish, the warp returns to the reconvergence point with the original mask.

The minimal simulator should implement at least a simplified version of this behavior.

---

## 4. Blocks, grids, and kernel launch

### 4.1 Thread block

A block is a group of threads that:

- can cooperate through shared memory,
- can synchronize with barriers,
- is scheduled on one SM.

Blocks do not synchronize with each other during a kernel.

### 4.2 Grid

A grid is the set of all blocks launched by a kernel.

Common indexing terms:

| Special value | Meaning |
| --- | --- |
| `threadIdx.x/y/z` / `tid.x/y/z` | thread index within its block |
| `blockIdx.x/y/z` / `ctaid.x/y/z` | block index within the grid |
| `blockDim.x/y/z` / `ntid.x/y/z` | number of threads in a block dimension |
| `gridDim.x/y/z` / `nctaid.x/y/z` | number of blocks in a grid dimension |

These values are exposed to ISA code as special registers.

The minimal ISA exposes the 3D forms directly:

```text
ctaid.x/y/z      nctaid.x/y/z
tid.x/y/z        ntid.x/y/z
```

### 4.3 Launch model

A kernel launch contains:

- grid dimensions,
- block dimensions,
- kernel arguments,
- input/output buffer references,
- optionally shared-memory size.

The hardware distributes blocks across SMs. Blocks are independent, so the scheduler can run them in any order.

### 4.4 Block vs warp

A block is a programmer-visible group of threads. A warp is the hardware
scheduling unit that executes those threads.

```text
grid
 ├── block 0  -> SM
 │    ├── warp 0: threads  0..31
 │    ├── warp 1: threads 32..63
 │    └── ...
 ├── block 1  -> another SM
 │    └── ...
```

Key rules:

- A block is scheduled on one SM.
- All threads in a block can share that SM's shared memory.
- All threads in a block can synchronize with `BAR.SYNC`.
- The hardware partitions the block's threads into warps, normally 32 threads
  per warp.
- For a block of `N` threads, the number of warps is:

```text
ceil(N / 32)
```

If `N` is not a multiple of 32, the last warp is partially filled. The missing
lanes are masked inactive.

For a 1D block:

```text
warp_id = tid.x // 32
lane_id = tid.x % 32
```

So:

- block = memory-sharing and synchronization group,
- warp = lockstep execution group,
- one block contains one or more warps,
- different blocks do not share shared memory and do not synchronize with each
  other.

---

## 5. Memory hierarchy and address spaces

### 5.1 Register file

- Fastest storage.
- Private to each thread.
- Backs the ISA registers such as `R0`, `R1`, etc.
- Large aggregate capacity but fixed per thread.

### 5.2 Shared memory

- Private to a thread block.
- On-chip scratchpad memory.
- Used for data reuse, tiling, and inter-thread cooperation.
- Requires explicit barriers when threads communicate through it.

Example uses:

- tiled matrix multiplication,
- convolution input reuse,
- reduction trees,
- softmax row normalization.

### 5.3 Global memory

- Visible to all threads.
- Large and high-latency.
- Used for model weights, activations, inputs, and outputs.

### 5.4 Local memory

- Private per-thread storage.
- Physically backed by global memory.
- Used when a thread runs out of registers or has a large private array.

### 5.5 Constant and texture/read-only memory

- Read-only during a kernel.
- Optimized for uniform access and spatial locality.
- Useful for weights and lookup tables.

### 5.6 L1 and L2 caches are transparent

In the normal GPU programming model, L1 and L2 caches are transparent to the
programmer.

The programmer does not:

- allocate cache space,
- choose which data goes into L1/L2,
- explicitly flush or invalidate them for correctness,
- address them directly.

The hardware automatically manages:

- cache allocation,
- replacement policy,
- hits and misses,
- write-back/write-through behavior,
- coherence.

Programmers address only:

- global memory,
- shared memory,
- registers,
- local/constant memory.

Caches are a performance detail, not a correctness-visible memory space.

Performance is still affected by access patterns:

- coalesced accesses use cache lines efficiently,
- repeated accesses can hit L1/L2,
- strided or random accesses may miss,
- tiling can improve reuse.

Some GPU ISAs also expose optional cache hints or prefetch instructions, such
as `ld.global.ca`, `ld.global.cg`, and `ld.global.cs` in PTX. These are
performance hints, not required for correct execution.

---

## 6. Memory access efficiency

### 6.1 Coalescing

When adjacent threads in a warp access adjacent addresses, the memory system can combine those accesses into fewer transactions.

Good:

```text
thread 0 -> address 0
thread 1 -> address 4
thread 2 -> address 8
...
```

Bad:

```text
thread 0 -> address 0
thread 1 -> address 1024
thread 2 -> address 2048
...
```

Coalescing is one of the most important optimizations for GPU kernels.

### 6.2 Shared-memory bank conflicts

Shared memory is divided into banks. If multiple threads in a warp access different addresses that map to the same bank, the access is serialized.

The simulator should eventually detect bank conflicts and report them in the timing model.

### 6.3 Cache behavior

Repeated loads from the same memory region should hit L1/L2. The simulator can expose hit and miss counters to show why a kernel is memory-bound.

---

## 7. Scheduling and latency hiding

Each SM runs many warps, not just one. The warp scheduler selects a ready warp every cycle.

A warp is ready when:

- its next instruction is available,
- its operands are ready,
- it is not waiting on memory,
- it is not blocked at a barrier,
- it is not exited.

When a warp issues a memory instruction, it can stall. The scheduler then issues another warp. This hides memory latency.

Occupancy is the number of active warps per SM relative to the hardware maximum. Higher occupancy often improves latency hiding, but uses more registers and shared memory.

---

## 8. Tensor cores and matrix acceleration

Modern GPUs include specialized matrix units, often called tensor cores.

They accelerate operations such as:

```text
D = A * B + C
```

where A, B, C, and D are small matrix tiles.

The minimal ISA models this with an `MMA` instruction. A first implementation can use a small tile such as 16x16x16.

Tensor cores are important for:

- CNN convolution as GEMM,
- Transformer QKV projections,
- attention score computation,
- feed-forward layers.

---

## 9. Synchronization and memory consistency

### 9.1 Barriers

`BAR.SYNC` synchronizes all threads in a block.

Before a barrier:

- shared-memory writes must be visible,
- global-memory writes issued by the block should be visible to the block.

After the barrier, all threads continue together.

### 9.2 Kernel boundaries

At the end of a kernel, all writes must be visible to the host before the host reads the result.

### 9.3 Async copy

Modern GPUs can copy global memory to shared memory asynchronously while computation continues. Compilers often expose this as an asynchronous-copy instruction or intrinsic.

---

## 10. Minimal simulator mapping

The simulator maps modern GPU concepts to the following abstractions:

| Modern GPU concept | Minimal simulator equivalent |
| --- | --- |
| Device | `Device` |
| Global memory | `ByteMemory` |
| Shared memory | `ByteMemory` owned by each `SM` |
| Register file | `RegisterFile` per lane |
| Thread | `LaneState` |
| Warp | `WarpState` |
| SM | `SM` |
| Thread/block/grid IDs | special registers: `tid.x/y/z`, `ntid.x/y/z`, `ctaid.x/y/z`, `nctaid.x/y/z` |
| Warp scheduler | deterministic cycle loop in `Device.run` |
| SIMT execution | one instruction applied to all `active_lanes` |
| Tensor core | reserved `MMA` instruction |

This table describes the intended conceptual mapping without freezing the
implementation status in this reference document.

---

## 11. Why this model matters for the compiler

The MLIR compiler must eventually lower high-level operators to this execution model.

Key implications:

- Matmul and attention need tiled shared-memory kernels.
- Conv2d needs im2col or direct tiled convolution.
- Elementwise ops map naturally to one thread per element.
- Reductions use shared-memory trees and barriers.
- Softmax and LayerNorm need stable online reductions.
- Divergence should be avoided in hot loops.
- Global-memory accesses should be coalesced.
- Tensor-core `MMA` should be used for matrix-heavy ops.

The hardware model in this document therefore directly influences the compiler's scheduler, tiling passes, and ISA emitter.

---

## 12. Programming best practices

These guidelines apply to both hand-written GPU kernels and the code generated
by the MLIR compiler.

### 12.1 Keep enough parallelism

- Launch enough thread blocks to fill all SMs.
- Choose block sizes as multiples of 32.
- Avoid blocks that are too small because they waste scheduling slots.
- Avoid blocks that are too large if they reduce occupancy through excessive
  register or shared-memory use.

### 12.2 Make global-memory access coalesced

- Adjacent lanes in a warp should access adjacent addresses.
- Prefer contiguous layouts for the innermost loop.
- Use vectorized loads and stores where possible.
- Keep data in registers once loaded.

### 12.3 Minimize divergence

- Avoid data-dependent branches in hot loops.
- Prefer arithmetic or `SEL` over divergent `if/else`.
- If branching is necessary, make branches uniform across the whole warp.

### 12.4 Use shared memory for reuse

- Tile matrix multiplication, convolution, and attention.
- Load a tile into shared memory once and reuse it many times.
- Use `BAR.SYNC` to make shared-memory writes visible to the block.
- Pad shared-memory layouts to avoid bank conflicts.

### 12.5 Reduce memory traffic

- Keep frequently reused values in registers.
- Prefer shared memory over global memory for inter-thread reuse.
- Fuse elementwise operations with producers/consumers when possible.
- Prefetch the next tile while computing the current tile.

### 12.6 Use tensor cores for matrix-heavy work

- Map matmul, convolution-as-GEMM, and attention score/projection steps to
  tensor-core operations when available.
- Stage matrix tiles in shared memory.
- Use wider matrix tiles to amortize overhead.

### 12.7 Use numerically stable reductions

- Use online max/sum for softmax.
- Use two-pass or Welford-style reductions for mean and variance.
- Prefer tree reductions in shared memory.
- Keep accumulation in a wider type when available.

### 12.8 Use synchronization only when necessary

- Insert barriers only where shared memory is reused or inter-thread ordering
  is required.
- Avoid barriers inside divergent code.
- Let independent warps proceed so the scheduler can hide latency.

### 12.9 Profile, then optimize

- Measure cycles, memory traffic, occupancy, and stall reasons.
- Optimize the actual bottleneck rather than guessing.
- The simulator should eventually report cache hits, bank conflicts, and
  stall reasons to support this workflow.

---

## 13. Reference glossary

- **SM:** streaming multiprocessor.
- **SIMT:** single instruction, multiple threads.
- **Warp:** a lockstep group of threads, typically 32.
- **Lane:** one thread position inside a warp.
- **Active mask:** mask of lanes currently executing.
- **Divergence:** lanes in a warp taking different branch paths.
- **Reconvergence:** rejoining lanes after a divergent branch.
- **Coalescing:** combining adjacent thread memory accesses into fewer transactions.
- **Occupancy:** number of active warps relative to hardware capacity.
- **Tensor core:** specialized matrix multiply-accumulate unit.
