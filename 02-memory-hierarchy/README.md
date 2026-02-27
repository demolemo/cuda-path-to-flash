# Module 02 — Memory Hierarchy

## Learning Goals

1. Understand the full GPU memory hierarchy: registers → shared → L1/L2 → global (DRAM)
2. Write kernels that use shared memory effectively
3. Understand and fix bank conflicts
4. Achieve coalesced global memory access patterns
5. Apply corner turning optimization
6. Reason about compute-bound vs memory-bound kernels (roofline model)

## Theory

### The Memory Wall

The #1 bottleneck in GPU programming is almost always **memory bandwidth**, not compute.
A modern GPU (H100) can do ~1000 TFLOPS but only move ~3 TB/s from global memory.
That's a ratio of ~300 FLOPS per byte. Most kernels don't come close to that arithmetic intensity.

```
Registers:     ~0 cycles latency    | ~255 per thread     | Per-thread private
Shared mem:    ~5 cycles            | 48-228 KB per SM    | Per-block shared
L1 cache:      ~30 cycles           | 128-256 KB per SM   | Automatic
L2 cache:      ~200 cycles          | 4-50 MB total       | Automatic
Global (DRAM): ~400-800 cycles      | GBs                 | All threads
```

### Memory Coalescing

When threads in a warp access global memory, the hardware tries to **coalesce** their requests into
as few transactions as possible. If 32 threads each read consecutive 4-byte values, that's ONE
128-byte transaction. If they read scattered addresses, you get 32 separate transactions — 32x slower.

**Rule**: Thread `i` should access address `base + i` (or nearby). Stride-1 access = good. Random access = bad.

```cuda
// GOOD — coalesced (thread i reads element i)
float val = A[blockIdx.x * blockDim.x + threadIdx.x];

// BAD — strided (thread i reads element i*stride, huge gaps)
float val = A[threadIdx.x * stride];
```

### Corner Turning

When you need to access a matrix column-wise (which is non-coalesced in row-major layout),
load a tile into shared memory with coalesced reads, then access the shared memory in whatever
pattern you need. Shared memory has no coalescing requirement.

This is exactly what tiled matmul does — and you already wrote it! (Your `sharedMemTiledMatmul` in PMPP Ch.5)

### Shared Memory & Bank Conflicts

Shared memory is divided into 32 **banks** (one per warp lane). Address `addr` maps to bank `(addr / 4) % 32`.

- If all 32 threads access **different banks** → 1 cycle (no conflict)
- If 2+ threads access the **same bank** (different addresses) → serialized (N-way conflict)
- If all threads access the **exact same address** → broadcast (no conflict!)

**Common fix**: Pad shared memory arrays by 1 element to break conflict patterns.
```cuda
__shared__ float tile[32][33];  // 33 instead of 32 — breaks bank conflicts
```

### The Roofline Model

Every kernel is either:
- **Memory-bound**: limited by how fast you can move data (most kernels)
- **Compute-bound**: limited by how fast you can do math (dense matmul at large sizes)

**Arithmetic intensity** = FLOPs / bytes transferred

If your intensity is below the GPU's ridge point, you're memory-bound.
If above, you're compute-bound.

```
         Performance
(GFLOPS) |        ___________  ← compute ceiling
         |       /
         |      /
         |     /  ← memory bandwidth slope
         |    /
         |___/________________
              Arithmetic Intensity (FLOP/byte)
```

## Your Prior Work (from PMPP repo)

You've already nailed the core concepts:
- ✅ Tiled matmul with shared memory (chapter05/tiling_matmul.cu — 5 kernel variants!)
- ✅ Corner turning concept (chapter06/anki.md)
- ✅ Memory coalescing definition and strategies
- ✅ Thread coarsening concept
- ✅ The optimization checklist (Table 6.1 from PMPP)

Now let's drill deeper: bank conflicts, actual profiling, and transpose optimization.

## Reading List

1. **PMPP Chapter 4** — Compute architecture and scheduling
2. **PMPP Chapter 5** — Memory architecture and data locality (you've done this — review)
3. **PMPP Chapter 6** — Performance considerations (coalescing, occupancy, coarsening)
4. **NVIDIA Blog** — [Using Shared Memory in CUDA C/C++](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)
5. **NVIDIA Blog** — [Efficient Matrix Transpose in CUDA](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)
6. **NVIDIA Blog** — [How to Access Global Memory Efficiently](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/)
7. **Your own notes** — `~/projects/pmpp/chapter05/notes.md`, `chapter06/notes.md`

## Exercises

| File | What You'll Do | Difficulty |
|------|---------------|------------|
| `exercises/01_shmem_reduce.cu` | Reduction using shared memory (single block) | ⭐⭐ |
| `exercises/02_transpose.cu` | Matrix transpose: naive → coalesced → no bank conflicts | ⭐⭐⭐ |
| `exercises/03_tiled_matmul.cu` | Tiled matmul from scratch (you've done this — do it again from memory!) | ⭐⭐⭐ |
