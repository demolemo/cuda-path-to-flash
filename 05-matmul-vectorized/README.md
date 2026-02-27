# Module 05 — Matmul: Vectorized & Beyond (60%+ cuBLAS)

## Learning Goals

1. Use vectorized memory access (`float4`) for global memory loads
2. Implement double buffering (prefetch next tile while computing current)
3. Understand warp-level tiling
4. Resolve register pressure and bank conflict issues
5. Reach 60%+ cuBLAS performance

## Context

You already explored vectorized loads in your leet-gpu repo:
- `reinterpret_cast_2.cu` and `reinterpret_cast_4.cu` for vector add
- You noticed float4 was slower than float2 and hypothesized register spilling
- You found that `threadsPerBlock=128` with `float2` was optimal on H100

Now apply these ideas to matmul.

## Theory

### Vectorized Loads (float4)

Loading `float4` (128 bits) instead of `float` (32 bits) reduces the number of
memory transactions by 4x. The GPU issues fewer, wider loads.

```cuda
// Instead of:
As[row][col] = A[global_idx];

// Do:
float4 tmp = reinterpret_cast<const float4*>(&A[global_idx])[0];
As[row][col]     = tmp.x;
As[row][col + 1] = tmp.y;
As[row][col + 2] = tmp.z;
As[row][col + 3] = tmp.w;
```

Requires BK to be a multiple of 4 and addresses to be aligned.

### Transposing As During Load

When we later read `As[threadRow*TM + i][k]` in the inner loop, adjacent threads
read different rows — that's a bank conflict! Fix: store As transposed.

```cuda
// Load A into As transposed: As[BK][BM] instead of As[BM][BK]
As[col][row] = A[...];  // note the swap
// Now inner loop reads As[k][threadRow*TM + i] — adjacent threads read adjacent addresses
```

### Double Buffering

Overlap the NEXT tile load with the CURRENT tile computation:
1. Load tile 0 into buffer A
2. Loop:
   a. Start async load of tile i+1 into buffer B
   b. Compute using tile i from buffer A
   c. Swap buffers

On modern GPUs (Ampere+), use `cp.async` for truly asynchronous copies.
On older GPUs, you can still get benefit by restructuring the code.

### Warp Tiling

Within the 2D block-tile, organize the thread-tiles so that threads within
a warp compute adjacent elements. This improves shared memory access patterns.

Instead of mapping thread tiles row-by-row, map them in a warp-friendly pattern:
- Warp 0 handles a contiguous region of the output tile
- Adjacent threads within the warp handle adjacent columns

### Register Pressure

With TM=TN=8, each thread has:
- 64 result registers (TM × TN)
- 8 regA + 8 regB = 16 register
- Total: ~80+ registers per thread

At 256 threads/block, that's ~20K registers/block.
If you exceed the SM register file, you get **register spilling** to local memory (slow).
Use `--ptxas-options=-v` to check register usage.

## Reading List

1. **Simon Boehm** — [Continued: Kernels 6-10](https://siboehm.com/articles/22/CUDA-MMM)
2. **Your own experiments** — `~/projects/leet-gpu/vector-addition/` (float4 work)
3. **NVIDIA** — [CUTLASS documentation](https://github.com/NVIDIA/cutlass/blob/main/media/docs/fundamental_types.md)
4. **wangzyon** — [NVIDIA_SGEMM_PRACTICE](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE)

## Exercises

| File | What You'll Do | Target |
|------|---------------|--------|
| `exercises/01_vectorized.cu` | Add float4 loads to 2D block-tiled kernel | ~45% cuBLAS |
| `exercises/02_resolve_conflicts.cu` | Transpose As, pad shared memory | ~55% cuBLAS |
| `exercises/03_warptiling.cu` | Warp-level tiling for better locality | ~60%+ cuBLAS |

## Profiling Checklist

At this stage you MUST use Nsight Compute. Key metrics to check:
```bash
ncu --set full -o profile ./your_kernel
```

- **Memory throughput** — are you saturating bandwidth?
- **Compute throughput** — are you saturating FMA units?
- **Occupancy** — how many warps are active?
- **Bank conflicts** — check shared memory efficiency
- **Register usage** — `--ptxas-options=-v` during compilation
