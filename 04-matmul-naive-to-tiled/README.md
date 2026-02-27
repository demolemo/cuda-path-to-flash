# Module 04 — Matmul: Naive → Tiled (~30% cuBLAS)

## Learning Goals

1. Profile naive matmul and understand why it's slow
2. Implement SGEMM with shared memory tiling (you've done this — now benchmark it)
3. Implement 1D block-tiling: each thread computes a column of the output tile
4. Understand the compute-to-memory ratio and why tiling helps
5. Get to ~30% cuBLAS performance

## Context

You've already written:
- `sgemm_naive` in your `~/projects/cuda/` repo
- Multiple tiled matmul variants in `~/projects/pmpp/chapter05/`
- Vectorized vector add with `float4` in `~/projects/leet-gpu/`

Now we follow **Simon Boehm's blog post** systematically and BENCHMARK every step.

The full progression (Kernels 1-5 in this module):
```
Kernel 1: Naive                        → ~1-2% cuBLAS
Kernel 2: Global mem coalescing fix    → ~3-5% cuBLAS
Kernel 3: Shared memory tiling         → ~10-15% cuBLAS
Kernel 4: 1D block-tiling (TM)         → ~25-30% cuBLAS
Kernel 5: 2D block-tiling (TM × TN)   → ~30-40% cuBLAS
```

## Theory

### Why Matmul?

Matrix multiplication is THE kernel to optimize because:
1. It's the core of neural network training & inference
2. It has high arithmetic intensity (O(N³) compute / O(N²) data)
3. cuBLAS is incredibly optimized — matching it teaches you everything
4. Every optimization technique shows up here

### SGEMM Convention

We compute: `C = alpha * A @ B + beta * C`

Where A is M×K, B is K×N, C is M×N, all row-major float32.

For simplicity we'll use alpha=1, beta=0 in exercises (just C = A×B).

### Kernel 1 → 2: Coalescing Fix

In naive matmul, if thread (x,y) computes C[x][y], adjacent threads (same warp)
have adjacent `x` values but access `A[x * K + i]` — that's a stride-K access pattern.

Fix: swap the mapping so adjacent threads have adjacent `y` (column) values,
making B accesses coalesced.

### Kernel 2 → 3: Shared Memory Tiling

You know this from PMPP Ch.5. Load a TILE×TILE block from A and B into shared memory,
compute partial products, slide the tile along K dimension.

Reduces global memory reads from O(K) per thread to O(K/TILE_SIZE) per thread.

### Kernel 3 → 4: 1D Block-tiling

Instead of each thread computing ONE output element, each thread computes
TM elements (a column of the output tile). This increases the work per thread
and amortizes the shared memory loads.

```
Block tile: BM × BN (e.g., 64×64)
Each thread: computes TM elements (e.g., TM=8 → each thread does 8 rows)
Threads per block: (BM/TM) × BN = (64/8) × 64 = 512
```

### Kernel 4 → 5: 2D Block-tiling

Each thread computes a TM × TN sub-tile of the output. Now you load values
from A into registers (TM values) and B into registers (TN values), and compute
the outer product: TM × TN FMAs per inner loop step.

```
Each thread: TM × TN = 8 × 8 = 64 output elements
Threads per block: (BM/TM) × (BN/TN) = 8 × 8 = 64
```

## Reading List

1. **Simon Boehm** — [How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM) — Kernels 1-5
2. **NVIDIA Blog** — [CUTLASS: Fast Linear Algebra in CUDA C++](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)
3. **wangzyon/NVIDIA_SGEMM_PRACTICE** — [GitHub](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE)
4. Your own code: `~/projects/pmpp/chapter05/tiling_matmul.cu`, `~/projects/cuda/matrix_mul_naive.cu`

## Exercises

| File | What You'll Do | Target |
|------|---------------|--------|
| `exercises/01_naive.cu` | Naive SGEMM + benchmark vs cuBLAS | Baseline |
| `exercises/02_coalesced.cu` | Fix memory access pattern | ~5% cuBLAS |
| `exercises/03_shmem_tiled.cu` | Shared memory tiling | ~15% cuBLAS |
| `exercises/04_1d_blocktile.cu` | 1D block-tiling (TM) | ~30% cuBLAS |
| `exercises/05_2d_blocktile.cu` | 2D block-tiling (TM×TN) | ~40% cuBLAS |
