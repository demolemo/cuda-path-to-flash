"""
Exercise 02 — Matrix Transpose

GOAL: Write 3 transpose kernels, each faster:
  1. naive — non-coalesced writes (or reads)
  2. shmem — shared memory tile (has bank conflicts)
  3. shmem_nobc — padded shared memory (no bank conflicts)

B = A^T where A is H×W, B is W×H
"""

import torch
import sys; sys.path.insert(0, "../.."); from lib import compile_cuda_raw, check_close, bench_compare

TILE = 32

# ============================================================
# YOUR CUDA KERNELS
# ============================================================
CUDA_SRC = r"""
#include <torch/extension.h>

#define TILE 32

// VERSION 1: Naive transpose
__global__ void naive_transpose(const float *A, float *B, int H, int W) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // TODO: B[col * H + row] = A[row * W + col]
    // YOUR CODE HERE
}

// VERSION 2: Shared memory (with bank conflicts)
__global__ void shmem_transpose(const float *A, float *B, int H, int W) {
    __shared__ float tile[TILE][TILE];

    // TODO:
    // 1. Load tile from A (coalesced read)
    // 2. __syncthreads()
    // 3. Write transposed tile to B (coalesced write, swap block indices)
    // YOUR CODE HERE
}

// VERSION 3: Shared memory, padded (no bank conflicts)
__global__ void shmem_transpose_nobc(const float *A, float *B, int H, int W) {
    __shared__ float tile[TILE][TILE + 1];  // +1 padding!

    // TODO: Same as version 2 but with padded tile
    // YOUR CODE HERE
}

void launch_naive(const at::Tensor& A, at::Tensor& B) {
    int H = A.size(0), W = A.size(1);
    dim3 block(TILE, TILE);
    dim3 grid((W + TILE - 1) / TILE, (H + TILE - 1) / TILE);
    naive_transpose<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), H, W);
}

void launch_shmem(const at::Tensor& A, at::Tensor& B) {
    int H = A.size(0), W = A.size(1);
    dim3 block(TILE, TILE);
    dim3 grid((W + TILE - 1) / TILE, (H + TILE - 1) / TILE);
    shmem_transpose<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), H, W);
}

void launch_nobc(const at::Tensor& A, at::Tensor& B) {
    int H = A.size(0), W = A.size(1);
    dim3 block(TILE, TILE);
    dim3 grid((W + TILE - 1) / TILE, (H + TILE - 1) / TILE);
    shmem_transpose_nobc<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), H, W);
}

TORCH_LIBRARY(transpose_mod, m) {
    m.def("naive(Tensor A, Tensor(a!) B) -> ()");
    m.impl("naive", &launch_naive);
    m.def("shmem(Tensor A, Tensor(a!) B) -> ()");
    m.impl("shmem", &launch_shmem);
    m.def("nobc(Tensor A, Tensor(a!) B) -> ()");
    m.impl("nobc", &launch_nobc);
}
"""

compile_cuda_raw(CUDA_SRC, name="transpose_mod")

# ============================================================
# Test
# ============================================================
print("\n--- transpose tests ---")

for H, W in [(32, 32), (1024, 1024), (4096, 4096), (1023, 517)]:
    A = torch.randn(H, W, device="cuda")

    for name, fn in [("naive", torch.ops.transpose_mod.naive),
                     ("shmem", torch.ops.transpose_mod.shmem),
                     ("nobc",  torch.ops.transpose_mod.nobc)]:
        B = torch.empty(W, H, device="cuda")
        fn(A, B)
        check_close(B, A.T.contiguous(), name=f"{name} {H}×{W}")

# ============================================================
# Benchmark
# ============================================================
H, W = 4096, 4096
A = torch.randn(H, W, device="cuda")
B = torch.empty(W, H, device="cuda")

bench_compare({
    "naive":    (lambda: torch.ops.transpose_mod.naive(A, B),),
    "shmem":    (lambda: torch.ops.transpose_mod.shmem(A, B),),
    "nobc":     (lambda: torch.ops.transpose_mod.nobc(A, B),),
    "torch.T":  (lambda: A.T.contiguous(),),
})
