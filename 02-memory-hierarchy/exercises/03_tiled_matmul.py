"""
Exercise 03 — Tiled Matmul (From Memory!)

GOAL: Write tiled matmul again. No peeking at your PMPP code.
  1. Square (N % TILE == 0)
  2. General (arbitrary M, K, N)

Validated against torch.mm — the beauty of this approach.
"""

import torch
import sys; sys.path.insert(0, "../.."); from lib import compile_cuda_raw, check_close, bench_compare

# ============================================================
# YOUR CUDA KERNELS
# ============================================================
CUDA_SRC = r"""
#include <torch/extension.h>

#define TILE_SIZE 16

// C(M×N) = A(M×K) × B(K×N)
__global__ void tiled_matmul_kernel(const float *A, const float *B, float *C,
                                     int M, int K, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // TODO: Tiled matmul with boundary checks
    // YOUR CODE HERE
}

void launch_matmul(const at::Tensor& A, const at::Tensor& B, at::Tensor& C) {
    int M = A.size(0), K = A.size(1), N = B.size(1);
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    tiled_matmul_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
}

TORCH_LIBRARY(tiled_mm_mod, m) {
    m.def("launch(Tensor A, Tensor B, Tensor(a!) C) -> ()");
    m.impl("launch", &launch_matmul);
}
"""

compile_cuda_raw(CUDA_SRC, name="tiled_mm_mod")

# ============================================================
# Test
# ============================================================
print("\n--- tiled matmul tests ---")

for M, K, N in [(512, 512, 512), (237, 419, 173), (1, 1024, 1), (1024, 64, 2048)]:
    A = torch.randn(M, K, device="cuda")
    B = torch.randn(K, N, device="cuda")
    C = torch.zeros(M, N, device="cuda")

    torch.ops.tiled_mm_mod.launch(A, B, C)
    expected = torch.mm(A, B)

    check_close(C, expected, name=f"{M}×{K} @ {K}×{N}", atol=1e-2, rtol=1e-2)

# ============================================================
# Benchmark
# ============================================================
M = K = N = 2048
A = torch.randn(M, K, device="cuda")
B = torch.randn(K, N, device="cuda")
C = torch.zeros(M, N, device="cuda")

bench_compare({
    "your tiled": (lambda: torch.ops.tiled_mm_mod.launch(A, B, C.zero_()),),
    "torch.mm":   (lambda: torch.mm(A, B),),
})
