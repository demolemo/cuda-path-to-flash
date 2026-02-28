"""
Kernel 3 — Shared Memory Tiled SGEMM

GOAL: Standard tiled matmul. You've done this many times. Now benchmark it.
"""

import torch
import sys; sys.path.insert(0, "../.."); from lib import compile_cuda_raw, check_close, bench_compare

CUDA_SRC = r"""
#include <torch/extension.h>

#define TILE 32

__global__ void sgemm_shmem(const float *A, const float *B, float *C,
                             int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    // TODO: Tiled matmul — you've written this 5 times in PMPP
    // YOUR CODE HERE
}

void launch(const at::Tensor& A, const at::Tensor& B, at::Tensor& C) {
    int M = A.size(0), K = A.size(1), N = B.size(1);
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    sgemm_shmem<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
}

TORCH_LIBRARY(sgemm_shmem_mod, m) {
    m.def("launch(Tensor A, Tensor B, Tensor(a!) C) -> ()");
    m.impl("launch", &launch);
}
"""

compile_cuda_raw(CUDA_SRC, name="sgemm_shmem_mod")

print("\n--- shmem tiled SGEMM tests ---")
for M, K, N in [(128, 128, 128), (512, 256, 512), (1024, 1024, 1024)]:
    A = torch.randn(M, K, device="cuda")
    B = torch.randn(K, N, device="cuda")
    C = torch.zeros(M, N, device="cuda")
    torch.ops.sgemm_shmem_mod.launch(A, B, C)
    check_close(C, torch.mm(A, B), name=f"{M}×{K}×{N}", atol=1e-2, rtol=1e-2)

M = K = N = 4096
A = torch.randn(M, K, device="cuda"); B = torch.randn(K, N, device="cuda"); C = torch.zeros(M, N, device="cuda")
bench_compare({
    "shmem tiled": (lambda: torch.ops.sgemm_shmem_mod.launch(A, B, C),),
    "torch.mm":    (lambda: torch.mm(A, B),),
})
