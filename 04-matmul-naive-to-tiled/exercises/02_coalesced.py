"""
Kernel 2 — Coalesced SGEMM

GOAL: Fix memory access pattern. threadIdx.x → columns (consecutive in memory).
"""

import torch
import sys; sys.path.insert(0, "../.."); from lib import compile_cuda_raw, check_close, bench_compare

CUDA_SRC = r"""
#include <torch/extension.h>

__global__ void sgemm_coalesced(const float *A, const float *B, float *C,
                                 int M, int N, int K) {
    // KEY CHANGE: threadIdx.x maps to col, threadIdx.y maps to row
    // This makes adjacent threads access adjacent memory locations
    // TODO:
    // int col = blockIdx.x * blockDim.x + threadIdx.x;
    // int row = blockIdx.y * blockDim.y + threadIdx.y;
    // YOUR CODE HERE
}

void launch(const at::Tensor& A, const at::Tensor& B, at::Tensor& C) {
    int M = A.size(0), K = A.size(1), N = B.size(1);
    dim3 block(32, 32);
    dim3 grid((N + 31) / 32, (M + 31) / 32);
    sgemm_coalesced<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
}

TORCH_LIBRARY(sgemm_coal_mod, m) {
    m.def("launch(Tensor A, Tensor B, Tensor(a!) C) -> ()");
    m.impl("launch", &launch);
}
"""

compile_cuda_raw(CUDA_SRC, name="sgemm_coal_mod")

print("\n--- coalesced SGEMM tests ---")
for M, K, N in [(128, 128, 128), (1024, 1024, 1024)]:
    A = torch.randn(M, K, device="cuda")
    B = torch.randn(K, N, device="cuda")
    C = torch.zeros(M, N, device="cuda")
    torch.ops.sgemm_coal_mod.launch(A, B, C)
    check_close(C, torch.mm(A, B), name=f"{M}×{K}×{N}", atol=1e-2, rtol=1e-2)

M = K = N = 4096
A = torch.randn(M, K, device="cuda"); B = torch.randn(K, N, device="cuda"); C = torch.zeros(M, N, device="cuda")
bench_compare({
    "coalesced SGEMM": (lambda: torch.ops.sgemm_coal_mod.launch(A, B, C),),
    "torch.mm":        (lambda: torch.mm(A, B),),
})
