"""
Kernel 1 — Naive SGEMM + benchmark vs torch.mm

GOAL: Simplest possible matmul. One thread, one output element.
Measure how far from torch.mm (which uses cuBLAS internally).
"""

import torch
import sys; sys.path.insert(0, "../.."); from lib import compile_cuda_raw, check_close, bench_compare, gpu_info

gpu_info()

CUDA_SRC = r"""
#include <torch/extension.h>

__global__ void sgemm_naive(const float *A, const float *B, float *C,
                             int M, int N, int K) {
    // TODO: Each thread computes C[row][col]
    // row = blockIdx.y * blockDim.y + threadIdx.y
    // col = blockIdx.x * blockDim.x + threadIdx.x
    // C[row*N+col] = sum_k A[row*K+k] * B[k*N+col]
    // YOUR CODE HERE
}

void launch(const at::Tensor& A, const at::Tensor& B, at::Tensor& C) {
    int M = A.size(0), K = A.size(1), N = B.size(1);
    dim3 block(32, 32);
    dim3 grid((N + 31) / 32, (M + 31) / 32);
    sgemm_naive<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
}

TORCH_LIBRARY(sgemm_naive_mod, m) {
    m.def("launch(Tensor A, Tensor B, Tensor(a!) C) -> ()");
    m.impl("launch", &launch);
}
"""

compile_cuda_raw(CUDA_SRC, name="sgemm_naive_mod")

print("\n--- naive SGEMM tests ---")

for M, K, N in [(128, 128, 128), (512, 256, 512), (1024, 1024, 1024)]:
    A = torch.randn(M, K, device="cuda")
    B = torch.randn(K, N, device="cuda")
    C = torch.zeros(M, N, device="cuda")

    torch.ops.sgemm_naive_mod.launch(A, B, C)
    expected = torch.mm(A, B)
    check_close(C, expected, name=f"{M}×{K}×{N}", atol=1e-2, rtol=1e-2)

# Benchmark
M = K = N = 4096
A = torch.randn(M, K, device="cuda")
B = torch.randn(K, N, device="cuda")
C = torch.zeros(M, N, device="cuda")

def my_mm(): torch.ops.sgemm_naive_mod.launch(A, B, C)
def torch_mm(): torch.mm(A, B)

bench_compare({
    "naive SGEMM": (my_mm,),
    "torch.mm":    (torch_mm,),
})
