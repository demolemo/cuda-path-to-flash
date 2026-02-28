"""
Kernel 5 — 2D Block-Tiling

GOAL: Each thread computes TM × TN output elements via outer product.
This is where matmul gets REAL.
"""

import torch
import sys; sys.path.insert(0, "../.."); from lib import compile_cuda_raw, check_close, bench_compare

CUDA_SRC = r"""
#include <torch/extension.h>

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

__global__ void sgemm_2d_blocktile(const float *A, const float *B, float *C,
                                    int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // threadRow = threadIdx.x / (BN/TN)
    // threadCol = threadIdx.x % (BN/TN)
    // float regA[TM], regB[TN], results[TM*TN] = {0}
    //
    // Outer loop over K/BK tiles:
    //   Collaboratively load As, Bs (each thread loads multiple elements)
    //   Inner loop over BK:
    //     Load regA[TM] from As, regB[TN] from Bs
    //     Outer product: results[i*TN+j] += regA[i] * regB[j]
    //   Write results to C
    //
    // TODO: YOUR CODE HERE
}

void launch(const at::Tensor& A, const at::Tensor& B, at::Tensor& C) {
    int M = A.size(0), K = A.size(1), N = B.size(1);
    int threads_per_block = (BM / TM) * (BN / TN);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_2d_blocktile<<<grid, threads_per_block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
}

TORCH_LIBRARY(sgemm_2dbt_mod, m) {
    m.def("launch(Tensor A, Tensor B, Tensor(a!) C) -> ()");
    m.impl("launch", &launch);
}
"""

compile_cuda_raw(CUDA_SRC, name="sgemm_2dbt_mod")

print("\n--- 2D block-tiled SGEMM tests ---")
for M, K, N in [(128, 128, 128), (512, 256, 512), (1024, 1024, 1024)]:
    A = torch.randn(M, K, device="cuda")
    B = torch.randn(K, N, device="cuda")
    C = torch.zeros(M, N, device="cuda")
    torch.ops.sgemm_2dbt_mod.launch(A, B, C)
    check_close(C, torch.mm(A, B), name=f"{M}×{K}×{N}", atol=1e-1, rtol=1e-2)

M = K = N = 4096
A = torch.randn(M, K, device="cuda"); B = torch.randn(K, N, device="cuda"); C = torch.zeros(M, N, device="cuda")
bench_compare({
    "2D blocktile": (lambda: torch.ops.sgemm_2dbt_mod.launch(A, B, C),),
    "torch.mm":     (lambda: torch.mm(A, B),),
})
