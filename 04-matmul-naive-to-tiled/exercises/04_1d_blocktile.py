"""
Kernel 4 — 1D Block-Tiling

GOAL: Each thread computes TM output elements (a column of the block tile).
"""

import torch
import sys; sys.path.insert(0, "../.."); from lib import compile_cuda_raw, check_close, bench_compare

CUDA_SRC = r"""
#include <torch/extension.h>

#define BM 64
#define BN 64
#define BK 8
#define TM 8

__global__ void sgemm_1d_blocktile(const float *A, const float *B, float *C,
                                    int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // threadRow = threadIdx.x / BN, threadCol = threadIdx.x % BN
    // Each thread computes TM elements: C[threadRow*TM+0..TM-1][threadCol]
    //
    // Outer loop over K in steps of BK:
    //   Collaboratively load As, Bs
    //   Inner loop: for each k in BK, reuse Bs[k][threadCol] across TM rows
    //
    // TODO: YOUR CODE HERE
}

void launch(const at::Tensor& A, const at::Tensor& B, at::Tensor& C) {
    int M = A.size(0), K = A.size(1), N = B.size(1);
    int threads_per_block = (BM / TM) * BN;
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_1d_blocktile<<<grid, threads_per_block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
}

TORCH_LIBRARY(sgemm_1dbt_mod, m) {
    m.def("launch(Tensor A, Tensor B, Tensor(a!) C) -> ()");
    m.impl("launch", &launch);
}
"""

compile_cuda_raw(CUDA_SRC, name="sgemm_1dbt_mod")

print("\n--- 1D block-tiled SGEMM tests ---")
for M, K, N in [(128, 128, 128), (512, 256, 512), (1024, 1024, 1024)]:
    A = torch.randn(M, K, device="cuda")
    B = torch.randn(K, N, device="cuda")
    C = torch.zeros(M, N, device="cuda")
    torch.ops.sgemm_1dbt_mod.launch(A, B, C)
    check_close(C, torch.mm(A, B), name=f"{M}×{K}×{N}", atol=1e-1, rtol=1e-2)

M = K = N = 4096
A = torch.randn(M, K, device="cuda"); B = torch.randn(K, N, device="cuda"); C = torch.zeros(M, N, device="cuda")
bench_compare({
    "1D blocktile": (lambda: torch.ops.sgemm_1dbt_mod.launch(A, B, C),),
    "torch.mm":     (lambda: torch.mm(A, B),),
})
