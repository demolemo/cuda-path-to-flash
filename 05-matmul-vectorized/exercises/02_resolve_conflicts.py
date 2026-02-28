"""
Kernel 7 — Resolving Bank Conflicts

GOAL: Transpose As during load (As[BK][BM] instead of As[BM][BK])
to eliminate shared memory bank conflicts.
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

__global__ void sgemm_no_bc(const float *A, const float *B, float *C,
                             int M, int N, int K) {
    __shared__ float As[BK][BM];  // transposed!
    __shared__ float Bs[BK][BN];

    // During load: As[col][row] = A[...]  (store transposed)
    // During compute: regA[i] = As[k][threadRow * TM + i]
    // This eliminates bank conflicts on As reads.
    //
    // TODO: YOUR CODE HERE
}

void launch(const at::Tensor& A, const at::Tensor& B, at::Tensor& C) {
    int M = A.size(0), K = A.size(1), N = B.size(1);
    int tpb = (BM / TM) * (BN / TN);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_no_bc<<<grid, tpb>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
}

TORCH_LIBRARY(sgemm_nobc_mod, m) {
    m.def("launch(Tensor A, Tensor B, Tensor(a!) C) -> ()");
    m.impl("launch", &launch);
}
"""

compile_cuda_raw(CUDA_SRC, name="sgemm_nobc_mod")

print("\n--- no-bank-conflict SGEMM tests ---")
for M, K, N in [(128, 128, 128), (1024, 1024, 1024)]:
    A = torch.randn(M, K, device="cuda"); B = torch.randn(K, N, device="cuda"); C = torch.zeros(M, N, device="cuda")
    torch.ops.sgemm_nobc_mod.launch(A, B, C)
    check_close(C, torch.mm(A, B), name=f"{M}×{K}×{N}", atol=1e-1, rtol=1e-2)

M = K = N = 4096
A = torch.randn(M, K, device="cuda"); B = torch.randn(K, N, device="cuda"); C = torch.zeros(M, N, device="cuda")
bench_compare({"no-bc": (lambda: torch.ops.sgemm_nobc_mod.launch(A, B, C),), "torch.mm": (lambda: torch.mm(A, B),)})
