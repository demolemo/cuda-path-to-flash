"""
Kernel 8 — Warp Tiling

GOAL: Organize thread-tiles within warps for optimal shared memory access.
The hardest matmul kernel. Read Simon Boehm's blog carefully before this.
"""

import torch
import sys; sys.path.insert(0, "../.."); from lib import compile_cuda_raw, check_close, bench_compare

CUDA_SRC = r"""
#include <torch/extension.h>

#define BM 128
#define BN 128
#define BK 8
#define WM 64
#define WN 32
#define TM 8
#define TN 4

__global__ void sgemm_warptile(const float *A, const float *B, float *C,
                                int M, int N, int K) {
    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];

    // Map warp to WM×WN region of output
    // Map lane within warp to TM×TN thread tile
    // Same loading + outer product pattern, but warp-aware indexing
    //
    // TODO: YOUR CODE HERE (this is the hardest kernel)
}

void launch(const at::Tensor& A, const at::Tensor& B, at::Tensor& C) {
    int M = A.size(0), K = A.size(1), N = B.size(1);
    int warps_m = BM / WM, warps_n = BN / WN;
    int tpb = warps_m * warps_n * 32;
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_warptile<<<grid, tpb>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
}

TORCH_LIBRARY(sgemm_warp_mod, m) {
    m.def("launch(Tensor A, Tensor B, Tensor(a!) C) -> ()");
    m.impl("launch", &launch);
}
"""

compile_cuda_raw(CUDA_SRC, name="sgemm_warp_mod")

print("\n--- warp-tiled SGEMM tests ---")
for M, K, N in [(128, 128, 128), (1024, 1024, 1024)]:
    A = torch.randn(M, K, device="cuda"); B = torch.randn(K, N, device="cuda"); C = torch.zeros(M, N, device="cuda")
    torch.ops.sgemm_warp_mod.launch(A, B, C)
    check_close(C, torch.mm(A, B), name=f"{M}×{K}×{N}", atol=1e-1, rtol=1e-2)

M = K = N = 4096
A = torch.randn(M, K, device="cuda"); B = torch.randn(K, N, device="cuda"); C = torch.zeros(M, N, device="cuda")
bench_compare({"warptile": (lambda: torch.ops.sgemm_warp_mod.launch(A, B, C),), "torch.mm": (lambda: torch.mm(A, B),)})
