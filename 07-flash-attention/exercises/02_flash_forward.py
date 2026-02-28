"""
Exercise 02 — Flash Attention Forward Pass ⚡

THE FINAL BOSS.

Compute exact attention WITHOUT materializing the N×N matrix.
Tile over K/V blocks, use online softmax to maintain running statistics.
Validated against F.scaled_dot_product_attention.

Start SMALL (N=64, d=32). Get it correct. Then scale up.
"""

import torch
import torch.nn.functional as F
import sys; sys.path.insert(0, "../.."); from lib import compile_cuda_raw, check_close, bench_compare

Br = 32  # Q tile rows
Bc = 32  # K/V tile rows

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cfloat>

#define Br 32
#define Bc 32
#define MAX_D 128

// Flash Attention Forward
//
// Grid: one block per Q tile
// Each thread handles one query row within the tile
//
// Algorithm:
//   Load Q row into registers
//   Init: m = -inf, l = 0, O[d] = 0
//   For each K/V tile j:
//     Compute S = q · K[j:j+Bc]^T / sqrt(d)  →  Bc scores
//     m_new = max(m, max(S))
//     P = exp(S - m_new)
//     l_new = l * exp(m - m_new) + sum(P)
//     Rescale: O = O * (l * exp(m - m_new) / l_new) + (P @ V_tile) / l_new
//     Update m, l
//   Write O to global memory

__global__ void flash_attn_fwd(const float *Q, const float *K,
                                const float *V, float *O,
                                int N, int d) {
    int tid = threadIdx.x;
    int q_row = blockIdx.x * Br + tid;
    if (q_row >= N) return;

    float scale = 1.0f / sqrtf((float)d);

    // TODO: Implement Flash Attention forward
    //
    // Hint (simplest version — one thread per query row):
    //   float q[MAX_D], o[MAX_D];
    //   Load q from Q[q_row]
    //   float m_i = -FLT_MAX, l_i = 0;
    //   Zero out o[d]
    //
    //   for (int j = 0; j < N; j += Bc):
    //     float S[Bc], P[Bc]
    //     Compute S[c] = dot(q, K[j+c]) * scale for c in 0..Bc
    //     m_ij = max(S)
    //     P[c] = exp(S[c] - m_ij)
    //     l_ij = sum(P)
    //
    //     m_new = max(m_i, m_ij)
    //     l_new = l_i * exp(m_i - m_new) + l_ij * exp(m_ij - m_new)
    //
    //     float scale_old = l_i * exp(m_i - m_new) / l_new
    //     float scale_new = exp(m_ij - m_new) / l_new
    //     for k in 0..d:
    //       pv = sum_c P[c] * V[(j+c)*d + k]
    //       o[k] = o[k] * scale_old + pv * scale_new
    //
    //     m_i = m_new, l_i = l_new
    //
    //   Write o[d] → O[q_row * d + k]
    //
    // YOUR CODE HERE
}

void launch(const at::Tensor& Q, const at::Tensor& K,
            const at::Tensor& V, at::Tensor& O) {
    int N = Q.size(0), d = Q.size(1);
    int num_q_tiles = (N + Br - 1) / Br;
    flash_attn_fwd<<<num_q_tiles, Br>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(),
        V.data_ptr<float>(), O.data_ptr<float>(), N, d);
}

TORCH_LIBRARY(flash_fwd_mod, m) {
    m.def("launch(Tensor Q, Tensor K, Tensor V, Tensor(a!) O) -> ()");
    m.impl("launch", &launch);
}
"""

compile_cuda_raw(CUDA_SRC, name="flash_fwd_mod")

print("\n--- Flash Attention forward tests ---")

for N, d in [(32, 32), (64, 32), (128, 64), (256, 64)]:
    Q = torch.randn(N, d, device="cuda") * 0.3
    K = torch.randn(N, d, device="cuda") * 0.3
    V = torch.randn(N, d, device="cuda") * 0.3
    O = torch.zeros(N, d, device="cuda")

    torch.ops.flash_fwd_mod.launch(Q, K, V, O)

    with torch.no_grad():
        expected = F.scaled_dot_product_attention(
            Q.unsqueeze(0).unsqueeze(0), K.unsqueeze(0).unsqueeze(0),
            V.unsqueeze(0).unsqueeze(0)
        ).squeeze(0).squeeze(0)

    check_close(O, expected, name=f"N={N} d={d}", atol=5e-2, rtol=5e-2)

# Benchmark
N, d = 512, 64
Q = torch.randn(N, d, device="cuda")*0.3; K = torch.randn(N, d, device="cuda")*0.3; V = torch.randn(N, d, device="cuda")*0.3
O = torch.zeros(N, d, device="cuda")
q4 = Q.unsqueeze(0).unsqueeze(0); k4 = K.unsqueeze(0).unsqueeze(0); v4 = V.unsqueeze(0).unsqueeze(0)

bench_compare({
    "flash_attn":     (lambda: torch.ops.flash_fwd_mod.launch(Q, K, V, O.zero_()),),
    "F.sdpa":         (lambda: F.scaled_dot_product_attention(q4, k4, v4),),
})

print("\n🔥 If all tests pass, you just wrote Flash Attention from scratch!")
