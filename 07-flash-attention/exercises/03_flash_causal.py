"""
Exercise 03 — Flash Attention with Causal Masking

GOAL: Add causal mask — position i only attends to j <= i.
Optimization: skip entire K/V tiles that are fully masked.
"""

import torch
import torch.nn.functional as F
import sys; sys.path.insert(0, "../.."); from lib import compile_cuda_raw, check_close, bench_compare

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cfloat>

#define Br 32
#define Bc 32
#define MAX_D 128

__global__ void flash_attn_causal(const float *Q, const float *K,
                                   const float *V, float *O,
                                   int N, int d) {
    int tid = threadIdx.x;
    int q_row = blockIdx.x * Br + tid;
    if (q_row >= N) return;

    // TODO: Same as 02_flash_forward but:
    //   - After computing S[c], apply causal mask:
    //     if (j + c > q_row) S[c] = -FLT_MAX;
    //   - Optimization: if j > q_row (entire tile is masked), skip
    //
    // YOUR CODE HERE
}

void launch(const at::Tensor& Q, const at::Tensor& K,
            const at::Tensor& V, at::Tensor& O) {
    int N = Q.size(0), d = Q.size(1);
    int num_q_tiles = (N + Br - 1) / Br;
    flash_attn_causal<<<num_q_tiles, Br>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(),
        V.data_ptr<float>(), O.data_ptr<float>(), N, d);
}

TORCH_LIBRARY(flash_causal_mod, m) {
    m.def("launch(Tensor Q, Tensor K, Tensor V, Tensor(a!) O) -> ()");
    m.impl("launch", &launch);
}
"""

compile_cuda_raw(CUDA_SRC, name="flash_causal_mod")

print("\n--- Flash Attention causal tests ---")

for N, d in [(32, 32), (64, 32), (128, 64), (256, 64)]:
    Q = torch.randn(N, d, device="cuda") * 0.3
    K = torch.randn(N, d, device="cuda") * 0.3
    V = torch.randn(N, d, device="cuda") * 0.3
    O = torch.zeros(N, d, device="cuda")

    torch.ops.flash_causal_mod.launch(Q, K, V, O)

    # PyTorch reference with causal mask
    with torch.no_grad():
        expected = F.scaled_dot_product_attention(
            Q.unsqueeze(0).unsqueeze(0), K.unsqueeze(0).unsqueeze(0),
            V.unsqueeze(0).unsqueeze(0), is_causal=True
        ).squeeze(0).squeeze(0)

    check_close(O, expected, name=f"N={N} d={d}", atol=5e-2, rtol=5e-2)

print("\n🎉 Flash Attention with causal masking — DONE.")
print("You went from 'what is a thread' to Flash Attention. No university needed.")
