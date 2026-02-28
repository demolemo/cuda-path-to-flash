"""
Exercise 01 — Standard (Naive) Attention

GOAL: Materialize full N×N attention matrix as baseline.
S = Q @ K^T / sqrt(d), P = softmax(S), O = P @ V

Validated against F.scaled_dot_product_attention.
"""

import torch
import torch.nn.functional as F
import sys; sys.path.insert(0, "../.."); from lib import compile_cuda_raw, check_close, bench_compare

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cfloat>

// Step 1: S = Q @ K^T / sqrt(d)
__global__ void qk_matmul(const float *Q, const float *K, float *S,
                           int N, int d) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < d; i++)
            sum += Q[row * d + i] * K[col * d + i];
        S[row * N + col] = sum / sqrtf((float)d);
    }
}

// Step 2: row-wise softmax
__global__ void softmax_kernel(float *S, int N) {
    __shared__ float sdata[256];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    float *s = S + row * N;

    // Max
    float m = -FLT_MAX;
    for (int i = tid; i < N; i += blockDim.x) m = fmaxf(m, s[i]);
    sdata[tid] = m;
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] = fmaxf(sdata[tid], sdata[tid+stride]);
        __syncthreads();
    }
    m = sdata[0]; __syncthreads();

    // Sum
    float sum = 0;
    for (int i = tid; i < N; i += blockDim.x) sum += expf(s[i] - m);
    sdata[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid+stride];
        __syncthreads();
    }
    sum = sdata[0]; __syncthreads();

    // Normalize
    for (int i = tid; i < N; i += blockDim.x)
        s[i] = expf(s[i] - m) / sum;
}

// Step 3: O = P @ V
__global__ void pv_matmul(const float *P, const float *V, float *O,
                           int N, int d) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < d) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++)
            sum += P[row * N + i] * V[i * d + col];
        O[row * d + col] = sum;
    }
}

void launch_naive_attn(const at::Tensor& Q, const at::Tensor& K,
                       const at::Tensor& V, at::Tensor& O, at::Tensor& S) {
    int N = Q.size(0), d = Q.size(1);
    dim3 block(16, 16);
    dim3 grid_s((N+15)/16, (N+15)/16);
    dim3 grid_o((d+15)/16, (N+15)/16);

    qk_matmul<<<grid_s, block>>>(Q.data_ptr<float>(), K.data_ptr<float>(),
                                  S.data_ptr<float>(), N, d);
    softmax_kernel<<<N, 256>>>(S.data_ptr<float>(), N);
    pv_matmul<<<grid_o, block>>>(S.data_ptr<float>(), V.data_ptr<float>(),
                                  O.data_ptr<float>(), N, d);
}

TORCH_LIBRARY(naive_attn_mod, m) {
    m.def("launch(Tensor Q, Tensor K, Tensor V, Tensor(a!) O, Tensor(b!) S) -> ()");
    m.impl("launch", &launch_naive_attn);
}
"""

compile_cuda_raw(CUDA_SRC, name="naive_attn_mod")

print("\n--- naive attention tests ---")

for N, d in [(64, 32), (128, 64), (256, 64)]:
    Q = torch.randn(N, d, device="cuda") * 0.5
    K = torch.randn(N, d, device="cuda") * 0.5
    V = torch.randn(N, d, device="cuda") * 0.5
    O = torch.empty(N, d, device="cuda")
    S = torch.empty(N, N, device="cuda")

    torch.ops.naive_attn_mod.launch(Q, K, V, O, S)

    # PyTorch reference
    with torch.no_grad():
        expected = F.scaled_dot_product_attention(
            Q.unsqueeze(0).unsqueeze(0), K.unsqueeze(0).unsqueeze(0),
            V.unsqueeze(0).unsqueeze(0)
        ).squeeze(0).squeeze(0)

    check_close(O, expected, name=f"N={N} d={d}", atol=1e-2, rtol=1e-2)

N, d = 256, 64
Q = torch.randn(N, d, device="cuda")*0.5; K = torch.randn(N, d, device="cuda")*0.5; V = torch.randn(N, d, device="cuda")*0.5
O = torch.empty(N, d, device="cuda"); S = torch.empty(N, N, device="cuda")
q4 = Q.unsqueeze(0).unsqueeze(0); k4 = K.unsqueeze(0).unsqueeze(0); v4 = V.unsqueeze(0).unsqueeze(0)

bench_compare({
    "naive 3-kernel": (lambda: torch.ops.naive_attn_mod.launch(Q, K, V, O, S),),
    "F.sdpa":         (lambda: F.scaled_dot_product_attention(q4, k4, v4),),
})
