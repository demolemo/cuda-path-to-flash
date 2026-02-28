"""
Exercise 02 — Online Softmax (2-pass, Milakov & Gimelshein)

GOAL: Compute max + sum in a SINGLE pass using the online algorithm.
Then normalize in a second pass. 2 passes instead of 3.

The online update: m_new = max(m, x_i), d = d * exp(m-m_new) + exp(x_i-m_new)
The merge rule:    m = max(m1,m2), d = d1*exp(m1-m) + d2*exp(m2-m)

This is THE algorithm that makes Flash Attention possible.
"""

import torch
import sys; sys.path.insert(0, "../.."); from lib import compile_cuda_raw, check_close, bench_compare

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cfloat>

#define BLOCK_SIZE 256

__global__ void online_softmax_kernel(const float *X, float *Y, int rows, int cols) {
    __shared__ float s_max[BLOCK_SIZE];
    __shared__ float s_sum[BLOCK_SIZE];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float *x = X + row * cols;
    float *y = Y + row * cols;

    // PASS 1: Online max + sum
    float local_m = -FLT_MAX;
    float local_d = 0.0f;

    // TODO: Scan elements with online update rule
    // for (int i = tid; i < cols; i += blockDim.x) {
    //     float val = x[i];
    //     float m_new = fmaxf(local_m, val);
    //     local_d = local_d * expf(local_m - m_new) + expf(val - m_new);
    //     local_m = m_new;
    // }
    // YOUR CODE HERE

    s_max[tid] = local_m;
    s_sum[tid] = local_d;
    __syncthreads();

    // TODO: Reduce (m, d) pairs with merge rule
    // float m_new = fmaxf(m1, m2);
    // d_new = d1 * expf(m1 - m_new) + d2 * expf(m2 - m_new);
    // YOUR CODE HERE

    float row_max = s_max[0];
    float row_sum = s_sum[0];
    __syncthreads();

    // PASS 2: Normalize
    // TODO: y[i] = exp(x[i] - row_max) / row_sum
    // YOUR CODE HERE
}

void launch(const at::Tensor& X, at::Tensor& Y) {
    int rows = X.size(0), cols = X.size(1);
    online_softmax_kernel<<<rows, BLOCK_SIZE>>>(
        X.data_ptr<float>(), Y.data_ptr<float>(), rows, cols);
}

TORCH_LIBRARY(online_sm_mod, m) {
    m.def("launch(Tensor X, Tensor(a!) Y) -> ()");
    m.impl("launch", &launch);
}
"""

compile_cuda_raw(CUDA_SRC, name="online_sm_mod")

print("\n--- online softmax tests ---")
for R, C in [(1, 128), (1024, 2048), (512, 4096)]:
    X = torch.randn(R, C, device="cuda") * 5
    Y = torch.empty_like(X)
    torch.ops.online_sm_mod.launch(X, Y)
    expected = torch.softmax(X, dim=-1)
    check_close(Y, expected, name=f"{R}×{C}", atol=1e-4, rtol=1e-4)

R, C = 1024, 4096
X = torch.randn(R, C, device="cuda") * 5; Y = torch.empty_like(X)
bench_compare({
    "online 2-pass": (lambda: torch.ops.online_sm_mod.launch(X, Y),),
    "torch.softmax": (lambda: torch.softmax(X, dim=-1),),
})
