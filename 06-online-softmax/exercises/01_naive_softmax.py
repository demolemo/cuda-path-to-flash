"""
Exercise 01 — Naive Softmax (3-pass)

GOAL: Row-wise softmax. 3 passes: max, sum(exp), normalize.
One block per row. Validated against torch.softmax.
"""

import torch
import sys; sys.path.insert(0, "../.."); from lib import compile_cuda_raw, check_close, bench_compare

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cfloat>

#define BLOCK_SIZE 256

__global__ void naive_softmax_kernel(const float *X, float *Y, int rows, int cols) {
    __shared__ float sdata[BLOCK_SIZE];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float *x = X + row * cols;
    float *y = Y + row * cols;

    // PASS 1: Find max
    float local_max = -FLT_MAX;
    for (int i = tid; i < cols; i += blockDim.x)
        local_max = fmaxf(local_max, x[i]);
    sdata[tid] = local_max;
    __syncthreads();
    // TODO: Reduce for max
    // YOUR CODE HERE
    float row_max = sdata[0];
    __syncthreads();

    // PASS 2: Sum of exp(x - max)
    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x)
        local_sum += expf(x[i] - row_max);
    sdata[tid] = local_sum;
    __syncthreads();
    // TODO: Reduce for sum
    // YOUR CODE HERE
    float row_sum = sdata[0];
    __syncthreads();

    // PASS 3: Normalize
    // TODO: y[i] = exp(x[i] - row_max) / row_sum
    // YOUR CODE HERE
}

void launch(const at::Tensor& X, at::Tensor& Y) {
    int rows = X.size(0), cols = X.size(1);
    naive_softmax_kernel<<<rows, BLOCK_SIZE>>>(
        X.data_ptr<float>(), Y.data_ptr<float>(), rows, cols);
}

TORCH_LIBRARY(naive_sm_mod, m) {
    m.def("launch(Tensor X, Tensor(a!) Y) -> ()");
    m.impl("launch", &launch);
}
"""

compile_cuda_raw(CUDA_SRC, name="naive_sm_mod")

print("\n--- naive softmax tests ---")
for R, C in [(1, 128), (1024, 2048), (512, 4096)]:
    X = torch.randn(R, C, device="cuda") * 5
    Y = torch.empty_like(X)
    torch.ops.naive_sm_mod.launch(X, Y)
    expected = torch.softmax(X, dim=-1)
    check_close(Y, expected, name=f"{R}×{C}", atol=1e-4, rtol=1e-4)

R, C = 1024, 4096
X = torch.randn(R, C, device="cuda") * 5; Y = torch.empty_like(X)
bench_compare({
    "naive 3-pass":  (lambda: torch.ops.naive_sm_mod.launch(X, Y),),
    "torch.softmax": (lambda: torch.softmax(X, dim=-1),),
})
