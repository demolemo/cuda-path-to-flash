"""
Exercise 01 — Shared Memory Reduction

GOAL: Implement parallel reduction sum using shared memory.
Write THREE versions, each better than the last.

Reference: your PMPP Ch.10 reduction kernels
"""

import torch
import sys; sys.path.insert(0, "../.."); from lib import compile_cuda_raw, check_close, bench_compare

BLOCK_SIZE = 256

# ============================================================
# YOUR CUDA KERNELS
# ============================================================
CUDA_SRC = r"""
#include <torch/extension.h>

#define BLOCK_SIZE 256
#define COARSEN_FACTOR 4

// VERSION 1: Naive reduction (divergent branching)
__global__ void naive_reduce(const float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    // TODO: stride = 1, 2, 4, ...
    // if (tid % (2*stride) == 0) sdata[tid] += sdata[tid + stride]
    // YOUR CODE HERE

    if (tid == 0) atomicAdd(output, sdata[0]);
}

// VERSION 2: Improved reduction (non-divergent, better access pattern)
__global__ void improved_reduce(const float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    // TODO: stride = blockDim.x/2 down to 1
    // if (tid < stride) sdata[tid] += sdata[tid + stride]
    // YOUR CODE HERE

    if (tid == 0) atomicAdd(output, sdata[0]);
}

// VERSION 3: Coarsened — each thread sums multiple elements first
__global__ void coarsened_reduce(const float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x * COARSEN_FACTOR + threadIdx.x;

    // TODO: Sum COARSEN_FACTOR elements per thread
    float sum = 0.0f;
    // YOUR CODE HERE

    sdata[tid] = sum;
    __syncthreads();

    // TODO: Tree reduction
    // YOUR CODE HERE

    if (tid == 0) atomicAdd(output, sdata[0]);
}

void launch_naive(const at::Tensor& input, at::Tensor& output) {
    int n = input.size(0);
    output.zero_();
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    naive_reduce<<<blocks, BLOCK_SIZE>>>(input.data_ptr<float>(), output.data_ptr<float>(), n);
}

void launch_improved(const at::Tensor& input, at::Tensor& output) {
    int n = input.size(0);
    output.zero_();
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    improved_reduce<<<blocks, BLOCK_SIZE>>>(input.data_ptr<float>(), output.data_ptr<float>(), n);
}

void launch_coarsened(const at::Tensor& input, at::Tensor& output) {
    int n = input.size(0);
    output.zero_();
    int blocks = (n + BLOCK_SIZE * COARSEN_FACTOR - 1) / (BLOCK_SIZE * COARSEN_FACTOR);
    coarsened_reduce<<<blocks, BLOCK_SIZE>>>(input.data_ptr<float>(), output.data_ptr<float>(), n);
}

TORCH_LIBRARY(reduce_mod, m) {
    m.def("naive(Tensor input, Tensor(a!) output) -> ()");
    m.impl("naive", &launch_naive);
    m.def("improved(Tensor input, Tensor(a!) output) -> ()");
    m.impl("improved", &launch_improved);
    m.def("coarsened(Tensor input, Tensor(a!) output) -> ()");
    m.impl("coarsened", &launch_coarsened);
}
"""

compile_cuda_raw(CUDA_SRC, name="reduce_mod")

# ============================================================
# Test
# ============================================================
print("\n--- reduction tests ---")

for N in [1023, 65536, 1 << 20]:
    data = torch.randn(N, device="cuda")
    expected = data.sum()

    for name, fn in [("naive", torch.ops.reduce_mod.naive),
                     ("improved", torch.ops.reduce_mod.improved),
                     ("coarsened", torch.ops.reduce_mod.coarsened)]:
        out = torch.zeros(1, device="cuda")
        fn(data, out)
        # Relaxed tolerance — atomicAdd accumulates float error
        check_close(out, expected.unsqueeze(0), name=f"{name} N={N}", atol=1e-1, rtol=1e-2)

# ============================================================
# Benchmark
# ============================================================
N = 1 << 24
data = torch.randn(N, device="cuda")
out = torch.zeros(1, device="cuda")

bench_compare({
    "naive":     (lambda: torch.ops.reduce_mod.naive(data, out.zero_()),),
    "improved":  (lambda: torch.ops.reduce_mod.improved(data, out.zero_()),),
    "coarsened": (lambda: torch.ops.reduce_mod.coarsened(data, out.zero_()),),
    "torch.sum": (lambda: data.sum(),),
})
