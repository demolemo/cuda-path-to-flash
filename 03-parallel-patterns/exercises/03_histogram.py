"""
Exercise 03 — Parallel Histogram

GOAL: Count occurrences. Practice atomics + privatization.
  1. naive — atomicAdd to global memory
  2. privatized — shared memory per block, then merge
"""

import torch
import sys; sys.path.insert(0, "../.."); from lib import compile_cuda_raw, check_close, bench_compare

CUDA_SRC = r"""
#include <torch/extension.h>

#define BLOCK_SIZE 256
#define NUM_BINS 256

// VERSION 1: Naive — direct atomicAdd to global histogram
__global__ void naive_hist(const int *input, int *hist, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    // TODO: atomicAdd(&hist[input[gid]], 1)
    // YOUR CODE HERE
}

// VERSION 2: Privatized — shared memory histogram per block
__global__ void private_hist(const int *input, int *hist, int n) {
    __shared__ int local_hist[NUM_BINS];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // TODO: Init local_hist to 0
    // YOUR CODE HERE
    __syncthreads();

    // TODO: Accumulate into local histogram
    // YOUR CODE HERE
    __syncthreads();

    // TODO: Merge local → global
    // YOUR CODE HERE
}

void launch_naive(const at::Tensor& input, at::Tensor& hist) {
    int n = input.size(0);
    hist.zero_();
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    naive_hist<<<blocks, BLOCK_SIZE>>>(
        input.data_ptr<int>(), hist.data_ptr<int>(), n);
}

void launch_private(const at::Tensor& input, at::Tensor& hist) {
    int n = input.size(0);
    hist.zero_();
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    private_hist<<<blocks, BLOCK_SIZE>>>(
        input.data_ptr<int>(), hist.data_ptr<int>(), n);
}

TORCH_LIBRARY(hist_mod, m) {
    m.def("naive(Tensor input, Tensor(a!) hist) -> ()");
    m.impl("naive", &launch_naive);
    m.def("privatized(Tensor input, Tensor(a!) hist) -> ()");
    m.impl("privatized", &launch_private);
}
"""

compile_cuda_raw(CUDA_SRC, name="hist_mod")

print("\n--- histogram tests ---")

N = 1 << 22
data = torch.randint(0, 256, (N,), device="cuda", dtype=torch.int32)
expected = torch.bincount(data, minlength=256).int()

for name, fn in [("naive", torch.ops.hist_mod.naive),
                 ("privatized", torch.ops.hist_mod.privatized)]:
    hist = torch.zeros(256, device="cuda", dtype=torch.int32)
    fn(data, hist)
    check_close(hist.float(), expected.float(), name=f"{name} N={N}", atol=0, rtol=0)

bench_compare({
    "naive":      (lambda: torch.ops.hist_mod.naive(data, torch.zeros(256, device="cuda", dtype=torch.int32)),),
    "privatized": (lambda: torch.ops.hist_mod.privatized(data, torch.zeros(256, device="cuda", dtype=torch.int32)),),
    "torch.bincount": (lambda: torch.bincount(data, minlength=256),),
})
