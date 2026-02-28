"""
Exercise 01 — Multi-Block Reduction

GOAL: Reduce arrays of ANY size. Coarsened + multi-block.
"""

import torch
import sys; sys.path.insert(0, "../.."); from lib import compile_cuda_raw, check_close, bench_compare

CUDA_SRC = r"""
#include <torch/extension.h>

#define BLOCK_SIZE 256
#define COARSEN 8

__global__ void reduce_kernel(const float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x * COARSEN + threadIdx.x;

    // TODO: Coarsened loading — sum COARSEN elements per thread
    float sum = 0.0f;
    // YOUR CODE HERE

    sdata[tid] = sum;
    __syncthreads();

    // TODO: Tree reduction
    // YOUR CODE HERE

    if (tid == 0) atomicAdd(output, sdata[0]);
}

void launch_reduce(const at::Tensor& input, at::Tensor& output) {
    int n = input.size(0);
    output.zero_();
    int blocks = (n + BLOCK_SIZE * COARSEN - 1) / (BLOCK_SIZE * COARSEN);
    reduce_kernel<<<blocks, BLOCK_SIZE>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), n);
}

TORCH_LIBRARY(mb_reduce_mod, m) {
    m.def("launch(Tensor input, Tensor(a!) output) -> ()");
    m.impl("launch", &launch_reduce);
}
"""

compile_cuda_raw(CUDA_SRC, name="mb_reduce_mod")

print("\n--- multi-block reduction tests ---")

for N in [1023, 65536, 1 << 20, 1 << 24]:
    data = torch.randn(N, device="cuda")
    out = torch.zeros(1, device="cuda")
    torch.ops.mb_reduce_mod.launch(data, out)
    expected = data.sum()
    check_close(out, expected.unsqueeze(0), name=f"N={N}", atol=1.0, rtol=1e-2)

N = 1 << 24
data = torch.randn(N, device="cuda")
out = torch.zeros(1, device="cuda")
bench_compare({
    "your reduce": (lambda: torch.ops.mb_reduce_mod.launch(data, out.zero_()),),
    "torch.sum":   (lambda: data.sum(),),
})
