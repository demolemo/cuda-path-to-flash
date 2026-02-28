"""
Exercise 02 — Parallel Prefix Sum (Scan)

GOAL: Implement inclusive (Hillis-Steele) and exclusive (Blelloch) scan.
Single block for now.
"""

import torch
import sys; sys.path.insert(0, "../.."); from lib import compile_cuda_raw, check_close

BLOCK_SIZE = 512

CUDA_SRC = r"""
#include <torch/extension.h>

#define BLOCK_SIZE 512

// Hillis-Steele: inclusive scan, double-buffered
__global__ void hillis_steele_kernel(const float *input, float *output, int n) {
    __shared__ float buf[2][BLOCK_SIZE];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    buf[0][tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    int in_buf = 0;
    // TODO: For stride = 1, 2, 4, ...
    // out[tid] = in[tid] + in[tid - stride]  (if tid >= stride)
    // YOUR CODE HERE

    if (gid < n) output[gid] = buf[in_buf][tid];
}

// Blelloch: exclusive scan, work-efficient
__global__ void blelloch_kernel(const float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    // TODO: Up-sweep (reduce phase)
    // YOUR CODE HERE

    if (tid == blockDim.x - 1) sdata[tid] = 0.0f;
    __syncthreads();

    // TODO: Down-sweep
    // YOUR CODE HERE

    if (gid < n) output[gid] = sdata[tid];
}

void launch_hillis_steele(const at::Tensor& input, at::Tensor& output) {
    int n = input.size(0);
    hillis_steele_kernel<<<1, n>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), n);
}

void launch_blelloch(const at::Tensor& input, at::Tensor& output) {
    int n = input.size(0);
    blelloch_kernel<<<1, n>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), n);
}

TORCH_LIBRARY(scan_mod, m) {
    m.def("hillis_steele(Tensor input, Tensor(a!) output) -> ()");
    m.impl("hillis_steele", &launch_hillis_steele);
    m.def("blelloch(Tensor input, Tensor(a!) output) -> ()");
    m.impl("blelloch", &launch_blelloch);
}
"""

compile_cuda_raw(CUDA_SRC, name="scan_mod")

print("\n--- scan tests ---")

N = BLOCK_SIZE
data = torch.randint(0, 10, (N,), device="cuda", dtype=torch.float32)

# Inclusive scan (Hillis-Steele)
out = torch.empty(N, device="cuda")
torch.ops.scan_mod.hillis_steele(data, out)
expected = torch.cumsum(data, dim=0)
check_close(out, expected, name="hillis_steele (inclusive)", atol=1e-2, rtol=1e-2)

# Exclusive scan (Blelloch)
out = torch.empty(N, device="cuda")
torch.ops.scan_mod.blelloch(data, out)
expected_excl = torch.zeros(N, device="cuda")
expected_excl[1:] = torch.cumsum(data[:-1], dim=0)
check_close(out, expected_excl, name="blelloch (exclusive)", atol=1e-2, rtol=1e-2)
