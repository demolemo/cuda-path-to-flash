"""
Exercise 01 — Hello GPU

GOAL: Launch your very first CUDA kernel.
Each thread writes its global index into an output array.

Fill in the CUDA kernel body where it says YOUR CODE HERE.
"""

import torch
import sys; sys.path.insert(0, "../.."); from lib import compile_cuda_raw, check_close, gpu_info

gpu_info()

# ============================================================
# YOUR CUDA KERNEL — fill in the body of fill_index
# ============================================================
CUDA_SRC = r"""
#include <torch/extension.h>

__global__ void fill_index_kernel(int *out, int n) {
    // TODO: Compute global index and write it to out[idx]
    // int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Don't forget bounds check: if (idx < n)
    // YOUR CODE HERE
}

void launch_fill_index(at::Tensor out) {
    int n = out.size(0);
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    fill_index_kernel<<<blocks, threads>>>(out.data_ptr<int>(), n);
}

TORCH_LIBRARY(fill_index_mod, m) {
    m.def("launch(Tensor(a!) out) -> ()");
    m.impl("launch", &launch_fill_index);
}
"""

compile_cuda_raw(CUDA_SRC, name="fill_index_mod")

# ============================================================
# Test
# ============================================================
print("\n--- fill_index tests ---")

for N in [1, 32, 256, 1024, 1_000_003]:
    out = torch.zeros(N, dtype=torch.int32, device="cuda")
    torch.ops.fill_index_mod.launch(out)

    expected = torch.arange(N, dtype=torch.int32, device="cuda")
    check_close(out.float(), expected.float(), name=f"N={N}")
