"""
Exercise 02 — Vector Addition

GOAL: Compute C[i] = A[i] + B[i] on the GPU.
The "hello world" of CUDA.

Fill in the kernel body. Handle non-power-of-2 sizes.
"""

import torch
import os, sys; sys.path.insert(0, "../.."); from lib import compile_cuda_raw, check_close, bench_compare

# ============================================================
# YOUR CUDA KERNEL
# ============================================================
CUDA_SRC = r"""
#include <torch/extension.h>

__global__ void vector_add_kernel(const float *A, const float *B, float *C, int n) {
    // TODO: C[i] = A[i] + B[i]
    // YOUR CODE HERE
}

void launch_vector_add(const at::Tensor& A, const at::Tensor& B, at::Tensor& C) {
    int n = A.size(0);
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vector_add_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), n);
}

TORCH_LIBRARY(vec_add_mod, m) {
    m.def("launch(Tensor A, Tensor B, Tensor(a!) C) -> ()");
    m.impl("launch", &launch_vector_add);
}
"""

compile_cuda_raw(CUDA_SRC, name="vec_add_mod")

# ============================================================
# Test
# ============================================================
print("\n--- vector_add tests ---")

for N in [1, 33, 1024, 1_000_003]:
    A = torch.randn(N, device="cuda")
    B = torch.randn(N, device="cuda")
    C = torch.empty(N, device="cuda")

    torch.ops.vec_add_mod.launch(A, B, C)
    expected = A + B  # PyTorch reference — one line!

    check_close(C, expected, name=f"N={N}")

# ============================================================
# Benchmark: your kernel vs PyTorch
# ============================================================
N = 1 << 24
A = torch.randn(N, device="cuda")
B = torch.randn(N, device="cuda")
C = torch.empty(N, device="cuda")

def my_add():
    torch.ops.vec_add_mod.launch(A, B, C)

def torch_add():
    torch.add(A, B, out=C)

bench_compare({
    "your kernel": (my_add,),
    "torch.add":   (torch_add,),
})
