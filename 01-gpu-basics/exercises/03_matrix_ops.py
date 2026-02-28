"""
Exercise 03 — Matrix Operations (2D Indexing)

GOAL: Practice 2D thread indexing.
  1. matrix_add:  C = A + B  (element-wise)
  2. matrix_scale: B = alpha * A

Matrices are row-major: element (row, col) = data[row * width + col]
"""

import torch
import sys; sys.path.insert(0, "../.."); from lib import compile_cuda_raw, check_close

# ============================================================
# YOUR CUDA KERNELS
# ============================================================
CUDA_SRC = r"""
#include <torch/extension.h>

__global__ void matrix_add_kernel(const float *A, const float *B, float *C,
                                   int height, int width) {
    // TODO: 2D indexing
    // int row = blockIdx.y * blockDim.y + threadIdx.y;
    // int col = blockIdx.x * blockDim.x + threadIdx.x;
    // Bounds check both row < height AND col < width
    // C[row * width + col] = A[...] + B[...]
    // YOUR CODE HERE
}

__global__ void matrix_scale_kernel(const float *A, float *B, float alpha,
                                     int height, int width) {
    // TODO: B[row][col] = alpha * A[row][col]
    // YOUR CODE HERE
}

void launch_matrix_add(const at::Tensor& A, const at::Tensor& B, at::Tensor& C) {
    int H = A.size(0), W = A.size(1);
    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);
    matrix_add_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), H, W);
}

void launch_matrix_scale(const at::Tensor& A, at::Tensor& B, double alpha) {
    int H = A.size(0), W = A.size(1);
    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);
    matrix_scale_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), (float)alpha, H, W);
}

TORCH_LIBRARY(matrix_ops_mod, m) {
    m.def("add(Tensor A, Tensor B, Tensor(a!) C) -> ()");
    m.impl("add", &launch_matrix_add);
    m.def("scale(Tensor A, Tensor(a!) B, float alpha) -> ()");
    m.impl("scale", &launch_matrix_scale);
}
"""

compile_cuda_raw(CUDA_SRC, name="matrix_ops_mod")

# ============================================================
# Test
# ============================================================
print("\n--- matrix_add tests ---")

for H, W in [(1, 1), (16, 16), (1023, 517), (1, 10000)]:
    A = torch.randn(H, W, device="cuda")
    B = torch.randn(H, W, device="cuda")
    C = torch.empty(H, W, device="cuda")

    torch.ops.matrix_ops_mod.add(A, B, C)
    check_close(C, A + B, name=f"add {H}×{W}")

print("\n--- matrix_scale tests ---")

alpha = 2.5
for H, W in [(1, 1), (16, 16), (1023, 517), (1, 10000)]:
    A = torch.randn(H, W, device="cuda")
    B = torch.empty(H, W, device="cuda")

    torch.ops.matrix_ops_mod.scale(A, B, alpha)
    check_close(B, alpha * A, name=f"scale {H}×{W}")
