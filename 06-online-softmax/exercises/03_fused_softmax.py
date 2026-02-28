"""
Exercise 03 — Causal Softmax

GOAL: Softmax with causal mask — position i can only attend to j <= i.
Bridges to Flash Attention.
"""

import torch
import sys; sys.path.insert(0, "../.."); from lib import compile_cuda_raw, check_close

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cfloat>

#define BLOCK_SIZE 256

__global__ void causal_softmax_kernel(const float *scores, float *output,
                                       int num_rows, int seq_len) {
    __shared__ float s_max[BLOCK_SIZE];
    __shared__ float s_sum[BLOCK_SIZE];
    int row = blockIdx.x;
    int query_pos = row % seq_len;
    int tid = threadIdx.x;
    const float *s = scores + row * seq_len;
    float *o = output + row * seq_len;

    // Online softmax with causal mask
    // Skip elements where j > query_pos
    float local_m = -FLT_MAX;
    float local_d = 0.0f;

    // TODO: Online scan, skipping masked positions
    // YOUR CODE HERE

    s_max[tid] = local_m;
    s_sum[tid] = local_d;
    __syncthreads();

    // TODO: Reduce (m, d) pairs
    // YOUR CODE HERE

    float row_max = s_max[0];
    float row_sum = s_sum[0];
    __syncthreads();

    // TODO: Normalize (masked positions → 0)
    // YOUR CODE HERE
}

void launch(const at::Tensor& scores, at::Tensor& output, int64_t seq_len) {
    int num_rows = scores.size(0);
    causal_softmax_kernel<<<num_rows, BLOCK_SIZE>>>(
        scores.data_ptr<float>(), output.data_ptr<float>(), num_rows, seq_len);
}

TORCH_LIBRARY(causal_sm_mod, m) {
    m.def("launch(Tensor scores, Tensor(a!) output, int seq_len) -> ()");
    m.impl("launch", &launch);
}
"""

compile_cuda_raw(CUDA_SRC, name="causal_sm_mod")

print("\n--- causal softmax tests ---")

for BATCH, SEQ in [(1, 64), (4, 128), (8, 512)]:
    NUM_ROWS = BATCH * SEQ
    scores = torch.randn(NUM_ROWS, SEQ, device="cuda") * 3
    output = torch.empty_like(scores)

    torch.ops.causal_sm_mod.launch(scores, output, SEQ)

    # PyTorch reference with causal mask
    mask = torch.triu(torch.ones(SEQ, SEQ, device="cuda"), diagonal=1).bool()
    # Each row i applies mask based on i % seq_len
    expected = torch.empty_like(scores)
    for r in range(NUM_ROWS):
        qpos = r % SEQ
        row = scores[r].clone()
        row[qpos+1:] = float('-inf')
        expected[r] = torch.softmax(row, dim=-1)
        expected[r][qpos+1:] = 0.0

    check_close(output, expected, name=f"batch={BATCH} seq={SEQ}", atol=1e-3, rtol=1e-3)
