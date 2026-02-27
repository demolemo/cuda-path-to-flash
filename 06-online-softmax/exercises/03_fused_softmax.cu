/**
 * Exercise 03 — Fused Softmax for Attention Scores
 *
 * GOAL: Apply softmax to attention scores S = Q × K^T, row by row.
 *       This simulates what Flash Attention needs to do.
 *
 * Setup: Given attention scores of shape (batch*heads, seq_len, seq_len),
 *        apply softmax along the last dimension.
 *
 * Bonus challenges:
 *   1. Handle causal masking (set future positions to -inf before softmax)
 *   2. Handle the case where seq_len > BLOCK_SIZE (multiple iterations per thread)
 *   3. Compare performance: naive 3-pass vs online 2-pass
 *
 * This exercise bridges Module 06 → Module 07 (Flash Attention).
 *
 * Compile: nvcc -O2 -o fused_softmax 03_fused_softmax.cu
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define BLOCK_SIZE 256

// ============================================================
// TODO: Fused softmax with causal mask
//
// For row i, the causal mask means:
//   score[i][j] = -inf  if j > i  (can't attend to future)
//   score[i][j] = score[i][j]  if j <= i
//
// Implement using online softmax.
// Skip the -inf values entirely in the online scan
// (they don't affect max and contribute 0 to the sum).
// ============================================================
__global__ void causal_softmax(const float *scores, float *output,
                               int num_rows, int seq_len) {
    __shared__ float s_max[BLOCK_SIZE];
    __shared__ float s_sum[BLOCK_SIZE];

    int row = blockIdx.x;          // which row (0..num_rows-1)
    int query_pos = row % seq_len; // position in sequence (for causal mask)
    int tid = threadIdx.x;

    const float *s_row = scores + row * seq_len;
    float *o_row = output + row * seq_len;

    // TODO: Online softmax with causal masking
    // For elements j > query_pos, treat as -inf (skip them)
    // YOUR CODE HERE

    // TODO: Reduce (m, d) pairs
    // YOUR CODE HERE

    // TODO: Normalize (masked positions should output 0)
    // YOUR CODE HERE
}

// ============================================================
// CPU reference
// ============================================================
void cpu_causal_softmax(const float *scores, float *output,
                        int num_rows, int seq_len) {
    for (int r = 0; r < num_rows; r++) {
        int qpos = r % seq_len;
        const float *s = scores + r * seq_len;
        float *o = output + r * seq_len;

        float m = -FLT_MAX;
        for (int c = 0; c <= qpos; c++) m = fmaxf(m, s[c]);

        float sum = 0;
        for (int c = 0; c <= qpos; c++) sum += expf(s[c] - m);

        for (int c = 0; c < seq_len; c++) {
            if (c <= qpos) o[c] = expf(s[c] - m) / sum;
            else o[c] = 0.0f;
        }
    }
}

int main() {
    const int BATCH_HEADS = 8;
    const int SEQ_LEN = 512;
    const int NUM_ROWS = BATCH_HEADS * SEQ_LEN;
    const size_t bytes = NUM_ROWS * SEQ_LEN * sizeof(float);

    float *h_scores = (float *)malloc(bytes);
    float *h_out_gpu = (float *)malloc(bytes);
    float *h_out_cpu = (float *)malloc(bytes);

    srand(42);
    for (int i = 0; i < NUM_ROWS * SEQ_LEN; i++)
        h_scores[i] = ((float)rand() / RAND_MAX) * 10.0f - 5.0f;

    cpu_causal_softmax(h_scores, h_out_cpu, NUM_ROWS, SEQ_LEN);

    float *d_scores, *d_out;
    CUDA_CHECK(cudaMalloc(&d_scores, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_scores, h_scores, bytes, cudaMemcpyHostToDevice));

    causal_softmax<<<NUM_ROWS, BLOCK_SIZE>>>(d_scores, d_out, NUM_ROWS, SEQ_LEN);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_out_gpu, d_out, bytes, cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < NUM_ROWS * SEQ_LEN; i++) {
        if (fabsf(h_out_gpu[i] - h_out_cpu[i]) > 1e-3f) {
            if (errors < 10)
                printf("  MISMATCH at %d: gpu=%f cpu=%f\n", i, h_out_gpu[i], h_out_cpu[i]);
            errors++;
        }
    }

    if (errors == 0) printf("✅ causal_softmax — correct (%d rows × %d cols)\n", NUM_ROWS, SEQ_LEN);
    else printf("❌ causal_softmax — %d errors\n", errors);

    free(h_scores); free(h_out_gpu); free(h_out_cpu);
    CUDA_CHECK(cudaFree(d_scores)); CUDA_CHECK(cudaFree(d_out));
    return 0;
}
