/**
 * Exercise 03 — Flash Attention with Causal Masking
 *
 * GOAL: Add causal masking to Flash Attention.
 *       Position i can only attend to positions j <= i.
 *
 * Changes from Exercise 02:
 *   1. When computing S_ij, set S[r][c] = -inf if (query_pos + r) < (kv_start + c)
 *   2. Skip entire K/V tiles that are fully masked (optimization)
 *   3. For partially masked tiles, the online softmax handles -inf correctly
 *      (exp(-inf) = 0, doesn't affect sum)
 *
 * Bonus optimization: for causal, each Q tile only needs to process
 * K/V tiles up to (and including) the diagonal tile. Skip the rest entirely.
 *
 * Compile: nvcc -O2 -o flash_causal 03_flash_causal.cu
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

#define Br 32
#define Bc 32

// ============================================================
// TODO: Flash Attention with Causal Mask
//
// Same as 02_flash_forward.cu but:
//   - After computing Sij = Qi × Kj^T, apply causal mask
//   - For each element Sij[r][c]:
//       global_query_pos = blockIdx.x * Br + r
//       global_key_pos = kv_tile_start + c
//       if global_key_pos > global_query_pos: Sij[r][c] = -infinity
//
//   - Optimization: if kv_tile_start > (blockIdx.x + 1) * Br - 1,
//     the entire tile is masked → skip it
//
// YOUR CODE HERE
// ============================================================
__global__ void flash_attention_causal(const float *Q, const float *K,
                                        const float *V, float *O,
                                        int N, int d) {
    // YOUR CODE HERE
}

// CPU reference with causal mask
void cpu_causal_attention(const float *Q, const float *K, const float *V,
                          float *O, int N, int d) {
    float scale = 1.0f / sqrtf((float)d);

    for (int i = 0; i < N; i++) {
        float m = -FLT_MAX;
        for (int j = 0; j <= i; j++) {
            float s = 0;
            for (int k = 0; k < d; k++) s += Q[i * d + k] * K[j * d + k];
            s *= scale;
            if (s > m) m = s;
        }

        float l = 0;
        for (int j = 0; j <= i; j++) {
            float s = 0;
            for (int k = 0; k < d; k++) s += Q[i * d + k] * K[j * d + k];
            l += expf(s * scale - m);
        }

        for (int k = 0; k < d; k++) {
            float sum = 0;
            for (int j = 0; j <= i; j++) {
                float s = 0;
                for (int p = 0; p < d; p++) s += Q[i * d + p] * K[j * d + p];
                sum += expf(s * scale - m) / l * V[j * d + k];
            }
            O[i * d + k] = sum;
        }
    }
}

int main() {
    const int N = 128, d = 64;
    size_t bytes = N * d * sizeof(float);

    printf("Flash Attention (Causal): N=%d, d=%d\n\n", N, d);

    float *h_Q = (float *)malloc(bytes);
    float *h_K = (float *)malloc(bytes);
    float *h_V = (float *)malloc(bytes);
    float *h_O_gpu = (float *)malloc(bytes);
    float *h_O_cpu = (float *)malloc(bytes);

    srand(42);
    for (int i = 0; i < N * d; i++) {
        h_Q[i] = ((float)rand() / RAND_MAX) * 0.5f;
        h_K[i] = ((float)rand() / RAND_MAX) * 0.5f;
        h_V[i] = ((float)rand() / RAND_MAX) * 0.5f;
    }

    cpu_causal_attention(h_Q, h_K, h_V, h_O_cpu, N, d);

    float *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, bytes));
    CUDA_CHECK(cudaMalloc(&d_K, bytes));
    CUDA_CHECK(cudaMalloc(&d_V, bytes));
    CUDA_CHECK(cudaMalloc(&d_O, bytes));
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_O, 0, bytes));

    int num_q_tiles = (N + Br - 1) / Br;
    flash_attention_causal<<<num_q_tiles, Br>>>(d_Q, d_K, d_V, d_O, N, d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_O_gpu, d_O, bytes, cudaMemcpyDeviceToHost));

    int errors = 0;
    float max_err = 0;
    for (int i = 0; i < N * d; i++) {
        float diff = fabsf(h_O_gpu[i] - h_O_cpu[i]);
        if (diff > max_err) max_err = diff;
        if (diff > 5e-2f) errors++;
    }

    if (errors == 0) printf("🔥 Flash Attention (Causal) CORRECT! (max_err=%e)\n", max_err);
    else printf("❌ %d errors (max_err=%e)\n", errors, max_err);

    free(h_Q); free(h_K); free(h_V); free(h_O_gpu); free(h_O_cpu);
    CUDA_CHECK(cudaFree(d_Q)); CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V)); CUDA_CHECK(cudaFree(d_O));
    return 0;
}
