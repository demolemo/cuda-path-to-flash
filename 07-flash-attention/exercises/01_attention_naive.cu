/**
 * Exercise 01 — Standard (Naive) Attention
 *
 * GOAL: Implement the textbook attention mechanism as your baseline.
 *       S = Q × K^T, P = softmax(S), O = P × V
 *
 * This materializes the full N×N attention matrix — the thing
 * Flash Attention avoids. But you need it for correctness checking.
 *
 * For simplicity: single head, batch_size=1.
 * Q, K, V: (N, d)
 * S, P: (N, N)
 * O: (N, d)
 *
 * Compile: nvcc -O2 -lcublas -o naive_attn 01_attention_naive.cu
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

// ============================================================
// Step 1: Q × K^T → S  (N×d × d×N = N×N)
// ============================================================
__global__ void matmul_qkt(const float *Q, const float *K, float *S,
                           int N, int d) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < d; i++) {
            sum += Q[row * d + i] * K[col * d + i];  // K^T means we read K row-wise
        }
        S[row * N + col] = sum / sqrtf((float)d);  // scale by 1/sqrt(d)
    }
}

// ============================================================
// Step 2: softmax(S) → P  (row-wise)
// ============================================================
__global__ void softmax_rows(const float *S, float *P, int N) {
    // TODO: One block per row, online or 3-pass softmax
    // You've done this in module 06!
    __shared__ float smax[256];
    __shared__ float ssum[256];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    // YOUR CODE HERE — same as module 06
}

// ============================================================
// Step 3: P × V → O  (N×N × N×d = N×d)
// ============================================================
__global__ void matmul_pv(const float *P, const float *V, float *O,
                          int N, int d) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < d) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += P[row * N + i] * V[i * d + col];
        }
        O[row * d + col] = sum;
    }
}

// ============================================================
// CPU reference attention
// ============================================================
void cpu_attention(const float *Q, const float *K, const float *V,
                   float *O, int N, int d) {
    float *S = (float *)malloc(N * N * sizeof(float));

    // S = Q × K^T / sqrt(d)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < d; k++) sum += Q[i * d + k] * K[j * d + k];
            S[i * N + j] = sum / sqrtf((float)d);
        }
    }

    // P = softmax(S) row-wise, O = P × V
    for (int i = 0; i < N; i++) {
        float m = -FLT_MAX;
        for (int j = 0; j < N; j++) m = fmaxf(m, S[i * N + j]);

        float s = 0;
        for (int j = 0; j < N; j++) s += expf(S[i * N + j] - m);

        for (int k = 0; k < d; k++) {
            float sum = 0;
            for (int j = 0; j < N; j++)
                sum += expf(S[i * N + j] - m) / s * V[j * d + k];
            O[i * d + k] = sum;
        }
    }
    free(S);
}

int main() {
    const int N = 256, d = 64;
    size_t qkv_bytes = N * d * sizeof(float);
    size_t s_bytes = N * N * sizeof(float);

    printf("Standard Attention: N=%d, d=%d\n\n", N, d);

    // Host allocation
    float *h_Q = (float *)malloc(qkv_bytes);
    float *h_K = (float *)malloc(qkv_bytes);
    float *h_V = (float *)malloc(qkv_bytes);
    float *h_O_gpu = (float *)malloc(qkv_bytes);
    float *h_O_cpu = (float *)malloc(qkv_bytes);

    srand(42);
    for (int i = 0; i < N * d; i++) {
        h_Q[i] = ((float)rand() / RAND_MAX) * 2 - 1;
        h_K[i] = ((float)rand() / RAND_MAX) * 2 - 1;
        h_V[i] = ((float)rand() / RAND_MAX) * 2 - 1;
    }

    cpu_attention(h_Q, h_K, h_V, h_O_cpu, N, d);

    // Device allocation
    float *d_Q, *d_K, *d_V, *d_S, *d_P, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, qkv_bytes));
    CUDA_CHECK(cudaMalloc(&d_K, qkv_bytes));
    CUDA_CHECK(cudaMalloc(&d_V, qkv_bytes));
    CUDA_CHECK(cudaMalloc(&d_S, s_bytes));
    CUDA_CHECK(cudaMalloc(&d_P, s_bytes));
    CUDA_CHECK(cudaMalloc(&d_O, qkv_bytes));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, qkv_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, qkv_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, qkv_bytes, cudaMemcpyHostToDevice));

    // Step 1: S = Q × K^T / sqrt(d)
    dim3 block(16, 16);
    dim3 grid_s((N + 15) / 16, (N + 15) / 16);
    matmul_qkt<<<grid_s, block>>>(d_Q, d_K, d_S, N, d);

    // Step 2: P = softmax(S)
    softmax_rows<<<N, 256>>>(d_S, d_P, N);

    // Step 3: O = P × V
    dim3 grid_o((d + 15) / 16, (N + 15) / 16);
    matmul_pv<<<grid_o, block>>>(d_P, d_V, d_O, N, d);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_O_gpu, d_O, qkv_bytes, cudaMemcpyDeviceToHost));

    // Verify
    int errors = 0;
    float max_err = 0;
    for (int i = 0; i < N * d; i++) {
        float diff = fabsf(h_O_gpu[i] - h_O_cpu[i]);
        if (diff > max_err) max_err = diff;
        if (diff > 1e-2f) errors++;
    }

    if (errors == 0) printf("✅ Naive attention correct (max_err=%e)\n", max_err);
    else printf("❌ Naive attention — %d errors (max_err=%e)\n", errors, max_err);

    // Cleanup
    free(h_Q); free(h_K); free(h_V); free(h_O_gpu); free(h_O_cpu);
    CUDA_CHECK(cudaFree(d_Q)); CUDA_CHECK(cudaFree(d_K)); CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_S)); CUDA_CHECK(cudaFree(d_P)); CUDA_CHECK(cudaFree(d_O));

    return 0;
}
