/**
 * Exercise 03 — Tiled Matrix Multiplication (From Memory!)
 *
 * GOAL: You already wrote 5 versions of tiled matmul in your PMPP repo.
 *       Now write it again from scratch, WITHOUT looking at your old code.
 *       This should be muscle memory by the end.
 *
 * Write TWO versions:
 *   1. tiled_matmul_square    — assumes N % TILE_SIZE == 0
 *   2. tiled_matmul_general   — arbitrary M, N, K (with boundary checks)
 *
 * C (M×N) = A (M×K) × B (K×N)
 *
 * Recall your own notes from chapter05:
 *   "not drawing a picture from the start"
 *   "incorrectly naming column and row"
 *   "went through indexing in shared matrices by vibe,
 *    should have deconstructed them and worked with them by hand"
 *
 * Don't repeat those mistakes. Draw the picture first!
 *
 * Compile: nvcc -O2 -o tiled_matmul 03_tiled_matmul.cu
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define TILE_SIZE 16

// ============================================================
// VERSION 1: Square tiled matmul (N×N × N×N = N×N)
//   Assumes N % TILE_SIZE == 0
//
//   Algorithm:
//     for each tile phase:
//       1. Load tile from A into shared memory (coalesced)
//       2. Load tile from B into shared memory (coalesced)
//       3. __syncthreads()
//       4. Multiply tiles, accumulate into register
//       5. __syncthreads()
//     Write final value to C
// ============================================================
__global__ void tiled_matmul_square(const float *A, const float *B, float *C, int N) {
    // YOUR CODE HERE
    // Hint: you've done this 3 times before. Trust your understanding.
}

// ============================================================
// VERSION 2: General tiled matmul (M×K × K×N = M×N)
//   Arbitrary dimensions — need boundary checks everywhere
//   Load 0.0f for out-of-bounds tiles
//
//   This is your sharedMemTiledMatmulArbSizes kernel, rewritten.
// ============================================================
__global__ void tiled_matmul_general(const float *A, const float *B, float *C,
                                     int M, int K, int N) {
    // YOUR CODE HERE
}

// ============================================================
// Reference CPU matmul
// ============================================================
void cpu_matmul(const float *A, const float *B, float *C, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int p = 0; p < K; p++) {
                sum += A[i * K + p] * B[p * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void verify(const float *gpu, const float *cpu, int size, const char *name) {
    int errors = 0;
    float max_err = 0;
    for (int i = 0; i < size; i++) {
        float diff = fabsf(gpu[i] - cpu[i]);
        if (diff > max_err) max_err = diff;
        if (diff > 1e-2f) {  // matmul accumulates error
            if (errors < 5) printf("  %s MISMATCH at %d: gpu=%f cpu=%f\n", name, i, gpu[i], cpu[i]);
            errors++;
        }
    }
    if (errors == 0) printf("  ✅ %s — correct (max_err=%e)\n", name, max_err);
    else printf("  ❌ %s — %d errors (max_err=%e)\n", name, errors, max_err);
}

int main() {
    printf("=== Test 1: Square (512×512) ===\n");
    {
        int N = 512;
        size_t bytes = N * N * sizeof(float);
        float *h_A = (float *)malloc(bytes);
        float *h_B = (float *)malloc(bytes);
        float *h_C_gpu = (float *)malloc(bytes);
        float *h_C_cpu = (float *)malloc(bytes);

        srand(42);
        for (int i = 0; i < N * N; i++) {
            h_A[i] = ((float)rand() / RAND_MAX) * 2 - 1;
            h_B[i] = ((float)rand() / RAND_MAX) * 2 - 1;
        }

        cpu_matmul(h_A, h_B, h_C_cpu, N, N, N);

        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, bytes));
        CUDA_CHECK(cudaMalloc(&d_B, bytes));
        CUDA_CHECK(cudaMalloc(&d_C, bytes));
        CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid(N / TILE_SIZE, N / TILE_SIZE);

        tiled_matmul_square<<<grid, block>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost));
        verify(h_C_gpu, h_C_cpu, N * N, "tiled_matmul_square");

        free(h_A); free(h_B); free(h_C_gpu); free(h_C_cpu);
        CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    }

    printf("\n=== Test 2: Non-square (237×419 × 419×173) ===\n");
    {
        int M = 237, K = 419, N = 173;
        float *h_A = (float *)malloc(M * K * sizeof(float));
        float *h_B = (float *)malloc(K * N * sizeof(float));
        float *h_C_gpu = (float *)malloc(M * N * sizeof(float));
        float *h_C_cpu = (float *)malloc(M * N * sizeof(float));

        srand(123);
        for (int i = 0; i < M * K; i++) h_A[i] = ((float)rand() / RAND_MAX) * 2 - 1;
        for (int i = 0; i < K * N; i++) h_B[i] = ((float)rand() / RAND_MAX) * 2 - 1;

        cpu_matmul(h_A, h_B, h_C_cpu, M, K, N);

        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));

        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

        tiled_matmul_general<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
        verify(h_C_gpu, h_C_cpu, M * N, "tiled_matmul_general");

        free(h_A); free(h_B); free(h_C_gpu); free(h_C_cpu);
        CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    }

    return 0;
}
