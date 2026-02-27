/**
 * Kernel 6 — Vectorized Loads (float4)
 *
 * GOAL: Take the 2D block-tiled kernel from module 04 and add float4 loads.
 *
 * Changes from Kernel 5:
 *   - Load 4 elements at a time from global memory using float4
 *   - BK must be multiple of 4
 *   - Addresses must be 16-byte aligned
 *
 * You've already experimented with float4 in your leet-gpu repo.
 * Now apply it to matmul.
 *
 * Compile: nvcc -O2 -lcublas -o vectorized 01_vectorized.cu
 */

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

// ============================================================
// TODO: 2D Block-Tiled SGEMM with float4 loads
//
// Key changes from Kernel 5:
//   1. When loading As tile from A:
//      float4 tmp = reinterpret_cast<const float4*>(&A[row * K + col])[0];
//      Store 4 elements at once into shared memory
//
//   2. When loading Bs tile from B:
//      Same idea — load 4 consecutive elements
//
//   3. BK=8 means each thread loads 8/4 = 2 float4 per tile
//      (or adjust loading strategy)
//
//   4. The inner computation loop is unchanged — it operates on shared memory
//
// Remember from your leet-gpu experiments:
//   "Different grid sizes affect performance"
//   "threadsPerBlock = 128 was optimal on H100"
// ============================================================
__global__ void sgemm_vectorized(int M, int N, int K,
                                 const float *A, const float *B, float *C) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // YOUR CODE HERE
}

int main() {
    int M = 4096, N = 4096, K = 4096;
    printf("SGEMM Vectorized: %d × %d × %d\n", M, N, K);
    printf("TODO: Add benchmark infrastructure\n");
    printf("Expected: ~40-50%% cuBLAS\n");
    return 0;
}
