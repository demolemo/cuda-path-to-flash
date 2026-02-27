/**
 * Kernel 3 — Shared Memory Tiled SGEMM
 *
 * GOAL: Standard tiled matmul with shared memory.
 *       You've written this many times. Now benchmark it against cuBLAS.
 *
 * Parameters to tune:
 *   TILE_SIZE (BK): 16 or 32 — experiment!
 *
 * Compile: nvcc -O2 -lcublas -o shmem_tiled 03_shmem_tiled.cu
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

#define TILE 32

// ============================================================
// TODO: Shared memory tiled SGEMM
//   You know this. BM=BN=BK=TILE.
//   Each thread computes one element of C.
// ============================================================
__global__ void sgemm_shmem(int M, int N, int K,
                            const float *A, const float *B, float *C) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    // YOUR CODE HERE
}

int main() {
    int M = 4096, N = 4096, K = 4096;
    printf("SGEMM Shared Memory Tiled: %d × %d × %d\n", M, N, K);
    printf("TODO: Add benchmark infrastructure\n");
    printf("Expected: ~10-15%% cuBLAS\n");
    return 0;
}
