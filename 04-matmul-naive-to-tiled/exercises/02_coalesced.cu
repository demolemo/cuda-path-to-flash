/**
 * Kernel 2 — Coalesced Global Memory Access
 *
 * GOAL: Fix the access pattern so that threads in a warp
 *       access consecutive memory addresses.
 *
 * The fix: swap row/col mapping so that threadIdx.x maps to columns (consecutive in memory).
 * Adjacent threads → adjacent columns → coalesced reads from B and writes to C.
 *
 * Compile: nvcc -O2 -lcublas -o coalesced 02_coalesced.cu
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

// ============================================================
// TODO: Coalesced SGEMM
//   Key change from naive:
//     row = blockIdx.y * blockDim.y + threadIdx.y  (not threadIdx.x!)
//     col = blockIdx.x * blockDim.x + threadIdx.x  (threadIdx.x → columns)
//
//   This ensures threads 0-31 in a warp access consecutive columns.
// ============================================================
__global__ void sgemm_coalesced(int M, int N, int K,
                                const float *A, const float *B, float *C) {
    // YOUR CODE HERE
}

// (Benchmark boilerplate same as 01_naive.cu — copy or #include a common header)

int main() {
    int M = 4096, N = 4096, K = 4096;
    printf("SGEMM Coalesced: %d × %d × %d\n", M, N, K);
    printf("TODO: Copy benchmark infrastructure from 01_naive.cu and test this kernel\n");
    printf("Expected: ~3-5x improvement over naive\n");
    return 0;
}
