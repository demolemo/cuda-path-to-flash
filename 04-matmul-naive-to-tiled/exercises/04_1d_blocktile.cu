/**
 * Kernel 4 — 1D Block-Tiling
 *
 * GOAL: Each thread computes TM elements (a column of the output tile)
 *       instead of just one element. This increases arithmetic intensity.
 *
 * Layout:
 *   Block tile: BM × BN (output tile this block computes)
 *   Shared memory: As[BM][BK], Bs[BK][BN]
 *   Each thread: loads from A and B, computes TM output elements
 *   Thread count: (BM/TM) × BN per block
 *
 * Suggested starting parameters:
 *   BM = 64, BN = 64, BK = 8, TM = 8
 *   Threads per block = (64/8) * 64 = 512
 *
 * Compile: nvcc -O2 -lcublas -o blocktile_1d 04_1d_blocktile.cu
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

#define BM 64
#define BN 64
#define BK 8
#define TM 8

// ============================================================
// TODO: 1D Block-Tiled SGEMM
//
// Algorithm:
//   1. This block computes a BM×BN tile of C
//   2. Shared memory: As[BM][BK], Bs[BK][BN]
//   3. Thread mapping:
//      - threadRow = threadIdx.x / BN  (which row-group of TM)
//      - threadCol = threadIdx.x % BN  (which column)
//   4. Each thread has a register array: float results[TM] = {0}
//   5. Loop over K in steps of BK:
//      a. Collaboratively load As and Bs tiles
//      b. __syncthreads()
//      c. Inner loop over BK:
//         For each of TM rows: results[i] += As[threadRow*TM + i][k] * Bs[k][threadCol]
//      d. __syncthreads()
//   6. Write results[TM] back to C
//
// The key insight: each thread reuses Bs[k][threadCol] across TM rows.
// That value is loaded once from shared mem and used TM times.
// ============================================================
__global__ void sgemm_1d_blocktile(int M, int N, int K,
                                   const float *A, const float *B, float *C) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // YOUR CODE HERE
}

int main() {
    int M = 4096, N = 4096, K = 4096;
    printf("SGEMM 1D Block-Tiled: %d × %d × %d\n", M, N, K);
    printf("  BM=%d BN=%d BK=%d TM=%d → %d threads/block\n",
           BM, BN, BK, TM, (BM / TM) * BN);
    printf("TODO: Add benchmark infrastructure\n");
    printf("Expected: ~25-30%% cuBLAS\n");
    return 0;
}
