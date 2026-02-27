/**
 * Kernel 5 — 2D Block-Tiling
 *
 * GOAL: Each thread computes a TM × TN sub-tile of the output.
 *       This maximizes register reuse via outer product computation.
 *
 * Layout:
 *   Block tile: BM × BN
 *   Thread tile: TM × TN
 *   Threads per block: (BM/TM) × (BN/TN)
 *
 * Suggested parameters:
 *   BM=128, BN=128, BK=8, TM=8, TN=8
 *   Threads = (128/8) × (128/8) = 16 × 16 = 256
 *
 * The key: load TM values from As column and TN values from Bs row
 * into registers, then compute the outer product: TM × TN FMAs.
 *
 * Compile: nvcc -O2 -lcublas -o blocktile_2d 05_2d_blocktile.cu
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
// TODO: 2D Block-Tiled SGEMM
//
// Algorithm:
//   1. Thread mapping:
//      threadRow = threadIdx.x / (BN/TN)   — which TM-row group
//      threadCol = threadIdx.x % (BN/TN)   — which TN-col group
//
//   2. Register arrays:
//      float regA[TM], regB[TN]
//      float results[TM][TN] = {0}
//
//   3. Collaborative tile loading:
//      Need to load BM×BK floats for As and BK×BN floats for Bs
//      With 256 threads, each thread loads multiple elements
//      Use linear indexing: innerRow = tid / BK, innerCol = tid % BK
//
//   4. Inner loop (over BK):
//      for k in 0..BK:
//        Load regA[0..TM] from As[threadRow*TM + i][k]
//        Load regB[0..TN] from Bs[k][threadCol*TN + j]
//        Outer product: results[i][j] += regA[i] * regB[j]
//
//   5. Write results back to C
//
// This is where it gets REAL. Take your time.
// ============================================================
__global__ void sgemm_2d_blocktile(int M, int N, int K,
                                   const float *A, const float *B, float *C) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // YOUR CODE HERE
}

int main() {
    int M = 4096, N = 4096, K = 4096;
    printf("SGEMM 2D Block-Tiled: %d × %d × %d\n", M, N, K);
    printf("  BM=%d BN=%d BK=%d TM=%d TN=%d → %d threads/block\n",
           BM, BN, BK, TM, TN, (BM / TM) * (BN / TN));
    printf("TODO: Add benchmark infrastructure\n");
    printf("Expected: ~30-40%% cuBLAS\n");
    return 0;
}
