/**
 * Kernel 8+ — Warp Tiling
 *
 * GOAL: Organize thread-tile assignment so that threads within a warp
 *       access shared memory optimally. Push toward 60%+ cuBLAS.
 *
 * Key idea: instead of mapping thread tiles in a flat row-major pattern,
 * organize them so that a warp (32 threads) handles a contiguous rectangular
 * region of the output tile.
 *
 * Warp tile dimensions: WM × WN
 *   - Each warp computes a WM × WN region of the BM × BN block tile
 *   - Warps per block: (BM/WM) × (BN/WN)
 *   - Within the warp, 32 threads divide the WM × WN region into thread tiles
 *
 * Example:
 *   BM=128, BN=128, WM=64, WN=32, TM=8, TN=4
 *   Threads per warp tile: (WM/TM) × (WN/TN) = 8 × 8 = 64... hmm, too many.
 *   Need: exactly 32 threads per warp.
 *   So: WM=32, WN=32, TM=8, TN=4 → (32/8)×(32/4) = 4×8 = 32 ✓
 *
 * This is where Simon Boehm's blog gets really detailed.
 * Read it carefully before implementing.
 *
 * Compile: nvcc -O2 -lcublas --ptxas-options=-v -o warptile 03_warptiling.cu
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

// Block tile
#define BM 128
#define BN 128
#define BK 8

// Warp tile
#define WM 64
#define WN 32

// Thread tile
#define TM 8
#define TN 4

// Derived
#define NUM_WARPS ((BM / WM) * (BN / WN))  // warps per block

// ============================================================
// TODO: Warp-tiled SGEMM
//
// This is the hardest kernel in the matmul series.
// Take it step by step:
//
// 1. Map warp to its WM×WN output region
//    warpId = threadIdx.x / 32
//    warpRow = warpId / (BN/WN)  → which row of warps
//    warpCol = warpId % (BN/WN)  → which col of warps
//
// 2. Within the warp, map lane to thread tile
//    laneId = threadIdx.x % 32
//    laneRow = laneId / (WN/TN)
//    laneCol = laneId % (WN/TN)
//
// 3. Each thread computes TM×TN output elements
//    float results[TM * TN] = {0}
//    float regA[TM], regB[TN]
//
// 4. Same outer loop over BK tiles
//    Same collaborative loading (vectorized)
//    Inner loop: load from As/Bs into regA/regB, outer product
//
// 5. Write back to C at the correct warp + lane offset
//
// If you get this working and it's faster than Kernel 7, celebrate.
// If it hits 60%+ cuBLAS, you're doing amazing for self-taught.
// ============================================================
__global__ void sgemm_warptile(int M, int N, int K,
                               const float *A, const float *B, float *C) {
    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];

    // YOUR CODE HERE
}

int main() {
    int M = 4096, N = 4096, K = 4096;
    printf("SGEMM Warp-Tiled: %d × %d × %d\n", M, N, K);
    printf("  BM=%d BN=%d BK=%d WM=%d WN=%d TM=%d TN=%d\n",
           BM, BN, BK, WM, WN, TM, TN);
    printf("TODO: Add benchmark infrastructure\n");
    printf("Expected: ~60%%+ cuBLAS\n");
    return 0;
}
