/**
 * Kernel 7 — Resolving Bank Conflicts
 *
 * GOAL: Eliminate shared memory bank conflicts by:
 *   1. Transposing As during load: store as As[BK][BM] instead of As[BM][BK]
 *   2. Optionally padding shared memory arrays
 *
 * Why transposing As helps:
 *   In the inner loop, we read As[threadRow*TM + i][k] — adjacent threads (i values)
 *   hit different rows but same column → bank conflict!
 *
 *   After transpose, we read As[k][threadRow*TM + i] — adjacent accesses are now
 *   in consecutive addresses → no bank conflict!
 *
 * Compile: nvcc -O2 -lcublas --ptxas-options=-v -o resolve 02_resolve_conflicts.cu
 *          (--ptxas-options=-v shows register usage)
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
// TODO: Bank-conflict-free SGEMM
//
// Changes:
//   1. As is now [BK][BM] instead of [BM][BK]
//   2. During load: As[loadCol][loadRow] = A[...]  (transposed!)
//   3. During compute: regA[i] = As[k][threadRow * TM + i]
//   4. Optionally: Bs[BK][BN+1] for padding
//
// Profile with ncu to verify bank conflicts are gone:
//   ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld ./resolve
// ============================================================
__global__ void sgemm_no_bc(int M, int N, int K,
                            const float *A, const float *B, float *C) {
    __shared__ float As[BK][BM];    // transposed!
    __shared__ float Bs[BK][BN];

    // YOUR CODE HERE
}

int main() {
    int M = 4096, N = 4096, K = 4096;
    printf("SGEMM No Bank Conflicts: %d × %d × %d\n", M, N, K);
    printf("TODO: Add benchmark infrastructure\n");
    printf("Expected: ~50-55%% cuBLAS\n");
    return 0;
}
