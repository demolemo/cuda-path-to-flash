/**
 * Exercise 02 — Vector Addition
 *
 * GOAL: The "hello world" of GPU computing.
 *       Compute C[i] = A[i] + B[i] for all i.
 *
 * This exercise practices:
 *   - Allocating and transferring multiple arrays
 *   - Writing a kernel that reads AND writes global memory
 *   - Handling sizes that aren't a perfect multiple of block size
 *
 * STEPS:
 *   1. Fill in the vector_add kernel
 *   2. Complete the memory allocation and transfers in main()
 *   3. Launch with correct grid dimensions
 *   4. Verify results
 *
 * Compile:  nvcc -o vec_add 02_vector_add.cu
 * Run:      ./vec_add
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

// ============================================================
// TODO: Write the vector_add kernel
//   C[i] = A[i] + B[i]
//   Don't forget bounds checking!
// ============================================================

__global__ void vector_add(const float *A, const float *B, float *C, int n) {
    // YOUR CODE HERE
}

int main() {
    // Deliberately NOT a multiple of 256 — you need to handle this
    const int N = 1000003;
    const int THREADS = 256;
    const size_t bytes = N * sizeof(float);

    printf("Vector addition: N = %d\n", N);

    // ---- Host allocation & initialization ----
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    for (int i = 0; i < N; i++) {
        h_A[i] = sinf(i) * sinf(i);    // some non-trivial values
        h_B[i] = cosf(i) * cosf(i);
    }

    // ---- Device allocation ----
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

    // TODO: Allocate device memory for d_A, d_B, d_C
    // YOUR CODE HERE

    // TODO: Copy h_A → d_A, h_B → d_B
    // YOUR CODE HERE

    // ---- Launch kernel ----
    // TODO: Calculate grid size and launch vector_add
    // YOUR CODE HERE

    CUDA_CHECK(cudaGetLastError());  // catch launch errors

    // TODO: Copy d_C → h_C
    // YOUR CODE HERE

    // ---- Verify ----
    int errors = 0;
    float max_err = 0.0f;
    for (int i = 0; i < N; i++) {
        float expected = h_A[i] + h_B[i];
        float diff = fabsf(h_C[i] - expected);
        if (diff > 1e-5) {
            if (errors < 5) {
                printf("MISMATCH at %d: got %f, expected %f (diff=%e)\n",
                       i, h_C[i], expected, diff);
            }
            errors++;
        }
        if (diff > max_err) max_err = diff;
    }

    if (errors == 0) {
        printf("✅ SUCCESS — all %d elements correct (max error: %e)\n", N, max_err);
    } else {
        printf("❌ FAILED — %d / %d errors\n", errors, N);
    }

    // ---- Cleanup ----
    free(h_A); free(h_B); free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return errors > 0 ? 1 : 0;
}
