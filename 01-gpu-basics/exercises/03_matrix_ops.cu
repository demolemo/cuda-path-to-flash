/**
 * Exercise 03 — Matrix Operations (2D Indexing)
 *
 * GOAL: Practice 2D thread indexing with two kernels:
 *   1. matrix_add:   C = A + B   (element-wise)
 *   2. matrix_scale:  B = alpha * A
 *
 * This is your first time working with 2D grids and blocks.
 * Matrices are stored in ROW-MAJOR order:
 *   element (row, col) is at index [row * width + col]
 *
 * STEPS:
 *   1. Fill in both kernels using 2D thread indexing
 *   2. Complete the 2D grid/block configuration in main()
 *   3. Launch both kernels and verify results
 *
 * Compile:  nvcc -o matrix_ops 03_matrix_ops.cu
 * Run:      ./matrix_ops
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
// TODO: matrix_add kernel
//   C[row][col] = A[row][col] + B[row][col]
//
//   Use 2D indexing:
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;
//
//   Bounds check BOTH row < height AND col < width
//   Row-major: index = row * width + col
// ============================================================

__global__ void matrix_add(const float *A, const float *B, float *C,
                           int height, int width) {
    // YOUR CODE HERE
}

// ============================================================
// TODO: matrix_scale kernel
//   B[row][col] = alpha * A[row][col]
// ============================================================

__global__ void matrix_scale(const float *A, float *B, float alpha,
                             int height, int width) {
    // YOUR CODE HERE
}

// Helper: fill matrix with values
void fill_matrix(float *M, int h, int w) {
    for (int i = 0; i < h * w; i++) {
        M[i] = (float)(rand() % 100) / 10.0f;
    }
}

int main() {
    // Non-square, non-power-of-2 dimensions — keeps you honest
    const int HEIGHT = 1023;
    const int WIDTH = 517;
    const float ALPHA = 2.5f;
    const size_t bytes = HEIGHT * WIDTH * sizeof(float);

    printf("Matrix operations: %d × %d\n", HEIGHT, WIDTH);

    // ---- Host allocation ----
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    srand(42);
    fill_matrix(h_A, HEIGHT, WIDTH);
    fill_matrix(h_B, HEIGHT, WIDTH);

    // ---- Device allocation ----
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

    // TODO: Allocate device memory
    // YOUR CODE HERE

    // TODO: Copy h_A → d_A, h_B → d_B
    // YOUR CODE HERE

    // ---- Configure 2D grid ----
    // TODO: Set up block and grid dimensions
    //   Hint: a common choice is 16×16 or 32×32 threads per block
    //   Grid must cover the full matrix
    dim3 blockDim(1, 1);  // FIX THIS
    dim3 gridDim(1, 1);   // FIX THIS

    // ---- Test 1: Matrix Addition ----
    // TODO: Launch matrix_add kernel
    // YOUR CODE HERE

    CUDA_CHECK(cudaGetLastError());

    // TODO: Copy d_C → h_C
    // YOUR CODE HERE

    // Verify addition
    int errors = 0;
    for (int i = 0; i < HEIGHT * WIDTH; i++) {
        float expected = h_A[i] + h_B[i];
        if (fabsf(h_C[i] - expected) > 1e-5) {
            if (errors < 5) printf("ADD MISMATCH at %d: got %f expected %f\n",
                                   i, h_C[i], expected);
            errors++;
        }
    }
    if (errors == 0) printf("✅ matrix_add PASSED\n");
    else printf("❌ matrix_add FAILED (%d errors)\n", errors);

    // ---- Test 2: Matrix Scale ----
    // TODO: Launch matrix_scale kernel (output into d_C, or reuse d_B)
    //   matrix_scale<<<gridDim, blockDim>>>(d_A, d_C, ALPHA, HEIGHT, WIDTH);
    // YOUR CODE HERE

    CUDA_CHECK(cudaGetLastError());

    // TODO: Copy result → h_C
    // YOUR CODE HERE

    // Verify scale
    errors = 0;
    for (int i = 0; i < HEIGHT * WIDTH; i++) {
        float expected = ALPHA * h_A[i];
        if (fabsf(h_C[i] - expected) > 1e-5) {
            if (errors < 5) printf("SCALE MISMATCH at %d: got %f expected %f\n",
                                   i, h_C[i], expected);
            errors++;
        }
    }
    if (errors == 0) printf("✅ matrix_scale PASSED\n");
    else printf("❌ matrix_scale FAILED (%d errors)\n", errors);

    // ---- Cleanup ----
    free(h_A); free(h_B); free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
