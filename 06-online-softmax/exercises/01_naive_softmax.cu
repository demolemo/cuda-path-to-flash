/**
 * Exercise 01 — Naive Softmax (3-pass)
 *
 * GOAL: Implement row-wise softmax on a matrix (batch of vectors).
 *       Input: matrix X of shape (rows, cols), row-major
 *       Output: matrix Y where each row is softmax of the corresponding input row
 *
 * Three passes per row:
 *   1. Find max of the row
 *   2. Compute sum of exp(x - max)
 *   3. Divide each exp(x - max) by sum
 *
 * Parallelism: one block per row, threads collaborate within a row.
 * Use shared memory reduction for max and sum.
 *
 * Compile: nvcc -O2 -o naive_softmax 01_naive_softmax.cu
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define BLOCK_SIZE 256

// ============================================================
// TODO: Naive 3-pass softmax
//   - One block per row
//   - Each thread handles multiple elements (if cols > BLOCK_SIZE)
//   - Use shared memory for reductions (max, sum)
//
// Pass 1: Each thread finds local max → reduce to row max
// Pass 2: Each thread computes local sum of exp(x-max) → reduce to row sum
// Pass 3: Each thread writes exp(x-max)/sum
// ============================================================
__global__ void naive_softmax(const float *X, float *Y, int rows, int cols) {
    __shared__ float sdata[BLOCK_SIZE];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float *x_row = X + row * cols;
    float *y_row = Y + row * cols;

    // PASS 1: Find max
    // TODO: Each thread scans its elements, finds local max
    //       Then reduce in shared memory
    float local_max = -FLT_MAX;
    // YOUR CODE HERE

    // Reduce to get row max
    sdata[tid] = local_max;
    __syncthreads();
    // YOUR CODE HERE — tree reduction for max

    float row_max = sdata[0];
    __syncthreads();

    // PASS 2: Compute sum of exp(x - max)
    float local_sum = 0.0f;
    // YOUR CODE HERE

    // Reduce to get row sum
    sdata[tid] = local_sum;
    __syncthreads();
    // YOUR CODE HERE — tree reduction for sum

    float row_sum = sdata[0];
    __syncthreads();

    // PASS 3: Normalize
    // YOUR CODE HERE — y_row[i] = exp(x_row[i] - row_max) / row_sum
}

// ============================================================
// CPU reference
// ============================================================
void cpu_softmax(const float *X, float *Y, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        const float *x = X + r * cols;
        float *y = Y + r * cols;

        float m = -FLT_MAX;
        for (int c = 0; c < cols; c++) m = fmaxf(m, x[c]);

        float s = 0;
        for (int c = 0; c < cols; c++) s += expf(x[c] - m);

        for (int c = 0; c < cols; c++) y[c] = expf(x[c] - m) / s;
    }
}

int main() {
    const int ROWS = 1024, COLS = 2048;
    size_t bytes = ROWS * COLS * sizeof(float);

    float *h_X = (float *)malloc(bytes);
    float *h_Y_gpu = (float *)malloc(bytes);
    float *h_Y_cpu = (float *)malloc(bytes);

    srand(42);
    for (int i = 0; i < ROWS * COLS; i++)
        h_X[i] = ((float)rand() / RAND_MAX) * 20.0f - 10.0f;  // [-10, 10]

    cpu_softmax(h_X, h_Y_cpu, ROWS, COLS);

    float *d_X, *d_Y;
    CUDA_CHECK(cudaMalloc(&d_X, bytes));
    CUDA_CHECK(cudaMalloc(&d_Y, bytes));
    CUDA_CHECK(cudaMemcpy(d_X, h_X, bytes, cudaMemcpyHostToDevice));

    naive_softmax<<<ROWS, BLOCK_SIZE>>>(d_X, d_Y, ROWS, COLS);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_Y_gpu, d_Y, bytes, cudaMemcpyDeviceToHost));

    // Verify: each row should sum to ~1.0 and match CPU
    int errors = 0;
    for (int r = 0; r < ROWS; r++) {
        float row_sum = 0;
        for (int c = 0; c < COLS; c++) {
            row_sum += h_Y_gpu[r * COLS + c];
            float diff = fabsf(h_Y_gpu[r * COLS + c] - h_Y_cpu[r * COLS + c]);
            if (diff > 1e-4f) errors++;
        }
        if (fabsf(row_sum - 1.0f) > 1e-3f) {
            if (r < 5) printf("Row %d sum = %f (expected ~1.0)\n", r, row_sum);
            errors++;
        }
    }

    if (errors == 0) printf("✅ naive_softmax — correct (%d rows × %d cols)\n", ROWS, COLS);
    else printf("❌ naive_softmax — %d errors\n", errors);

    free(h_X); free(h_Y_gpu); free(h_Y_cpu);
    CUDA_CHECK(cudaFree(d_X)); CUDA_CHECK(cudaFree(d_Y));
    return 0;
}
