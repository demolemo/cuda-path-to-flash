/**
 * Exercise 02 — Online Softmax (2-pass, Milakov & Gimelshein)
 *
 * GOAL: Implement the online algorithm that computes max and sum in a single pass.
 *
 * The online update rule (per element):
 *   m_new = max(m_old, x_i)
 *   d_new = d_old * exp(m_old - m_new) + exp(x_i - m_new)
 *
 * After scanning all elements: m = true max, d = true sum
 * Then one more pass to normalize: y_i = exp(x_i - m) / d
 *
 * For the GPU version, each thread maintains its own (m, d) pair,
 * scans its assigned elements, then we REDUCE the (m, d) pairs across threads.
 *
 * The (m, d) reduction is special — you can't just add d values because they
 * might have different m bases. The merge rule is:
 *   m_merged = max(m1, m2)
 *   d_merged = d1 * exp(m1 - m_merged) + d2 * exp(m2 - m_merged)
 *
 * Compile: nvcc -O2 -o online_softmax 02_online_softmax.cu
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
// TODO: Online softmax (2-pass)
//
// PASS 1 (online max + sum):
//   1. Each thread scans its elements, maintaining local (m, d)
//      using the online update rule
//   2. Reduce (m, d) pairs across threads in shared memory
//      using the merge rule
//   3. Now thread 0 has the final (row_max, row_sum)
//
// PASS 2 (normalize):
//   4. Broadcast row_max and row_sum to all threads
//   5. Each thread writes: y[i] = exp(x[i] - row_max) / row_sum
// ============================================================
__global__ void online_softmax(const float *X, float *Y, int rows, int cols) {
    // Shared memory for reducing (m, d) pairs
    __shared__ float s_max[BLOCK_SIZE];
    __shared__ float s_sum[BLOCK_SIZE];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float *x_row = X + row * cols;
    float *y_row = Y + row * cols;

    // PASS 1: Online scan
    float local_m = -FLT_MAX;
    float local_d = 0.0f;

    // TODO: Each thread scans its elements using online update
    // for (int i = tid; i < cols; i += blockDim.x) {
    //     float x = x_row[i];
    //     float m_new = fmaxf(local_m, x);
    //     local_d = local_d * expf(local_m - m_new) + expf(x - m_new);
    //     local_m = m_new;
    // }
    // YOUR CODE HERE

    // Store in shared memory
    s_max[tid] = local_m;
    s_sum[tid] = local_d;
    __syncthreads();

    // TODO: Reduce (m, d) pairs using the merge rule
    // for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    //     if (tid < stride) {
    //         float m1 = s_max[tid], d1 = s_sum[tid];
    //         float m2 = s_max[tid + stride], d2 = s_sum[tid + stride];
    //         float m_new = fmaxf(m1, m2);
    //         s_sum[tid] = d1 * expf(m1 - m_new) + d2 * expf(m2 - m_new);
    //         s_max[tid] = m_new;
    //     }
    //     __syncthreads();
    // }
    // YOUR CODE HERE

    float row_max = s_max[0];
    float row_sum = s_sum[0];
    __syncthreads();

    // PASS 2: Normalize
    // YOUR CODE HERE
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
    const int ROWS = 1024, COLS = 4096;
    size_t bytes = ROWS * COLS * sizeof(float);

    float *h_X = (float *)malloc(bytes);
    float *h_Y_gpu = (float *)malloc(bytes);
    float *h_Y_cpu = (float *)malloc(bytes);

    srand(42);
    for (int i = 0; i < ROWS * COLS; i++)
        h_X[i] = ((float)rand() / RAND_MAX) * 20.0f - 10.0f;

    cpu_softmax(h_X, h_Y_cpu, ROWS, COLS);

    float *d_X, *d_Y;
    CUDA_CHECK(cudaMalloc(&d_X, bytes));
    CUDA_CHECK(cudaMalloc(&d_Y, bytes));
    CUDA_CHECK(cudaMemcpy(d_X, h_X, bytes, cudaMemcpyHostToDevice));

    online_softmax<<<ROWS, BLOCK_SIZE>>>(d_X, d_Y, ROWS, COLS);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_Y_gpu, d_Y, bytes, cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < ROWS * COLS; i++) {
        if (fabsf(h_Y_gpu[i] - h_Y_cpu[i]) > 1e-4f) errors++;
    }

    if (errors == 0) printf("✅ online_softmax — correct (%d × %d)\n", ROWS, COLS);
    else printf("❌ online_softmax — %d errors\n", errors);

    // Benchmark both versions
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int iters = 100;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++)
        online_softmax<<<ROWS, BLOCK_SIZE>>>(d_X, d_Y, ROWS, COLS);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Online softmax: %.3f ms avg\n", ms / iters);

    free(h_X); free(h_Y_gpu); free(h_Y_cpu);
    CUDA_CHECK(cudaFree(d_X)); CUDA_CHECK(cudaFree(d_Y));
    return 0;
}
