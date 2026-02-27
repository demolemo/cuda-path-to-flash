/**
 * Exercise 01 — Multi-Block Reduction
 *
 * GOAL: Reduce an array of ANY size to a single sum.
 *       Your PMPP ch10 code only worked on a single block.
 *       Now make it work for millions of elements.
 *
 * Approach:
 *   1. Each block reduces its chunk → one partial sum per block
 *   2. Option A: atomicAdd partial sums (simple)
 *   3. Option B: two-pass reduction (launch a second kernel on partial sums)
 *
 * Also implement coarsening: each thread processes COARSEN elements
 * before the tree reduction starts (better hardware utilization).
 *
 * Compile: nvcc -O2 -o reduce 01_multi_block_reduce.cu
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

#define BLOCK_SIZE 256
#define COARSEN 8

// ============================================================
// Kernel: coarsened multi-block reduction with atomicAdd
// ============================================================
__global__ void reduce_atomic(const float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;

    // TODO: Coarsened loading — each thread sums COARSEN elements
    // Global starting index for this thread:
    //   gid = blockIdx.x * blockDim.x * COARSEN + threadIdx.x
    // Then stride by blockDim.x for each coarsened element
    float sum = 0.0f;
    // YOUR CODE HERE

    sdata[tid] = sum;
    __syncthreads();

    // TODO: Tree reduction in shared memory
    // YOUR CODE HERE

    // TODO: Thread 0 atomicAdd to output
    // YOUR CODE HERE
}

// ============================================================
// Kernel: two-pass reduction (no atomics)
// Pass 1: reduce blocks → partial sums array
// Pass 2: reduce partial sums → final result
// ============================================================
__global__ void reduce_pass(const float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x * COARSEN + threadIdx.x;

    // TODO: Coarsened load
    float sum = 0.0f;
    // YOUR CODE HERE

    sdata[tid] = sum;
    __syncthreads();

    // TODO: Tree reduction
    // YOUR CODE HERE

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// ============================================================
// Host code
// ============================================================

float cpu_sum(const float *data, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) sum += data[i];
    return (float)sum;
}

int main() {
    int sizes[] = {1023, 65536, 1 << 20, 1 << 24};

    for (int t = 0; t < 4; t++) {
        int N = sizes[t];
        float *h_data = (float *)malloc(N * sizeof(float));
        srand(42);
        for (int i = 0; i < N; i++) h_data[i] = ((float)rand() / RAND_MAX) * 2 - 1;

        float expected = cpu_sum(h_data, N);

        float *d_input, *d_output;
        CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_input, h_data, N * sizeof(float), cudaMemcpyHostToDevice));

        // Test atomic version
        CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));
        int blocks = (N + BLOCK_SIZE * COARSEN - 1) / (BLOCK_SIZE * COARSEN);
        reduce_atomic<<<blocks, BLOCK_SIZE>>>(d_input, d_output, N);
        CUDA_CHECK(cudaGetLastError());

        float result;
        CUDA_CHECK(cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost));

        float rel_err = fabsf(result - expected) / (fabsf(expected) + 1e-8f);
        printf("N=%8d | atomic: got=%10.3f expected=%10.3f rel_err=%e %s\n",
               N, result, expected, rel_err, rel_err < 1e-2 ? "✅" : "❌");

        // Test two-pass version
        float *d_partial;
        CUDA_CHECK(cudaMalloc(&d_partial, blocks * sizeof(float)));
        reduce_pass<<<blocks, BLOCK_SIZE>>>(d_input, d_partial, N);
        CUDA_CHECK(cudaGetLastError());

        // Second pass: reduce partial sums
        int blocks2 = (blocks + BLOCK_SIZE * COARSEN - 1) / (BLOCK_SIZE * COARSEN);
        CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));
        reduce_atomic<<<blocks2, BLOCK_SIZE>>>(d_partial, d_output, blocks);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost));
        rel_err = fabsf(result - expected) / (fabsf(expected) + 1e-8f);
        printf("         | 2-pass: got=%10.3f expected=%10.3f rel_err=%e %s\n",
               result, expected, rel_err, rel_err < 1e-2 ? "✅" : "❌");

        free(h_data);
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));
        CUDA_CHECK(cudaFree(d_partial));
    }

    return 0;
}
