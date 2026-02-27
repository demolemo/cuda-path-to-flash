/**
 * Exercise 01 — Shared Memory Reduction
 *
 * GOAL: Implement a parallel reduction sum using shared memory.
 *       You already wrote reduction kernels in your PMPP Ch.10 work.
 *       Now do it again, cleaner, from memory.
 *
 * You'll write THREE versions:
 *   1. naive_reduce     — divergent branching (stride doubles each iteration)
 *   2. improved_reduce  — non-divergent (stride halves each iteration)
 *   3. coarsened_reduce — each thread loads & sums multiple elements first
 *
 * Remember from your PMPP notes:
 *   "Naive reduction - stride 1, grows by 2, bad memory access pattern"
 *   "Patched reduction - stride is max first, then reduces to one"
 *
 * Compile: nvcc -O2 -o shmem_reduce 01_shmem_reduce.cu
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

// ============================================================
// VERSION 1: Naive reduction (bad access pattern, divergent)
//   - stride starts at 1, doubles each iteration
//   - threads with even indices do work
//   - This is what you wrote as naiveReductionSum in ch10
// ============================================================
__global__ void naive_reduce(const float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load from global to shared
    sdata[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    // TODO: Reduction loop
    // stride = 1, 2, 4, 8, ...
    // if (tid % (2*stride) == 0) → add sdata[tid] += sdata[tid + stride]
    // YOUR CODE HERE

    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

// ============================================================
// VERSION 2: Improved reduction (better access, less divergence)
//   - stride starts at blockDim.x/2, halves each iteration
//   - contiguous threads do work
//   - This is your reductionSumMemPattern from ch10
// ============================================================
__global__ void improved_reduce(const float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    // TODO: Reduction loop
    // for stride = blockDim.x/2 down to 1
    // if (tid < stride) → sdata[tid] += sdata[tid + stride]
    // YOUR CODE HERE

    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

// ============================================================
// VERSION 3: Coarsened reduction
//   - Each thread loads COARSEN_FACTOR elements and sums them
//   - Then do the tree reduction on the partial sums
//   - Better hardware utilization (your ch10 notes mention this!)
// ============================================================
#define COARSEN_FACTOR 4

__global__ void coarsened_reduce(const float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x * COARSEN_FACTOR + threadIdx.x;

    // TODO: Each thread sums COARSEN_FACTOR elements
    float sum = 0.0f;
    // YOUR CODE HERE — loop over COARSEN_FACTOR, accumulate into sum
    // Don't forget bounds checking!

    sdata[tid] = sum;
    __syncthreads();

    // TODO: Tree reduction (same as improved_reduce)
    // YOUR CODE HERE

    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

// ============================================================
// Host code — tests all three versions
// ============================================================

float cpu_reduce(const float *data, int n) {
    // Use Kahan summation for accurate reference
    float sum = 0.0f, c = 0.0f;
    for (int i = 0; i < n; i++) {
        float y = data[i] - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return sum;
}

void test_reduction(const char *name,
                    void (*launch)(const float*, float*, int, int),
                    const float *d_input, int n, float expected) {
    float *d_output;
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    launch(d_input, d_output, n, blocks);
    CUDA_CHECK(cudaGetLastError());

    float result;
    CUDA_CHECK(cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost));

    float rel_err = fabsf(result - expected) / (fabsf(expected) + 1e-8f);
    if (rel_err < 1e-3f) {
        printf("  ✅ %s: sum = %f (expected %f, rel_err = %e)\n", name, result, expected, rel_err);
    } else {
        printf("  ❌ %s: sum = %f (expected %f, rel_err = %e)\n", name, result, expected, rel_err);
    }

    CUDA_CHECK(cudaFree(d_output));
}

void launch_naive(const float *in, float *out, int n, int blocks) {
    naive_reduce<<<blocks, BLOCK_SIZE>>>(in, out, n);
}

void launch_improved(const float *in, float *out, int n, int blocks) {
    improved_reduce<<<blocks, BLOCK_SIZE>>>(in, out, n);
}

void launch_coarsened(const float *in, float *out, int n, int blocks) {
    int coarse_blocks = (n + BLOCK_SIZE * COARSEN_FACTOR - 1) / (BLOCK_SIZE * COARSEN_FACTOR);
    coarsened_reduce<<<coarse_blocks, BLOCK_SIZE>>>(in, out, n);
}

int main() {
    const int N = 1 << 20;  // ~1M elements
    float *h_data = (float *)malloc(N * sizeof(float));

    srand(42);
    for (int i = 0; i < N; i++) {
        h_data[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;  // [-1, 1]
    }

    float expected = cpu_reduce(h_data, N);
    printf("Reduction test (N = %d, expected sum = %f)\n", N, expected);

    float *d_input;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_data, N * sizeof(float), cudaMemcpyHostToDevice));

    test_reduction("naive_reduce",    launch_naive,     d_input, N, expected);
    test_reduction("improved_reduce", launch_improved,  d_input, N, expected);
    test_reduction("coarsened_reduce",launch_coarsened,  d_input, N, expected);

    free(h_data);
    CUDA_CHECK(cudaFree(d_input));

    return 0;
}
