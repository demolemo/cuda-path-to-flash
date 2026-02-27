/**
 * Exercise 03 — Parallel Histogram
 *
 * GOAL: Count occurrences of values in a large array.
 *       Practice atomics and the privatization optimization pattern.
 *
 * Input: array of integers in range [0, NUM_BINS)
 * Output: histogram[i] = count of elements equal to i
 *
 * Three versions:
 *   1. naive_histogram     — atomicAdd directly to global memory (slow, lots of contention)
 *   2. private_histogram   — each block builds local histogram in shared mem, then merges
 *   3. coarsened_histogram — each thread processes multiple elements before any atomic
 *
 * Compile: nvcc -O2 -o histogram 03_histogram.cu
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define BLOCK_SIZE 256
#define NUM_BINS 256
#define COARSEN 8

// ============================================================
// VERSION 1: Naive — direct atomicAdd to global histogram
// ============================================================
__global__ void naive_histogram(const int *input, int *histogram, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // TODO: if in bounds, atomicAdd to histogram[input[gid]]
    // YOUR CODE HERE
}

// ============================================================
// VERSION 2: Privatized — shared memory histogram per block
//   1. Initialize shared histogram to 0
//   2. Each thread atomicAdd to shared histogram (much less contention)
//   3. __syncthreads()
//   4. Merge: each thread adds one bin from shared → global
// ============================================================
__global__ void private_histogram(const int *input, int *histogram, int n) {
    __shared__ int local_hist[NUM_BINS];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // TODO: Initialize shared histogram to 0
    // (might need multiple iterations if NUM_BINS > BLOCK_SIZE)
    // YOUR CODE HERE

    __syncthreads();

    // TODO: Accumulate into local histogram
    // YOUR CODE HERE

    __syncthreads();

    // TODO: Merge local → global
    // Each thread merges one (or more) bins
    // YOUR CODE HERE
}

// ============================================================
// VERSION 3: Coarsened + privatized
//   Same as version 2, but each thread processes COARSEN elements
// ============================================================
__global__ void coarsened_histogram(const int *input, int *histogram, int n) {
    __shared__ int local_hist[NUM_BINS];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x * COARSEN + threadIdx.x;

    // TODO: Init shared histogram
    // YOUR CODE HERE

    __syncthreads();

    // TODO: Each thread processes COARSEN elements
    // YOUR CODE HERE

    __syncthreads();

    // TODO: Merge
    // YOUR CODE HERE
}

// ============================================================
// Host
// ============================================================

void cpu_histogram(const int *input, int *hist, int n) {
    memset(hist, 0, NUM_BINS * sizeof(int));
    for (int i = 0; i < n; i++) hist[input[i]]++;
}

void verify(const int *gpu, const int *cpu, const char *name) {
    int errors = 0;
    for (int i = 0; i < NUM_BINS; i++) {
        if (gpu[i] != cpu[i]) {
            if (errors < 5) printf("  %s bin %d: gpu=%d cpu=%d\n", name, i, gpu[i], cpu[i]);
            errors++;
        }
    }
    if (errors == 0) printf("  ✅ %s\n", name);
    else printf("  ❌ %s (%d bins wrong)\n", name, errors);
}

typedef void (*HistKernel)(const int*, int*, int);

void test_kernel(const char *name, HistKernel kernel, const int *d_input, int N,
                 const int *cpu_hist, int blocks) {
    int *d_hist;
    CUDA_CHECK(cudaMalloc(&d_hist, NUM_BINS * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_hist, 0, NUM_BINS * sizeof(int)));

    kernel<<<blocks, BLOCK_SIZE>>>(d_input, d_hist, N);
    CUDA_CHECK(cudaGetLastError());

    int h_hist[NUM_BINS];
    CUDA_CHECK(cudaMemcpy(h_hist, d_hist, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost));
    verify(h_hist, cpu_hist, name);
    CUDA_CHECK(cudaFree(d_hist));
}

int main() {
    const int N = 1 << 22;  // 4M elements
    int *h_input = (int *)malloc(N * sizeof(int));
    srand(42);
    for (int i = 0; i < N; i++) h_input[i] = rand() % NUM_BINS;

    int cpu_hist[NUM_BINS];
    cpu_histogram(h_input, cpu_hist, N);

    int *d_input;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice));

    printf("Histogram test (N = %d, bins = %d)\n\n", N, NUM_BINS);

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int blocks_coarse = (N + BLOCK_SIZE * COARSEN - 1) / (BLOCK_SIZE * COARSEN);

    test_kernel("naive_histogram",     naive_histogram,     d_input, N, cpu_hist, blocks);
    test_kernel("private_histogram",   private_histogram,   d_input, N, cpu_hist, blocks);
    test_kernel("coarsened_histogram", coarsened_histogram, d_input, N, cpu_hist, blocks_coarse);

    free(h_input);
    CUDA_CHECK(cudaFree(d_input));
    return 0;
}
