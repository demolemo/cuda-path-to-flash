/**
 * Benchmark — Module 01
 *
 * Nothing fancy here — just measures kernel launch overhead and
 * throughput for vector_add at different sizes. Gives you a feel
 * for how fast the GPU actually is.
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

__global__ void vector_add(const float *A, const float *B, float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = A[idx] + B[idx];
}

void bench_vector_add(int N, int warmup_iters, int bench_iters) {
    size_t bytes = N * sizeof(float);
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);

    for (int i = 0; i < N; i++) {
        h_A[i] = sinf(i);
        h_B[i] = cosf(i);
    }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        vector_add<<<blocks, threads>>>(d_A, d_B, d_C, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < bench_iters; i++) {
        vector_add<<<blocks, threads>>>(d_A, d_B, d_C, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    float avg_ms = ms / bench_iters;
    // vector_add reads 2 arrays + writes 1 = 3N floats = 12N bytes
    double gb = 3.0 * N * sizeof(float) / 1e9;
    double bandwidth = gb / (avg_ms / 1000.0);

    printf("  N = %10d | avg = %8.3f ms | bandwidth = %6.1f GB/s\n",
           N, avg_ms, bandwidth);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    free(h_A); free(h_B);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

int main() {
    // Print device info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("  SMs: %d, Max threads/block: %d, Warp size: %d\n",
           prop.multiProcessorCount, prop.maxThreadsPerBlock, prop.warpSize);
    printf("  Global memory: %.1f GB\n", prop.totalGlobalMem / 1e9);
    printf("  Memory bandwidth (theoretical): %.0f GB/s\n\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6);

    printf("Vector Add Benchmark:\n");
    int sizes[] = {1024, 65536, 1 << 20, 1 << 24, 1 << 26};
    for (int i = 0; i < 5; i++) {
        bench_vector_add(sizes[i], 10, 100);
    }

    return 0;
}
