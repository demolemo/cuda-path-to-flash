/**
 * Kernel 1 — Naive SGEMM + cuBLAS benchmark
 *
 * GOAL: Write the simplest possible matmul and measure how far you are
 *       from cuBLAS. This is your baseline.
 *
 * C = A × B where A(M×K), B(K×N), C(M×N), row-major
 *
 * Compile: nvcc -O2 -lcublas -o naive 01_naive.cu
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cublas_v2.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// ============================================================
// TODO: Naive SGEMM kernel
//   - Each thread computes one element of C
//   - C[row][col] = sum over k of A[row][k] * B[k][col]
//
//   You already wrote this in ~/projects/cuda/matrix_mul_naive.cu
//   Do it again cleaner.
// ============================================================
__global__ void sgemm_naive(int M, int N, int K,
                            const float *A, const float *B, float *C) {
    // YOUR CODE HERE
}

// ============================================================
// Benchmarking infrastructure
// ============================================================

float benchmark_kernel(int M, int N, int K,
                       const float *d_A, const float *d_B, float *d_C,
                       int iters) {
    int BLOCK = 32;
    dim3 block(BLOCK, BLOCK);
    dim3 grid((N + BLOCK - 1) / BLOCK, (M + BLOCK - 1) / BLOCK);

    // Warmup
    for (int i = 0; i < 5; i++)
        sgemm_naive<<<grid, block>>>(M, N, K, d_A, d_B, d_C);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < iters; i++)
        sgemm_naive<<<grid, block>>>(M, N, K, d_A, d_B, d_C);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / iters;
}

float benchmark_cublas(int M, int N, int K,
                       const float *d_A, const float *d_B, float *d_C,
                       int iters) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;

    // cuBLAS expects column-major. For row-major: compute B^T × A^T = (AB)^T
    // i.e., cublasSgemm(handle, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N)
    for (int i = 0; i < 5; i++)
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                    &alpha, d_B, N, d_A, K, &beta, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < iters; i++)
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                    &alpha, d_B, N, d_A, K, &beta, d_C, N);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    cublasDestroy(handle);
    return ms / iters;
}

int main() {
    int M = 4096, N = 4096, K = 4096;
    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    printf("SGEMM: %d × %d × %d\n\n", M, N, K);

    float *h_A = (float *)malloc(bytes_A);
    float *h_B = (float *)malloc(bytes_B);
    for (int i = 0; i < M * K; i++) h_A[i] = ((float)rand() / RAND_MAX) * 2 - 1;
    for (int i = 0; i < K * N; i++) h_B[i] = ((float)rand() / RAND_MAX) * 2 - 1;

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B, bytes_B));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

    int iters = 20;
    float naive_ms = benchmark_kernel(M, N, K, d_A, d_B, d_C, iters);
    float cublas_ms = benchmark_cublas(M, N, K, d_A, d_B, d_C, iters);

    double flops = 2.0 * M * N * K;
    double naive_tflops = flops / (naive_ms / 1000.0) / 1e12;
    double cublas_tflops = flops / (cublas_ms / 1000.0) / 1e12;

    printf("Naive:   %8.3f ms  → %6.2f TFLOPS\n", naive_ms, naive_tflops);
    printf("cuBLAS:  %8.3f ms  → %6.2f TFLOPS\n", cublas_ms, cublas_tflops);
    printf("Ratio:   %.1f%% of cuBLAS\n", 100.0 * cublas_ms / naive_ms);

    free(h_A); free(h_B);
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    return 0;
}
