/**
 * Exercise 01 — Hello GPU
 *
 * GOAL: Launch your very first CUDA kernel.
 *
 * The kernel should write a value into an output array so the CPU can read it
 * back and verify it worked. Each thread writes its global index.
 *
 * STEPS:
 *   1. Allocate device memory with cudaMalloc
 *   2. Fill in the kernel body
 *   3. Launch the kernel with <<<blocks, threads>>>
 *   4. Copy results back with cudaMemcpy
 *   5. Verify on CPU
 *
 * Compile:  nvcc -o hello_gpu 01_hello_gpu.cu
 * Run:      ./hello_gpu
 */

#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// ============================================================
// TODO: Write a kernel called `fill_index`
//   - Each thread computes its global index: blockIdx.x * blockDim.x + threadIdx.x
//   - Writes that index into out[global_index]
//   - Must bounds-check: only write if global_index < n
// ============================================================

__global__ void fill_index(int *out, int n) {
    // YOUR CODE HERE
}

int main() {
    const int N = 1024;
    const int THREADS = 256;

    // TODO: Calculate how many blocks you need to cover N elements
    int blocks = 0; // FIX THIS

    // Host array for results
    int *h_out = (int *)malloc(N * sizeof(int));

    // TODO: Allocate device memory (d_out)
    int *d_out = nullptr;
    // YOUR CODE HERE — cudaMalloc

    // TODO: Launch the kernel
    // YOUR CODE HERE — fill_index<<<blocks, THREADS>>>(...)

    // TODO: Copy results back to host
    // YOUR CODE HERE — cudaMemcpy

    // Verify
    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (h_out[i] != i) {
            printf("MISMATCH at index %d: got %d, expected %d\n", i, h_out[i], i);
            errors++;
            if (errors > 10) { printf("... (stopping)\n"); break; }
        }
    }

    if (errors == 0) {
        printf("✅ SUCCESS — all %d elements correct!\n", N);
    } else {
        printf("❌ FAILED — %d errors\n", errors);
    }

    // Cleanup
    free(h_out);
    CUDA_CHECK(cudaFree(d_out));

    return errors > 0 ? 1 : 0;
}
