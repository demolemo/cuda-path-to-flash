/**
 * Exercise 02 — Matrix Transpose
 *
 * GOAL: Write 3 transpose kernels, each faster than the last.
 *       This is THE classic exercise for understanding coalescing and bank conflicts.
 *
 * VERSION 1: naive_transpose
 *   - Read rows (coalesced), write columns (non-coalesced) — or vice versa
 *   - Baseline performance
 *
 * VERSION 2: shmem_transpose
 *   - Load a tile into shared memory (coalesced read)
 *   - Write from shared memory (coalesced write)
 *   - But: has bank conflicts!
 *
 * VERSION 3: shmem_transpose_nobc
 *   - Same as version 2, but pad shared memory to avoid bank conflicts
 *   - __shared__ float tile[TILE][TILE+1]  ← the +1 trick
 *
 * Matrix B = transpose(A), where:
 *   A is HEIGHT × WIDTH  (row-major)
 *   B is WIDTH × HEIGHT  (row-major)
 *   B[col][row] = A[row][col]
 *   B[col * HEIGHT + row] = A[row * WIDTH + col]
 *
 * Compile: nvcc -O2 -o transpose 02_transpose.cu
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

#define TILE 32

// ============================================================
// VERSION 1: Naive transpose
//   Each thread copies one element: B[col*H + row] = A[row*W + col]
//   One of the accesses (read or write) will be non-coalesced.
// ============================================================
__global__ void naive_transpose(const float *A, float *B, int height, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // TODO: bounds check, then copy A[row][col] → B[col][row]
    // YOUR CODE HERE
}

// ============================================================
// VERSION 2: Shared memory transpose (with bank conflicts)
//   - Load tile from A into shared memory (coalesced global read)
//   - Write tile from shared memory into B (coalesced global write)
//   - Key insight: the BLOCK that reads tile (bx, by) from A
//     writes to tile (by, bx) in B
// ============================================================
__global__ void shmem_transpose(const float *A, float *B, int height, int width) {
    __shared__ float tile[TILE][TILE];  // <-- has bank conflicts!

    // TODO:
    // 1. Calculate source position in A
    // 2. Load into tile[threadIdx.y][threadIdx.x] (coalesced read from A)
    // 3. __syncthreads()
    // 4. Calculate destination position in B (swap block indices!)
    // 5. Write tile[threadIdx.x][threadIdx.y] to B (coalesced write to B)
    //    Note the index swap in the tile access — this is the transpose!
    // YOUR CODE HERE
}

// ============================================================
// VERSION 3: Shared memory transpose WITHOUT bank conflicts
//   Only change: pad the shared memory by 1
// ============================================================
__global__ void shmem_transpose_nobc(const float *A, float *B, int height, int width) {
    __shared__ float tile[TILE][TILE + 1];  // <-- +1 padding kills bank conflicts

    // TODO: Same logic as version 2, but using the padded tile
    // YOUR CODE HERE
}

// ============================================================
// Copy kernel (for measuring peak bandwidth as reference)
// ============================================================
__global__ void copy_kernel(const float *A, float *B, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) B[idx] = A[idx];
}

// ============================================================
// Verification & benchmarking
// ============================================================

void verify_transpose(const float *h_A, const float *h_B, int H, int W, const char *name) {
    int errors = 0;
    for (int r = 0; r < H && errors < 5; r++) {
        for (int c = 0; c < W && errors < 5; c++) {
            float expected = h_A[r * W + c];
            float got = h_B[c * H + r];
            if (fabsf(got - expected) > 1e-5f) {
                printf("  %s MISMATCH at (%d,%d): expected %f got %f\n",
                       name, r, c, expected, got);
                errors++;
            }
        }
    }
    if (errors == 0) printf("  ✅ %s — correct\n", name);
    else printf("  ❌ %s — %d+ errors\n", name, errors);
}

float benchmark_kernel(void (*launcher)(const float*, float*, int, int,
                                        dim3, dim3),
                       const float *d_A, float *d_B, int H, int W,
                       dim3 grid, dim3 block) {
    // Warmup
    for (int i = 0; i < 5; i++) launcher(d_A, d_B, H, W, grid, block);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int iters = 50;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) launcher(d_A, d_B, H, W, grid, block);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / iters;
}

void launch_naive(const float *A, float *B, int H, int W, dim3 g, dim3 b) {
    naive_transpose<<<g, b>>>(A, B, H, W);
}
void launch_shmem(const float *A, float *B, int H, int W, dim3 g, dim3 b) {
    shmem_transpose<<<g, b>>>(A, B, H, W);
}
void launch_shmem_nobc(const float *A, float *B, int H, int W, dim3 g, dim3 b) {
    shmem_transpose_nobc<<<g, b>>>(A, B, H, W);
}

int main() {
    const int H = 4096, W = 4096;
    const size_t bytes = H * W * sizeof(float);

    printf("Matrix Transpose: %d × %d\n\n", H, W);

    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    for (int i = 0; i < H * W; i++) h_A[i] = (float)i;

    float *d_A, *d_B;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);  // Note: 32×32 = 1024 threads (max for most GPUs)
    dim3 grid((W + TILE - 1) / TILE, (H + TILE - 1) / TILE);

    // Block size 32x8 is often better (each thread handles multiple rows)
    // but 32x32 is simpler to understand first
    // If your GPU limits blocks to 1024, use dim3 block(TILE, 8) and loop in kernel

    struct { const char *name; void (*fn)(const float*, float*, int, int, dim3, dim3); } tests[] = {
        {"naive_transpose",      launch_naive},
        {"shmem_transpose",      launch_shmem},
        {"shmem_transpose_nobc", launch_shmem_nobc},
    };

    for (auto &t : tests) {
        CUDA_CHECK(cudaMemset(d_B, 0, bytes));
        t.fn(d_A, d_B, H, W, grid, block);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_B, d_B, bytes, cudaMemcpyDeviceToHost));

        verify_transpose(h_A, h_B, H, W, t.name);

        float ms = benchmark_kernel(t.fn, d_A, d_B, H, W, grid, block);
        double gb = 2.0 * bytes / 1e9;  // read + write
        printf("    → %.3f ms | %.1f GB/s\n\n", ms, gb / (ms / 1000.0));
    }

    free(h_A); free(h_B);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));

    return 0;
}
