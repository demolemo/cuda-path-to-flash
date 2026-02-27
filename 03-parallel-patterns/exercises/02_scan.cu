/**
 * Exercise 02 — Parallel Prefix Sum (Scan)
 *
 * GOAL: Implement both inclusive and exclusive scan.
 *
 * From your PMPP ch11 notes:
 *   "make a visualization of the algorithm (cool parallel stuff)"
 *   "is the complexity of this algo proved? can any parallel algo work faster than log N?"
 *   "double buffering for partial sums"
 *
 * You'll implement:
 *   1. hillis_steele_scan  — inclusive, work-inefficient O(N log N), simple
 *   2. blelloch_scan       — exclusive, work-efficient O(N), two phases
 *   3. (Bonus) multi-block scan using block-level scan + block sums
 *
 * Compile: nvcc -O2 -o scan 02_scan.cu
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

#define BLOCK_SIZE 512

// ============================================================
// VERSION 1: Hillis-Steele (inclusive scan)
//   - Simple double-buffered approach
//   - At each step d: out[i] = in[i] + in[i - stride]  (if i >= stride)
//   - Needs double buffering (can't read and write same array)
//   - O(N log N) work but only O(log N) steps
// ============================================================
__global__ void hillis_steele_scan(float *input, float *output, int n) {
    // Use two shared memory buffers for double buffering
    __shared__ float buf[2][BLOCK_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Load into buffer 0
    buf[0][tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    int in_buf = 0;  // which buffer to read from

    // TODO: For each stride = 1, 2, 4, 8, ...
    //   - Read from buf[in_buf], write to buf[1-in_buf]
    //   - If tid >= stride: out[tid] = in[tid] + in[tid - stride]
    //   - Else: out[tid] = in[tid]
    //   - Swap buffers
    // YOUR CODE HERE

    // Write result
    if (gid < n) {
        output[gid] = buf[in_buf][tid];
    }
}

// ============================================================
// VERSION 2: Blelloch scan (exclusive scan)
//   - Two phases: up-sweep (reduce) and down-sweep
//   - Work-efficient: O(N) total work
//
//   Up-sweep (reduction):
//     for d = 0 to log2(n)-1:
//       stride = 2^(d+1)
//       for all k where k % stride == 0:
//         a[k + stride - 1] += a[k + stride/2 - 1]
//
//   Set last element to 0
//
//   Down-sweep:
//     for d = log2(n)-1 down to 0:
//       stride = 2^(d+1)
//       for all k where k % stride == 0:
//         temp = a[k + stride/2 - 1]
//         a[k + stride/2 - 1] = a[k + stride - 1]
//         a[k + stride - 1] += temp
// ============================================================
__global__ void blelloch_scan(float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (gid < n) ? input[gid] : 0.0f;
    __syncthreads();

    // TODO: Up-sweep phase
    // YOUR CODE HERE

    // Set last element to 0 (for exclusive scan)
    if (tid == blockDim.x - 1) sdata[tid] = 0.0f;
    __syncthreads();

    // TODO: Down-sweep phase
    // YOUR CODE HERE

    if (gid < n) {
        output[gid] = sdata[tid];
    }
}

// ============================================================
// CPU reference
// ============================================================
void cpu_inclusive_scan(const float *in, float *out, int n) {
    out[0] = in[0];
    for (int i = 1; i < n; i++) out[i] = out[i-1] + in[i];
}

void cpu_exclusive_scan(const float *in, float *out, int n) {
    out[0] = 0;
    for (int i = 1; i < n; i++) out[i] = out[i-1] + in[i-1];
}

void verify(const float *gpu, const float *cpu, int n, const char *name) {
    int errors = 0;
    for (int i = 0; i < n; i++) {
        if (fabsf(gpu[i] - cpu[i]) > 1e-2f * (fabsf(cpu[i]) + 1.0f)) {
            if (errors < 5)
                printf("  %s MISMATCH at %d: gpu=%f cpu=%f\n", name, i, gpu[i], cpu[i]);
            errors++;
        }
    }
    if (errors == 0) printf("  ✅ %s\n", name);
    else printf("  ❌ %s (%d errors)\n", name, errors);
}

int main() {
    // Single-block test (N <= BLOCK_SIZE)
    const int N = BLOCK_SIZE;
    float *h_in = (float *)malloc(N * sizeof(float));
    float *h_out_gpu = (float *)malloc(N * sizeof(float));
    float *h_out_cpu = (float *)malloc(N * sizeof(float));

    srand(42);
    for (int i = 0; i < N; i++) h_in[i] = ((float)(rand() % 10));  // small ints for easy debugging

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    printf("Scan tests (N = %d)\n\n", N);

    // Test Hillis-Steele (inclusive)
    hillis_steele_scan<<<1, N>>>(d_in, d_out, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_out_gpu, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    cpu_inclusive_scan(h_in, h_out_cpu, N);
    verify(h_out_gpu, h_out_cpu, N, "hillis_steele (inclusive)");

    // Test Blelloch (exclusive)
    blelloch_scan<<<1, N>>>(d_in, d_out, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_out_gpu, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    cpu_exclusive_scan(h_in, h_out_cpu, N);
    verify(h_out_gpu, h_out_cpu, N, "blelloch (exclusive)");

    free(h_in); free(h_out_gpu); free(h_out_cpu);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}
