/**
 * Test suite for Module 01
 *
 * Tests all three exercises with various edge cases.
 * Compile: nvcc -o test_01 test_01.cu
 * Run:     ./test_01
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

// ---------- Kernel declarations (solutions should match these signatures) ----------

__global__ void fill_index(int *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = idx;
}

__global__ void vector_add(const float *A, const float *B, float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = A[idx] + B[idx];
}

__global__ void matrix_add(const float *A, const float *B, float *C,
                           int height, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < height && col < width) {
        int idx = row * width + col;
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void matrix_scale(const float *A, float *B, float alpha,
                             int height, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < height && col < width) {
        int idx = row * width + col;
        B[idx] = alpha * A[idx];
    }
}

// ---------- Test helpers ----------

int total_tests = 0;
int passed_tests = 0;

void test_result(const char *name, bool passed) {
    total_tests++;
    if (passed) {
        passed_tests++;
        printf("  ✅ %s\n", name);
    } else {
        printf("  ❌ %s\n", name);
    }
}

// ---------- Tests ----------

void test_fill_index() {
    printf("\n--- fill_index tests ---\n");

    int test_sizes[] = {1, 32, 256, 1024, 1000003};

    for (int t = 0; t < 5; t++) {
        int N = test_sizes[t];
        int *d_out, *h_out = (int *)malloc(N * sizeof(int));

        CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_out, 0xFF, N * sizeof(int)));  // poison

        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        fill_index<<<blocks, threads>>>(d_out, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N; i++) {
            if (h_out[i] != i) { ok = false; break; }
        }

        char buf[64];
        snprintf(buf, sizeof(buf), "fill_index N=%d", N);
        test_result(buf, ok);

        free(h_out);
        CUDA_CHECK(cudaFree(d_out));
    }
}

void test_vector_add() {
    printf("\n--- vector_add tests ---\n");

    int test_sizes[] = {1, 33, 1000003};

    for (int t = 0; t < 3; t++) {
        int N = test_sizes[t];
        size_t bytes = N * sizeof(float);
        float *h_A = (float *)malloc(bytes);
        float *h_B = (float *)malloc(bytes);
        float *h_C = (float *)malloc(bytes);

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
        vector_add<<<blocks, threads>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < N; i++) {
            if (fabsf(h_C[i] - (h_A[i] + h_B[i])) > 1e-5) { ok = false; break; }
        }

        char buf[64];
        snprintf(buf, sizeof(buf), "vector_add N=%d", N);
        test_result(buf, ok);

        free(h_A); free(h_B); free(h_C);
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
    }
}

void test_matrix_ops() {
    printf("\n--- matrix ops tests ---\n");

    struct { int h, w; } sizes[] = {{1, 1}, {16, 16}, {1023, 517}, {1, 10000}};

    for (int t = 0; t < 4; t++) {
        int H = sizes[t].h, W = sizes[t].w;
        size_t bytes = H * W * sizeof(float);
        float *h_A = (float *)malloc(bytes);
        float *h_B = (float *)malloc(bytes);
        float *h_C = (float *)malloc(bytes);

        for (int i = 0; i < H * W; i++) {
            h_A[i] = (float)(rand() % 1000) / 100.0f;
            h_B[i] = (float)(rand() % 1000) / 100.0f;
        }

        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, bytes));
        CUDA_CHECK(cudaMalloc(&d_B, bytes));
        CUDA_CHECK(cudaMalloc(&d_C, bytes));
        CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

        dim3 block(16, 16);
        dim3 grid((W + 15) / 16, (H + 15) / 16);

        // Test matrix_add
        matrix_add<<<grid, block>>>(d_A, d_B, d_C, H, W);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < H * W; i++) {
            if (fabsf(h_C[i] - (h_A[i] + h_B[i])) > 1e-5) { ok = false; break; }
        }
        char buf[64];
        snprintf(buf, sizeof(buf), "matrix_add %dx%d", H, W);
        test_result(buf, ok);

        // Test matrix_scale
        float alpha = 3.14f;
        matrix_scale<<<grid, block>>>(d_A, d_C, alpha, H, W);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

        ok = true;
        for (int i = 0; i < H * W; i++) {
            if (fabsf(h_C[i] - alpha * h_A[i]) > 1e-4) { ok = false; break; }
        }
        snprintf(buf, sizeof(buf), "matrix_scale %dx%d", H, W);
        test_result(buf, ok);

        free(h_A); free(h_B); free(h_C);
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
    }
}

int main() {
    printf("🧪 Module 01 — Test Suite\n");

    test_fill_index();
    test_vector_add();
    test_matrix_ops();

    printf("\n========================================\n");
    printf("Results: %d / %d passed\n", passed_tests, total_tests);
    if (passed_tests == total_tests) {
        printf("🎉 ALL TESTS PASSED — Module 01 complete!\n");
    }
    printf("========================================\n");

    return (passed_tests == total_tests) ? 0 : 1;
}
