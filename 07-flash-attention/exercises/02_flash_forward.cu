/**
 * Exercise 02 — Flash Attention Forward Pass ⚡
 *
 * THE EXERCISE. Everything leads here.
 *
 * GOAL: Compute exact attention WITHOUT materializing the N×N attention matrix.
 *       Single head, batch_size=1 for simplicity.
 *
 * Q, K, V: (N, d) in HBM
 * O: (N, d) output in HBM
 *
 * Algorithm (Flash Attention 1, simplified):
 *   - Tile K/V into blocks of Bc rows
 *   - Tile Q into blocks of Br rows
 *   - For each (Q_block, K_block), compute partial attention in SRAM
 *   - Use online softmax to maintain running statistics (m, l)
 *   - Rescale output as statistics update
 *
 * Start SMALL: N=64, d=32, Br=Bc=16. Verify against naive.
 * Then scale up.
 *
 * Compile: nvcc -O2 -o flash_fwd 02_flash_forward.cu
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

// Tile sizes — start small, increase later
#define Br 32   // rows of Q per tile
#define Bc 32   // rows of K/V per tile

// ============================================================
// Flash Attention Forward Kernel
//
// Grid: one block per Q tile (gridDim.x = ceil(N / Br))
// Block: Br threads (one thread per query row in the tile)
//        OR Br × d threads if you want to parallelize over d
//
// Shared memory:
//   Qi[Br][d]     — current Q tile
//   Kj[Bc][d]     — current K tile
//   Vj[Bc][d]     — current V tile
//   Sij[Br][Bc]   — local attention scores
//
// Per-thread state (in registers):
//   m_i     — running max for this query row
//   l_i     — running sum for this query row
//   O_i[d]  — running output for this query row (or in shared mem)
//
// Pseudocode for one block (processing Q_tile i):
//   Load Qi from HBM → shared
//   Initialize: m = -inf, l = 0, O = 0
//
//   For each K/V tile j:
//     Load Kj, Vj from HBM → shared
//     Compute Sij = Qi × Kj^T / sqrt(d)  (Br × Bc matmul in SRAM)
//     m_new = max(m, rowmax(Sij))
//     P_ij = exp(Sij - m_new)
//     l_new = l * exp(m - m_new) + rowsum(P_ij)
//     O = O * (l * exp(m - m_new) / l_new) + P_ij × Vj / l_new
//     m = m_new, l = l_new
//
//   Write O back to HBM
// ============================================================

__global__ void flash_attention_fwd(const float *Q, const float *K,
                                     const float *V, float *O,
                                     int N, int d) {
    // TODO: This is your final boss. Write it.
    //
    // Suggestion for thread mapping (simplest version):
    //   - Each thread handles one row of the Q tile (one query)
    //   - Thread tid handles Q_row = blockIdx.x * Br + tid
    //   - Inner dimension d is handled with a loop
    //   - This means Br threads per block
    //
    // More advanced: parallelize over d with 2D blocks
    //   - But start simple!
    //
    // YOUR CODE HERE
}

// ============================================================
// CPU reference (naive attention)
// ============================================================
void cpu_attention(const float *Q, const float *K, const float *V,
                   float *O, int N, int d) {
    float scale = 1.0f / sqrtf((float)d);

    for (int i = 0; i < N; i++) {
        float m = -FLT_MAX;
        // Compute S[i,:] and find max
        for (int j = 0; j < N; j++) {
            float s = 0;
            for (int k = 0; k < d; k++) s += Q[i * d + k] * K[j * d + k];
            s *= scale;
            if (s > m) m = s;
        }
        // Compute softmax denominator
        float l = 0;
        for (int j = 0; j < N; j++) {
            float s = 0;
            for (int k = 0; k < d; k++) s += Q[i * d + k] * K[j * d + k];
            s *= scale;
            l += expf(s - m);
        }
        // Compute output
        for (int k = 0; k < d; k++) {
            float sum = 0;
            for (int j = 0; j < N; j++) {
                float s = 0;
                for (int p = 0; p < d; p++) s += Q[i * d + p] * K[j * d + p];
                s *= scale;
                sum += expf(s - m) / l * V[j * d + k];
            }
            O[i * d + k] = sum;
        }
    }
}

int main() {
    // Start small!
    const int N = 128, d = 64;
    size_t bytes = N * d * sizeof(float);

    printf("Flash Attention Forward: N=%d, d=%d, Br=%d, Bc=%d\n\n", N, d, Br, Bc);

    float *h_Q = (float *)malloc(bytes);
    float *h_K = (float *)malloc(bytes);
    float *h_V = (float *)malloc(bytes);
    float *h_O_gpu = (float *)malloc(bytes);
    float *h_O_cpu = (float *)malloc(bytes);

    srand(42);
    for (int i = 0; i < N * d; i++) {
        h_Q[i] = ((float)rand() / RAND_MAX) * 0.5f;  // small values to avoid overflow
        h_K[i] = ((float)rand() / RAND_MAX) * 0.5f;
        h_V[i] = ((float)rand() / RAND_MAX) * 0.5f;
    }

    printf("Computing CPU reference...\n");
    cpu_attention(h_Q, h_K, h_V, h_O_cpu, N, d);
    printf("Done.\n\n");

    float *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, bytes));
    CUDA_CHECK(cudaMalloc(&d_K, bytes));
    CUDA_CHECK(cudaMalloc(&d_V, bytes));
    CUDA_CHECK(cudaMalloc(&d_O, bytes));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_O, 0, bytes));

    // Launch
    int num_q_tiles = (N + Br - 1) / Br;
    flash_attention_fwd<<<num_q_tiles, Br>>>(d_Q, d_K, d_V, d_O, N, d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_O_gpu, d_O, bytes, cudaMemcpyDeviceToHost));

    // Verify
    int errors = 0;
    float max_err = 0;
    for (int i = 0; i < N * d; i++) {
        float diff = fabsf(h_O_gpu[i] - h_O_cpu[i]);
        if (diff > max_err) max_err = diff;
        if (diff > 5e-2f) {  // relaxed tolerance for accumulated error
            if (errors < 10)
                printf("  MISMATCH at [%d][%d]: gpu=%f cpu=%f diff=%e\n",
                       i / d, i % d, h_O_gpu[i], h_O_cpu[i], diff);
            errors++;
        }
    }

    printf("\n");
    if (errors == 0) printf("🔥 Flash Attention CORRECT! (max_err=%e)\n", max_err);
    else printf("❌ Flash Attention — %d / %d errors (max_err=%e)\n", errors, N * d, max_err);

    free(h_Q); free(h_K); free(h_V); free(h_O_gpu); free(h_O_cpu);
    CUDA_CHECK(cudaFree(d_Q)); CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V)); CUDA_CHECK(cudaFree(d_O));
    return 0;
}
