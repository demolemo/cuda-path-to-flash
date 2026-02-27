# Module 01 — GPU Architecture & Basics

## Learning Goals

By the end of this module you should be able to:

1. Explain how a GPU differs from a CPU (and why that matters for parallelism)
2. Understand the CUDA execution model: threads, blocks, grids, warps
3. Write, compile, and launch a CUDA kernel
4. Use `threadIdx`, `blockIdx`, `blockDim`, `gridDim` to index into data
5. Understand basic error checking and synchronization

## Theory

### CPU vs GPU — The Big Picture

A CPU has a few powerful cores optimized for **sequential** work (branch prediction, out-of-order execution, big caches). A GPU has **thousands** of simple cores optimized for **throughput** — doing the same thing to lots of data simultaneously.

```
CPU:  4-64 cores   × very smart   = great for complex serial tasks
GPU:  thousands    × very simple   = great for data-parallel tasks
```

Think of it like this: a CPU is a few PhD students who can each solve any problem. A GPU is a stadium full of people who can each do simple arithmetic — but there are 10,000 of them working at once.

### The CUDA Execution Model

CUDA organizes parallel work in a hierarchy:

```
Grid
 └── Block (up to 1024 threads)
      └── Thread (smallest unit of execution)
```

- **Thread**: runs your kernel function once. Has a unique ID.
- **Block**: a group of threads that can cooperate (shared memory, synchronization).
- **Grid**: all blocks launched by one kernel call.

You launch a kernel like this:
```cuda
myKernel<<<numBlocks, threadsPerBlock>>>(args...);
```

### Warps — The Hidden Unit

The GPU doesn't execute individual threads — it executes **warps** of 32 threads in lockstep (SIMT). All 32 threads in a warp execute the same instruction at the same time. If threads in a warp take different branches (`if/else`), both paths run and results are masked — this is **warp divergence** and it's expensive.

```
Block (256 threads)
 └── Warp 0: threads 0-31
 └── Warp 1: threads 32-63
 └── ...
 └── Warp 7: threads 224-255
```

### Thread Indexing

Every thread knows where it is:

```cuda
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

- `threadIdx.x` — thread's position within its block (0 to blockDim.x-1)
- `blockIdx.x` — which block this thread belongs to
- `blockDim.x` — how many threads per block
- `gridDim.x` — how many blocks in the grid

For 2D grids (useful for matrices):
```cuda
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

### Memory (Preview)

For now, just know there are several memory spaces (we'll deep-dive in Module 02):

| Memory | Scope | Speed | Size |
|--------|-------|-------|------|
| Registers | Per thread | Fastest | ~255 per thread |
| Shared | Per block | Very fast | 48-164 KB |
| Global (DRAM) | All threads | Slow (~400 cycles) | GBs |

### Error Checking

CUDA calls don't throw exceptions — they return error codes silently. Always check:

```cuda
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
```

### Compilation

CUDA code (`.cu` files) is compiled with `nvcc`:
```bash
nvcc -o my_program my_program.cu
```

`nvcc` splits host code (CPU) and device code (GPU), compiles them separately, and links them together.

## Reading List

1. **PMPP Chapter 1** — Introduction (CPU vs GPU, why heterogeneous computing)
2. **PMPP Chapter 2** — Heterogeneous data parallel computing (kernel launch, thread org)
3. **PMPP Chapter 3** — Multidimensional grids and data (indexing patterns)
4. **CUDA Programming Guide** — [Section 2: Programming Model](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model)
5. **GPU Puzzles** — [github.com/srush/GPU-Puzzles](https://github.com/srush/GPU-Puzzles) — great warm-up

## Additional Resources

- [NVIDIA: An Even Easier Introduction to CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/) — first kernel walkthrough
- [NVIDIA: CUDA Refresher - The GPU Computing Ecosystem](https://developer.nvidia.com/blog/cuda-refresher-the-gpu-computing-ecosystem/)
- [Fabien Sanglard — GPU timeline visualization](https://fabiensanglard.net/cuda/)

## Exercises

Work through them in order. Each builds on the last.

| File | What You'll Do | Difficulty |
|------|---------------|------------|
| `exercises/01_hello_gpu.cu` | Launch your very first kernel | ⭐ |
| `exercises/02_vector_add.cu` | Classic vector addition — the "hello world" of CUDA | ⭐⭐ |
| `exercises/03_matrix_ops.cu` | 2D indexing: matrix add + scale | ⭐⭐⭐ |

Run tests with `make test` and check hints in `hints/` only if you're truly stuck.
