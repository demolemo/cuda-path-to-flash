# 📚 CUDA Learning Resources — The Full Stack

Everything we could find. Books, blogs, papers, repos, videos, courses, Twitter threads.
No gatekeeping. No university needed.

---

## 🏗️ Books

| Book | Authors | Why It's Good |
|------|---------|---------------|
| **Programming Massively Parallel Processors (PMPP)** | Kirk & Hwu | THE textbook. 4th edition covers modern GPUs. Our primary source for modules 01-03. |
| **CUDA by Example** | Sanders & Kandrot | Gentler intro, good if PMPP feels dense at first |
| **Professional CUDA C Programming** | Cheng, Grossman, McKercher | More applied, good for after PMPP |
| **The CUDA Handbook** | Nicholas Wilt | Deep hardware-level reference, gets into PTX/SASS |
| **GPU Gems 1, 2, 3** | NVIDIA | Classic series, free online. GPU Gems 3 especially relevant |
| **Parallel and High Performance Computing** | Robey & Zamora | Broader HPC context, good for understanding where CUDA fits |

## 🌐 Blog Posts & Articles (The Good Shit)

### Simon Boehm (siboehm.com) — Our Matmul Bible
- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM) — Step-by-step from naive to 95% cuBLAS. This is modules 04-05.
- Covers: naive → shared mem tiling → 1D blocktiling → 2D blocktiling → vectorized loads → double buffering → warp tiling

### Towernest — Kernel Optimization Deep Dives
- Active in the CUDA kernel optimization space, produces content on GPU programming and optimization techniques
- Look them up on Twitter/GitHub — posts about low-level kernel tuning, warp-level primitives, memory access patterns
- Worth following for the practical optimization mindset

### Lei Mao (leimao.github.io)
- [CUDA Matrix Multiplication Optimization](https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/)
- [CUDA Shared Memory Bank Conflicts](https://leimao.github.io/blog/CUDA-Shared-Memory-Bank/)
- [CUDA Stream](https://leimao.github.io/blog/CUDA-Stream/)
- [CUDA Event](https://leimao.github.io/blog/CUDA-Event/)
- [NVIDIA CUDA Warp-Level Primitives](https://leimao.github.io/blog/CUDA-Warp-Level-Primitives/)
- Basically just read everything on this blog. Lei Mao is goated.

### Mark Harris / NVIDIA Developer Blog
- [An Even Easier Introduction to CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/) — Absolute starter
- [An Efficient Matrix Transpose in CUDA](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/) — Classic bank conflict lesson
- [Using Shared Memory in CUDA C/C++](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)
- [How to Optimize Data Transfers in CUDA C/C++](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/)
- [How to Access Global Memory Efficiently](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/)
- [How to Query Device Properties and Handle Errors](https://developer.nvidia.com/blog/how-query-device-properties-and-handle-errors-cuda-cc/)
- [Faster Parallel Reductions on Kepler](https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/)
- [CUDA Pro Tip: Optimized Filtering with Warp-Aggregated Atomics](https://developer.nvidia.com/blog/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/)

### Fabien Sanglard
- [CUDA — a timeline](https://fabiensanglard.net/cuda/) — Visual walkthrough of GPU execution

### Bruce Dawson (randomascii.wordpress.com)
- Posts on floating-point, profiling, GPU debugging — essential background knowledge

### Colfax Research
- [CUTLASS tutorial series](https://research.colfax-intl.com/cutlass-tutorial/) — When you're ready for templated GEMM

### Finbarr Timbers
- [Writing a CUDA Kernel](https://finbarr.ca/cuda-kernel/) — Clean, practical walkthrough

### Tom's Hardware / Chips and Cheese
- GPU architecture deep dives (die shots, cache hierarchy, SM internals)
- [Chips and Cheese](https://chipsandcheese.com/) — Incredible hardware analysis

## 📝 Papers

### Core Papers for This Course
| Paper | Why |
|-------|-----|
| [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135) (Tri Dao et al., 2022) | Our final boss. Module 07. |
| [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) (Tri Dao, 2023) | The sequel. Better warp partitioning. |
| [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867) (Milakov & Gimelshein, 2018) | Key algorithm for module 06. Numerically stable online softmax. |
| [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08691) (Tri Dao et al., 2024) | Hopper-specific tricks, FP8, warp specialization |

### Foundational GPU/CUDA Papers
| Paper | Why |
|-------|-----|
| [Volkov — Understanding Latency Hiding on GPUs](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-143.pdf) | Deep dive into occupancy vs. ILP. Changes how you think about perf. |
| [Yan, Max, Suri — Demystifying Tensor Cores](https://arxiv.org/abs/2203.15234) | When you get to tensor core matmuls |
| [NVIDIA — Parallel Thread Execution ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) | The PTX reference. You'll need this eventually. |
| [Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf) (Mark Harris) | THE reduction talk. Classic. Module 03. |

### Attention & Transformer Kernel Papers
| Paper | Why |
|-------|-----|
| [Self-attention Does Not Need O(n²) Memory](https://arxiv.org/abs/2112.05682) (Rabe & Staats, 2021) | Predecessor to Flash Attention idea |
| [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102) (Pope et al., 2022) | Multi-query attention, KV cache optimization |
| [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) | Grouped-query attention |
| [PagedAttention / vLLM](https://arxiv.org/abs/2309.06180) | How serving systems manage KV cache |

## 🎥 Videos & Talks

### YouTube Channels
- **Umar Jamil** — Great visual explanations of attention, transformers, CUDA concepts
- **Andrej Karpathy** — [Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) — not CUDA but essential context
- **Yannic Kilcher** — Paper walkthroughs including Flash Attention
- **3Blue1Brown** — Linear algebra visualizations (you'll need this for matmul intuition)

### Conference Talks
- [NVIDIA GTC talks](https://www.nvidia.com/en-us/on-demand/) — Free. Search for CUDA optimization, CUTLASS, etc.
- Mark Harris — "Optimizing Parallel Reduction in CUDA" (GTC classic, also a PDF above)
- Tri Dao — Flash Attention talks at various ML conferences

### Specific Videos
- [CUDA Crash Course by CoffeeBeforeArch](https://www.youtube.com/playlist?list=PLxNPSjHT5qvtYRVdNN1yDcdSl39uHV_sU) — YouTube playlist, very practical
- [GPU Programming lectures (Wen-mei Hwu)](https://www.youtube.com/playlist?list=PLRRuQYjFhpmubuwx-w8X964ofVkW1T8O4) — PMPP author's actual lectures!

## 🛠️ Repos & Tools

### Learning Repos
| Repo | What |
|------|------|
| [srush/GPU-Puzzles](https://github.com/srush/GPU-Puzzles) | Interactive GPU programming exercises. Great warmup. |
| [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples) | Official examples for every CUDA feature |
| [siboehm/CUDA-MMM](https://github.com/siboehm/CUDA-MMM) | Simon Boehm's matmul code from the blog post |
| [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) | The actual Flash Attention implementation |
| [NVIDIA/DALI](https://github.com/NVIDIA/DALI) | Data loading pipeline — see how pros write CUDA |
| [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) | Templated GEMM library. Dense but gold standard. |
| [wangzyon/NVIDIA_SGEMM_PRACTICE](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE) | Step-by-step SGEMM optimization, similar to Simon's blog |
| [Bruce-Lee-LY/cuda_hgemm](https://github.com/Bruce-Lee-LY/cuda_hgemm) | Half-precision GEMM implementations |
| [jundaf2/CUDA-Flash-Attention](https://github.com/jundaf2/CUDA-Flash-Attention) | Minimal Flash Attention in plain CUDA |
| [tspeterkim/flash-attention-minimal](https://github.com/tspeterkim/flash-attention-minimal) | ~100 lines Flash Attention for learning |
| [ademeure/RNNoise-CUDA](https://github.com/ademeure) | ademeure — GPU optimization wizard, check their repos |
| [cuda-mode](https://github.com/cuda-mode) | Community CUDA learning group repos |

### Profiling Tools (You WILL need these)
| Tool | What |
|------|------|
| `nvprof` | Classic CUDA profiler (deprecated but still works) |
| **Nsight Compute** (`ncu`) | Modern kernel profiler. Shows memory throughput, occupancy, everything. |
| **Nsight Systems** (`nsys`) | System-level timeline profiler. See CPU↔GPU interaction. |
| `cuda-memcheck` / `compute-sanitizer` | Find memory bugs (out-of-bounds, race conditions) |
| **NVIDIA Visual Profiler** (legacy) | GUI profiler, good for beginners |

### Compiler Explorer
- [godbolt.org](https://godbolt.org/) — Supports CUDA/PTX! See what your kernel compiles to.

## 🐦 People to Follow (Twitter/X)

These people regularly post about CUDA, GPU optimization, and kernel engineering:

| Handle | Known For |
|--------|-----------|
| **@tri_dao** | Flash Attention creator |
| **@siaborhm** | The matmul blog post guy |
| **@towernest** | CUDA kernel optimization content |
| **@mark_harris** | NVIDIA, CUDA ecosystem |
| **@caborhm** | GPU performance analysis |
| **@_lhw** | Wen-mei Hwu, PMPP author |
| **@georghotz** | tinygrad, low-level GPU hacking |
| **@kaborhm** | ML systems, kernel engineering |
| **@sraborhm** | Sasha Rush, GPU Puzzles creator |

(Handles approximate — search by name if they don't resolve)

## 🎓 Courses (Free / Online)

| Course | Where | Notes |
|--------|-------|-------|
| **Caltech CS179 — GPU Programming** | [Course page](http://courses.cms.caltech.edu/cs179/) | Assignments available, well-structured |
| **Stanford CS149 — Parallel Computing** | [cs149.stanford.edu](https://cs149.stanford.edu/) | Broader than CUDA but excellent |
| **UIUC ECE408 — Applied Parallel Programming** | Various mirrors | Hwu's actual course (PMPP in action) |
| **Heterogeneous Parallel Programming (Coursera)** | Wen-mei Hwu | Based on PMPP, sometimes available |
| **CUDA Training Series** | [OLCF](https://www.olcf.ornl.gov/cuda-training-series/) | Oak Ridge National Lab. Free. Excellent. |
| **GPU Programming (Aalto University)** | [gpucomputing.fi](https://gpucomputing.fi/) | Nordic efficiency: clear, concise |

## 🗺️ Learning Path Cheat Sheet

```
Week 1-2:  PMPP Ch.1-3 + Module 01 exercises + GPU Puzzles
           → "I can write and launch a kernel"

Week 3-4:  PMPP Ch.4-5 + Module 02 + NVIDIA shared memory blogs
           → "I understand memory hierarchy"

Week 5-6:  PMPP Ch.8-9 + Module 03 + Mark Harris reduction PDF
           → "I can write reductions and scans"

Week 7-9:  Simon Boehm's blog + Module 04-05
           → "My matmul hits 60%+ cuBLAS"

Week 10:   Milakov paper + Module 06
           → "I understand online softmax"

Week 11+:  Tri Dao papers + Module 07
           → "I wrote Flash Attention from scratch"

Beyond:    CUTLASS, Tensor Cores, Triton, warp specialization,
           FlashAttention-3, FP8, async copies (cp.async), TMA
```

## 💡 Random Tips We've Collected

1. **Always profile before optimizing.** `ncu` is your best friend.
2. **Memory bandwidth is usually the bottleneck**, not compute. Know your GPU's theoretical bandwidth.
3. **Ceiling division**: `(N + BLOCK - 1) / BLOCK` — you'll write this 10,000 times.
4. **Warp size is 32. Always 32.** Design everything around this.
5. **Occupancy isn't everything.** Read Volkov's paper on latency hiding.
6. **Start stupid, then optimize.** Write the naive version first. Make it correct. THEN make it fast.
7. **Read the PTX.** When you're stuck on perf, look at what the compiler actually generated (`nvcc --ptx` or godbolt).
8. **Bank conflicts are real.** You'll debug one eventually. Padding shared memory by 1 column fixes most of them.
9. **The CUDA programming guide is actually readable.** Not a textbook — a reference. Ctrl+F is your friend.
10. **Don't trust the internet for CUDA.** A lot of blog posts have bugs. Always verify with profiler.

## 🔗 Quick Reference

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [CUDA Runtime API Reference](https://docs.nvidia.com/cuda/cuda-runtime-api/)
- [PTX ISA Reference](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [CUDA Compute Capability Table](https://developer.nvidia.com/cuda-gpus)
- [GPU Architecture Whitepapers](https://developer.nvidia.com/cuda-toolkit-archive) (Ampere, Hopper, etc.)
