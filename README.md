# cuda-path-to-flash 🔥

Self-taught CUDA course — from "what's a thread?" to writing Flash Attention from scratch.

No university degree needed. Just a GPU and stubbornness.

## Roadmap

| # | Topic | Goal | Status |
|---|-------|------|--------|
| 01 | GPU Architecture & Basics | Understand threads/blocks/warps, write first kernel | 🔨 |
| 02 | Memory Hierarchy | Shared memory, coalescing, bank conflicts | ⬜ |
| 03 | Parallel Patterns | Reduction, scan, histogram | ⬜ |
| 04 | Matmul: Naive → Tiled | Write matmul from 1% to ~30% cuBLAS | ⬜ |
| 05 | Matmul: Vectorized & Beyond | Vectorized loads, double buffering, warp tiling → 60%+ cuBLAS | ⬜ |
| 06 | Online Softmax | Numerically stable softmax, online algorithm | ⬜ |
| 07 | Flash Attention | Tiling + online softmax + fused kernel | ⬜ |

## Key Resources

- **PMPP** — *Programming Massively Parallel Processors* (Kirk & Hwu) — the textbook
- **Simon Boehm** — [How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM) — the matmul bible
- **Tri Dao** — Flash Attention [1](https://arxiv.org/abs/2205.14135) & [2](https://arxiv.org/abs/2307.08691) papers
- **Milakov & Gimelshein** — [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)
- **Towernest** — CUDA kernel optimization deep dives
- **Lei Mao** — [leimao.github.io](https://leimao.github.io/) — incredible CUDA blog
- **Mark Harris / NVIDIA** — Classic CUDA blog posts (reduction, shared memory, transpose)
- **GPU Puzzles** — [srush/GPU-Puzzles](https://github.com/srush/GPU-Puzzles) — interactive exercises

👉 **[RESOURCES.md](RESOURCES.md)** — The full resource bible: books, blogs, papers, repos, videos, courses, people to follow. Everything.

## How to Use

```bash
cd 01-gpu-basics
make test        # run correctness checks
make bench       # run benchmarks
```

Each module has exercises (skeleton → fill in the kernel), tests, benchmarks, and hints you should only open when stuck.
