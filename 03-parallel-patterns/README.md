# Module 03 — Parallel Patterns

## Learning Goals

1. Implement parallel reduction (you've started this — now master it)
2. Implement parallel prefix sum (scan) — inclusive and exclusive
3. Implement parallel histogram
4. Understand the work-efficiency tradeoff in parallel algorithms
5. Use atomics correctly (and understand when NOT to use them)

## Theory

### Reduction (Review + Complete)

You know this from Ch.10. The key insight: a tree-based approach reduces N elements in O(log N) parallel steps.

**Your unfinished business** (from your ch10 notes):
- ✅ Naive reduction (divergent)
- ✅ Better access pattern
- ✅ Shared memory version
- ⬜ Multi-block with atomics (you started this but it had bugs)
- ⬜ Coarsened version (you noted "write coarsening kernel after a little while")

Mark Harris's "Optimizing Parallel Reduction" PDF has 7 optimization levels. You should know all of them.

### Prefix Sum (Scan)

Given input `[a₀, a₁, a₂, a₃, ...]`:
- **Inclusive scan**: `[a₀, a₀+a₁, a₀+a₁+a₂, ...]`
- **Exclusive scan**: `[0, a₀, a₀+a₁, a₀+a₁+a₂, ...]`

Two classic algorithms:
1. **Hillis-Steele** (inclusive): work-inefficient O(N log N) but fewer steps
2. **Blelloch** (exclusive): work-efficient O(N) — two phases: up-sweep + down-sweep

Scan is one of the most important parallel primitives. It's used inside: sort, compact/filter, radix sort, sparse matrix ops, and more.

### Histogram

Count occurrences of values. The GPU challenge: many threads want to increment the same counter → contention.

Approaches:
1. **Global atomics only** — simple but slow (serialization)
2. **Privatization** — each block builds a local histogram in shared memory, then merges
3. **Aggregation** — coarsen: each thread accumulates locally before atomic

From your PMPP Ch.6 anki: *"Privatization — applying partial updates to a private copy of the data and then updating the universal copy when done"*

## Reading List

1. **PMPP Chapter 10** — Reduction (you have notes + kernels)
2. **PMPP Chapter 11** — Prefix sum / scan (you have notes)
3. **PMPP Chapter 9** — Histogram (privatization pattern)
4. **Mark Harris** — [Optimizing Parallel Reduction (PDF)](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
5. **GPU Gems 3 Ch.39** — [Parallel Prefix Sum with CUDA](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda)

## Exercises

| File | What You'll Do | Difficulty |
|------|---------------|------------|
| `exercises/01_multi_block_reduce.cu` | Full reduction across any input size (multi-block + coarsening) | ⭐⭐ |
| `exercises/02_scan.cu` | Inclusive + exclusive scan (single block, then multi-block) | ⭐⭐⭐ |
| `exercises/03_histogram.cu` | Histogram: naive atomics → privatized → coarsened | ⭐⭐⭐ |
