# Module 06 — Online Softmax

## Learning Goals

1. Implement naive softmax (3 passes: max, sum, normalize)
2. Understand numerical stability (why subtract max)
3. Implement the online softmax algorithm (1 pass for max+sum, 1 for normalize → 2 passes total)
4. Understand how online softmax enables Flash Attention's tiling strategy

## Theory

### Softmax Definition

```
softmax(x)_i = exp(x_i) / sum_j(exp(x_j))
```

### Numerical Stability Problem

If `x_i` is large (e.g., 100), `exp(100)` overflows float32. Solution: subtract max first.

```
softmax(x)_i = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))
```

This is mathematically identical but numerically stable.

### Naive Implementation (3 passes over data)

```
Pass 1: m = max(x)           — find maximum
Pass 2: d = Σ exp(x_i - m)   — compute denominator
Pass 3: y_i = exp(x_i - m)/d — normalize
```

For attention with N tokens and d dimensions, each pass reads the full row.
Memory bandwidth = 3 × N × d reads. For long sequences, this is painful.

### Online Softmax (Milakov & Gimelshein 2018)

**Key insight**: you can compute max AND sum in a SINGLE pass by correcting
the running sum whenever you find a new maximum.

```python
m = -inf  # running max
d = 0     # running sum of exp(x_i - m)

for x_i in x:
    m_new = max(m, x_i)
    d = d * exp(m - m_new) + exp(x_i - m_new)  # correct old sum + add new term
    m = m_new
```

Why does the correction work?
- Previously: d = Σ exp(x_j - m_old)
- We want: d = Σ exp(x_j - m_new)
- Multiply: d * exp(m_old - m_new) converts old sum to new base
- Then add: exp(x_i - m_new) for the new element

After the loop: normalize with `y_i = exp(x_i - m) / d` (one more pass).

**Result: 2 passes instead of 3.**

### Why This Matters for Flash Attention

Flash Attention tiles the attention computation along the sequence dimension.
At each tile, we only see PART of the row. We can't compute the final max or sum yet.

Online softmax lets us:
1. Process tile 1: get partial (m₁, d₁)
2. Process tile 2: update to (m₂, d₂) using the correction formula
3. Continue tiling... at the end we have the exact global (m, d)
4. Rescale output tiles to account for updated normalization

This is THE algorithm that makes Flash Attention possible.

## Reading List

1. **Milakov & Gimelshein** — [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867) — short and clear
2. **Tri Dao** — Flash Attention paper Section 3.1 (online softmax in context)
3. **NVIDIA** — [Online Softmax blog post](https://developer.nvidia.com/blog/) (search for it)

## Exercises

| File | What You'll Do | Difficulty |
|------|---------------|------------|
| `exercises/01_naive_softmax.cu` | 3-pass softmax (max, sum, normalize) | ⭐⭐ |
| `exercises/02_online_softmax.cu` | 2-pass online softmax (Milakov algorithm) | ⭐⭐⭐ |
| `exercises/03_fused_softmax.cu` | Fused softmax for attention rows (batch of rows) | ⭐⭐⭐ |
