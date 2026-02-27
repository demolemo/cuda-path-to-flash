# Module 07 — Flash Attention 🔥

## The Final Boss

Everything you've learned leads here. This is where:
- Tiled matmul meets online softmax
- Memory hierarchy mastery pays off
- Warp-level thinking becomes essential
- You write a kernel that fuses Q×K^T, softmax, and ×V into ONE pass over SRAM

## Learning Goals

1. Understand WHY Flash Attention exists (memory bandwidth bottleneck of standard attention)
2. Understand the tiling strategy (tile over N — sequence length)
3. Implement the forward pass of Flash Attention from scratch
4. Handle the online softmax rescaling of partial outputs
5. (Bonus) Implement backward pass

## Theory

### Standard Attention — The Problem

```
S = Q × K^T       (N×d × d×N = N×N)      ← materialize full N×N matrix
P = softmax(S)     (N×N)                   ← another N×N matrix
O = P × V          (N×N × N×d = N×d)      ← read full N×N matrix
```

Memory: O(N²) — for N=4096 and 32 heads, that's 4096² × 32 × 4 bytes = 2 GB just for S.
The N×N matrices don't fit in SRAM and must live in slow HBM → memory-bound.

### Flash Attention — The Solution

**Never materialize the full N×N attention matrix.**

Instead, tile over the key/value sequence dimension:
1. Load a block of Q (Br rows)
2. For each block of K, V (Bc columns):
   a. Compute partial S = Q_block × K_block^T (Br × Bc — fits in SRAM!)
   b. Update running softmax statistics (m, l) using online algorithm
   c. Update running output O using rescaling

The key equations (from Tri Dao's paper):

```
For each K/V block j:
  S_ij = Q_i × K_j^T                          (Br × Bc matmul in SRAM)
  m_ij = rowmax(S_ij)                          (local max)
  P_ij = exp(S_ij - m_ij)                      (local softmax numerator)
  l_ij = rowsum(P_ij)                          (local sum)

  m_new = max(m_old, m_ij)                     (update running max)
  l_new = l_old * exp(m_old - m_new) + l_ij * exp(m_ij - m_new)  (update running sum)

  O_new = O_old * (l_old * exp(m_old - m_new) / l_new)           (rescale old output)
        + P_ij * exp(m_ij - m_new) / l_new × V_j                 (add new contribution)
```

After processing all K/V blocks, O contains the exact attention output.
Total HBM access: O(N²d/M) where M is SRAM size. For typical M, this is O(N) per element!

### Algorithm Pseudocode

```python
# Q, K, V: (N, d) — in HBM
# O: (N, d) — output in HBM
# Br, Bc: tile sizes

# Initialize
O = zeros(N, d)
l = zeros(N)      # running sum
m = full(N, -inf) # running max

# Outer loop: over K/V blocks
for j in range(0, N, Bc):
    Kj = K[j:j+Bc]    # load from HBM → SRAM
    Vj = V[j:j+Bc]    # load from HBM → SRAM

    # Inner loop: over Q blocks
    for i in range(0, N, Br):
        Qi = Q[i:i+Br]          # load from HBM → SRAM
        Oi = O[i:i+Br]          # load from HBM → SRAM
        li = l[i:i+Br]
        mi = m[i:i+Br]

        # Compute attention for this tile
        Sij = Qi @ Kj.T         # (Br, Bc) — in SRAM
        mij = rowmax(Sij)       # (Br,)
        Pij = exp(Sij - mij)    # (Br, Bc)
        lij = rowsum(Pij)       # (Br,)

        # Update statistics
        mi_new = max(mi, mij)
        li_new = li * exp(mi - mi_new) + lij * exp(mij - mi_new)

        # Rescale and update output
        Oi = Oi * (li * exp(mi - mi_new) / li_new) + (Pij * exp(mij - mi_new) / li_new) @ Vj

        # Write back
        O[i:i+Br] = Oi
        l[i:i+Br] = li_new
        m[i:i+Br] = mi_new
```

### Complexity

| | Standard Attention | Flash Attention |
|---|---|---|
| HBM reads | O(N²) + O(N²) + O(N²) | O(N²d/M) |
| Extra memory | O(N²) for S, P | O(N) for m, l |
| Numerically exact? | Yes | Yes (same output!) |

## Reading List

1. **Tri Dao et al.** — [FlashAttention paper](https://arxiv.org/abs/2205.14135) — Section 3 (Algorithm 1)
2. **Tri Dao** — [FlashAttention-2](https://arxiv.org/abs/2307.08691) — better parallelism, swap loop order
3. **tspeterkim** — [flash-attention-minimal](https://github.com/tspeterkim/flash-attention-minimal) — ~100 lines, great reference
4. **jundaf2** — [CUDA-Flash-Attention](https://github.com/jundaf2/CUDA-Flash-Attention) — clean CUDA implementation
5. **Dao-AILab** — [Official implementation](https://github.com/Dao-AILab/flash-attention)

## Exercises

| File | What You'll Do | Difficulty |
|------|---------------|------------|
| `exercises/01_attention_naive.cu` | Standard attention (materialize S, P) — baseline | ⭐⭐ |
| `exercises/02_flash_forward.cu` | Flash Attention forward pass — THE exercise | ⭐⭐⭐⭐⭐ |
| `exercises/03_flash_causal.cu` | Add causal masking to Flash Attention | ⭐⭐⭐⭐ |

## Tips Before You Start

1. **Start with small sizes** (N=64, d=32, Br=Bc=16) and verify against naive
2. **Print intermediate values** — check m, l, O after each tile
3. **The rescaling is the hard part** — get the math on paper first
4. **Don't optimize prematurely** — get it CORRECT first, then make it fast
5. **Reference implementations exist** — if stuck for hours, read tspeterkim's 100-line version

## 🎉 When You're Done

If you've completed this module, you've gone from "what's a thread?" to writing
Flash Attention in CUDA. That's a university-level GPU programming course
plus state-of-the-art research, done on your own terms.

You're not done learning — there's always Tensor Cores, Triton, FlashAttention-3,
FP8, cp.async, warp specialization... but the foundation is SOLID.
