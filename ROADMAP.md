# Roadmap — Real Goals, Real Timelines

No bullshit. Here's where we are and where we're going.

## Current State (February 2026)

- ✅ PMPP chapters 2-11 read, questions answered
- ✅ Written: tiled matmul (5 variants), reductions, vectorized vector add
- ✅ Course repo built with 23 exercises
- ❌ Never trained a model from scratch
- ❌ Never used custom CUDA kernels in a real training loop
- ❌ No dedicated GPU access

## The Actual Goal

**Train a small language model (GPT-2 124M scale) where at least one custom CUDA kernel is used in the training loop.**

That's the finish line. Everything else is a waypoint.

---

## Phase 1: Finish the Kernel Exercises (2-3 weeks)

**Time commitment**: 1-1.5 hours/day, using Kaggle free tier

### Week 1-2: Modules 01-03
The fundamentals. You've done most of this in PMPP already, so it should go fast.
- [ ] Module 01: hello GPU, vector add, matrix ops (warm-up, 1 day)
- [ ] Module 02: reduction, transpose, tiled matmul from memory (2-3 days)
- [ ] Module 03: multi-block reduction, scan, histogram (2-3 days)

**Checkpoint**: all exercises in 01-03 show ✅

### Week 3: Modules 04-05 (Matmul deep dive)
This is where you spend real time. Follow Simon Boehm kernel by kernel.
- [ ] Kernel 1-3: naive → coalesced → shmem tiled (1-2 days)
- [ ] Kernel 4-5: 1D and 2D block-tiling (2-3 days)
- [ ] Kernel 6-8: vectorized, bank conflicts, warp tiling (3-4 days)
- [ ] Profile with ncu, understand WHERE your kernel is bottlenecked

**Checkpoint**: your best matmul hits 30%+ of torch.mm

### Plane reading
- [ ] Online softmax paper (9 pages)
- [ ] Harris parallel reduction (38 slides)
- [ ] Flash Attention 1 paper

---

## Phase 2: Softmax + Flash Attention (1-2 weeks)

### Week 4: Module 06
- [ ] Naive softmax (should be quick after module 02-03)
- [ ] Online softmax — THE key algorithm
- [ ] Causal softmax

### Week 5: Module 07
- [ ] Naive attention baseline
- [ ] Flash Attention forward — take your time, get it correct
- [ ] Flash Attention causal
- [ ] Read FA2 paper, understand the loop order change

**Checkpoint**: Flash Attention output matches F.scaled_dot_product_attention ✅

---

## Phase 3: Train Something (2-3 weeks)

This is where the CUDA course ends and the real world begins.

### Week 6: Learn the Training Stack
You don't need to write everything from scratch. Learn the parts:
- [ ] Understand PyTorch training loop (forward, loss, backward, step)
- [ ] Read Andrej Karpathy's nanoGPT (~300 lines that train GPT-2)
  - https://github.com/karpathy/nanoGPT
- [ ] Run nanoGPT on a GPU, see a loss curve go down
- [ ] Understand: what are the hot kernels? (matmul, attention, layernorm, softmax)

**This requires a GPU for hours.** Options:
1. **Kaggle**: 30h/week free, enough for GPT-2 124M on Shakespeare/OpenWebText subset
2. **Colab Pro**: ~$10/mo, more hours
3. **Vast.ai**: ~$0.20-0.50/hr for a decent GPU, SSH access
4. **Lambda / RunPod**: similar pricing
5. **University compute** if accessible (some have free GPU programs)

For GPT-2 124M on a single GPU, you need maybe 4-8 hours of A100/H100 time.
On a T4 (Kaggle free), more like 12-24 hours (split over multiple sessions).

### Week 7: Plug in a Custom Kernel
Pick ONE kernel to replace in the training loop:
- **Option A**: Replace torch.mm with your SGEMM (easiest to validate, hardest to beat PyTorch)
- **Option B**: Replace F.softmax with your online softmax (more realistic win)
- **Option C**: Replace F.scaled_dot_product_attention with your Flash Attention (the boss move)

Steps:
- [ ] Write a PyTorch autograd Function that calls your CUDA kernel
- [ ] Swap it into nanoGPT
- [ ] Verify: training loss matches the original (your kernel is correct if loss curves overlap)
- [ ] Benchmark: is your kernel faster/slower? By how much?

### Week 8: Make it Work End-to-End
- [ ] Train GPT-2 124M with your custom kernel for at least 1000 steps
- [ ] Generate text from the trained model
- [ ] Write up what you learned

**Checkpoint**: you have a model that generates text, trained with YOUR kernel.

---

## Phase 4: What's Next (ongoing)

Once you've done the above, you'll know what interests you. Possible directions:

### Go Deeper on Kernels
- Tensor Cores (WMMA / MMA instructions)
- FP16/BF16/FP8 matmul
- Fused kernels (attention + MLP, fused Adam)
- Triton (write kernels in Python, compiles to PTX)

### Go Wider on Training
- Multi-GPU training (NCCL, FSDP, tensor parallelism)
- Train a bigger model (GPT-2 350M, 1.5B)
- Train on a real dataset (FineWeb, RedPajama)
- Implement your own optimizer kernel (fused AdamW)

### Go into Inference
- KV cache optimization
- Quantization kernels (INT8, INT4)
- Speculative decoding
- Serving (vLLM, TGI)

---

## GPU Budget Reality Check

| Task | GPU Hours | Cost (Vast.ai A100) |
|------|-----------|-------------------|
| Exercises (modules 01-07) | ~10-15h | Free (Kaggle) |
| Run nanoGPT baseline | ~4-8h | ~$4-8 |
| Custom kernel integration + debug | ~5-10h | ~$5-10 |
| Full training run with custom kernel | ~4-8h | ~$4-8 |
| **Total** | **~25-40h** | **~$15-25 + Kaggle free** |

That's it. Under $30 to go from zero to "I trained a language model with my own CUDA kernel."

---

## Weekly Check-in Template

Every Sunday, fill this in honestly:

```
Week of: ___________
Hours spent: ___
Exercises completed: ___
What I learned: ___
What I'm stuck on: ___
Next week's plan: ___
```

---

## Non-Negotiable Rules

1. **Do the exercises in order.** Don't skip to Flash Attention because it's cool.
2. **Write kernels from memory.** If you can't write tiled matmul without looking, you don't know it.
3. **Profile before optimizing.** "I think it's slow because..." is not a reason. ncu output is.
4. **Correct first, fast second.** A fast wrong kernel is useless.
5. **Ship something.** A trained model with a janky kernel beats a perfect kernel that never ran.
