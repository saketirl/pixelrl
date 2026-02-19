# ViT Architecture Search: Findings Summary

## Final Architecture

```
Input (84×84×12)
    ↓ Conv stem: 2× Conv(64ch, 3×3), stride 2 → 42×42×64
    ↓ Patch embed: patch_size=21 → 2×2 = 4 tokens
    ↓ Transformer: 4 layers, 8 heads, hidden=128, mlp=512
    ↓ Mean pool → Dense(512) → LayerNorm
    ↓ 512-dim representation
```

**Key config:**
```
--encoder-type vit --vit-patch-size 21 --vit-use-conv-stem
--vit-hidden-size 128 --vit-mlp-dim 512 --vit-num-heads 8 --vit-num-layers 4
--vit-conv-stem-channels 64 --vit-conv-stem-kernel 3
--no-vit-use-cls-token --no-vit-apply-output-tanh
--encoder-lr 5e-5 --encoder-tanh-scale 0.25
```

---

## What Worked

- **Conv stem is essential.** The stem (CNN, stride 2) processes local structure before the transformer sees anything. Configs with stem consistently outperformed no-stem configs.
- **Very few tokens (4).** 2×2=4 patches after the stem is optimal — the transformer acts as a global aggregator over 4 coarse feature locations, not a spatial reasoner.
- **More depth, not more tokens.** 4–6 transformer layers work well. Increasing patches (to 16 or 36) hurt performance and throughput.
- **Small hidden dim (128, 8 heads).** Literature-inspired; head_dim=16. Wider configs (192, heads=3) performed worse.
- **Mean pooling over CLS token.** Marginally better and simpler.
- **No output tanh.** Removing the tanh bottleneck on the 512-dim output helped.
- **Low encoder LR (5e-5).** ViTs are sensitive to LR early in training; low LR with warmup (500 updates) was important.
- **Adam for encoder.** Muon was kept only for actor/critic head matrices.

---

## What Didn't Work

- **Large patch count (36–49 tokens).** Too slow and no better than 4 tokens. Attention is O(n²) — going from 4→36 tokens is ~81× more expensive in attention.
- **No conv stem with direct patch embedding.** Without the stem, 16 tokens is competitive but loses to 4-token + stem configs.
- **Large hidden size (192, heads=3, head_dim=64).** Worse than 128/8/16 despite more parameters.
- **CLS token readout.** No benefit over mean pooling.
- **Output tanh.** Hurt performance.
- **Muon / Q/K Stiefel on ViT.** All Stiefel LRs (1e-4, 1e-3, 1e-2) gave similar results to the baseline, but the constraint added significant wall-clock overhead. Not worth it.
- **Very deep transformers (>6 layers).** Depth plateaus around 4–6 layers; beyond that, no gain.

---

## Key Insight

The winning architecture is **not really a ViT** in the traditional sense. It is:

> **CNN stem → 4 global attention tokens → deep transformer**

The conv stem does the heavy lifting for local spatial features (as in a standard CNN). The transformer provides a small amount of learned global mixing over 4 coarse representations. This is closer to **CNN + attention pooling** than to the patch-based ViT designed for ImageNet.

This explains why the literature recommendations (many patches, no stem) didn't transfer: those configs assumed the transformer would learn spatial structure from scratch, which requires far more data than online RL provides.

---

## Comparison vs CNN Baseline

The CNN baseline (Conv 32→64→64, strides 4,2,1, Dense 512) is still the strongest overall. The best ViT configs are competitive but have not consistently beaten the CNN.

| Encoder | Notes |
|---------|-------|
| CNN | Best overall; strong inductive bias, fast |
| ViT (stem + 4 patches, 4 layers) | Competitive; slower to compile, similar SPS |
| ViT (no stem, 16+ patches) | Worse; too slow for online RL |
