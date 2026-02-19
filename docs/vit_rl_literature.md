# Vision Transformers for RL from Pixels: Literature Review

## Key Papers

### 1. Evaluating Vision Transformer Methods for Deep RL from Pixels (2022)
**Link:** [arxiv.org/abs/2204.04905](https://arxiv.org/abs/2204.04905)

The most relevant paper for online ViT-based RL from pixels.

**Key Findings:**
- CNN (with RAD augmentation) still generally outperforms ViT
- Reconstruction-based auxiliary losses (MAE, Data2Vec) significantly help ViT
- Contrastive learning is much less effective for ViT in RL

**Recommended ViT Architecture (84x84 images):**
| Parameter | Value |
|-----------|-------|
| Patch size | 12x12 |
| Depth | 4 layers |
| Attention heads | 8 |
| MLP dimension | 128 |
| Latent dimension | 128 |
| Encoder LR | 1e-3 |

**Auxiliary Loss Settings:**
- MAE: 75% masking ratio (best)
- Data2Vec: 40% masking ratio (competitive)

**Implementation Notes:**
- Use SAC as RL algorithm
- Random cropping as augmentation
- Block gradient from actor to encoder (only critic updates encoder)
- Smaller batch for auxiliary (128) vs RL updates (512)

---

### 2. MVP: Masked Visual Pre-training for Motor Control (2022)
**Link:** [arxiv.org/abs/2203.06173](https://arxiv.org/abs/2203.06173)

**Key Findings:**
- MAE pre-training on natural images transfers well to motor control
- Frozen encoder + RL on top works surprisingly well
- Outperforms supervised pre-training by up to 80% success rate

**Limitation for Online RL:**
- Requires offline pre-training on ImageNet-scale data
- Not directly applicable to pure online learning

---

### 3. Real-World Robot Learning with Masked Visual Pre-training (2022)
**Link:** [openreview.net/forum?id=KWCZfuqshd](https://openreview.net/forum?id=KWCZfuqshd)

Extension of MVP to real-world robotics.

---

## Key Insights for Online ViT RL

### Why CNNs Still Win
1. **Inductive biases**: Translation equivariance is built-in for CNNs
2. **Sample efficiency**: ViTs need more data to learn spatial structure
3. **Small images**: 84x84 gives very few patches (9-49 tokens)

### What Helps ViTs

1. **More patches**: Smaller patch size = more tokens for attention
   - 12x12 patches on 84x84 → 49 tokens
   - 14x14 patches on 42x42 (after stem) → 9 tokens (too few!)

2. **Reconstruction auxiliary losses**: MAE-style losses help most
   - LeJEPA/JEPA are in this family

3. **Conv stem**: Helps with small images, provides local features before patching

4. **Proper attention capacity**: 8 heads performs better than fewer

### Recommended Changes (vs our previous config)

| Parameter | Previous | Literature | Rationale |
|-----------|----------|------------|-----------|
| patch_size | 14 | 7 | More patches (36 vs 9) |
| num_heads | 3 | 6-8 | More attention capacity |
| hidden_size | 192 | 128 | Matches paper, prevents overfitting |
| mlp_dim | 768 | 512 | Proportional to hidden_size |

---

## Open Questions

1. Can online auxiliary losses (LeJEPA) match offline pre-training (MVP)?
2. Is there a ViT architecture that matches CNN without pre-training?
3. Do Stiefel constraints on Q/K help or hurt in RL's non-stationary setting?

---

## References

```bibtex
@article{tao2022evaluating,
  title={Evaluating Vision Transformer Methods for Deep Reinforcement Learning from Pixels},
  author={Tao, Jordan and Chockalingam, Sai and Bhatt, Shubham and Sycara, Katia},
  journal={arXiv preprint arXiv:2204.04905},
  year={2022}
}

@article{xiao2022masked,
  title={Masked Visual Pre-training for Motor Control},
  author={Xiao, Tete and Radosavovic, Ilija and Darrell, Trevor and Malik, Jitendra},
  journal={arXiv preprint arXiv:2203.06173},
  year={2022}
}
```
