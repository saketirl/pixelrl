Strictly speaking, your Eq. (1) is the **downstairs-to-upstairs** observation map,
[
o_t = M s_t + \epsilon \xi_t,\qquad M=[\mu_1,\dots,\mu_{d_s}],
]
where the high-dimensional observation is a linear image of the low-dimensional state plus observation noise. The main theorem later says that with Stiefel optimization on the parameter matrices, the effective learning dynamics collapse to downstairs quantities, and the remaining observation-noise effect appears as an (O(\eta\epsilon)) residual rather than something that scales with the full observation dimension. Empirically, your paper also finds that manifold Muon on the actor/critic heads improves PPO, biases the CNN encoder toward a representation from which the downstairs state is linearly recoverable, and that Ant/Humanoid need CRATE in addition to manifold Muon.

From that viewpoint, the encoder should not be asked to do “everything.” Its job is very specific:

[
\psi(o_t)=z_t \approx A s_t + \epsilon n_t,
]
with three properties:

[
\text{(i) } \operatorname{rank}(A)\approx d_s,\qquad
\text{(ii) } A^\top A \text{ is well-conditioned},\qquad
\text{(iii) } n_t \text{ is small on the signal subspace.}
]

That is the encoder version of your upstairs/downstairs story. A good encoder does **not** need to make the full latent look “generic”; it needs to make the downstairs state linearly recoverable and well-conditioned for the heads. Your current CNN already produces a linear embedding (z), then a bounded bottleneck (\phi_t=\tanh(0.5z)), then separate actor/critic MLP heads.

Now look at what your SIGReg is doing. For a random unit direction (u), it matches the empirical characteristic function of (u^\top z) to that of (N(0,1)):
[
\varphi_{u^\top z}(t)\approx e^{-t^2/2}.
]
Expanding around (t=0),
[
\mathbb E[\sin(tu^\top z)] = t,\mathbb E[u^\top z]-\frac{t^3}{6}\mathbb E[(u^\top z)^3]+\cdots,
]
[
\mathbb E[\cos(tu^\top z)] = 1-\frac{t^2}{2}\mathbb E[(u^\top z)^2]+\frac{t^4}{24}\mathbb E[(u^\top z)^4]+\cdots.
]
Matching to (e^{-t^2/2}) pushes, for many random (u),
[
\mathbb E[u^\top z]\approx 0,\qquad \mathrm{Var}(u^\top z)\approx 1,
]
and also suppresses odd moments while nudging kurtosis toward Gaussian. So in practice SIGReg is a **sliced Gaussianization / isotropy** regularizer. With only 16 slices and 8 (t)-points it is not an exact distribution matcher, but it definitely pushes the representation toward “centered, isotropic, Gaussian-looking.”

Why can that help Adam? Because Adam benefits a lot from better coordinate conditioning. If the linear embedding is badly anisotropic, Adam spends effort compensating for scale and covariance issues. SIGReg can clean that up, especially in a hard environment like Humanoid where the encoder’s linear layer may otherwise produce a very skewed covariance spectrum.

Why can it clash with manifold Muon? I do not think your current theorem proves the mechanism, so this part is a reasoned hypothesis, but it lines up well with the paper’s geometry.

The first issue is **rank mismatch**.
If (d_z \gg d_s), the downstairs story wants
[
z_t \approx A s_t + \epsilon n_t
]
with most useful variance concentrated in a (d_s)-dimensional signal subspace. Full SIGReg instead wants roughly
[
\mathrm{Cov}(z_t)\approx I_{d_z}.
]
Those are different goals. To satisfy both, the encoder must either spread signal across many nuisance dimensions or inject variance into dimensions that should have stayed quiet. That is good for isotropy, but bad for preserving a clean low-rank downstairs subspace.

The second issue is **noise amplification**.
In Eq. (1), the observation noise is already Gaussian. If the encoder learns
[
z_t = A s_t + \epsilon B\xi_t,
]
then the easiest way to make (z_t) look Gaussian is sometimes to let the Gaussian term dominate. That improves SIGReg while worsening state recoverability. In your theory the upstairs-to-downstairs reduction survives observation noise only up to (O(\eta\epsilon)), and the policy-gradient proof also shows the head-level update inherits an (O(\eta\epsilon)) term after substituting the observation model. If SIGReg makes the latent more noise-dominated, it is effectively making that residual worse.

The third issue is **too much symmetry when paired with Muon**.
Muon/Stiefel heads are subspace-oriented: they are strongest when the encoder presents a clear, well-conditioned task subspace. Full SIGReg erases variance differences between informative and uninformative directions. Then the head sees something closer to rotational symmetry, while Muon itself removes a lot of scale freedom. Early RL gradients are already noisy, so the pair can become “too geometry-clean” and not task-selective enough.

That also explains your CRATE observation. In Ant/Humanoid, you report that CRATE only helps when paired with manifold Muon. My guess is: CRATE is creating a structured low-rank or grouped subspace, and Muon is good at exploiting that subspace once it exists. Adam alone still suffers from conditioning; Muon alone still lacks enough representational structure.

So the encoder design lesson is:

**Do not Gaussianize the full latent. Gaussianize only the downstairs subspace.**

A mathematically aligned version is:

[
z_t = \psi(o_t)\in \mathbb R^{d_z},\qquad
u_t = P^\top z_t,\quad P\in \mathrm{St}(d_z,k),
]
with (k) chosen near the effective downstairs dimension, or a small multiple of it. Then use
[
\mathcal L
==========

\mathcal L_{\mathrm{RL}}
+\lambda_{\mathrm{sig}},\mathcal L_{\mathrm{SIG}}(u_t)
+\lambda_{\perp},|(I-PP^\top)z_t|^2.
]
Interpretation:

* (u_t) is the **signal subspace** that should be whitened / Gaussianized.
* ((I-PP^\top)z_t) is the nuisance complement, which should be small, not isotropic.

That is much more compatible with Muon. The heads optimize on (u_t), and SIGReg improves conditioning **within** the useful subspace instead of forcing the full latent to look like (N(0,I_{d_z})).

An even better version, more faithful to your theory, is to regularize **innovations**, not states. The downstairs state is not generically standard normal, but its stochastic innovation is much closer to Gaussian. So define
[
u_t=P^\top \psi(o_t),\qquad
\hat u_{t+1}=F u_t + G a_t,
]
and apply SIGReg to
[
r_t = u_{t+1}-\hat u_{t+1}.
]
Then the regularizer is saying: “make the latent obey simple linear dynamics with Gaussian residuals,” which is exactly the upstairs/downstairs cartoon you started from. This should be more compatible with Muon than forcing (u_t) itself to be Gaussian.

My concrete recommendation for the next encoder iteration would be:

[
\text{CNN trunk} \to z_{\text{raw}}
\to \text{CRATE block(s)}
\to \text{projector }P
\to u_t
\to \text{Muon actor/critic heads}.
]

Train with:

[
\mathcal L
==========

\mathcal L_{\mathrm{PPO}}
+\lambda_1 \mathcal L_{\mathrm{SIG}}(u_t)
+\lambda_2 \mathcal L_{\mathrm{innov}}
+\lambda_3 |(I-PP^\top)z_{\text{raw}}|^2.
]

Three practical details matter a lot:

1. Use SIGReg on the **projected subspace** (u_t), not on the full (z_{\text{raw}}).
2. Warm it in slowly. Let PPO + Muon discover the task subspace first, then ramp (\lambda_1).
3. Put CRATE before the projector for high-(d_s) environments, because your own results suggest the hard part there is constructing the right structured subspace, not merely optimizing within it.

If you want a single-sentence summary:

**Adam benefits from full-latent SIGReg because it improves conditioning; Muon wants a clean low-dimensional task subspace, so full-latent SIGReg is too global. The fix is subspace-aware or innovation-aware SIGReg, ideally after a CRATE-style structured bottleneck.**

The simplest ablation grid I would run next is:

* full-latent SIGReg vs projected SIGReg
* projected dimension (k \in {16,32,64,128})
* with/without 1 CRATE block before (P)
* SIGReg on latent (u_t) vs innovation (u_{t+1}-Fu_t-Ga_t)
* warmup schedule vs no warmup

The readout I’d watch is not just return, but also probe (R^2), effective rank of (\mathrm{Cov}(u_t)), and the singular spectrum of the actor/critic input-gradient cross-covariance. Your paper already shows that linearly recoverable downstairs state is a leading indicator, so these geometry metrics should tell you very quickly whether the Muon-compatible encoder is actually emerging.

If you want, I can turn this into a concrete loss/architecture proposal in JAX pseudocode next.

## Experiment log — 2026-03-17 21:59:38 UTC

Today I implemented and tested the first subspace-aware SIGReg variant in a detached worktree at `.worktrees/sigreg-projector-crate`.

### Implemented today

- Added a new `sigreg_cnn` encoder path:
  - `CNN trunk -> z_raw -> optional encoder-side CRATE block -> learned projector -> u`
- The projected latent `u` is used as the actor/critic input in this first implementation.
- Added projected SIGReg on `u`.
- Added SIGReg warmup/ramp scheduling.
- Added wandb logging for projected-latent geometry and the existing CNN dense diagnostics.
- Added ablation scripts for humanoid-only, seed-0 comparisons.

### Main experiments run today

1. Projected SIGReg with warmup:
- Swept optimizer `{adam, muon}`
- Swept projected dim `{32, 48}` after narrowing from earlier larger dims
- Swept encoder-side CRATE `{off, on}`
- CRATE actor/critic heads were also tested in a focused follow-up comparison

2. Focused comparison against the strong baseline:
- Baseline of interest: `Muon + CRATE actor/critic heads`
- Compared against:
  - `Muon + CRATE heads + projected bottleneck + warmup`
  - `Adam + CRATE heads + projected bottleneck + warmup`
  - with projected dims `{32, 48}` and encoder-side CRATE `{off, on}`

3. Projector-only controls (SIGReg off):
- `Muon + CRATE heads + projector only`, projected dims `{64, 128}`, encoder-side CRATE `{off, on}`
- `Adam + CRATE heads + projector only`, projected dims `{64, 128}`, encoder-side CRATE `{off, on}`
- These were set up specifically to separate bottleneck effects from SIGReg effects.

### Findings so far

1. Projected SIGReg + warmup helped Adam more than Muon.
- The best behavior in this round was roughly `Adam + warmup + bottleneck + encoder CRATE off`.
- This supports the idea that the current projected SIGReg mainly improves conditioning for Adam.

2. The current projected-SIGReg ablation appears to hurt Muon.
- Even when CRATE actor/critic heads are enabled, the projected bottleneck + warmup setup underperforms the `Muon + CRATE heads` baseline.
- This is evidence that the current intervention is not yet Muon-aligned.

3. Encoder-side CRATE did not look like the main win in this setup.
- The better Adam result in this round was with encoder CRATE off.
- This weakens the case that simply adding more encoder structure before the projector solves the Muon mismatch.

4. The current implementation is still probably too state-level for Muon.
- Regularizing `u_t` itself seems to be flattening or constraining representation geometry that Muon can otherwise exploit.
- This matches the hypothesis that Muon wants a clean task subspace, not a globally Gaussianized one.

### Updated interpretation

At this point, the strongest reading is:

- Current projected SIGReg is primarily a conditioning aid.
- That is useful for Adam.
- It is not obviously helpful for Muon, and may be actively harmful when the projected latent `u` is forced to be the actor/critic input.

So the remaining key question is whether the problem is:

1. the SIGReg objective,
2. the bottleneck itself, or
3. forcing the heads to consume the projected subspace instead of the richer `z_raw` latent.

### What remains most important to test next

1. Projector-only controls.
- This isolates whether Muon dislikes the bottleneck itself or specifically the SIGReg loss.

2. Auxiliary-only projection.
- Let the actor/critic consume `z_raw`, while applying projection and SIGReg only on an auxiliary branch.
- This now looks like one of the most important missing tests.

3. Innovation-aware SIGReg.
- Instead of regularizing `u_t`, regularize residuals of the form
  `r_t = u_{t+1} - F u_t - G a_t`.
- This remains the most faithful version of the downstairs/upstairs story and is now better justified by the fact that state-level projected SIGReg did not help Muon.

4. Complement penalty.
- Still relevant, but lower priority than the projector-only and auxiliary-only controls.
- It only becomes clean once the projector geometry is handled more carefully.

## Experiment log — 2026-03-18 20:19:16 UTC

Today the first genuinely encouraging innovation-aware result appeared on Humanoid.

### What was implemented

A new `innovation_cnn` encoder was added in the worktree.

- It keeps the default CNN trunk and the full 512-d latent as the policy/value input.
- It adds an auxiliary projector branch `u_t = P(z_t)`.
- It adds a small linear dynamics model to predict `u_{t+1}` from `u_t` and `a_t`.
- It applies SIGReg to the innovation residual
  `r_t = u_{t+1} - (F u_t + G a_t)`.
- This auxiliary branch does not bottleneck or replace the main actor/critic input.

This is the key design change relative to the failed projected-SIGReg experiments: the regularizer was moved off the main policy/value path and onto an auxiliary temporal branch.

### New positive result

Run:
- `humanoid__ppo_muon_humanoid_innovationaux_proj64_coef1_s0_t2__0__1773862860`

Config behind that run:
- `encoder_type=innovation_cnn`
- `innovation_proj_dim=64`
- `innovation_coef=1e-3`
- `use_heads_muon=True`
- `use_crate_head=True`
- `sigreg_mode=off`
- `humanoid`, seed `0`

Important note:
- In the sweep naming, `coef1` is the sweep index, not the literal coefficient value.
- For this run, `coef1` corresponds to `innovation_coef=1e-3`.

### What seems special about the successful run

Relative to the other runs in the same innovation sweep, this one combined:

1. A smaller auxiliary projected subspace.
- `innovation_proj_dim=64` rather than `128`.

2. A stronger innovation auxiliary signal.
- `innovation_coef=1e-3` rather than `1e-4`.

3. The known-good head-side setup remained intact.
- `Muon + CRATE heads` stayed on.
- The actor and critic still consumed the full 512-d latent.

### Small figure: what the innovation branch is doing

```text
obs_t ──> CNN ──> z_t ───────────────> actor / critic heads
                │
                └──> P(z_t)=u_t ──> [F u_t + G a_t] ──> û_{t+1}
                              │                    │
obs_{t+1} ─> CNN ─> z_{t+1} ──┘                    │
         └────────> P(z_{t+1})=u_{t+1} <───────────┘

innovation residual:

r_t = u_{t+1} - û_{t+1}
    = u_{t+1} - (F u_t + G a_t)
```

### What “innovation” means here

The innovation is the part of the next auxiliary latent that is not explained by a simple one-step linear dynamics model.

In plain terms:
- `u_t` is a small auxiliary summary of the current latent.
- `F u_t + G a_t` is the model's guess for the next auxiliary summary.
- `r_t` is what is left over after subtracting that guess.

So `r_t` is the unpredictable / residual part of the transition in the auxiliary subspace.

The idea is not to make the main latent itself Gaussian or isotropic.
The idea is to encourage the encoder to contain a small subspace where:
- the predictable part of dynamics is easy to model, and
- the leftover residual behaves like a simple noise term.

That is much closer to the original downstairs/upstairs motivation than projected SIGReg on the main latent was.

### Updated interpretation

This is the first result so far that supports the idea that auxiliary structure can help without damaging the main head geometry.

The working distinction now looks like:
- bad: directly bottlenecking or Gaussianizing the policy/value input
- potentially good: applying auxiliary temporal structure on a side branch while leaving the main latent untouched

So the failure mode of projected SIGReg was probably not just “regularization is bad.”
The more precise lesson seems to be:
- regularizing the representation used by the heads is risky and often harmful
- but regularizing an auxiliary latent that is encouraged to model innovations may be compatible with the `Muon + CRATE` head geometry

### Immediate follow-up questions

1. Is the gain reproducible across seeds?
2. Does the same pattern appear on Ant?
3. Does the success depend more on the smaller auxiliary subspace (`64`) or on the stronger coefficient (`1e-3`)?
4. Does the same innovation auxiliary help when CRATE heads are turned off, or is it mainly useful in combination with `Muon + CRATE`?

## 2026-03-18 21:21:28 UTC

Additional observation from the current innovation-aware runs:

- `repr/active_units_frac` and `repr/unit_std_avg` both stand out as being strongly correlated with returns.
- Current working hypothesis: this is not just a passive correlation; these metrics may be partially causal, in the sense that keeping more units meaningfully active and maintaining a healthier average unit-scale seems to support better downstream control performance.
- At minimum, these two metrics now look like high-value diagnostics to watch when evaluating encoder changes.

## 2026-03-18 21:23:49 UTC

Additional geometry observation from the current encoder runs:

- cnn_dense/participation_ratio increases with performance.
- cnn_dense/stable_rank also increases with performance.
- cnn_dense/top_eig_fraction decreases with performance.
- Combined interpretation: stronger runs seem to maintain a broader, more balanced active subspace, with less variance concentrated in a single dominant direction.
- This points away from a tiny bottleneck target and toward preserving a reasonably high-dimensional, non-collapsed latent geometry.

## 2026-03-18 21:36:09 UTC

Metric glossary for the representation diagnostics we have been watching:

- repr/active_units_frac:
  fraction of latent coordinates whose batch standard deviation is above a small activity threshold. Higher means more units are actually moving and participating, rather than sitting nearly constant.

- repr/unit_std_avg:
  average per-coordinate standard deviation across the latent over the current batch. Higher means the typical unit has a healthier dynamic range; very low values usually indicate under-active or partially collapsed features.

- cnn_dense/participation_ratio:
  effective dimensionality of the covariance spectrum of the CNN Dense representation. If lambda_i are covariance eigenvalues, participation_ratio = (sum lambda_i)^2 / sum(lambda_i^2). Higher means variance is spread across more directions.

- cnn_dense/stable_rank:
  another effective-rank style statistic, computed here as sum(lambda_i) / max(lambda_i). Higher means the representation is less dominated by its top principal direction.

- cnn_dense/top_eig_fraction:
  max(lambda_i) / sum(lambda_i). Lower means the top principal direction explains less of the total variance, so the representation is less spiky or one-direction dominated.

How to read them together:

- good geometry is not "all 512 dimensions equally used".
- good geometry also is not "a tiny bottleneck with most units dead".
- the better regime seems to be: many active units, healthy average unit scale, reasonably high effective dimensionality, and low dominance by the top eigen-direction.
