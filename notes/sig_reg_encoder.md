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


## 2026-03-22 UTC

Ant update from the new 1M diagnosis runs (`cnn + CRATE heads`, innovation off, anneal off, small VICReg variance-floor auxiliary):

- The key new ant insight is that the winning/losing split shows up earliest in the `cnn_dense/*` metrics, before return gives any useful signal.
- In the successful seed, `cnn_dense/feature_var_max` is already materially larger than the losing seeds very early:
  - more than `2x` all other seeds by `3840` steps
  - more than `5x` by `6400`
  - more than `10x` by `20480`
  - more than `50x` by `34560`
- `cnn_dense/feature_var_ratio` also separates early, though slightly later than `feature_var_max`.
- The visible encoder "rescue" happens later, around `40960-42240` steps:
  - `repr/unit_std_avg` jumps up
  - `repr/dead_units_frac` drops sharply
  - `policy/obs_to_noise_ratio` becomes nontrivial
  - `cnn_dense/pre_ln_std` falls instead of continuing to blow up
  - `losses/vicreg_var_total` drops below its saturated value
- The losing ant seeds never make that transition. They keep:
  - near-zero `repr/unit_std_avg`
  - `repr/dead_units_frac = 1`
  - near-zero `policy/obs_to_noise_ratio`
  - much larger `cnn_dense/pre_ln_std`
  - saturated `losses/vicreg_var_total`

Updated ant hypothesis:

- The earliest branching event is probably in the CNN dense bottleneck, not in the policy head and not in episodic return.
- Current candidate causal chain:
  - early preservation of variance in at least one dense-layer direction (`cnn_dense/feature_var_max`, `cnn_dense/feature_var_ratio`)
  - then encoder rescue around `~40k`
  - then a genuinely observation-conditioned policy
  - then the later performance phase change around `~550k-650k`
- So for ant, `cnn_dense/feature_var_max`, `cnn_dense/feature_var_ratio`, and `cnn_dense/pre_ln_std` currently look like the earliest high-value branch predictors. `repr/*` and `policy/*` still matter, but they appear to be later confirmations of a split that has already started inside the CNN dense bottleneck.

One caution:

- The variance-floor VICReg term did not reliably prevent collapse by itself; `3/4` seeds still collapsed. So the ant result is better read as a diagnosis of where branching happens than as strong evidence that VICReg is the fix.


## 2026-03-23 UTC

Matched ant baseline update (`cnn + CRATE heads`, innovation off, anneal off, `vicreg_var_coef=0`):

- The matched `vicreg=0` sweep reproduced the same qualitative result as the previous `vicreg=1e-3` sweep.
- In both sweeps:
  - one seed entered the alive / strong branch and finished around `900+` return by `1M`
  - three seeds collapsed early and still reached only the moderate-return local-minimum gait branch (`~480-510` return)
- So the good ant behavior is not being caused by the VICReg variance-floor term.
- The strong result is coming from the base configuration change:
  - plain `cnn`
  - `CRATE` heads on
  - innovation off
  - LR annealing off

Important confirmation from the matched baseline:

- The same early branch structure appears with `vicreg=0`:
  - good seed has healthy `repr/unit_std_avg`, low `dead_units_frac`, nontrivial `policy/obs_to_noise_ratio`, and much broader `cnn_dense` geometry
  - bad seeds still show collapsed `repr/*`, near-zero `policy/*`, exploding `cnn_dense/pre_ln_std`, and near-zero `cnn_dense/participation_ratio`
- This confirms that the earlier ant diagnosis was real and not an artifact of the auxiliary VICReg term.

Updated ant conclusion:

- For ant, the key problem is still the early CNN dense bottleneck branch.
- VICReg, as implemented here, did not materially change the branch probabilities.
- The next intervention should therefore target the early dense bottleneck dynamics directly rather than relying on the current variance-floor auxiliary.


## 2026-03-24 UTC

Ant follow-up from the `cnn + CRATE heads`, innovation-off, anneal-off sweeps after splitting `init_seed` and `data_seed`:

- Both initialization randomness and data / rollout randomness can flip the run into the good or bad branch.
- With fixed init and varying data randomness, one seed still became strong while the others stayed in the weaker branch.
- With fixed data randomness and varying init, some inits became strong while others did not.
- So the ant inconsistency is not "just an init issue" and not "just a rollout noise issue". It is better understood as an early instability that is sensitive to both.

Encoder-final-Muon update:

- Applying manifold Muon only to the final CNN bottleneck matrix appears to remove the catastrophic encoder-collapse branch.
- Across the factorized sweeps, all runs kept healthy representation statistics:
  - nontrivial `repr/unit_std_avg`
  - `repr/dead_units_frac` near zero
  - controlled `cnn_dense/pre_ln_std`
  - healthy `cnn_dense/participation_ratio`
- But this did not solve ant end-to-end. Returns still ranged from poor to strong, so the remaining problem is not simply encoder death.
- Updated interpretation:
  - encoder collapse was a real bottleneck
  - encoder-final Muon largely fixes that bottleneck
  - the remaining failure mode is policy-side: healthy latent, but still getting stuck in a bad control regime

Actor-conditionality update (`encoder_final_muon` baseline plus early actor conditionality floor):

- The actor-side conditionality regularizer substantially improved the floor of performance.
- Compared to the encoder-final-Muon baseline, the very poor runs disappeared:
  - before actor conditionality: returns included about `-135`, `-117`, `-111`, and `15`
  - with actor conditionality: all 8 runs finished positive, roughly `263-838`
- Aggregate shift:
  - mean return improved from about `234` to `518`
  - median improved from about `212` to `467`
  - worst-case return improved from about `-135` to `263`
- Representation health stayed strong, so this gain is not from further encoder stabilization.
- The main policy-side shift is that the weak low-conditionality / high-noise regime was reduced:
  - bad encoder-final-Muon runs had very low `policy/mean_action_obs_std_avg`, very high `policy/action_noise_std_avg`, and tiny `policy/obs_to_noise_ratio`
  - actor conditionality raises the effective state dependence of the actor and removes the catastrophic low-return branch

Updated ant conclusion:

- There now seem to be at least three qualitatively different ant regimes:
  - dead encoder / collapsed branch
  - alive encoder but weakly state-conditioned middling branch
  - alive encoder and strong branch
- Encoder-final Muon appears to remove the first regime.
- Early actor conditionality appears to remove most of the catastrophic policy-side failures and lift the floor into the middling regime.
- The remaining open problem is the second phase change:
  - how to move reliably from a healthy encoder plus middling policy into the strong ant regime.


## 2026-04-01 UTC

Ant innovation-direct update:

- Revisited the innovation encoder family with a new `innovation_direct_cnn` variant.
- The old `innovation_cnn` structure was:
  - CNN trunk -> `hidden` (`512`-d)
  - PPO actor/critic read `hidden`
  - separate projector produced `u` (`innovation_proj_dim`, typically `64`)
  - innovation auxiliary modeled `u_{t+1}` from `(u_t, a_t)` and regularized the residual `u_{t+1} - \hat{u}_{t+1}`
- The key issue with that design is that the innovation branch was indirect:
  - PPO optimized `hidden`
  - the innovation objective optimized `u`
  - so the auxiliary could shape the encoder, but the policy did not directly act on the innovation bottleneck

What changed in `innovation_direct_cnn`:

- The encoder still builds the same CNN trunk and `hidden` representation.
- It still projects to `u` through `innovation_projector`.
- But PPO now reads `u` directly instead of `hidden`.
- So the new structure is effectively:
  - pixels -> CNN trunk -> `hidden` -> `innovation_projector` -> `u`
  - actor / critic read `u`
  - innovation dynamics also operate on `u`

Why this is more principled:

- The policy and the innovation auxiliary now act on the same bottleneck.
- This removes the mismatch in the old design where the innovation latent was only a side objective.
- If the innovation objective helps, it now helps the actual control representation directly.

Additional implementation detail:

- Encoder-final Muon is routed to the `innovation_projector` kernel for `innovation_direct_cnn`.
- That means Muon now acts on the actual policy bottleneck for this encoder, rather than on the upstream `512`-d dense layer.

Experiment setup:

- Added a dedicated ant 1M seed-factorization launcher using:
  - `innovation_direct_cnn`
  - `CRATE` heads
  - innovation on (`coef=1e-3`, `proj_dim=64`, warmup/ramp `500/500`)
  - no LR anneal
  - encoder-final Muon on the projector / policy bottleneck
  - same `fixed_init` / `fixed_data` 8-seed layout as the recent ant diagnostics

Current status:

- The direct innovation ant sweep was launched from this new baseline for comparison against the earlier `encoder_final_muon`, `actor_cond`, and split-encoder experiments.


Innovation-direct results:

- The direct innovation sweep was the strongest ant result so far in terms of mean / median performance.
- Aggregate comparison against recent ant baselines:
  - `encoder_final_muon`: mean about `234`, median about `212`
  - `actor_cond`: mean about `518`, median about `467`
  - `innovation_direct_cnn`: mean about `669`, median about `864`
- Final returns for `innovation_direct_cnn`:
  - fixed init (`init_seed=0`, vary data): `937.8`, `892.9`, `914.0`, `835.0`
  - fixed data (`data_seed=0`, vary init): `457.3`, `-80.5`, `911.4`, `483.9`

Most important qualitative observation:

- This is the first ant setup where a good initialization appears robust across data randomness.
- With fixed good init and varying data randomness, all `4/4` runs became strong.
- So compared with the earlier seed-factorized sweeps, `innovation_direct_cnn` appears to widen the good basin substantially along the data / rollout axis.

But the init dependence is not gone:

- With fixed data randomness and varying init, only `1/4` inits became strong.
- Two inits fell into the moderate-return local-minimum gait branch (`~457` and `~484`).
- One init still failed badly (`~-81`).
- So the remaining instability now looks much more init-dominated than before.

Representation / policy signature of the two branches:

- Strong runs have a healthy direct policy bottleneck:
  - `repr/unit_std_avg` on the projected policy latent around `0.19-0.24`
  - `repr/dead_units_frac = 0`
  - `policy/obs_to_noise_ratio` around `0.57-0.86`
  - `policy/action_noise_std_avg` around `0.11-0.16`
- Failed or moderate runs collapse the projected bottleneck:
  - `repr/unit_std_avg` around `1e-5`
  - `repr/dead_units_frac = 1`
  - near-zero `policy/obs_to_noise_ratio`
  - moderate action noise (`~0.31-0.43`) for the local-minimum branch, or somewhat larger for the worst failed run
  - effectively zero `cnn_dense/participation_ratio` on the projected bottleneck

Updated ant interpretation:

- Making the innovation latent the actual policy latent was a meaningful architectural improvement.
- It appears to convert the ant problem from "both init and data randomness strongly matter" into something closer to "good init is robust, but bad init still collapses".
- That is a real step forward, even though it is not yet a full solution to the init sensitivity.

## 2026-04-01 UTC - Ant direct innovation projector variant and two-regime picture

Current best direct-innovation ant setup to keep in mind:
- `innovation_direct_cnn`
- `innovation_proj_dim=64`
- `innovation_coef=1e-3`
- `innovation_warmup_updates=500`, `innovation_ramp_updates=500`
- `innovation_num_slices=16`, `innovation_num_t=8`, `innovation_t_max=5.0`
- `encoder_final_muon` on `innovation_projector` (`encoder_muon_lr=1e-3`, `encoder_muon_max_grad_norm=1.0`)
- `CRATE` heads on actor/critic
- no LR anneal
- projector variant from the latest sweep: remove post-projector `LayerNorm`, use `orthogonal(1.0)` on `innovation_projector`
- other important run settings stayed as in the recent ant diagnostics: `backend=spring`, `frame_stack=4`, `action_repeat=4`, `n_envs=128`, `num_steps=10`, `num_minibatches=32`, `update_epochs=4`, `encoder_lr=3e-4`, `heads_adam_lr=3e-4`, `heads_muon_lr=1e-3`, `ent_coef=0.0`

The recent 2M ant sweep with that projector variant clarified the branch structure. There are two clear regimes:

1. Good-representation regime
- `cnn_dense/participation_ratio` becomes large
- projected latent `u` rescues early (`repr/unit_std_avg` rises, `dead_units_frac -> 0`)
- `policy/obs_to_noise_ratio` rises later
- these runs become the strong ant policies by 2M (`~900-1085`)

2. Bad-representation but serviceable-policy regime
- projected latent stays effectively dead
- `cnn_dense/participation_ratio` stays near zero
- `repr/unit_std_avg` stays near zero and `dead_units_frac = 1`
- `policy/obs_to_noise_ratio` stays near zero
- PPO still finds a serviceable local-minimum gait, ending around `~500` return

So the current direct-innovation ant problem still looks representation-gated, not primarily policy-gated. The key open question is how to reliably push runs into the high-participation-ratio / rescued-latent regime early.

## 2026-04-01 UTC - Ant direct innovation instrumented hidden-vs-projector result

To answer whether the bad branch is upstream encoder collapse or projector-only collapse, I added separate pre-projector metrics on `hidden` alongside the existing post-projector metrics on `u`.

Setup:
- same direct-innovation ant baseline as above
- `innovation_direct_cnn`, `proj_dim=64`, warmup/ramp `500/500`
- no projector `LayerNorm`, `orthogonal(1.0)` projector init
- `encoder_final_muon`, `CRATE` heads, no LR anneal
- new logs:
  - `encoder_hidden/*`
  - `encoder_hidden_repr/*`
  - existing `repr/*`, `cnn_dense/*`, and `innovation_proj/*` remain on projected policy latent `u`

Main finding:
- catastrophic collapse is primarily upstream of the projector.
- In the instrumented 8-seed sweep:
  - `3/8` runs had dead `hidden` and dead `u`
  - `5/8` runs rescued `hidden` and `u`
  - `0/8` runs showed healthy `hidden` with dead `u`
- Rescue times for `hidden` and `u` matched almost exactly in the successful runs, e.g. `30,720`, `44,800`, `61,440`, `120,320`, and `~292k`.

Interpretation:
- The projector is not the main source of the catastrophic branch.
- When the direct-innovation ant run dies, the upstream encoder bottleneck `hidden` is already collapsed.
- The projected policy latent `u` mostly mirrors that upstream branch.

Important nuance:
- Healthy representation is still not sufficient for a top run.
- In the same sweep, one run had healthy `hidden` and healthy `u` but only reached a middling return (`~410`), so there is still a second policy-quality stage after representation rescue.

Updated conclusion:
- If the goal is to stop the catastrophic branch, the intervention should target the upstream encoder bottleneck, not just the projector.
- The current ant picture is now:
  1. upstream encoder `hidden` either rescues or collapses
  2. if it rescues, policy quality can still vary from middling to strong

## 2026-04-02 UTC - Ant direct innovation with upstream + projector Muon

I tested a variant of the direct-innovation ant setup that applies encoder MUON to both:
- upstream encoder bottleneck `Dense_0`
- projected policy bottleneck `innovation_projector`

Setup:
- `innovation_direct_cnn`
- `innovation_proj_dim=64`
- `innovation_warmup_updates=500`, `innovation_ramp_updates=500`
- no projector `LayerNorm`, projector init `orthogonal(1.0)`
- `CRATE` heads
- no LR anneal
- same instrumented logs on:
  - `encoder_hidden/*`, `encoder_hidden_repr/*`
  - `repr/*`, `cnn_dense/*`, `innovation_proj/*`

Main result:
- This was not a performance win.
- 2M final returns:
  - `-138`, `922`, `552`, `-166`, `-135`, `775`, `-136`, `-133`
- mean / median about `193 / -134`, much worse than the earlier 2M direct baseline (`~796 / 916`).

What improved:
- Hard upstream encoder collapse was largely removed.
- All 8 runs rescued both `hidden` and `u` early, roughly in the `5k-45k` step range for `hidden`, and `32k-45k` for `u` in most runs.
- So applying MUON to `Dense_0` does directly attack the stage-1 collapse issue.

What failed:
- The bad branch changed rather than disappeared.
- Failed runs no longer had dead representations; instead they had weak, low-dimensional latents plus very poor policy commitment.
- Bad-run signature at 2M:
  - `encoder_hidden_repr/unit_std_avg ~ 0.042-0.058`
  - `encoder_hidden/participation_ratio ~ 3.7-6.8`
  - `repr/unit_std_avg ~ 0.059-0.102`
  - `cnn_dense/participation_ratio ~ 2.1-3.6`
  - `policy/obs_to_noise_ratio ~ 0.003-0.010`
  - `policy/action_noise_std_avg ~ 1.3-5.8`
- So upstream MUON prevented catastrophic collapse, but often left the actor with a weak latent and a very noisy / undercommitted policy.

Interpretation:
- Stage 1 and stage 2 are now clearly separable.
- Upstream MUON helps stage 1 (prevent hard encoder death), but in this form it harms stage 2 enough that overall performance gets worse.
- So this variant should not replace the current direct baseline.

Updated ant read:
- The plain direct baseline still looks better overall.
- If the next intervention targets upstream encoder collapse, it should probably be gentler than full-strength MUON on `Dense_0`.



## 2026-04-02 UTC - Ant direct innovation partial 2M rerun with trajectory metrics

I reran the current best direct-innovation ant baseline with the new `traj/*` rollout logging enabled. The jobs were cancelled early, but they still reached about `0.79M-0.82M` env steps, which was enough to diagnose the branch.

Setup:
- same current best direct baseline:
  - `innovation_direct_cnn`
  - `innovation_proj_dim=64`
  - `innovation_coef=1e-3`
  - `innovation_warmup_updates=500`, `innovation_ramp_updates=500`
  - projector variant: no post-projector `LayerNorm`, `orthogonal(1.0)` init
  - `encoder_final_muon` on `innovation_projector`
  - `CRATE` heads
  - no LR anneal
- new trajectory logs:
  - `traj/action_*`
  - `traj/reward_*`, `traj/raw_reward_*`
  - `traj/done_frac`, `traj/env_done_frac`
  - `traj/frame_delta_*`, `traj/frame_var_*`

Main result:
- The same two-regime split is already obvious by about `800k` steps.
- `5/8` runs rescued upstream `hidden` and projected `u`.
- `3/8` runs stayed fully collapsed.
- Again there were `0/8` runs with healthy `hidden` but dead `u`.

Returns at cancellation:
- rescued branch: about `866.6`, `854.4`, `743.5`, `732.3`, and one late-rescue run at `246.9`
- collapsed branch: about `-145.6`, `-121.3`, `-114.7`

Trajectory metrics now separate the branches as well.

Rescued runs had:
- `traj/action_dim_std_avg ~ 0.18-0.31`
- `traj/done_frac ~ 0.004-0.007`
- `traj/raw_reward_mean ~ 1.27-3.51`
- `traj/frame_delta_l2_mean ~ 56.4-60.0`

Collapsed runs had:
- `traj/action_dim_std_avg ~ 0.49-0.55`
- `traj/done_frac ~ 0.024-0.038`
- `traj/raw_reward_mean ~ -4.63 to -3.38`
- `traj/frame_delta_l2_mean ~ 63.1-64.3`

Interpretation:
- The bad branch is not just an isolated encoder failure.
- It is already paired with a different early trajectory distribution: noisier actions, more episode termination, worse rewards, and larger frame-to-frame changes.
- This supports the current view that the ant issue is an exploration-coupled representation problem rather than a purely static representation problem.

One useful nuance from this partial rerun:
- rescue timing still matters even within the rescued branch.
- The weakest rescued run (`~246.9`) only rescued `hidden` at about `186,880` steps, much later than the stronger rescued runs (`32k`, `39.7k`, `79.4k`, `181.8k`).
- So late rescue appears materially weaker than early rescue.

Updated takeaway:
- early upstream `hidden` rescue is still the main gate
- but the trajectory metrics suggest that the gate is coupled to the rollout distribution from the very beginning
- the next diagnostics should compare early rollout statistics of future-rescued vs future-collapsed runs, not just encoder geometry alone


## 2026-04-02 UTC - Ant direct innovation with upstream SiLU bottleneck

I replaced the upstream innovation-encoder bottleneck activation with `SiLU` while keeping the current best direct baseline otherwise unchanged.

Setup:
- `innovation_direct_cnn`
- upstream bottleneck changed from `LayerNorm -> tanh(scale * ·)` to `LayerNorm -> SiLU`
- `innovation_proj_dim=64`
- `innovation_coef=1e-3`
- `innovation_warmup_updates=500`, `innovation_ramp_updates=500`
- no post-projector `LayerNorm`
- projector init `orthogonal(1.0)`
- `encoder_final_muon` on `innovation_projector`
- `CRATE` heads
- no LR anneal
- 2M seed-factorized ant sweep

Main result:
- This is the strongest ant result so far.
- Final returns at 2M:
  - `1149`, `1105`, `1093`, `917`, `914`, `883`, `510`, `493`
- mean / median about `882.8 / 915.2`

Most important structural finding:
- the catastrophic dead-representation branch disappeared.
- All `8/8` runs rescued upstream `hidden`.
- Rescue happened early in every run, roughly `23k-65k` steps.
- So the stage-1 representation lottery appears to be largely solved by replacing the upstream `tanh` bottleneck with `SiLU`.

Seed-factorized breakdown:
- `fixed_init` with `init_seed=0`, varying data: `4/4` strong
  - `1105`, `1093`, `917`, `914`
- `fixed_data` with `data_seed=0`, varying init: `2/4` strong, `2/4` middling
  - `1149`, `883`, `510`, `493`

Interpretation:
- We no longer see the old pattern of some runs having dead `hidden` and dead `u`.
- What remains is a policy-quality split on top of healthy representations.
- The two middling runs still had healthy `hidden` and `u`, but much weaker policy commitment:
  - `policy/obs_to_noise_ratio ~ 0.51`
  - `policy/action_noise_std_avg ~ 0.30`
- The strong runs had much stronger commitment:
  - `obs_to_noise_ratio ~ 1.48-2.98`
  - `action_noise_std_avg ~ 0.058-0.065`

Updated takeaway:
- `LayerNorm -> SiLU` is a much better upstream bottleneck than `LayerNorm -> tanh` in this ant direct-innovation line.
- This appears to solve stage 1 (representation rescue) much more reliably.
- The remaining inconsistency is now mostly stage 2: policy commitment / policy quality on top of a healthy latent.
