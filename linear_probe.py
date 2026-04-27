"""
Linear Probe Analysis for PixelBrax CNN Encoder.

Tests whether the frozen encoder learns a low-dimensional linear subspace
that predicts true simulator state.

Usage (importable):
    from linear_probe import run_online_probe
    metrics = run_online_probe(network, network_params, envs, vars(args))
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import optax
from flax.training.train_state import TrainState


# --------------------------------------------------------
# FrameStack — copied from training script to avoid import entanglement
# --------------------------------------------------------

@flax.struct.dataclass
class FrameStack:
    """Frame stacking buffer for temporal information in pixel-based RL."""
    frames: jnp.ndarray  # Shape: (n_envs, num_frames, H, W, C)

    @classmethod
    def create(cls, n_envs: int, num_frames: int, obs_shape: tuple):
        h, w, c = obs_shape
        frames = jnp.zeros((n_envs, num_frames, h, w, c), dtype=jnp.uint8)
        return cls(frames=frames)

    def reset(self, obs: jnp.ndarray, done: jnp.ndarray = None):
        n_envs = obs.shape[0]
        num_frames = self.frames.shape[1]
        new_frames = jnp.broadcast_to(
            obs[:, None, :, :, :],
            (n_envs, num_frames, obs.shape[1], obs.shape[2], obs.shape[3])
        )
        if done is None:
            return self.replace(frames=new_frames)
        frames = jnp.where(
            done[:, None, None, None, None],
            new_frames,
            self.frames
        )
        return self.replace(frames=frames)

    def push(self, obs: jnp.ndarray):
        new_frames = jnp.concatenate([
            self.frames[:, 1:, :, :, :],
            obs[:, None, :, :, :]
        ], axis=1)
        return self.replace(frames=new_frames)

    def get_stacked(self) -> jnp.ndarray:
        n_envs, num_frames, h, w, c = self.frames.shape
        frames_transposed = jnp.transpose(self.frames, (0, 2, 3, 1, 4))
        return frames_transposed.reshape(n_envs, h, w, c * num_frames)


# --------------------------------------------------------
# JIT-compiled encoder (stop_gradient on output)
# --------------------------------------------------------

def _make_encode_fn(network):
    @jax.jit
    def _encode(params, obs):
        features = network.apply(params, obs)
        if isinstance(features, tuple):
            features = features[0]
        return jax.lax.stop_gradient(features)
    return _encode


# --------------------------------------------------------
# Data collection — single device_get at end
# --------------------------------------------------------

def collect_probe_data(
    network,
    network_params,
    envs,
    args_dict: dict,
    n_eval_steps: int,
    seed: int,
):
    """
    Run a random-action rollout and collect (features, low_dim_state) pairs.

    All JAX arrays are accumulated on-device; a single jax.device_get
    is issued at the end to avoid per-step host syncs.

    Returns:
        features_np: np.ndarray  (n_envs * n_eval_steps, feature_dim)
        states_np:   np.ndarray  (n_envs * n_eval_steps, state_dim)
    """
    n_envs = args_dict["n_envs"]
    frame_stack_n = args_dict["frame_stack"]
    max_action = args_dict.get("max_action", 1.0)

    encode_fn = _make_encode_fn(network)

    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)
    reset_rngs = jax.random.split(reset_key, n_envs)
    env_state = envs.reset(reset_rngs)

    raw_obs_shape = env_state.pixels.shape[1:]  # (H, W, C_env_stacked)

    frame_stack = FrameStack.create(n_envs, frame_stack_n, raw_obs_shape)
    frame_stack = frame_stack.reset(env_state.pixels)
    obs = frame_stack.get_stacked()

    # Accumulate JAX arrays on-device; no device_get in the loop
    all_features = []   # list of jnp arrays (n_envs, feat_dim)
    all_states = []     # list of jnp arrays (n_envs, state_dim)

    try:
        action_dim = envs.action_size
    except AttributeError:
        action_dim = envs.env.action_size

    for _ in range(n_eval_steps):
        features = encode_fn(network_params, obs)   # (n_envs, feat_dim) — on device
        state_vec = env_state.obs                   # (n_envs, state_dim) — on device, this is the downstairs state vector

        all_features.append(features)
        all_states.append(state_vec)

        key, action_key = jax.random.split(key)
        action = jax.random.uniform(
            action_key, shape=(n_envs, action_dim),
            minval=-max_action, maxval=max_action,
        )

        env_state = envs.step(env_state, action)
        raw_obs = env_state.pixels
        done = env_state.done.astype(jnp.bool_)
        frame_stack = frame_stack.push(raw_obs)
        frame_stack = frame_stack.reset(raw_obs, done)
        obs = frame_stack.get_stacked()

    # Single host transfer for all data
    features_np = jax.device_get(jnp.concatenate(all_features, axis=0))
    states_np   = jax.device_get(jnp.concatenate(all_states,   axis=0))

    return features_np, states_np


# --------------------------------------------------------
# Ridge regression (pure numpy/scipy)
# --------------------------------------------------------

def ridge_fit_predict(X_train, y_train, X_test, y_test, alpha: float = 1e-3):
    """
    Fit ridge regression A s.t. X @ A ≈ y.
    Uses the augmented least-squares formulation solved via SVD (lstsq),
    which is robust to singular/rank-deficient feature matrices (e.g. at init).
    Returns (A, train_R2, test_R2).
    """
    n_feat = X_train.shape[1]
    y_2d = y_train if y_train.ndim == 2 else y_train[:, None]

    # Augmented system: [X; sqrt(alpha)*I] @ A = [y; 0]  ↔ ridge regression
    X_aug = np.vstack([X_train, np.sqrt(alpha) * np.eye(n_feat)])
    y_aug = np.vstack([y_2d,    np.zeros((n_feat, y_2d.shape[1]))])

    A, _, _, _ = np.linalg.lstsq(X_aug, y_aug, rcond=None)
    if y_train.ndim == 1:
        A = A[:, 0]

    def r2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean(axis=0)) ** 2)
        return 1.0 - ss_res / (ss_tot + 1e-12)

    def mse(y_true, y_pred):
        return float(np.mean((y_true - y_pred) ** 2))

    y_pred_train = X_train @ A
    y_pred_test  = X_test  @ A
    return A, r2(y_train, y_pred_train), r2(y_test, y_pred_test), mse(y_train, y_pred_train), mse(y_test, y_pred_test)


# --------------------------------------------------------
# PCA dimension sweep
# --------------------------------------------------------

def pca_r2_sweep(X_train, y_train, X_test, y_test, max_k: int, alpha: float = 1e-3):
    """
    Project features to top-k PCA components, fit ridge, record R².

    Returns (ks, r2_train_list, r2_test_list, singular_values).
    """
    X_mean = X_train.mean(axis=0, keepdims=True)
    X_c       = X_train - X_mean
    X_c_test  = X_test  - X_mean

    U, sv, Vt = np.linalg.svd(X_c, full_matrices=False)

    max_k = min(max_k, Vt.shape[0], X_train.shape[0])

    ks, r2_trains, r2_tests = [], [], []
    for k in range(1, max_k + 1):
        V_k   = Vt[:k].T             # (feat_dim, k)
        Z_tr  = X_c      @ V_k
        Z_te  = X_c_test @ V_k
        _, r2_tr, r2_te, _, _ = ridge_fit_predict(Z_tr, y_train, Z_te, y_test, alpha)
        ks.append(k)
        r2_trains.append(r2_tr)
        r2_tests.append(r2_te)

    return ks, r2_trains, r2_tests, sv


# --------------------------------------------------------
# Decoder SVD analysis
# --------------------------------------------------------

def decoder_svd_analysis(A: np.ndarray, X_train: np.ndarray, state_dim: int):
    """
    Analyse the fitted decoder matrix A (feat_dim, state_dim).

    sv_ratio        : sum of top-state_dim SVs / total SV sum (how concentrated A is)
    avg_offdiag_corr: mean |off-diag| of Pearson correlation matrix of the
                      top-d projected data coordinates Z = X_train @ U_top.
                      Non-trivial because Z columns can correlate even though
                      U_top columns are orthogonal.
    corr_matrix     : (top_d, top_d) correlation matrix for plotting
    sv              : all singular values of A
    """
    if A.ndim == 1:
        A = A[:, None]

    U, sv, Vt = np.linalg.svd(A, full_matrices=False)

    top_d   = min(state_dim, len(sv))
    sv_ratio = float(sv[:top_d].sum() / (sv.sum() + 1e-12))

    # Project training data onto top-d decoder directions in feature space
    U_top = U[:, :top_d]                      # (feat_dim, top_d)
    X_c   = X_train - X_train.mean(axis=0)    # centre
    Z     = X_c @ U_top                       # (n_train, top_d) — data coordinates

    if top_d >= 2:
        corr = np.corrcoef(Z.T)               # (top_d, top_d), non-trivial
        mask = 1 - np.eye(top_d)
        avg_offdiag = float(np.sum(np.abs(corr) * mask) / (mask.sum() + 1e-12))
    else:
        corr        = np.ones((1, 1))
        avg_offdiag = 0.0

    return sv_ratio, avg_offdiag, sv, corr


# --------------------------------------------------------
# Plotting
# --------------------------------------------------------

def save_plots(
    ks, r2_trains, r2_tests,
    sv_pca,
    corr_matrix,
    env_name: str,
    output_dir: str,
    state_dim: int,
):
    os.makedirs(output_dir, exist_ok=True)

    # --- R² vs k ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, r2_trains, label="train R²", marker=".")
    ax.plot(ks, r2_tests,  label="test R²",  marker=".")
    ax.axvline(state_dim, color="gray", linestyle="--", label=f"state_dim={state_dim}")
    ax.set_xlabel("PCA dimensions k")
    ax.set_ylabel("R²")
    ax.set_title(f"{env_name}: R² vs PCA dims")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{env_name}_r2_vs_k.png"), dpi=120)
    plt.close(fig)

    # --- Scree plot — explained variance uses sv² ---
    fig, ax = plt.subplots(figsize=(7, 4))
    sv2      = sv_pca ** 2
    ev_frac  = sv2 / (sv2.sum() + 1e-12)
    n_show   = min(len(ev_frac), 64)
    ax.bar(np.arange(1, n_show + 1), ev_frac[:n_show])
    ax.set_xlabel("PCA component")
    ax.set_ylabel("Explained variance fraction (σ²)")
    ax.set_title(f"{env_name}: Scree plot (PCA of encoder features)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{env_name}_scree.png"), dpi=120)
    plt.close(fig)

    # --- Correlation heatmap of projected data coordinates ---
    if corr_matrix.shape[0] >= 2:
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(corr_matrix, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        plt.colorbar(im, ax=ax)
        ax.set_title(f"{env_name}: corr(Z) — decoder top-{corr_matrix.shape[0]} dirs")
        ax.set_xlabel("Decoder direction")
        ax.set_ylabel("Decoder direction")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{env_name}_corr_heatmap.png"), dpi=120)
        plt.close(fig)


# --------------------------------------------------------
# Additive Component Decomposition Probe
# --------------------------------------------------------

class CoordMLP(nn.Module):
    """Tiny MLP: 1 → hidden → hidden → feat_dim, one per state coordinate."""
    feat_dim: int
    hidden: int = 32

    @nn.compact
    def __call__(self, s_k):   # s_k: (N, 1)
        x = nn.Dense(self.hidden)(s_k)
        x = nn.relu(x)
        x = nn.Dense(self.hidden)(x)
        x = nn.relu(x)
        return nn.Dense(self.feat_dim)(x)  # (N, feat_dim)


class AdditiveModel(nn.Module):
    """ĥ = c + Σ_k f_k(s_k), one CoordMLP per state coordinate."""
    feat_dim: int
    state_dim: int
    hidden: int = 32

    @nn.compact
    def __call__(self, S):   # S: (N, state_dim)
        c   = self.param("bias", nn.initializers.zeros, (self.feat_dim,))
        out = jnp.broadcast_to(c, (S.shape[0], self.feat_dim))
        for k in range(self.state_dim):
            out = out + CoordMLP(
                feat_dim=self.feat_dim, hidden=self.hidden, name=f"coord_{k}"
            )(S[:, k : k + 1])
        return out


def _get_component_outputs(params, S_np, feat_dim, state_dim, hidden=32):
    """
    Return per-coordinate outputs u_k for all k after training.
    Shape: (state_dim, N, feat_dim), numpy.
    """
    components = []
    coord_model = CoordMLP(feat_dim=feat_dim, hidden=hidden)
    for k in range(state_dim):
        s_k  = jnp.array(S_np[:, k : k + 1])
        u_k  = coord_model.apply(
            {"params": params["params"][f"coord_{k}"]}, s_k
        )
        components.append(jax.device_get(u_k))
    return np.stack(components, axis=0)   # (state_dim, N, feat_dim)


def _gram_matrices(u):
    """
    u: (state_dim, N, feat_dim) — centered component outputs.

    Returns:
        G_cos : (state_dim, state_dim) cosine Gram
        G_raw : (state_dim, state_dim) raw inner-product Gram
    """
    # dots[k, l, t] = <u_k_t, u_l_t>
    dots = np.einsum("kti,lti->klt", u, u)   # (d, d, N)
    G_raw = dots.mean(axis=-1)

    norms = np.linalg.norm(u, axis=-1)        # (d, N)
    denom = (
        norms[:, None, :] * norms[None, :, :] + 1e-8
    )                                          # (d, d, N)
    G_cos = (dots / denom).mean(axis=-1)

    return G_cos, G_raw


def _additive_r2(H_true, H_hat):
    ss_res = np.sum((H_true - H_hat) ** 2)
    ss_tot = np.sum((H_true - H_true.mean(axis=0)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def save_additive_plots(G_cos, G_raw, norm_means, norm_stds,
                        H_test, H_hat_test, env_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    state_dim = G_cos.shape[0]

    # 1. Reconstruction parity: true vs predicted feature norm (held-out)
    true_norms = np.linalg.norm(H_test,     axis=-1)
    pred_norms = np.linalg.norm(H_hat_test, axis=-1)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(true_norms, pred_norms, s=3, alpha=0.3)
    lim = [min(true_norms.min(), pred_norms.min()),
           max(true_norms.max(), pred_norms.max())]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set_xlabel("True feature norm")
    ax.set_ylabel("Reconstructed feature norm")
    ax.set_title(f"{env_name}: additive reconstruction parity (test)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{env_name}_additive_parity.png"), dpi=120)
    plt.close(fig)

    # 2. Cosine Gram heatmap
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(G_cos, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    plt.colorbar(im, ax=ax)
    ax.set_title(f"{env_name}: cosine Gram G")
    ax.set_xlabel("State coord k")
    ax.set_ylabel("State coord k")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{env_name}_cosine_gram.png"), dpi=120)
    plt.close(fig)

    # 3. Raw Gram heatmap
    fig, ax = plt.subplots(figsize=(4, 4))
    vabs = np.abs(G_raw).max()
    im = ax.imshow(G_raw, vmin=-vabs, vmax=vabs, cmap="RdBu_r", aspect="auto")
    plt.colorbar(im, ax=ax)
    ax.set_title(f"{env_name}: raw Gram G̃")
    ax.set_xlabel("State coord k")
    ax.set_ylabel("State coord k")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{env_name}_raw_gram.png"), dpi=120)
    plt.close(fig)

    # 4. Component norm bar plot (mean ± std per coordinate)
    fig, ax = plt.subplots(figsize=(max(4, state_dim), 4))
    xs = np.arange(state_dim)
    ax.bar(xs, norm_means, yerr=norm_stds, capsize=4)
    ax.set_xlabel("State coord k")
    ax.set_ylabel("‖u_k‖  (mean ± std)")
    ax.set_title(f"{env_name}: component norms")
    ax.set_xticks(xs)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{env_name}_component_norms.png"), dpi=120)
    plt.close(fig)


def run_additive_probe(
    features_np: np.ndarray,   # (N, feat_dim)  — already collected
    states_np:   np.ndarray,   # (N, state_dim)
    train_frac:  float = 0.8,
    n_epochs:    int   = 200,
    lr:          float = 1e-3,
    hidden:      int   = 32,
    batch_size:  int   = 2048,
    seed:        int   = 42,
    output_dir:  str   = None,
    env_name:    str   = "env",
) -> dict:
    """
    Fit ĥ = c + Σ_k f_k(s_k) to the frozen features, then measure:
      - reconstruction R² / MSE on held-out data
      - cosine Gram off-diagonal (angular orthogonality of components)
      - raw Gram off-diagonal (dominance check)
      - per-component norm mean / std (stability)
    """
    print(f"\n[additive] Starting additive probe for {env_name}...")

    N, feat_dim  = features_np.shape
    state_dim    = states_np.shape[1]
    n_train      = int(N * train_frac)

    H_train, H_test = features_np[:n_train], features_np[n_train:]
    S_train, S_test = states_np[:n_train],   states_np[n_train:]

    # Standardise state inputs (per-coord) using training stats
    S_mean = S_train.mean(axis=0)
    S_std  = S_train.std(axis=0) + 1e-8
    S_train_n = (S_train - S_mean) / S_std
    S_test_n  = (S_test  - S_mean) / S_std

    # Build model + optimizer — manage params/opt_state manually to avoid
    # FrozenDict/dict type mismatch inside JIT with some Flax versions.
    model  = AdditiveModel(feat_dim=feat_dim, state_dim=state_dim, hidden=hidden)
    key    = jax.random.PRNGKey(seed)
    dummy  = jnp.zeros((1, state_dim))
    # Use plain dicts throughout so jax.grad and optax see consistent pytree types.
    params = flax.core.unfreeze(model.init(key, dummy))

    tx        = optax.adam(lr)
    opt_state = tx.init(params)

    @jax.jit
    def train_step(params, opt_state, H_batch, S_batch):
        def loss_fn(p):
            return jnp.mean((model.apply(p, S_batch) - H_batch) ** 2)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    # Mini-batch training
    rng = np.random.default_rng(seed)
    for epoch in range(n_epochs):
        idx = rng.permutation(n_train)
        for start in range(0, n_train, batch_size):
            mb = idx[start : start + batch_size]
            H_b = jnp.array(H_train[mb])
            S_b = jnp.array(S_train_n[mb])
            params, opt_state, _ = train_step(params, opt_state, H_b, S_b)

        # Centering for identifiability: subtract empirical mean of each f_k,
        # absorb into global bias c.
        flat_params = flax.traverse_util.flatten_dict(params)
        S_all_n     = jnp.array(S_train_n)
        coord_model = CoordMLP(feat_dim=feat_dim, hidden=hidden)
        total_mean  = jnp.zeros(feat_dim)
        for k in range(state_dim):
            u_k = coord_model.apply(
                {"params": params["params"][f"coord_{k}"]},
                S_all_n[:, k : k + 1],
            )
            total_mean = total_mean + u_k.mean(axis=0)
        flat_params[("params", "bias")] = params["params"]["bias"] + total_mean
        for k in range(state_dim):
            u_k = coord_model.apply(
                {"params": params["params"][f"coord_{k}"]},
                S_all_n[:, k : k + 1],
            )
            bias_key = ("params", f"coord_{k}", "Dense_2", "bias")
            flat_params[bias_key] = flat_params[bias_key] - u_k.mean(axis=0)
        params = flax.traverse_util.unflatten_dict(flat_params)

    # Evaluate reconstruction
    H_hat_train = jax.device_get(model.apply(params, jnp.array(S_train_n)))
    H_hat_test  = jax.device_get(model.apply(params, jnp.array(S_test_n)))

    recon_r2_train  = _additive_r2(H_train, H_hat_train)
    recon_r2_test   = _additive_r2(H_test,  H_hat_test)
    recon_mse_train = float(np.mean((H_train - H_hat_train) ** 2))
    recon_mse_test  = float(np.mean((H_test  - H_hat_test)  ** 2))
    print(f"[additive] recon R²={recon_r2_test:.4f}  MSE={recon_mse_test:.4f} (test)")

    # Component outputs on test set (centered — mean already absorbed into bias above)
    u = _get_component_outputs(
        params, S_test_n, feat_dim, state_dim, hidden
    )   # (state_dim, N_test, feat_dim)

    # Center once more just in case (numerical residual after centering loop)
    u = u - u.mean(axis=1, keepdims=True)

    # Gram matrices
    G_cos, G_raw = _gram_matrices(u)
    mask = 1 - np.eye(state_dim)
    cos_offdiag = float(np.sum(np.abs(G_cos) * mask) / (mask.sum() + 1e-12))
    raw_offdiag = float(np.sum(np.abs(G_raw) * mask) / (mask.sum() + 1e-12))
    print(f"[additive] cosine Gram off-diag={cos_offdiag:.4f}, raw Gram off-diag={raw_offdiag:.4f}")

    # Component norm stats
    norms_per_k = np.linalg.norm(u, axis=-1)   # (state_dim, N_test)
    norm_means  = norms_per_k.mean(axis=-1)     # (state_dim,)
    norm_stds   = norms_per_k.std(axis=-1)
    norm_cv_mean = float(
        np.mean(norm_stds / (norm_means + 1e-8))
    )   # coefficient of variation — lower = more stable norms
    print(f"[additive] norm_cv_mean={norm_cv_mean:.4f}")

    # Plots
    if output_dir is not None:
        try:
            save_additive_plots(
                G_cos=G_cos, G_raw=G_raw,
                norm_means=norm_means, norm_stds=norm_stds,
                H_test=H_test, H_hat_test=H_hat_test,
                env_name=env_name, output_dir=output_dir,
            )
            print(f"[additive] Plots saved to {output_dir}")
        except Exception as e:
            print(f"[additive] Warning: plot saving failed: {e}")

    return {
        "additive/recon_r2_train":       float(recon_r2_train),
        "additive/recon_r2_test":        float(recon_r2_test),
        "additive/recon_mse_train":      float(recon_mse_train),
        "additive/recon_mse_test":       float(recon_mse_test),
        "additive/cosine_gram_offdiag":  cos_offdiag,
        "additive/raw_gram_offdiag":     raw_offdiag,
        "additive/norm_cv_mean":         norm_cv_mean,
    }


# --------------------------------------------------------
# Public API
# --------------------------------------------------------

def run_online_probe(
    network,
    network_params,
    envs,
    args_dict: dict,
    n_eval_envs: int = None,   # documented; must match envs.n_envs when reusing training envs
    n_eval_steps: int = 400,
    ridge_alpha: float = 1e-3,
    max_pca_dims: int = 64,
    seed: int = 42,
    train_frac: float = 0.8,
    output_dir: str = None,
) -> dict:
    """
    Run full probe analysis on a frozen encoder.

    network_params are used read-only; jax.lax.stop_gradient is applied
    inside collect_probe_data so no gradients can leak.

    n_eval_envs: informational — must equal the env count envs was created
                 with when reusing the training env instance.
    """
    env_name = args_dict.get("env_name", "env")
    print(f"\n[probe] Starting linear probe for {env_name}...")

    # 1. Collect data (single device_get at end)
    features_np, states_np = collect_probe_data(
        network=network,
        network_params=network_params,
        envs=envs,
        args_dict=args_dict,
        n_eval_steps=n_eval_steps,
        seed=seed,
    )
    n_total, feat_dim = features_np.shape
    state_dim = states_np.shape[1]
    print(f"[probe] Collected {n_total} samples: features={feat_dim}, state_dim={state_dim}")

    # 2. Temporal 80/20 split
    n_train  = int(n_total * train_frac)
    X_train, X_test = features_np[:n_train], features_np[n_train:]
    y_train, y_test = states_np[:n_train],   states_np[n_train:]

    # 3. Ridge regression on full features
    A_full, r2_full_train, r2_full_test, mse_full_train, mse_full_test = ridge_fit_predict(
        X_train, y_train, X_test, y_test, alpha=ridge_alpha
    )
    print(f"[probe] Full ridge: train R²={r2_full_train:.4f}, test R²={r2_full_test:.4f} | train MSE={mse_full_train:.4f}, test MSE={mse_full_test:.4f}")

    # 4. PCA dimension sweep
    ks, r2_trains, r2_tests, sv_pca = pca_r2_sweep(
        X_train, y_train, X_test, y_test,
        max_k=max_pca_dims, alpha=ridge_alpha,
    )

    threshold = 0.95 * max(r2_full_test, 1e-9)
    k_at_95   = max_pca_dims
    for k, r2_te in zip(ks, r2_tests):
        if r2_te >= threshold:
            k_at_95 = k
            break

    r2_at_state_dim = r2_tests[-1]   # fallback if state_dim > max_pca_dims
    for k, r2_te in zip(ks, r2_tests):
        if k == state_dim:
            r2_at_state_dim = r2_te
            break

    print(f"[probe] k_at_95pct_r2={k_at_95}, r2_at_state_dim={r2_at_state_dim:.4f}")

    # 5. Decoder SVD analysis (passes X_train so correlation is over data coords)
    sv_ratio, avg_offdiag_corr, sv_decoder, corr_matrix = decoder_svd_analysis(
        A_full, X_train, state_dim
    )
    print(f"[probe] decoder_sv_ratio={sv_ratio:.4f}, avg_offdiag_corr={avg_offdiag_corr:.4f}")

    # 6. Additive component decomposition probe (reuses same collected data)
    additive_metrics = run_additive_probe(
        features_np=features_np,
        states_np=states_np,
        train_frac=train_frac,
        seed=seed,
        output_dir=output_dir,
        env_name=env_name,
    )

    # 7. Save linear probe plots
    if output_dir is not None:
        try:
            save_plots(
                ks=ks,
                r2_trains=r2_trains,
                r2_tests=r2_tests,
                sv_pca=sv_pca,
                corr_matrix=corr_matrix,
                env_name=env_name,
                output_dir=output_dir,
                state_dim=state_dim,
            )
            print(f"[probe] Plots saved to {output_dir}")
        except Exception as e:
            print(f"[probe] Warning: plot saving failed: {e}")

    # 8. Return all metrics
    return {
        "probe/r2_full_test":    float(r2_full_test),
        "probe/r2_full_train":   float(r2_full_train),
        "probe/mse_full_test":   float(mse_full_test),
        "probe/mse_full_train":  float(mse_full_train),
        "probe/k_at_95pct_r2":  int(k_at_95),
        "probe/r2_at_state_dim": float(r2_at_state_dim),
        "probe/decoder_sv_ratio": float(sv_ratio),
        "probe/avg_offdiag_corr": float(avg_offdiag_corr),
        **additive_metrics,
    }
