#!/usr/bin/env python
"""
PPO for PixelBrax environments.

Optimizer split:
- Adam/AdamW for encoder (network)
- selectable Adam, Optax Muon, or manifold Stiefel for actor/critic head matrices
- Adam/AdamW for actor/critic vectors/scalars
"""
import json
import os
import random
import time
from dataclasses import dataclass
from functools import partial
from typing import Literal, Optional

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
import distrax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

import sys
sys.path.insert(0, "/users/stiwari4/data/stiwari4/pixelenvs/pixelbrax/pixelbrax/brax")
import pixelbrax
from pixelbrax.continual_dynamics import (
    apply_dynamics_task,
    continual_log_metrics,
    load_continual_dynamics_config,
    make_dynamics_task,
    resolve_schedule_seed,
    validate_continual_dynamics_config,
)
from pixelbrax.env_utils import make_pixel_brax
from configs.continual.slippery_ant_wrapper import (
    DEFAULT_SLIPPERY_FRICTIONS_CSV,
    load_seeded_friction_schedule_table,
)

from optimizers import aurora, manifold_stiefel, manifold_stiefel_admm
from encoders import build_encoder
from sigreg import sigreg_loss, sigreg_loss_masked
from spectrum.tracking import (
    add_hessian_histogram,
    add_histogram,
    add_matrix_singular_histogram,
    flatten_batch_tree,
    hessian_eigenvalues,
)

# Fix weird OOM https://github.com/google/jax/discussions/6332#discussioncomment-1279991
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")
# Fix CUDNN non-determinism
os.environ["TF_XLA_FLAGS"] = "--xla_gpu_autotune_level=2 --xla_gpu_deterministic_reductions"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"


HeadsOptimizer = Literal["adam", "stiefel", "stiefel_admm", "muon", "aurora"]


class CRATEFeedForward(nn.Module):
    """CRATE-style FeedForward layer implementing an ISTA step.

    Computes: output = swish(x + step_size * (W.T @ x - W.T @ W @ x))

    This implements a gradient descent step for sparse coding with dictionary W.
    Reference: https://github.com/Ma-Lab-Berkeley/CRATE/blob/main/model/crate.py
    """
    dim: int
    step_size: float = 0.1

    @nn.compact
    def __call__(self, x):
        # Weight matrix W of shape (dim, dim), initialized with Kaiming uniform
        weight = self.param(
            "weight",
            nn.initializers.kaiming_uniform(),
            (self.dim, self.dim)
        )

        # Compute D^T * D * x (W @ x then W.T @ result)
        x1 = x @ weight.T  # (batch, dim) @ (dim, dim) -> (batch, dim)
        grad_1 = x1 @ weight  # (batch, dim) @ (dim, dim) -> (batch, dim)

        # Compute D^T * x
        grad_2 = x @ weight  # (batch, dim) @ (dim, dim) -> (batch, dim)

        # Compute gradient update: step_size * (D^T * x - D^T * D * x)
        # lambda is set to 0.0 so we omit the - step_size * lambda term
        grad_update = self.step_size * (grad_2 - grad_1)

        # Apply swish activation (instead of ReLU in original CRATE)
        output = nn.swish(x + grad_update)
        return output


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 0
    """legacy seed used when init_seed/data_seed are not set explicitly"""
    init_seed: int = -1
    """parameter-initialization seed (-1 means use seed)"""
    data_seed: int = -1
    """environment/rollout/minibatch seed (-1 means use seed)"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "benchmark"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""

    # Environment arguments
    env_name: str = "inverted_pendulum"
    """the name of the environment"""
    backend: str = "generalized"
    """the physics backend (spring, generalized, positional)"""
    n_envs: int = 512
    """the number of parallel game environments"""
    hw: int = 84
    """height/width of the observation images"""

    # Algorithm specific arguments
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""

    # Learning rates
    encoder_lr: float = 3e-4
    """learning rate for encoder (Adam/AdamW)"""
    heads_optimizer: HeadsOptimizer = "stiefel"
    """head matrix optimizer for actor/critic matrices"""
    heads_stiefel_lr: float = 0.02
    """learning rate for actor/critic head matrices (Stiefel)"""
    heads_muon_lr: float = 1e-3
    """learning rate for actor/critic head matrices (Optax Muon or Aurora)"""
    heads_adam_lr: float = 3e-4
    """learning rate for actor/critic head vectors/scalars (Adam/AdamW)"""
    weight_decay: float = 0.0
    """weight decay for AdamW (0.0 uses Adam instead)"""

    num_steps: int = 256
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_eps: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function"""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for gradient clipping (Adam)"""
    actor_stiefel_max_grad_norm: float = 1.0
    """the maximum norm for gradient clipping (actor Stiefel)"""
    critic_stiefel_max_grad_norm: float = 1.0
    """the maximum norm for gradient clipping (critic Stiefel)"""
    actor_muon_max_grad_norm: float = 1.0
    """the maximum norm for gradient clipping (actor Optax Muon)"""
    critic_muon_max_grad_norm: float = 1.0
    """the maximum norm for gradient clipping (critic Optax Muon)"""
    max_action: float = 1.0
    """maximum action value for clipping"""
    log_interval: int = 10
    """logging interval (in updates)"""

    # Manifold Stiefel optimizer arguments
    use_encoder_final_stiefel: bool = False
    """if True, use manifold Stiefel for the encoder's final Dense kernel only"""
    use_encoder_upstream_stiefel: bool = False
    """if True, use manifold Stiefel for the encoder's upstream Dense bottleneck only"""
    encoder_stiefel_include_upstream: bool = False
    """If true for innovation_direct_cnn, also apply encoder-final Stiefel to the upstream Dense_0 bottleneck."""
    encoder_upstream_stiefel_lr: float = -1.0
    """Learning rate for the upstream Dense_0 Stiefel branch; if negative, reuse encoder_stiefel_lr."""
    encoder_stiefel_lr: float = 0.02
    """learning rate for the encoder's final Dense kernel when using Stiefel"""
    encoder_stiefel_max_grad_norm: float = 1.0
    """maximum norm for gradient clipping on the encoder final Dense Stiefel branch"""
    stiefel_dual_lr: float = 0.01
    """dual learning rate for Stiefel"""
    stiefel_dual_steps: int = 5
    """number of dual optimization steps for Stiefel"""
    stiefel_admm_steps: int = 10
    """number of ADMM iterations for Stiefel ADMM"""
    stiefel_msign_steps: int = 5
    """number of matrix sign iterations for Stiefel"""

    # Optax Muon optimizer arguments
    muon_ns_steps: int = 5
    """number of Newton-Schulz iterations for Optax Muon"""
    muon_beta: float = 0.95
    """momentum decay for Optax Muon"""
    muon_eps: float = 1e-8
    """epsilon for Optax Muon"""
    muon_weight_decay: float = 0.0
    """weight decay for Optax Muon matrix params"""
    muon_nesterov: bool = True
    """if True, use Nesterov momentum in Optax Muon"""
    muon_adaptive: bool = False
    """if True, use adaptive scaling in Optax Muon"""

    # Data augmentation (DrQ-style)
    use_augmentation: bool = False
    """Toggle random shift data augmentation (DrQ-style)"""
    augment_pad: int = 4
    """Padding size for random shift augmentation"""

    # Frame stacking
    frame_stack: int = 4
    """Number of frames to stack for temporal information"""

    # Action repeat
    action_repeat: int = 1
    """Number of times to repeat each action (frame skip)"""

    # Continual dynamics
    continual_dynamics_config: Optional[str] = None
    """Path to a YAML continual dynamics config"""

    # SlipperyAnt friction schedule
    slippery_ant: bool = False
    """Use the CSV-backed SlipperyAnt friction schedule."""
    slippery_change_every: int = 100_000
    """Number of underlying per-env timesteps between SlipperyAnt friction changes."""
    slippery_schedule_seed: int = 0
    """CSV row used for SlipperyAnt friction phases."""
    spectrum_lanczos_order: int = 20
    """Lanczos order for task-boundary Hessian spectrum diagnostics"""
    spectrum_lanczos_draws: int = 1
    """Number of random Lanczos draws for task-boundary Hessian diagnostics"""
    spectrum_batch_size: int = 0
    """Batch size for spectrum diagnostics; 0 uses one PPO minibatch"""
    spectrum_hist_bins: int = 64
    """Number of histogram bins for spectrum and activation WandB payloads"""

    # Debug/analysis flags
    debug_repr: bool = False
    """Toggle debug logging for encoder representations"""

    # Checkpoint saving
    save_checkpoint: bool = False
    """If true, save model params to disk at end of training"""
    checkpoint_dir: str = "checkpoints"
    """Directory for saved checkpoints"""

    # Online linear probe
    probe_interval: int = 0
    """Run linear probe every N global steps (0 disables). E.g. 100000."""
    probe_n_eval_steps: int = 400
    """Number of random-action rollout steps for probe data collection."""
    probe_output_dir: str = "probe_results"
    """Directory for probe plots (subdirectory per probe call is created)."""
    probe_save_plots: bool = False
    """If true, generate probe/additive plots to disk and log them to wandb when tracking."""

    # Encoder architecture
    encoder_type: str = "cnn"
    """encoder architecture to use: 'resnet', 'cnn', 'cnn_swish', 'cnn_swish_ta', 'cnn_swish_tb', 'cnn_swish_tc', 'crate_cnn', 'crate_cnn_tanh', 'crate_cnn_tanh_resid', 'cnn_swish_tanh', 'cnn_swish_tanh_resid', 'cnn_swish_tanh_resid_learned', 'split_cnn', 'separate_cnn', 'sigreg_cnn', 'innovation_cnn', 'innovation_direct_cnn', or 'mlp'"""
    encoder_tanh_scale: float = 0.5
    """Multiplier for encoder output before tanh (controls saturation)"""
    encoder_residual_scale: float = 0.1
    """Residual scale for encoder variants with a residual policy-facing path."""
    encoder_hidden_activation: str = "tanh"
    """Hidden bottleneck activation for innovation encoders: 'tanh' or 'swish' ('silu' accepted as alias)."""
    encoder_warmup_updates: int = 0
    """Warmup updates for encoder LR when annealing is enabled (0 disables warmup)."""
    encoder_use_crate_block: bool = False
    """If true, insert a CRATE block inside the encoder before the SIGReg projector."""
    encoder_crate_step_size: float = 0.1
    """Step size for the encoder CRATE block when enabled."""

    # SIGReg regularization
    sigreg_mode: str = "off"
    """SIGReg mode to use: 'off', 'projected', or 'shared_hidden'."""
    sigreg_coef: float = 0.0
    """Maximum coefficient for the SIGReg auxiliary loss."""
    sigreg_proj_dim: int = 64
    """Projected subspace dimension used by the SIGReg CNN encoder."""
    sigreg_num_slices: int = 16
    """Number of random projections used by SIGReg."""
    sigreg_num_t: int = 8
    """Number of evaluation points used by SIGReg."""
    sigreg_t_max: float = 5.0
    """Maximum absolute t value for the SIGReg characteristic-function grid."""
    sigreg_warmup_updates: int = 0
    """Number of PPO updates before enabling SIGReg."""
    sigreg_ramp_updates: int = 0
    """Number of PPO updates used to ramp SIGReg to its full coefficient."""

    # Innovation-aware auxiliary regularization
    innovation_coef: float = 0.0
    """Maximum coefficient for the innovation-aware auxiliary loss."""
    innovation_proj_dim: int = 64
    """Projected subspace dimension used by the innovation-aware CNN encoder."""
    innovation_num_slices: int = 16
    """Number of random projections used by the innovation-aware residual SIGReg."""
    innovation_num_t: int = 8
    """Number of evaluation points used by the innovation-aware residual SIGReg."""
    innovation_t_max: float = 5.0
    """Maximum absolute t value for the innovation-aware characteristic-function grid."""
    innovation_warmup_updates: int = 0
    """Number of PPO updates before enabling the innovation-aware auxiliary loss."""
    innovation_ramp_updates: int = 0
    """Number of PPO updates used to ramp innovation-aware regularization to its full coefficient."""

    # VICReg variance regularization
    vicreg_var_coef: float = 0.0
    """Maximum coefficient for the VICReg variance-floor loss on the shared hidden representation."""
    vicreg_var_target: float = 0.1
    """Target minimum per-feature standard deviation for the shared hidden representation."""
    vicreg_var_warmup_updates: int = 0
    """Number of PPO updates before enabling the VICReg variance loss."""
    vicreg_var_ramp_updates: int = 0
    """Number of PPO updates used to ramp VICReg variance regularization to its full coefficient."""

    # Early bottleneck stabilization
    bottleneck_var_coef: float = 0.0
    """Coefficient for an early lower-tail variance floor on the shared hidden representation."""
    bottleneck_var_target: float = 0.05
    """Target minimum std for the lowest-variance hidden features during early training."""
    bottleneck_var_bottom_frac: float = 0.25
    """Fraction of lowest-variance hidden features to penalize."""
    bottleneck_pre_ln_coef: float = 0.0
    """Coefficient for an early one-sided cap on encoder dense pre-LayerNorm scale."""
    bottleneck_pre_ln_max_std: float = 12.0
    """Maximum allowed std of encoder dense pre-LayerNorm activations before penalty."""
    bottleneck_reg_stop_updates: int = 0
    """Number of PPO updates for which bottleneck stabilization remains active (0 disables)."""

    # Early actor conditionality regularization
    actor_cond_coef: float = 0.0
    """Coefficient for an early actor-side conditionality floor on action means."""
    actor_cond_target: float = 0.05
    """Target minimum advantage-weighted std of actor means across a minibatch."""
    actor_cond_stop_updates: int = 0
    """Number of PPO updates for which actor conditionality regularization remains active (0 disables)."""
    actor_noise_coef: float = 0.0
    """Coefficient for an early one-sided cap on average actor action-noise std."""
    actor_noise_max_std: float = 0.25
    """Maximum desired average action-noise std during the early training window."""
    actor_noise_stop_updates: int = 0
    """Number of PPO updates for which actor noise regularization remains active (0 disables)."""

    # Policy std parameterization
    state_dependent_std: bool = False
    """If true, predict log-std from state instead of using a global per-action parameter."""
    state_std_tanh_scale: float = 0.5
    """Maximum per-state log-std deviation from the global bias when using state-dependent std."""
    actor_logstd_min: float = -5.0
    """Minimum log-std clamp for the actor policy."""
    actor_logstd_max: float = 2.0
    """Maximum log-std clamp for the actor policy."""
    clip_global_logstd: bool = False
    """If true, clamp the global actor log-std parameter with actor_logstd_min/max."""
    bounded_global_logstd: bool = False
    """If true, parameterize global log-std inside actor_logstd_min/max with a sigmoid."""
    actor_logstd_init: float = 0.0
    """Initializer for the global actor log-std parameter."""
    actor_mean_tanh: bool = False
    """If true, bound the actor mean with tanh(actor_mean) * actor_mean_scale."""
    actor_mean_scale: float = 1.0
    """Scale for tanh-bounded actor means."""

    # CRATE head architecture
    use_crate_head: bool = False
    """If true, use CRATE-style FeedForward as final hidden layer in actor/critic."""
    crate_step_size: float = 0.1
    """Step size for CRATE FeedForward ISTA update."""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_updates: int = 0
    """the number of updates (computed in runtime)"""


class Critic(nn.Module):
    """Value network with 2 hidden layers using Swish activation."""
    @nn.compact
    def __call__(self, x, return_intermediates: bool = False):
        dense_0 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        hidden_0 = nn.swish(dense_0)
        dense_1 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(hidden_0)
        hidden_1 = nn.swish(dense_1)
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(hidden_1)
        if return_intermediates:
            return {
                "value": value,
                "Dense_0": hidden_0,
                "Dense_1": hidden_1,
                "Dense_2": value,
            }
        return value


class Actor(nn.Module):
    """Continuous action actor with 2 hidden layers using Swish activation."""
    action_dim: int
    state_dependent_std: bool = False
    state_std_tanh_scale: float = 0.5
    logstd_min: float = -5.0
    logstd_max: float = 2.0
    clip_global_logstd: bool = False
    bounded_global_logstd: bool = False
    actor_logstd_init: float = 0.0
    actor_mean_tanh: bool = False
    actor_mean_scale: float = 1.0

    @nn.compact
    def __call__(self, x, return_intermediates: bool = False):
        dense_0 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        hidden_0 = nn.swish(dense_0)
        dense_1 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(hidden_0)
        hidden_1 = nn.swish(dense_1)
        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0)
        )(hidden_1)
        if self.actor_mean_tanh:
            actor_mean = self.actor_mean_scale * jnp.tanh(actor_mean)
        if self.state_dependent_std:
            log_std_bias = self.param(
                "log_std_bias",
                nn.initializers.zeros,
                (self.action_dim,)
            )
            log_std_delta = nn.Dense(
                self.action_dim,
                kernel_init=constant(0.0),
                bias_init=constant(0.0),
                name="log_std_head",
            )(hidden_1)
            actor_logstd = log_std_bias + self.state_std_tanh_scale * jnp.tanh(log_std_delta)
            actor_logstd = jnp.clip(actor_logstd, self.logstd_min, self.logstd_max)
        else:
            if self.bounded_global_logstd:
                eps = 1e-6
                init = np.clip(
                    self.actor_logstd_init,
                    self.logstd_min + eps,
                    self.logstd_max - eps,
                )
                frac = (init - self.logstd_min) / (self.logstd_max - self.logstd_min)
                raw_init = np.log(frac) - np.log1p(-frac)
                actor_logstd_raw = self.param(
                    "log_std_raw",
                    constant(float(raw_init)),
                    (self.action_dim,),
                )
                actor_logstd = self.logstd_min + (
                    self.logstd_max - self.logstd_min
                ) * jax.nn.sigmoid(actor_logstd_raw)
            else:
                actor_logstd = self.param(
                    "log_std",
                    constant(self.actor_logstd_init),
                    (self.action_dim,)
                )
                if self.clip_global_logstd:
                    actor_logstd = jnp.clip(actor_logstd, self.logstd_min, self.logstd_max)
        if return_intermediates:
            return {
                "actor_mean": actor_mean,
                "actor_logstd": actor_logstd,
                "Dense_0": hidden_0,
                "Dense_1": hidden_1,
                "Dense_2": actor_mean,
            }
        return actor_mean, actor_logstd


class CRATECritic(nn.Module):
    """Value network with CRATE FeedForward as final hidden layer."""
    crate_step_size: float = 0.1

    @nn.compact
    def __call__(self, x, return_intermediates: bool = False):
        dense_0 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        hidden_0 = nn.swish(dense_0)
        crate_hidden = CRATEFeedForward(dim=256, step_size=self.crate_step_size)(hidden_0)
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(crate_hidden)
        if return_intermediates:
            return {
                "value": value,
                "Dense_0": hidden_0,
                "CRATEFeedForward_0": crate_hidden,
                "Dense_1": value,
            }
        return value


class CRATEActor(nn.Module):
    """Continuous action actor with CRATE FeedForward as final hidden layer."""
    action_dim: int
    crate_step_size: float = 0.1
    state_dependent_std: bool = False
    state_std_tanh_scale: float = 0.5
    logstd_min: float = -5.0
    logstd_max: float = 2.0
    clip_global_logstd: bool = False
    bounded_global_logstd: bool = False
    actor_logstd_init: float = 0.0
    actor_mean_tanh: bool = False
    actor_mean_scale: float = 1.0

    @nn.compact
    def __call__(self, x, return_intermediates: bool = False):
        dense_0 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        hidden_0 = nn.swish(dense_0)
        crate_hidden = CRATEFeedForward(dim=256, step_size=self.crate_step_size)(hidden_0)
        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0)
        )(crate_hidden)
        if self.actor_mean_tanh:
            actor_mean = self.actor_mean_scale * jnp.tanh(actor_mean)
        if self.state_dependent_std:
            log_std_bias = self.param(
                "log_std_bias",
                nn.initializers.zeros,
                (self.action_dim,)
            )
            log_std_delta = nn.Dense(
                self.action_dim,
                kernel_init=constant(0.0),
                bias_init=constant(0.0),
                name="log_std_head",
            )(crate_hidden)
            actor_logstd = log_std_bias + self.state_std_tanh_scale * jnp.tanh(log_std_delta)
            actor_logstd = jnp.clip(actor_logstd, self.logstd_min, self.logstd_max)
        else:
            if self.bounded_global_logstd:
                eps = 1e-6
                init = np.clip(
                    self.actor_logstd_init,
                    self.logstd_min + eps,
                    self.logstd_max - eps,
                )
                frac = (init - self.logstd_min) / (self.logstd_max - self.logstd_min)
                raw_init = np.log(frac) - np.log1p(-frac)
                actor_logstd_raw = self.param(
                    "log_std_raw",
                    constant(float(raw_init)),
                    (self.action_dim,),
                )
                actor_logstd = self.logstd_min + (
                    self.logstd_max - self.logstd_min
                ) * jax.nn.sigmoid(actor_logstd_raw)
            else:
                actor_logstd = self.param(
                    "log_std",
                    constant(self.actor_logstd_init),
                    (self.action_dim,)
                )
                if self.clip_global_logstd:
                    actor_logstd = jnp.clip(actor_logstd, self.logstd_min, self.logstd_max)
        if return_intermediates:
            return {
                "actor_mean": actor_mean,
                "actor_logstd": actor_logstd,
                "Dense_0": hidden_0,
                "CRATEFeedForward_0": crate_hidden,
                "Dense_1": actor_mean,
            }
        return actor_mean, actor_logstd


class InnovationDynamics(nn.Module):
    """Linear latent dynamics model for innovation-aware auxiliary loss."""
    action_dim: int
    proj_dim: int

    @nn.compact
    def __call__(self, u_t, action, return_intermediates: bool = False):
        state_term = nn.Dense(
            self.proj_dim,
            use_bias=False,
            kernel_init=orthogonal(1.0),
            name="state_transition",
        )(u_t)
        action_term = nn.Dense(
            self.proj_dim,
            use_bias=False,
            kernel_init=orthogonal(1.0),
            name="action_transition",
        )(action)
        predicted_next_u = state_term + action_term
        if return_intermediates:
            return {
                "state_term": state_term,
                "action_term": action_term,
                "predicted_next_u": predicted_next_u,
            }
        return predicted_next_u


@flax.struct.dataclass
class Storage:
    obs: jnp.array
    next_obs: jnp.array
    actions: jnp.array
    logprobs: jnp.array
    dones: jnp.array
    next_dones: jnp.array
    values: jnp.array
    advantages: jnp.array
    returns: jnp.array
    rewards: jnp.array
    raw_rewards: jnp.array


@flax.struct.dataclass
class EpisodeStatistics:
    episode_returns: jnp.array
    episode_lengths: jnp.array
    returned_episode_returns: jnp.array
    returned_episode_lengths: jnp.array


@flax.struct.dataclass
class FrameStack:
    """Frame stacking buffer for temporal information in pixel-based RL."""
    frames: jnp.ndarray  # Shape: (n_envs, num_frames, H, W, C)

    @classmethod
    def create(cls, n_envs: int, num_frames: int, obs_shape: tuple):
        """Initialize frame stack with zeros."""
        h, w, c = obs_shape
        frames = jnp.zeros((n_envs, num_frames, h, w, c), dtype=jnp.uint8)
        return cls(frames=frames)

    def reset(self, obs: jnp.ndarray, done: jnp.ndarray = None):
        """Reset frame stack for environments that are done."""
        n_envs = obs.shape[0]
        num_frames = self.frames.shape[1]
        new_frames = jnp.broadcast_to(
            obs[:, None, :, :, :],
            (n_envs, num_frames, obs.shape[1], obs.shape[2], obs.shape[3])
        )

        if done is None:
            return self.replace(frames=new_frames)
        else:
            frames = jnp.where(
                done[:, None, None, None, None],
                new_frames,
                self.frames
            )
            return self.replace(frames=frames)

    def push(self, obs: jnp.ndarray):
        """Add a new frame to the stack, shifting out the oldest."""
        new_frames = jnp.concatenate([
            self.frames[:, 1:, :, :, :],
            obs[:, None, :, :, :]
        ], axis=1)
        return self.replace(frames=new_frames)

    def get_stacked(self) -> jnp.ndarray:
        """Get stacked observation by concatenating frames along channel dimension."""
        n_envs, num_frames, h, w, c = self.frames.shape
        frames_transposed = jnp.transpose(self.frames, (0, 2, 3, 1, 4))
        return frames_transposed.reshape(n_envs, h, w, c * num_frames)


@flax.struct.dataclass
class RewardNormalizer:
    """Reward normalization using discounted returns (CleanRL style)."""
    return_rms_mean: jnp.array
    return_rms_var: jnp.array
    return_rms_count: jnp.array
    discounted_return: jnp.array
    gamma: float = 0.99

    @classmethod
    def create(cls, n_envs, gamma=0.99):
        return cls(
            return_rms_mean=jnp.array(0.0),
            return_rms_var=jnp.array(1.0),
            return_rms_count=jnp.array(1e-4),
            discounted_return=jnp.zeros(n_envs),
            gamma=gamma,
        )

    def update(self, rewards, dones):
        """Update running statistics with a batch of rewards."""
        new_discounted_return = rewards + self.gamma * self.discounted_return * (1.0 - dones)

        batch_mean = jnp.mean(new_discounted_return)
        batch_var = jnp.var(new_discounted_return)
        batch_count = new_discounted_return.size

        delta = batch_mean - self.return_rms_mean
        tot_count = self.return_rms_count + batch_count

        new_mean = self.return_rms_mean + delta * batch_count / tot_count
        m_a = self.return_rms_var * self.return_rms_count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * self.return_rms_count * batch_count / tot_count
        new_var = M2 / tot_count

        new_discounted_return = jnp.where(dones, 0.0, new_discounted_return)

        return self.replace(
            return_rms_mean=new_mean,
            return_rms_var=new_var,
            return_rms_count=tot_count,
            discounted_return=new_discounted_return,
        )

    def normalize(self, rewards, clip=10.0, epsilon=1e-8):
        """Normalize rewards by std of discounted returns and clip."""
        normalized = rewards / jnp.sqrt(self.return_rms_var + epsilon)
        return jnp.clip(normalized, -clip, clip)


# --------------------------------------------------------
#  Debug Metrics for Encoder Representations
# --------------------------------------------------------

def encoder_repr_metrics(hidden: jnp.ndarray) -> dict:
    """
    Compute metrics for encoder representation health.

    Args:
        hidden: Encoder output of shape (batch, hidden_dim)

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Basic statistics
    metrics["repr/mean"] = jnp.mean(hidden)
    metrics["repr/std"] = jnp.std(hidden)
    metrics["repr/min"] = jnp.min(hidden)
    metrics["repr/max"] = jnp.max(hidden)

    # Per-unit statistics (across batch)
    unit_means = jnp.mean(hidden, axis=0)  # (hidden_dim,)
    unit_stds = jnp.std(hidden, axis=0)    # (hidden_dim,)

    metrics["repr/unit_mean_avg"] = jnp.mean(unit_means)
    metrics["repr/unit_std_avg"] = jnp.mean(unit_stds)
    metrics["repr/unit_std_min"] = jnp.min(unit_stds)
    metrics["repr/unit_std_max"] = jnp.max(unit_stds)

    # Dead units: units with very low variance (not learning)
    dead_threshold = 0.01
    dead_units = jnp.mean(unit_stds < dead_threshold)
    metrics["repr/dead_units_frac"] = dead_units

    # Saturated units: units always near boundaries (for tanh encoder output)
    saturated_high = jnp.mean(jnp.abs(hidden) > 0.95)
    metrics["repr/saturated_frac"] = saturated_high

    # Active units: units with reasonable variance
    active_units = jnp.mean(unit_stds > dead_threshold)
    metrics["repr/active_units_frac"] = active_units

    # Norms
    sample_norms = jnp.linalg.norm(hidden, axis=-1)  # (batch,)
    metrics["repr/norm_mean"] = jnp.mean(sample_norms)
    metrics["repr/norm_std"] = jnp.std(sample_norms)
    metrics["repr/norm_min"] = jnp.min(sample_norms)
    metrics["repr/norm_max"] = jnp.max(sample_norms)

    # Sparsity: fraction of near-zero activations
    sparsity = jnp.mean(jnp.abs(hidden) < 0.01)
    metrics["repr/sparsity"] = sparsity

    # Feature correlation (expensive, use subset)
    # Compute correlation between random pairs of features
    if hidden.shape[1] >= 2:
        # Normalize features
        hidden_centered = hidden - jnp.mean(hidden, axis=0, keepdims=True)
        hidden_norm = hidden_centered / (jnp.std(hidden_centered, axis=0, keepdims=True) + 1e-8)
        # Compute correlation matrix diagonal bands
        corr_matrix = jnp.corrcoef(hidden_norm.T)
        # Average absolute off-diagonal correlation
        mask = 1 - jnp.eye(corr_matrix.shape[0])
        avg_corr = jnp.sum(jnp.abs(corr_matrix) * mask) / (jnp.sum(mask) + 1e-8)
        metrics["repr/feature_correlation"] = avg_corr

    return metrics


def cnn_dense_metrics(
    dense_pre_ln: jnp.ndarray,
    hidden: jnp.ndarray,
    dense_kernel: jnp.ndarray,
    dense_kernel_grad: jnp.ndarray,
    prefix: str = "cnn_dense",
) -> dict:
    """Compute scalar debug metrics for a CNN encoder bottleneck."""
    eps = 1e-8
    metrics = {}

    metrics[f"{prefix}/pre_ln_mean"] = jnp.mean(dense_pre_ln)
    metrics[f"{prefix}/pre_ln_std"] = jnp.std(dense_pre_ln)
    metrics[f"{prefix}/pre_ln_abs_mean"] = jnp.mean(jnp.abs(dense_pre_ln))
    metrics[f"{prefix}/pre_ln_max_abs"] = jnp.max(jnp.abs(dense_pre_ln))
    metrics[f"{prefix}/pre_ln_l2_mean"] = jnp.mean(jnp.linalg.norm(dense_pre_ln, axis=-1))

    metrics[f"{prefix}/tanh_sat_frac_0.95"] = jnp.mean(jnp.abs(hidden) > 0.95)
    metrics[f"{prefix}/tanh_sat_frac_0.99"] = jnp.mean(jnp.abs(hidden) > 0.99)

    hidden_centered = hidden - jnp.mean(hidden, axis=0, keepdims=True)
    feature_var = jnp.var(hidden_centered, axis=0)
    metrics[f"{prefix}/feature_var_min"] = jnp.min(feature_var)
    metrics[f"{prefix}/feature_var_max"] = jnp.max(feature_var)
    metrics[f"{prefix}/feature_var_ratio"] = jnp.max(feature_var) / (jnp.min(feature_var) + eps)

    batch_denom = float(max(hidden.shape[0] - 1, 1))
    cov = (hidden_centered.T @ hidden_centered) / batch_denom
    feature_std = jnp.sqrt(feature_var + eps)
    corr = cov / (feature_std[:, None] * feature_std[None, :] + eps)
    corr_mask = 1.0 - jnp.eye(corr.shape[0], dtype=corr.dtype)
    metrics[f"{prefix}/mean_abs_corr"] = (
        jnp.sum(jnp.abs(corr) * corr_mask) / (jnp.sum(corr_mask) + eps)
    )

    eigvals = jnp.clip(jnp.linalg.eigvalsh(cov), a_min=0.0)
    eig_sum = jnp.sum(eigvals)
    eig_sq_sum = jnp.sum(jnp.square(eigvals))
    top_eig = jnp.max(eigvals)
    metrics[f"{prefix}/participation_ratio"] = (eig_sum ** 2) / (eig_sq_sum + eps)
    metrics[f"{prefix}/top_eig_fraction"] = top_eig / (eig_sum + eps)
    metrics[f"{prefix}/stable_rank"] = eig_sum / (top_eig + eps)

    singular_values = jnp.linalg.svd(dense_kernel, compute_uv=False)
    sigma_max = jnp.max(singular_values)
    metrics[f"{prefix}/weight_mean"] = jnp.mean(dense_kernel)
    metrics[f"{prefix}/weight_std"] = jnp.std(dense_kernel)
    metrics[f"{prefix}/weight_spectral_norm"] = sigma_max
    metrics[f"{prefix}/weight_stable_rank"] = (
        jnp.sum(jnp.square(singular_values)) / (jnp.square(sigma_max) + eps)
    )
    metrics[f"{prefix}/weight_grad_norm"] = jnp.linalg.norm(dense_kernel_grad)

    return metrics


def projected_latent_metrics(projected: jnp.ndarray, prefix: str = "sigreg_proj") -> dict:
    """Compute simple anisotropy metrics on an auxiliary projected latent."""
    eps = 1e-8
    metrics = {}
    centered = projected - jnp.mean(projected, axis=0, keepdims=True)
    feature_var = jnp.var(centered, axis=0)
    cov = (centered.T @ centered) / float(max(projected.shape[0] - 1, 1))
    feature_std = jnp.sqrt(feature_var + eps)
    corr = cov / (feature_std[:, None] * feature_std[None, :] + eps)
    corr_mask = 1.0 - jnp.eye(corr.shape[0], dtype=corr.dtype)
    eigvals = jnp.clip(jnp.linalg.eigvalsh(cov), a_min=0.0)
    eig_sum = jnp.sum(eigvals)
    eig_sq_sum = jnp.sum(jnp.square(eigvals))
    top_eig = jnp.max(eigvals)

    metrics[f"{prefix}/mean_abs_corr"] = (
        jnp.sum(jnp.abs(corr) * corr_mask) / (jnp.sum(corr_mask) + eps)
    )
    metrics[f"{prefix}/top_eig_fraction"] = top_eig / (eig_sum + eps)
    metrics[f"{prefix}/participation_ratio"] = (eig_sum ** 2) / (eig_sq_sum + eps)
    metrics[f"{prefix}/stable_rank"] = eig_sum / (top_eig + eps)
    return metrics


def rollout_trajectory_metrics(
    storage: Storage,
    frame_stack: int,
    obs_channels: int,
    max_action: float,
    prefix: str = "traj",
) -> dict:
    """Compute rollout-level trajectory diversity metrics from the collected batch."""
    eps = 1e-8
    metrics = {}

    actions = storage.actions.reshape((-1, storage.actions.shape[-1])).astype(jnp.float32)
    rewards = storage.rewards.reshape((-1,)).astype(jnp.float32)
    raw_rewards = storage.raw_rewards.reshape((-1,)).astype(jnp.float32)
    dones = storage.next_dones.reshape((-1,)).astype(jnp.float32)

    action_norms = jnp.linalg.norm(actions, axis=-1)
    action_dim_std = jnp.std(actions, axis=0)
    metrics[f"{prefix}/action_abs_mean"] = jnp.mean(jnp.abs(actions))
    metrics[f"{prefix}/action_l2_mean"] = jnp.mean(action_norms)
    metrics[f"{prefix}/action_l2_std"] = jnp.std(action_norms)
    metrics[f"{prefix}/action_dim_std_avg"] = jnp.mean(action_dim_std)
    metrics[f"{prefix}/action_clip_frac"] = jnp.mean(jnp.abs(actions) >= (max_action - 1e-6))

    metrics[f"{prefix}/reward_mean"] = jnp.mean(rewards)
    metrics[f"{prefix}/reward_std"] = jnp.std(rewards)
    metrics[f"{prefix}/raw_reward_mean"] = jnp.mean(raw_rewards)
    metrics[f"{prefix}/raw_reward_std"] = jnp.std(raw_rewards)
    metrics[f"{prefix}/done_frac"] = jnp.mean(dones)
    metrics[f"{prefix}/env_done_frac"] = jnp.mean(jnp.any(storage.next_dones, axis=0))

    obs_shape = storage.obs.shape
    curr_last = storage.obs.reshape(obs_shape[:-1] + (frame_stack, obs_channels))[..., -1, :]
    next_last = storage.next_obs.reshape(obs_shape[:-1] + (frame_stack, obs_channels))[..., -1, :]
    curr_last = curr_last.astype(jnp.float32) / 255.0
    next_last = next_last.astype(jnp.float32) / 255.0

    frame_delta = next_last - curr_last
    frame_delta_abs = jnp.abs(frame_delta)
    frame_delta_flat = frame_delta.reshape((frame_delta.shape[0] * frame_delta.shape[1], -1))
    curr_last_flat = curr_last.reshape((curr_last.shape[0] * curr_last.shape[1], -1))
    frame_delta_norms = jnp.linalg.norm(frame_delta_flat, axis=-1)
    frame_var = jnp.var(curr_last_flat, axis=0)

    metrics[f"{prefix}/frame_delta_abs_mean"] = jnp.mean(frame_delta_abs)
    metrics[f"{prefix}/frame_delta_l2_mean"] = jnp.mean(frame_delta_norms)
    metrics[f"{prefix}/frame_delta_l2_std"] = jnp.std(frame_delta_norms)
    metrics[f"{prefix}/frame_delta_gt_0p05_frac"] = jnp.mean(frame_delta_abs > 0.05)
    metrics[f"{prefix}/frame_var_mean"] = jnp.mean(frame_var)
    metrics[f"{prefix}/frame_var_max"] = jnp.max(frame_var)
    metrics[f"{prefix}/frame_var_min"] = jnp.min(frame_var)
    return metrics


def split_actor_critic_hiddens(encoded):
    """Return actor/critic hiddens from a shared or split encoder output."""
    if isinstance(encoded, tuple):
        return encoded
    return encoded, encoded


def resolve_heads_optimizer(heads_optimizer: str) -> str:
    """Validate and normalize the selected head matrix optimizer."""
    optimizer = heads_optimizer.lower()
    if optimizer not in {"adam", "stiefel", "stiefel_admm", "muon", "aurora"}:
        raise ValueError(
            f"Unsupported heads_optimizer='{heads_optimizer}'. "
            "Expected one of: adam, stiefel, stiefel_admm, muon, aurora."
        )
    return optimizer


def is_stiefel_matrix_param(param) -> bool:
    return param.ndim >= 2 and min(param.shape) > 1


def is_muon_matrix_param(param) -> bool:
    return param.ndim == 2 and min(param.shape) > 1


def is_aurora_matrix_param(param) -> bool:
    return param.ndim == 2 and min(param.shape) > 1


def encoder_final_stiefel_kernel_paths(encoder_type: str) -> tuple[tuple[str, ...], ...]:
    """Parameter paths that should receive the main encoder-final Stiefel branch."""
    kind = encoder_type.lower()
    if kind == "split_cnn":
        return (
            ("network", "params", "actor_dense", "kernel"),
            ("network", "params", "critic_dense", "kernel"),
        )
    if kind == "separate_cnn":
        return (
            ("network", "params", "actor_encoder", "Dense_0", "kernel"),
            ("network", "params", "critic_encoder", "Dense_0", "kernel"),
        )
    if kind == "innovation_direct_cnn":
        return (("network", "params", "innovation_projector", "kernel"),)
    if kind in {"cnn_swish_tanh", "cnn_swish_tanh_resid", "cnn_swish_tanh_resid_learned"}:
        return (("network", "params", "dense_stage2", "kernel"),)
    return (("network", "params", "Dense_0", "kernel"),)


def encoder_upstream_stiefel_kernel_paths(
    encoder_type: str,
    include_upstream_for_innovation_direct: bool = False,
) -> tuple[tuple[str, ...], ...]:
    """Optional upstream encoder paths that should receive a separate Stiefel branch."""
    kind = encoder_type.lower()
    if kind == "innovation_direct_cnn" and include_upstream_for_innovation_direct:
        return (("network", "params", "Dense_0", "kernel"),)
    if kind in {"cnn_swish_tanh", "cnn_swish_tanh_resid", "cnn_swish_tanh_resid_learned"}:
        return (("network", "params", "dense_stage1", "kernel"),)
    return ()


def policy_output_metrics(
    actor_mean: jnp.ndarray,
    actor_logstd: jnp.ndarray,
    values: jnp.ndarray,
) -> dict:
    """Measure how observation-conditioned the current policy/value outputs are."""
    eps = 1e-8
    metrics = {}

    action_mean_std = jnp.std(actor_mean, axis=0)
    action_noise_std = jnp.exp(actor_logstd)

    metrics["policy/mean_action_obs_std_avg"] = jnp.mean(action_mean_std)
    metrics["policy/mean_action_obs_std_min"] = jnp.min(action_mean_std)
    metrics["policy/mean_action_obs_std_max"] = jnp.max(action_mean_std)
    metrics["policy/action_noise_std_avg"] = jnp.mean(action_noise_std)
    metrics["policy/action_noise_std_min"] = jnp.min(action_noise_std)
    metrics["policy/action_noise_std_max"] = jnp.max(action_noise_std)
    metrics["policy/obs_to_noise_ratio"] = (
        jnp.mean(action_mean_std) / (jnp.mean(action_noise_std) + eps)
    )
    metrics["policy/mean_action_norm_mean"] = jnp.mean(jnp.linalg.norm(actor_mean, axis=-1))
    metrics["policy/mean_action_norm_std"] = jnp.std(jnp.linalg.norm(actor_mean, axis=-1))
    metrics["policy/logstd_mean"] = jnp.mean(actor_logstd)
    metrics["policy/logstd_std"] = jnp.std(actor_logstd)
    metrics["value/obs_std"] = jnp.std(values)
    metrics["value/obs_range"] = jnp.max(values) - jnp.min(values)

    return metrics


def vicreg_variance_penalty(
    hidden: jnp.ndarray,
    target: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """VICReg variance term on the shared hidden representation."""
    eps = 1e-4
    target = jnp.asarray(target, dtype=hidden.dtype)
    unit_std = jnp.sqrt(jnp.var(hidden, axis=0) + eps)
    gap = jax.nn.relu(target - unit_std)
    return jnp.mean(gap), jnp.mean(unit_std), jnp.mean(unit_std < target)



def lower_tail_variance_floor_penalty(
    hidden: jnp.ndarray,
    target: float,
    bottom_frac: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Encourage the weakest hidden directions to keep nontrivial variance."""
    eps = 1e-8
    target = jnp.asarray(target, dtype=hidden.dtype)
    unit_std = jnp.sqrt(jnp.var(hidden, axis=0) + eps)
    frac = float(np.clip(bottom_frac, 0.0, 1.0))
    bottom_k = max(1, int(np.ceil(hidden.shape[-1] * frac)))
    bottom_std = jnp.sort(unit_std)[:bottom_k]
    gap = jax.nn.relu(target - bottom_std)
    return (
        jnp.mean(jnp.square(gap)),
        jnp.mean(bottom_std),
        jnp.mean(bottom_std < target),
    )



def pre_ln_std_cap_penalty(
    dense_pre_ln: jnp.ndarray,
    max_std: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """One-sided cap on encoder dense pre-LayerNorm scale."""
    max_std = jnp.asarray(max_std, dtype=dense_pre_ln.dtype)
    pre_ln_std = jnp.std(dense_pre_ln)
    gap = jax.nn.relu(pre_ln_std - max_std)
    return jnp.square(gap), pre_ln_std



def actor_conditionality_penalty(
    actor_mean: jnp.ndarray,
    advantages: jnp.ndarray,
    target: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Encourage the actor mean to vary across important minibatch states."""
    eps = 1e-8
    target = jnp.asarray(target, dtype=actor_mean.dtype)
    weights = jax.lax.stop_gradient(jnp.abs(advantages)) + eps
    weights = weights / jnp.sum(weights)
    weighted_mean = jnp.sum(weights[:, None] * actor_mean, axis=0)
    centered = actor_mean - weighted_mean
    weighted_var = jnp.sum(weights[:, None] * jnp.square(centered), axis=0)
    mu_std = jnp.mean(jnp.sqrt(weighted_var + eps))
    gap = jax.nn.relu(target - mu_std)
    return jnp.square(gap), mu_std



def actor_noise_std_cap_penalty(
    actor_logstd: jnp.ndarray,
    max_std: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Discourage excessive action noise during the early training window."""
    action_noise_std = jnp.exp(actor_logstd)
    noise_std_avg = jnp.mean(action_noise_std)
    max_std = jnp.asarray(max_std, dtype=noise_std_avg.dtype)
    gap = jax.nn.relu(noise_std_avg - max_std)
    return jnp.square(gap), noise_std_avg



def innovation_dynamics_metrics(
    u_t: jnp.ndarray,
    next_u: jnp.ndarray,
    predicted_next_u: jnp.ndarray,
    state_term: jnp.ndarray,
    action_term: jnp.ndarray,
    next_done: jnp.ndarray,
) -> dict:
    """Summarize fit quality and state/action balance for the innovation model."""
    eps = 1e-8
    metrics = {}

    mask = (1.0 - next_done.astype(jnp.float32)).reshape((-1, 1))
    valid_count = jnp.maximum(jnp.sum(mask), 1.0)

    residual = next_u - predicted_next_u
    masked_next_u = next_u * mask
    masked_predicted = predicted_next_u * mask
    masked_residual = residual * mask

    next_u_mean = jnp.sum(masked_next_u, axis=0, keepdims=True) / valid_count
    centered_next_u = (next_u - next_u_mean) * mask

    residual_ss = jnp.sum(jnp.square(masked_residual))
    target_ss = jnp.sum(jnp.square(centered_next_u))

    residual_norms = jnp.linalg.norm(masked_residual, axis=-1)
    target_norms = jnp.linalg.norm(masked_next_u, axis=-1)
    state_term_norms = jnp.linalg.norm(state_term, axis=-1)
    action_term_norms = jnp.linalg.norm(action_term, axis=-1)

    cosine = jnp.sum(masked_predicted * masked_next_u, axis=-1) / (
        jnp.linalg.norm(masked_predicted, axis=-1) * target_norms + eps
    )
    valid_rows = jnp.squeeze(mask) > 0

    metrics["innovation_debug/target_norm_mean"] = jnp.sum(target_norms) / valid_count
    metrics["innovation_debug/pred_norm_mean"] = (
        jnp.sum(jnp.linalg.norm(masked_predicted, axis=-1)) / valid_count
    )
    metrics["innovation_debug/residual_norm_mean"] = jnp.sum(residual_norms) / valid_count
    metrics["innovation_debug/residual_to_target_ratio"] = (
        jnp.sum(residual_norms) / (jnp.sum(target_norms) + eps)
    )
    metrics["innovation_debug/residual_r2"] = 1.0 - residual_ss / (target_ss + eps)
    metrics["innovation_debug/cosine_pred_target_mean"] = jnp.mean(
        jnp.where(valid_rows, cosine, 0.0)
    )
    metrics["innovation_debug/state_term_norm_mean"] = jnp.mean(state_term_norms)
    metrics["innovation_debug/action_term_norm_mean"] = jnp.mean(action_term_norms)
    metrics["innovation_debug/action_to_state_norm_ratio"] = (
        jnp.mean(action_term_norms) / (jnp.mean(state_term_norms) + eps)
    )
    metrics["innovation_debug/u_std_avg"] = jnp.mean(jnp.std(u_t, axis=0))
    metrics["innovation_debug/next_u_std_avg"] = jnp.mean(jnp.std(next_u, axis=0))
    metrics["innovation_debug/residual_std_avg"] = jnp.mean(jnp.std(masked_residual, axis=0))

    return metrics


def compute_grad_norms(grads: dict) -> dict:
    """Compute gradient norms for key model components."""
    def tree_norm(tree):
        leaves = jax.tree_util.tree_leaves(tree)
        return jnp.sqrt(sum(jnp.sum(g**2) for g in leaves))

    metrics = {}
    if 'network' in grads:
        metrics["grads/encoder_norm"] = tree_norm(grads['network'])
    if 'actor' in grads:
        metrics["grads/actor_norm"] = tree_norm(grads['actor'])
    if 'critic' in grads:
        metrics["grads/critic_norm"] = tree_norm(grads['critic'])

    return metrics


# --------------------------------------------------------
#  Optimizer: Adam for encoder, selectable head matrices, Adam fallback vectors
# --------------------------------------------------------

def create_optimizer(
    encoder_lr,  # Can be float or schedule
    heads_stiefel_lr: float,
    heads_adam_lr,  # Can be float or schedule
    heads_muon_lr: float = 1e-3,
    stiefel_dual_lr: float = 0.01,
    stiefel_dual_steps: int = 5,
    stiefel_admm_steps: int = 10,
    stiefel_msign_steps: int = 5,
    adam_eps: float = 1e-5,
    max_grad_norm: float = 0.5,
    actor_stiefel_max_grad_norm: float = 1.0,
    critic_stiefel_max_grad_norm: float = 1.0,
    actor_muon_max_grad_norm: float = 1.0,
    critic_muon_max_grad_norm: float = 1.0,
    encoder_stiefel_lr: float = 0.02,
    encoder_upstream_stiefel_lr: float = -1.0,
    encoder_stiefel_max_grad_norm: float = 1.0,
    weight_decay: float = 0.0,
    heads_optimizer: str = "stiefel",
    use_encoder_final_stiefel: bool = False,
    use_encoder_upstream_stiefel: bool = False,
    muon_ns_steps: int = 5,
    muon_beta: float = 0.95,
    muon_eps: float = 1e-8,
    muon_weight_decay: float = 0.0,
    muon_nesterov: bool = True,
    muon_adaptive: bool = False,
    encoder_final_stiefel_paths: tuple[tuple[str, ...], ...] = (("network", "params", "Dense_0", "kernel"),),
    encoder_upstream_stiefel_paths: tuple[tuple[str, ...], ...] = (),
):
    """
    Create optimizer that uses:
    - Adam/AdamW for encoder (all params except optional final Dense kernel Stiefel branch) - supports lr schedule, with grad clipping
    - Manifold Stiefel for the encoder final Dense kernel only when enabled
    - Adam, Optax Muon, or manifold Stiefel for actor/critic head matrices
    - Adam/AdamW fallback for actor/critic params not handled by the selected head optimizer

    Uses AdamW when weight_decay > 0, otherwise uses Adam.
    """
    resolved_heads_optimizer = resolve_heads_optimizer(heads_optimizer)

    # Choose Adam or AdamW based on weight_decay
    if weight_decay > 0:
        adam_opt = lambda lr: optax.inject_hyperparams(optax.adamw)(
            learning_rate=lr, eps=adam_eps, weight_decay=weight_decay
        )
    else:
        adam_opt = lambda lr: optax.inject_hyperparams(optax.adam)(
            learning_rate=lr, eps=adam_eps
        )

    # Adam/AdamW for encoder (with schedule support and grad clipping)
    encoder_tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        adam_opt(encoder_lr),
    )

    # Adam/AdamW for head vectors/scalars (with schedule support and grad clipping)
    heads_adam_tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        adam_opt(heads_adam_lr),
    )

    # Stiefel for encoder final Dense kernel only (with separate grad clipping)
    encoder_final_stiefel_tx = optax.chain(
        optax.clip_by_global_norm(encoder_stiefel_max_grad_norm),
        manifold_stiefel(
            learning_rate=encoder_stiefel_lr,
            dual_lr=stiefel_dual_lr,
            dual_steps=stiefel_dual_steps,
            msign_steps=stiefel_msign_steps,
            min_ndim=2,
        ),
    )

    upstream_stiefel_lr = encoder_stiefel_lr if encoder_upstream_stiefel_lr < 0 else encoder_upstream_stiefel_lr
    encoder_upstream_stiefel_tx = optax.chain(
        optax.clip_by_global_norm(encoder_stiefel_max_grad_norm),
        manifold_stiefel(
            learning_rate=upstream_stiefel_lr,
            dual_lr=stiefel_dual_lr,
            dual_steps=stiefel_dual_steps,
            msign_steps=stiefel_msign_steps,
            min_ndim=2,
        ),
    )

    # Stiefel for actor head matrices (with separate grad clipping)
    actor_stiefel_tx = optax.chain(
        optax.clip_by_global_norm(actor_stiefel_max_grad_norm),
        manifold_stiefel(
            learning_rate=heads_stiefel_lr,
            dual_lr=stiefel_dual_lr,
            dual_steps=stiefel_dual_steps,
            msign_steps=stiefel_msign_steps,
            min_ndim=2,
        ),
    )

    # Stiefel for critic head matrices (with separate grad clipping)
    critic_stiefel_tx = optax.chain(
        optax.clip_by_global_norm(critic_stiefel_max_grad_norm),
        manifold_stiefel(
            learning_rate=heads_stiefel_lr,
            dual_lr=stiefel_dual_lr,
            dual_steps=stiefel_dual_steps,
            msign_steps=stiefel_msign_steps,
            min_ndim=2,
        ),
    )
    actor_stiefel_admm_tx = optax.chain(
        optax.clip_by_global_norm(actor_stiefel_max_grad_norm),
        manifold_stiefel_admm(
            learning_rate=heads_stiefel_lr,
            steps=stiefel_admm_steps,
            msign_steps=stiefel_msign_steps,
            min_ndim=2,
        ),
    )
    critic_stiefel_admm_tx = optax.chain(
        optax.clip_by_global_norm(critic_stiefel_max_grad_norm),
        manifold_stiefel_admm(
            learning_rate=heads_stiefel_lr,
            steps=stiefel_admm_steps,
            msign_steps=stiefel_msign_steps,
            min_ndim=2,
        ),
    )

    if not hasattr(optax, "contrib") or not hasattr(optax.contrib, "muon"):
        if resolved_heads_optimizer == "muon":
            raise ImportError(
                "heads_optimizer='muon' requires optax.contrib.muon. "
                "Install a Muon-capable Optax release, e.g. optax>=0.2.5."
            )
        muon_opt = None
    else:
        muon_opt = lambda lr: optax.contrib.muon(
            learning_rate=lr,
            ns_steps=muon_ns_steps,
            beta=muon_beta,
            eps=muon_eps,
            weight_decay=muon_weight_decay,
            nesterov=muon_nesterov,
            adaptive=muon_adaptive,
        )

    actor_muon_tx = optax.chain(
        optax.clip_by_global_norm(actor_muon_max_grad_norm),
        muon_opt(heads_muon_lr) if muon_opt is not None else optax.set_to_zero(),
    )
    critic_muon_tx = optax.chain(
        optax.clip_by_global_norm(critic_muon_max_grad_norm),
        muon_opt(heads_muon_lr) if muon_opt is not None else optax.set_to_zero(),
    )
    actor_aurora_tx = optax.chain(
        optax.clip_by_global_norm(actor_muon_max_grad_norm),
        aurora(
            learning_rate=heads_muon_lr,
            weight_decay=muon_weight_decay,
            mu=muon_beta,
            nesterov=muon_nesterov,
            eps=muon_eps,
            msign_steps=muon_ns_steps,
            adam_eps=adam_eps,
        ),
    )
    critic_aurora_tx = optax.chain(
        optax.clip_by_global_norm(critic_muon_max_grad_norm),
        aurora(
            learning_rate=heads_muon_lr,
            weight_decay=muon_weight_decay,
            mu=muon_beta,
            nesterov=muon_nesterov,
            eps=muon_eps,
            msign_steps=muon_ns_steps,
            adam_eps=adam_eps,
        ),
    )

    transforms = {
        'encoder': encoder_tx,
        'encoder_upstream_stiefel': encoder_upstream_stiefel_tx,
        'encoder_final_stiefel': encoder_final_stiefel_tx,
        'actor_stiefel': actor_stiefel_tx,
        'critic_stiefel': critic_stiefel_tx,
        'actor_stiefel_admm': actor_stiefel_admm_tx,
        'critic_stiefel_admm': critic_stiefel_admm_tx,
        'actor_muon': actor_muon_tx,
        'critic_muon': critic_muon_tx,
        'actor_aurora': actor_aurora_tx,
        'critic_aurora': critic_aurora_tx,
        'heads_adam': heads_adam_tx,
    }

    encoder_final_stiefel_paths = set(encoder_final_stiefel_paths)
    encoder_upstream_stiefel_paths = set(encoder_upstream_stiefel_paths)

    # Label function
    def label_fn(params):
        def _label(path, param):
            # path[0] is a top-level module key in params
            if path[0] == 'network':
                is_encoder_upstream_dense_kernel = path in encoder_upstream_stiefel_paths
                is_encoder_final_dense_kernel = path in encoder_final_stiefel_paths
                if use_encoder_upstream_stiefel and is_encoder_upstream_dense_kernel:
                    return 'encoder_upstream_stiefel'
                if use_encoder_final_stiefel and is_encoder_final_dense_kernel:
                    return 'encoder_final_stiefel'
                return 'encoder'
            if path[0] == 'innovation':
                return 'encoder'
            # For actor/critic heads, check if matrix or vector/scalar
            if path[0] == 'actor':
                if resolved_heads_optimizer == "stiefel" and is_stiefel_matrix_param(param):
                    return 'actor_stiefel'
                if resolved_heads_optimizer == "stiefel_admm" and is_stiefel_matrix_param(param):
                    return 'actor_stiefel_admm'
                if resolved_heads_optimizer == "muon" and is_muon_matrix_param(param):
                    return 'actor_muon'
                if resolved_heads_optimizer == "aurora" and is_aurora_matrix_param(param):
                    return 'actor_aurora'
                return 'heads_adam'
            if path[0] == 'critic':
                if resolved_heads_optimizer == "stiefel" and is_stiefel_matrix_param(param):
                    return 'critic_stiefel'
                if resolved_heads_optimizer == "stiefel_admm" and is_stiefel_matrix_param(param):
                    return 'critic_stiefel_admm'
                if resolved_heads_optimizer == "muon" and is_muon_matrix_param(param):
                    return 'critic_muon'
                if resolved_heads_optimizer == "aurora" and is_aurora_matrix_param(param):
                    return 'critic_aurora'
                return 'heads_adam'

            # Fallback for any extra top-level params.
            return 'heads_adam'

        flat = flax.traverse_util.flatten_dict(params)
        labeled = {k: _label(k, v) for k, v in flat.items()}
        return flax.core.freeze(flax.traverse_util.unflatten_dict(labeled))

    return optax.multi_transform(
        transforms=transforms,
        param_labels=label_fn,
    )


def make_pixelbrax_envs(args):
    """Create PixelBrax environments."""
    envs, _, _ = make_pixel_brax(
        backend=args.backend,
        env_name=args.env_name,
        n_envs=args.n_envs,
        seed=args.data_seed,
        hw=args.hw,
        distractor=None,
        video_path="datasets/DAVIS",
        video_set="train",
        return_float32=False,
        action_repeat=args.action_repeat,
    )

    try:
        action_dim = envs.action_size
    except AttributeError:
        action_dim = envs.env.action_size

    return envs, action_dim


def random_shift(key: jax.random.PRNGKey, x: jnp.ndarray, pad: int = 4) -> jnp.ndarray:
    """DrQ-style random shift augmentation."""
    b, h, w, c = x.shape
    x_padded = jnp.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='edge')

    key1, key2 = jax.random.split(key)
    crop_h = jax.random.randint(key1, (b,), 0, 2 * pad + 1)
    crop_w = jax.random.randint(key2, (b,), 0, 2 * pad + 1)

    def crop_single(x_pad, ch, cw):
        return jax.lax.dynamic_slice(x_pad, (ch, cw, 0), (h, w, c))

    return jax.vmap(crop_single)(x_padded, crop_h, crop_w)


def slippery_phase_from_timestep(
    timestep: int,
    change_every: int,
    num_seeded_phases: int,
) -> int:
    """Map a 1-indexed post-step timestep to the default-plus-CSV regime index."""
    zero_based_timestep = max(int(timestep) - 1, 0)
    phase = zero_based_timestep // change_every
    total_phases = num_seeded_phases + 1
    return min(phase, total_phases - 1)


def latest_slippery_pixel_metrics(
    global_step: int,
    args: Args,
    default_geom_friction: np.ndarray,
    friction_schedule: np.ndarray,
) -> dict[str, float]:
    timestep = int((global_step // args.n_envs) * args.action_repeat)
    phase = slippery_phase_from_timestep(
        timestep=timestep,
        change_every=args.slippery_change_every,
        num_seeded_phases=int(friction_schedule.shape[0]),
    )
    if phase == 0:
        geom_friction = default_geom_friction
    else:
        friction = float(friction_schedule[min(phase - 1, friction_schedule.shape[0] - 1)])
        geom_friction = default_geom_friction.copy()
        geom_friction[:, 0] = friction

    slide_friction = geom_friction[:, 0]
    return {
        "slippery/friction_slide_mean": float(np.mean(slide_friction)),
        "slippery/friction_slide_first_env": float(slide_friction[0]),
        "slippery/timestep_first_env": timestep,
        "slippery/phase_first_env": phase,
        "slippery/change_every": int(args.slippery_change_every),
    }


if __name__ == "__main__":
    args = tyro.cli(Args)

    if args.init_seed < 0:
        args.init_seed = args.seed
    if args.data_seed < 0:
        args.data_seed = args.seed
    args.heads_optimizer = resolve_heads_optimizer(args.heads_optimizer)
    if args.slippery_ant:
        if args.env_name != "ant":
            raise ValueError("--slippery-ant currently supports only --env-name ant.")
        if args.slippery_change_every <= 0:
            raise ValueError("--slippery-change-every must be positive.")
        if args.continual_dynamics_config is not None:
            raise ValueError("--slippery-ant cannot be combined with --continual-dynamics-config.")

    args.batch_size = int(args.n_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_updates = args.total_timesteps // args.batch_size
    if args.init_seed == args.seed and args.data_seed == args.seed:
        run_seed_tag = f"{args.seed}"
    else:
        run_seed_tag = f"s{args.seed}_i{args.init_seed}_d{args.data_seed}"
    run_name = f"{args.env_name}__{args.exp_name}__{run_seed_tag}__{int(time.time())}"

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=False,
            config=vars(args),
            name=run_name,
            save_code=True,
        )

    # Seeding
    random.seed(args.data_seed)
    np.random.seed(args.data_seed)
    key = jax.random.PRNGKey(args.data_seed)
    init_key = jax.random.PRNGKey(args.init_seed)
    network_key, innovation_key, actor_key, critic_key = jax.random.split(init_key, 4)

    # Environment setup
    print("=" * 60)
    print("PPO with Selectable Head Optimizer")
    print("=" * 60)
    print(f"JAX devices: {jax.devices()}")
    print(f"env name: {args.env_name}")
    print(f"total timesteps: {args.total_timesteps}")
    print(f"num steps per rollout: {args.num_steps}")
    print(f"num envs: {args.n_envs}")
    print(f"seed: {args.seed}")
    print(f"init_seed: {args.init_seed}")
    print(f"data_seed: {args.data_seed}")
    print(f"num_updates: {args.num_updates}")
    print(f"minibatch_size: {args.minibatch_size}")

    adam_type = "AdamW" if args.weight_decay > 0 else "Adam"
    use_encoder_upstream_stiefel = args.use_encoder_upstream_stiefel or (
        args.use_encoder_final_stiefel and args.encoder_stiefel_include_upstream
    )
    print(f"\nOptimizer config:")
    print(f"  encoder_lr ({adam_type}): {args.encoder_lr}")
    print(f"  heads_optimizer: {args.heads_optimizer}")
    print(f"  heads_stiefel_lr (Stiefel for actor/critic matrices): {args.heads_stiefel_lr}")
    print(f"  heads_muon_lr (Optax Muon for actor/critic matrices): {args.heads_muon_lr}")
    print(f"  heads_adam_lr ({adam_type} for actor/critic vectors): {args.heads_adam_lr}")
    print(f"  use_encoder_final_stiefel: {args.use_encoder_final_stiefel}")
    print(f"  use_encoder_upstream_stiefel: {use_encoder_upstream_stiefel}")
    print(f"  encoder_stiefel_include_upstream: {args.encoder_stiefel_include_upstream}")
    print(f"  encoder_upstream_stiefel_lr: {args.encoder_upstream_stiefel_lr}")
    print(f"  encoder_stiefel_lr: {args.encoder_stiefel_lr}")
    print(f"  encoder_stiefel_max_grad_norm: {args.encoder_stiefel_max_grad_norm}")
    print(f"  weight_decay: {args.weight_decay}")
    print(f"  stiefel_dual_lr: {args.stiefel_dual_lr}")
    print(f"  stiefel_dual_steps: {args.stiefel_dual_steps}")
    print(f"  stiefel_admm_steps: {args.stiefel_admm_steps}")
    print(f"  max_grad_norm ({adam_type}): {args.max_grad_norm}")
    print(f"  actor_stiefel_max_grad_norm: {args.actor_stiefel_max_grad_norm}")
    print(f"  critic_stiefel_max_grad_norm: {args.critic_stiefel_max_grad_norm}")
    print(f"  actor_muon_max_grad_norm: {args.actor_muon_max_grad_norm}")
    print(f"  critic_muon_max_grad_norm: {args.critic_muon_max_grad_norm}")
    print(f"  muon_ns_steps: {args.muon_ns_steps}")
    print(f"  muon_beta: {args.muon_beta}")
    print(f"  muon_eps: {args.muon_eps}")
    print(f"  muon_weight_decay: {args.muon_weight_decay}")
    print(f"  muon_nesterov: {args.muon_nesterov}")
    print(f"  muon_adaptive: {args.muon_adaptive}")
    print(f"  encoder_type: {args.encoder_type}")
    print(f"  encoder_tanh_scale: {args.encoder_tanh_scale}")
    print(f"  encoder_residual_scale: {args.encoder_residual_scale}")
    print(f"  encoder_hidden_activation: {args.encoder_hidden_activation}")
    print(f"  encoder_warmup_updates: {args.encoder_warmup_updates}")
    print(f"  encoder_use_crate_block: {args.encoder_use_crate_block}")
    print(f"  encoder_crate_step_size: {args.encoder_crate_step_size}")
    print(f"  sigreg_mode: {args.sigreg_mode}")
    print(f"  sigreg_coef: {args.sigreg_coef}")
    print(f"  sigreg_proj_dim: {args.sigreg_proj_dim}")
    print(f"  sigreg_num_slices: {args.sigreg_num_slices}")
    print(f"  sigreg_num_t: {args.sigreg_num_t}")
    print(f"  sigreg_t_max: {args.sigreg_t_max}")
    print(f"  sigreg_warmup_updates: {args.sigreg_warmup_updates}")
    print(f"  sigreg_ramp_updates: {args.sigreg_ramp_updates}")
    print(f"  innovation_coef: {args.innovation_coef}")
    print(f"  innovation_proj_dim: {args.innovation_proj_dim}")
    print(f"  innovation_num_slices: {args.innovation_num_slices}")
    print(f"  innovation_num_t: {args.innovation_num_t}")
    print(f"  innovation_t_max: {args.innovation_t_max}")
    print(f"  innovation_warmup_updates: {args.innovation_warmup_updates}")
    print(f"  innovation_ramp_updates: {args.innovation_ramp_updates}")
    print(f"  state_dependent_std: {args.state_dependent_std}")
    print(f"  state_std_tanh_scale: {args.state_std_tanh_scale}")
    print(f"  actor_logstd_min: {args.actor_logstd_min}")
    print(f"  actor_logstd_max: {args.actor_logstd_max}")
    print(f"  clip_global_logstd: {args.clip_global_logstd}")
    print(f"  bounded_global_logstd: {args.bounded_global_logstd}")
    print(f"  actor_logstd_init: {args.actor_logstd_init}")
    print(f"  actor_mean_tanh: {args.actor_mean_tanh}")
    print(f"  actor_mean_scale: {args.actor_mean_scale}")
    print(f"  anneal_lr: {args.anneal_lr}")
    print("=" * 60)

    encoder_type = args.encoder_type.lower()
    if encoder_type not in {"resnet", "cnn", "cnn_swish", "cnn_swish_ta", "cnn_swish_tb", "cnn_swish_tc", "crate_cnn", "crate_cnn_tanh", "crate_cnn_tanh_resid", "cnn_swish_tanh", "cnn_swish_tanh_resid", "cnn_swish_tanh_resid_learned", "split_cnn", "separate_cnn", "sigreg_cnn", "innovation_cnn", "innovation_direct_cnn", "mlp"}:
        raise ValueError(
            f"Unsupported encoder_type='{args.encoder_type}'. Expected one of: ['resnet', 'cnn', 'cnn_swish', 'cnn_swish_ta', 'cnn_swish_tb', 'cnn_swish_tc', 'crate_cnn', 'crate_cnn_tanh', 'crate_cnn_tanh_resid', 'cnn_swish_tanh', 'cnn_swish_tanh_resid', 'cnn_swish_tanh_resid_learned', 'split_cnn', 'separate_cnn', 'sigreg_cnn', 'innovation_cnn', 'innovation_direct_cnn', 'mlp']"
        )

    sigreg_mode = args.sigreg_mode.lower()
    if sigreg_mode not in {"off", "projected", "shared_hidden"}:
        raise ValueError(
            f"Unsupported sigreg_mode='{args.sigreg_mode}'. Expected one of: ['off', 'projected', 'shared_hidden']"
        )
    if sigreg_mode == "projected" and encoder_type != "sigreg_cnn":
        raise ValueError("Projected SIGReg requires encoder_type='sigreg_cnn'.")
    if args.innovation_coef > 0 and encoder_type not in {"innovation_cnn", "innovation_direct_cnn"}:
        raise ValueError("Innovation-aware regularization requires encoder_type in {'innovation_cnn', 'innovation_direct_cnn'}.")
    encoder_hidden_activation = args.encoder_hidden_activation.lower()
    if encoder_hidden_activation not in {"tanh", "swish", "silu", "swish_tanh", "swish_ta", "swish_tb", "swish_tc"}:
        raise ValueError(
            f"Unsupported encoder_hidden_activation='{args.encoder_hidden_activation}'. Expected one of: ['tanh', 'swish', 'silu', 'swish_tanh', 'swish_ta', 'swish_tb', 'swish_tc']"
        )

    envs, action_dim = make_pixelbrax_envs(args)
    print(f"action_dim: {action_dim}")

    base_sys = envs.base_sys
    current_sys = base_sys
    continual_config = None
    current_task = None
    current_task_index = 0
    task_start_step = 0
    updates_per_task = 0
    schedule_seed = args.data_seed
    slippery_friction_schedule = None
    slippery_default_geom_friction = None
    last_slippery_phase_logged = None

    if args.continual_dynamics_config is not None:
        loaded_config = load_continual_dynamics_config(args.continual_dynamics_config)
        if loaded_config.enabled:
            updates_per_task = validate_continual_dynamics_config(
                loaded_config,
                env_name=args.env_name,
                backend=args.backend,
                n_envs=args.n_envs,
                num_steps=args.num_steps,
                base_sys=base_sys,
            )
            continual_config = loaded_config
            schedule_seed = resolve_schedule_seed(continual_config, args.data_seed)
            current_task = make_dynamics_task(
                continual_config,
                base_sys,
                task_index=0,
                schedule_seed=schedule_seed,
            )
            current_sys = apply_dynamics_task(base_sys, current_task)
            print("\nContinual dynamics: ENABLED")
            print(f"  config: {continual_config.path}")
            print(f"  switch_every_env_steps: {continual_config.switch_every_env_steps}")
            print(f"  updates_per_task: {updates_per_task}")
            print(f"  schedule_seed: {schedule_seed}")
            print(
                "  task=0 step=0 default=True values="
                f"{json.dumps(current_task.values, sort_keys=True)}"
            )
        else:
            print("\nContinual dynamics: config disabled; using fixed dynamics")

    if args.slippery_ant:
        friction_table = load_seeded_friction_schedule_table(
            str(DEFAULT_SLIPPERY_FRICTIONS_CSV.resolve())
        )
        if args.slippery_schedule_seed < 0 or args.slippery_schedule_seed >= friction_table.shape[0]:
            raise ValueError(
                f"slippery_schedule_seed={args.slippery_schedule_seed} is out of range "
                f"for {friction_table.shape[0]} seeded friction rows."
            )
        slippery_friction_schedule = friction_table[args.slippery_schedule_seed].astype(np.float32)
        slippery_default_geom_friction = np.asarray(
            jax.device_get(base_sys.geom_friction),
            dtype=np.float32,
        )
        print("\nSlipperyAnt friction schedule: ENABLED")
        print(f"  source: {DEFAULT_SLIPPERY_FRICTIONS_CSV}")
        print(f"  change_every: {args.slippery_change_every}")
        print(f"  schedule_seed: {args.slippery_schedule_seed}")
        print(f"  num_csv_phases: {slippery_friction_schedule.shape[0]}")
        print("  phase friction_first_geom")
        for phase in range(min(6, slippery_friction_schedule.shape[0] + 1)):
            if phase == 0:
                friction = float(slippery_default_geom_friction[0, 0])
            else:
                friction = float(slippery_friction_schedule[phase - 1])
            print(f"  {phase:5d} {friction:.6g}")

    # Get observation shape
    reset_rng = jax.random.split(jax.random.PRNGKey(args.data_seed), args.n_envs)
    init_env_state = envs.reset_with_sys(current_sys, reset_rng)
    raw_obs_shape = init_env_state.pixels.shape[1:]  # (H, W, C)
    obs_shape = (raw_obs_shape[0], raw_obs_shape[1], raw_obs_shape[2] * args.frame_stack)
    print(f"raw_obs_shape: {raw_obs_shape}")
    print(f"frame_stack: {args.frame_stack}")
    print(f"action_repeat: {args.action_repeat}")
    print(f"obs_shape (with {args.frame_stack} stacked frames): {obs_shape}")

    episode_stats = EpisodeStatistics(
        episode_returns=jnp.zeros(args.n_envs, dtype=jnp.float32),
        episode_lengths=jnp.zeros(args.n_envs, dtype=jnp.int32),
        returned_episode_returns=jnp.zeros(args.n_envs, dtype=jnp.float32),
        returned_episode_lengths=jnp.zeros(args.n_envs, dtype=jnp.int32),
    )

    # Initialize networks
    network = build_encoder(
        encoder_type,
        args.encoder_tanh_scale,
        sigreg_proj_dim=args.sigreg_proj_dim,
        innovation_proj_dim=args.innovation_proj_dim,
        use_crate_block=args.encoder_use_crate_block,
        crate_step_size=args.encoder_crate_step_size,
        innovation_hidden_activation=encoder_hidden_activation,
        residual_scale=args.encoder_residual_scale,
    )
    if args.use_crate_head:
        actor = CRATEActor(
            action_dim=action_dim,
            crate_step_size=args.crate_step_size,
            state_dependent_std=args.state_dependent_std,
            state_std_tanh_scale=args.state_std_tanh_scale,
            logstd_min=args.actor_logstd_min,
            logstd_max=args.actor_logstd_max,
            clip_global_logstd=args.clip_global_logstd,
            bounded_global_logstd=args.bounded_global_logstd,
            actor_logstd_init=args.actor_logstd_init,
            actor_mean_tanh=args.actor_mean_tanh,
            actor_mean_scale=args.actor_mean_scale,
        )
        critic = CRATECritic(crate_step_size=args.crate_step_size)
        print(f"Using CRATE heads with step_size={args.crate_step_size}")
    else:
        actor = Actor(
            action_dim=action_dim,
            state_dependent_std=args.state_dependent_std,
            state_std_tanh_scale=args.state_std_tanh_scale,
            logstd_min=args.actor_logstd_min,
            logstd_max=args.actor_logstd_max,
            clip_global_logstd=args.clip_global_logstd,
            bounded_global_logstd=args.bounded_global_logstd,
            actor_logstd_init=args.actor_logstd_init,
            actor_mean_tanh=args.actor_mean_tanh,
            actor_mean_scale=args.actor_mean_scale,
        )
        critic = Critic()

    dummy_obs = jnp.zeros((1,) + obs_shape)
    network_params = network.init(network_key, dummy_obs)
    innovation_model = None
    dummy_actor_hidden, dummy_critic_hidden = split_actor_critic_hiddens(
        network.apply(network_params, dummy_obs)
    )
    encoder_stiefel_paths = encoder_final_stiefel_kernel_paths(encoder_type)
    encoder_upstream_stiefel_paths = encoder_upstream_stiefel_kernel_paths(
        encoder_type,
        include_upstream_for_innovation_direct=args.encoder_stiefel_include_upstream,
    )

    params_dict = {
        "network": network_params,
    }
    if encoder_type in {"innovation_cnn", "innovation_direct_cnn"}:
        innovation_model = InnovationDynamics(
            action_dim=action_dim,
            proj_dim=args.innovation_proj_dim,
        )
        dummy_debug = network.apply(network_params, dummy_obs, return_intermediates=True)
        params_dict["innovation"] = innovation_model.init(
            innovation_key,
            dummy_debug["u"],
            jnp.zeros((1, action_dim), dtype=jnp.float32),
        )
    params_dict["actor"] = actor.init(actor_key, dummy_actor_hidden)
    params_dict["critic"] = critic.init(critic_key, dummy_critic_hidden)

    all_params = flax.core.freeze(params_dict)

    # Count params by component and type
    def count_params(params, key):
        if key in params:
            return sum(p.size for p in jax.tree_util.tree_leaves(params[key]))
        return 0

    def count_head_optimizer_params(params, key, heads_optimizer):
        if key not in params:
            return 0, 0
        flat = flax.traverse_util.flatten_dict(params[key])
        total_count = sum(p.size for p in flat.values())
        if heads_optimizer in {"stiefel", "stiefel_admm"}:
            optimized_count = sum(p.size for p in flat.values() if is_stiefel_matrix_param(p))
        elif heads_optimizer == "muon":
            optimized_count = sum(p.size for p in flat.values() if is_muon_matrix_param(p))
        elif heads_optimizer == "aurora":
            optimized_count = sum(p.size for p in flat.values() if is_aurora_matrix_param(p))
        else:
            optimized_count = 0
        return optimized_count, total_count - optimized_count

    encoder_params = count_params(all_params, "network")
    innovation_params = count_params(all_params, "innovation")
    actor_params = count_params(all_params, "actor")
    critic_params = count_params(all_params, "critic")
    total_params = encoder_params + innovation_params + actor_params + critic_params

    actor_head_opt, actor_adam = count_head_optimizer_params(
        all_params,
        "actor",
        args.heads_optimizer,
    )
    critic_head_opt, critic_adam = count_head_optimizer_params(
        all_params,
        "critic",
        args.heads_optimizer,
    )
    encoder_final_stiefel_params = 0
    encoder_upstream_stiefel_params = 0
    if args.use_encoder_final_stiefel or use_encoder_upstream_stiefel:
        flat_all_params = flax.traverse_util.flatten_dict(all_params)
        if args.use_encoder_final_stiefel:
            encoder_final_stiefel_params = sum(
                flat_all_params[path].size
                for path in encoder_stiefel_paths
                if path in flat_all_params
            )
        if use_encoder_upstream_stiefel:
            encoder_upstream_stiefel_params = sum(
                flat_all_params[path].size
                for path in encoder_upstream_stiefel_paths
                if path in flat_all_params
            )
    encoder_adam_params = encoder_params - encoder_final_stiefel_params - encoder_upstream_stiefel_params

    print(f"\nParameter breakdown:")
    print(
        f"  Encoder total: {encoder_params:,} "
        f"(Stiefel upstream: {encoder_upstream_stiefel_params:,}, Stiefel final dense: {encoder_final_stiefel_params:,}, Adam rest: {encoder_adam_params:,})"
    )
    if innovation_params > 0:
        print(f"  Innovation dynamics (Adam): {innovation_params:,}")
    if args.heads_optimizer == "adam":
        print(f"  Actor total: {actor_params:,} (Adam: {actor_adam:,})")
        print(f"  Critic total: {critic_params:,} (Adam: {critic_adam:,})")
    else:
        head_opt_label = {
            "stiefel": "Stiefel",
            "stiefel_admm": "ADMM Stiefel",
            "muon": "Optax Muon",
            "aurora": "Aurora",
        }[args.heads_optimizer]
        print(
            f"  Actor total: {actor_params:,} "
            f"({head_opt_label}: {actor_head_opt:,}, Adam fallback: {actor_adam:,})"
        )
        print(
            f"  Critic total: {critic_params:,} "
            f"({head_opt_label}: {critic_head_opt:,}, Adam fallback: {critic_adam:,})"
        )
    print(f"  Total: {total_params:,}")

    # Create learning rate schedules for Adam optimizers
    steps_per_update = args.num_minibatches * args.update_epochs

    def lr_from_update(base_lr: float, update_idx: int, warmup_updates: int = 0):
        """Warmup + linear decay learning-rate helper."""
        update_idx = jnp.asarray(update_idx, dtype=jnp.float32)
        warmup = float(max(0, warmup_updates))
        decay_updates = float(max(1, args.num_updates - warmup_updates))

        warm_lr = base_lr * ((update_idx + 1.0) / jnp.maximum(1.0, warmup))
        decay_idx = jnp.maximum(0.0, update_idx - warmup)
        frac = 1.0 - jnp.minimum(1.0, decay_idx / decay_updates)
        decay_lr = base_lr * frac

        if warmup_updates <= 0:
            return decay_lr
        return jnp.where(update_idx < warmup, warm_lr, decay_lr)

    def make_linear_schedule(base_lr, warmup_updates: int = 0):
        """Schedule for Adam optimizers; optional warmup before linear decay."""
        def schedule(count):
            update_idx = count // steps_per_update
            return lr_from_update(base_lr, update_idx, warmup_updates)

        return schedule


    def sigreg_coef_from_update(
        base_coef: float,
        update_idx: int,
        warmup_updates: int = 0,
        ramp_updates: int = 0,
    ):
        """Warmup + linear ramp helper for the SIGReg coefficient."""
        if base_coef <= 0:
            return jnp.asarray(0.0, dtype=jnp.float32)
        update_idx = jnp.asarray(update_idx, dtype=jnp.float32)
        warmup = float(max(0, warmup_updates))
        ramp = float(max(0, ramp_updates))
        if ramp_updates <= 0:
            return jnp.where(update_idx < warmup, 0.0, base_coef)
        ramp_progress = jnp.clip((update_idx - warmup + 1.0) / jnp.maximum(1.0, ramp), 0.0, 1.0)
        return jnp.where(update_idx < warmup, 0.0, base_coef * ramp_progress)

    def early_window_coef(
        base_coef: float,
        update_idx: int,
        stop_updates: int = 0,
    ):
        """Keep a coefficient on only during the initial PPO update window."""
        if base_coef <= 0 or stop_updates <= 0:
            return jnp.asarray(0.0, dtype=jnp.float32)
        update_idx = jnp.asarray(update_idx, dtype=jnp.float32)
        stop = float(max(0, stop_updates))
        return jnp.where(update_idx < stop, base_coef, 0.0)

    if args.anneal_lr:
        encoder_lr = make_linear_schedule(args.encoder_lr, args.encoder_warmup_updates)
        heads_adam_lr = make_linear_schedule(args.heads_adam_lr, warmup_updates=0)
        if args.encoder_warmup_updates > 0:
            print(
                f"\nLearning rate annealing: ENABLED "
                f"(encoder warmup={args.encoder_warmup_updates} updates + linear decay)"
            )
        else:
            print("\nLearning rate annealing: ENABLED (linear decay for Adam)")
    else:
        encoder_lr = args.encoder_lr
        heads_adam_lr = args.heads_adam_lr
        print("\nLearning rate annealing: DISABLED")

    # Create optimizer
    tx = create_optimizer(
        encoder_lr=encoder_lr,
        heads_stiefel_lr=args.heads_stiefel_lr,
        heads_adam_lr=heads_adam_lr,
        heads_muon_lr=args.heads_muon_lr,
        stiefel_dual_lr=args.stiefel_dual_lr,
        stiefel_dual_steps=args.stiefel_dual_steps,
        stiefel_admm_steps=args.stiefel_admm_steps,
        stiefel_msign_steps=args.stiefel_msign_steps,
        adam_eps=1e-5,
        max_grad_norm=args.max_grad_norm,
        actor_stiefel_max_grad_norm=args.actor_stiefel_max_grad_norm,
        critic_stiefel_max_grad_norm=args.critic_stiefel_max_grad_norm,
        actor_muon_max_grad_norm=args.actor_muon_max_grad_norm,
        critic_muon_max_grad_norm=args.critic_muon_max_grad_norm,
        encoder_stiefel_lr=args.encoder_stiefel_lr,
        encoder_upstream_stiefel_lr=args.encoder_upstream_stiefel_lr,
        encoder_stiefel_max_grad_norm=args.encoder_stiefel_max_grad_norm,
        weight_decay=args.weight_decay,
        heads_optimizer=args.heads_optimizer,
        use_encoder_final_stiefel=args.use_encoder_final_stiefel,
        use_encoder_upstream_stiefel=use_encoder_upstream_stiefel,
        muon_ns_steps=args.muon_ns_steps,
        muon_beta=args.muon_beta,
        muon_eps=args.muon_eps,
        muon_weight_decay=args.muon_weight_decay,
        muon_nesterov=args.muon_nesterov,
        muon_adaptive=args.muon_adaptive,
        encoder_final_stiefel_paths=encoder_stiefel_paths,
        encoder_upstream_stiefel_paths=encoder_upstream_stiefel_paths,
    )

    agent_state = TrainState.create(
        apply_fn=None,
        params=all_params,
        tx=tx,
    )

    network_debug_apply = jax.jit(network.apply, static_argnames=("return_intermediates",))
    actor_debug_apply = jax.jit(actor.apply, static_argnames=("return_intermediates",))
    critic_debug_apply = jax.jit(critic.apply, static_argnames=("return_intermediates",))
    network.apply = jax.jit(network.apply)
    actor.apply = jax.jit(actor.apply)
    critic.apply = jax.jit(critic.apply)
    innovation_debug_apply = None
    if innovation_model is not None:
        innovation_debug_apply = jax.jit(
            innovation_model.apply,
            static_argnames=("return_intermediates",),
        )
        innovation_model.apply = jax.jit(innovation_model.apply)


    def encode_with_intermediates(network_params, obs):
        if encoder_type not in {"resnet", "cnn", "cnn_swish", "cnn_swish_ta", "cnn_swish_tb", "cnn_swish_tc", "crate_cnn", "crate_cnn_tanh", "crate_cnn_tanh_resid", "cnn_swish_tanh", "cnn_swish_tanh_resid", "cnn_swish_tanh_resid_learned", "split_cnn", "separate_cnn", "sigreg_cnn", "innovation_cnn", "innovation_direct_cnn", "mlp"}:
            raise ValueError(f"return_intermediates is not supported for encoder_type={encoder_type}")
        return network_debug_apply(network_params, obs, return_intermediates=True)

    def innovation_with_intermediates(innovation_params, u_t, action):
        if innovation_debug_apply is None:
            raise ValueError("innovation intermediates requested without innovation model")
        return innovation_debug_apply(
            innovation_params,
            u_t,
            action,
            return_intermediates=True,
        )

    def shared_hidden_sigreg(actor_hidden, critic_hidden, rng):
        """Apply SIGReg to the latent(s) consumed by the actor/critic heads."""
        if encoder_type in {"split_cnn", "separate_cnn"}:
            actor_key, critic_key = jax.random.split(rng)
            actor_total, actor_re, actor_im = sigreg_loss(
                actor_hidden,
                actor_key,
                num_slices=args.sigreg_num_slices,
                num_t=args.sigreg_num_t,
                t_max=args.sigreg_t_max,
            )
            critic_total, critic_re, critic_im = sigreg_loss(
                critic_hidden,
                critic_key,
                num_slices=args.sigreg_num_slices,
                num_t=args.sigreg_num_t,
                t_max=args.sigreg_t_max,
            )
            return (
                0.5 * (actor_total + critic_total),
                0.5 * (actor_re + critic_re),
                0.5 * (actor_im + critic_im),
            )
        return sigreg_loss(
            actor_hidden,
            rng,
            num_slices=args.sigreg_num_slices,
            num_t=args.sigreg_num_t,
            t_max=args.sigreg_t_max,
        )

    @jax.jit
    def get_action_and_value(
        agent_state: TrainState,
        next_obs: np.ndarray,
        key: jax.random.PRNGKey,
    ):
        """Sample action, calculate value, logprob, and return updated key."""
        actor_hidden, critic_hidden = split_actor_critic_hiddens(
            network.apply(agent_state.params["network"], next_obs)
        )
        actor_mean, actor_logstd = actor.apply(agent_state.params["actor"], actor_hidden)

        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logstd))

        key, subkey = jax.random.split(key)
        action = pi.sample(seed=subkey)
        logprob = pi.log_prob(action)
        value = critic.apply(agent_state.params["critic"], critic_hidden)

        action = jnp.clip(action, -args.max_action, args.max_action)

        return action, logprob, value.squeeze(-1), key

    @jax.jit
    def get_action_and_value2(
        params: flax.core.FrozenDict,
        x: np.ndarray,
        action: np.ndarray,
    ):
        """Calculate value, logprob of supplied action, and entropy."""
        actor_hidden, critic_hidden = split_actor_critic_hiddens(
            network.apply(params["network"], x)
        )
        actor_mean, actor_logstd = actor.apply(params["actor"], actor_hidden)

        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logstd))

        logprob = pi.log_prob(action)
        entropy = pi.entropy()
        value = critic.apply(params["critic"], critic_hidden).squeeze(-1)

        return logprob, entropy, value

    def compute_gae_once(carry, inp, gamma, gae_lambda):
        advantages = carry
        nextdone, nextvalues, curvalues, reward = inp
        nextnonterminal = 1.0 - nextdone

        delta = reward + gamma * nextvalues * nextnonterminal - curvalues
        advantages = delta + gamma * gae_lambda * nextnonterminal * advantages
        return advantages, advantages

    compute_gae_once = partial(compute_gae_once, gamma=args.gamma, gae_lambda=args.gae_lambda)

    @jax.jit
    def compute_gae(
        agent_state: TrainState,
        next_obs: np.ndarray,
        next_done: np.ndarray,
        storage: Storage,
    ):
        next_value = critic.apply(
            agent_state.params["critic"],
            split_actor_critic_hiddens(
                network.apply(agent_state.params["network"], next_obs)
            )[1],
        ).squeeze(-1)

        advantages = jnp.zeros((args.n_envs,))
        dones = jnp.concatenate([storage.dones, next_done[None, :]], axis=0)
        values = jnp.concatenate([storage.values, next_value[None, :]], axis=0)
        _, advantages = jax.lax.scan(
            compute_gae_once,
            advantages,
            (dones[1:], values[1:], values[:-1], storage.rewards),
            reverse=True,
        )
        storage = storage.replace(
            advantages=advantages,
            returns=advantages + storage.values,
        )
        return storage

    

    def ppo_loss(
        params,
        x,
        next_x,
        next_done,
        a,
        logp,
        mb_advantages,
        mb_returns,
        mb_values,
        aug_key,
        sigreg_coef_current,
        innovation_coef_current,
        vicreg_var_coef_current,
        bottleneck_var_coef_current,
        bottleneck_pre_ln_coef_current,
        actor_cond_coef_current,
        actor_noise_coef_current,
    ):
        """PPO loss function."""
        if args.use_augmentation:
            aug_key, next_aug_key, aux_key = jax.random.split(aug_key, 3)
            x = random_shift(aug_key, x, pad=args.augment_pad)
            next_x = random_shift(next_aug_key, next_x, pad=args.augment_pad)
        else:
            aux_key = aug_key

        sigreg_total = jnp.asarray(0.0, dtype=jnp.float32)
        sigreg_re = jnp.asarray(0.0, dtype=jnp.float32)
        sigreg_im = jnp.asarray(0.0, dtype=jnp.float32)
        innovation_total = jnp.asarray(0.0, dtype=jnp.float32)
        innovation_re = jnp.asarray(0.0, dtype=jnp.float32)
        innovation_im = jnp.asarray(0.0, dtype=jnp.float32)
        vicreg_var_total = jnp.asarray(0.0, dtype=jnp.float32)
        vicreg_var_unit_std_avg = jnp.asarray(0.0, dtype=jnp.float32)
        vicreg_var_below_target_frac = jnp.asarray(0.0, dtype=jnp.float32)
        bottleneck_var_total = jnp.asarray(0.0, dtype=jnp.float32)
        bottleneck_bottom_std_avg = jnp.asarray(0.0, dtype=jnp.float32)
        bottleneck_bottom_below_target_frac = jnp.asarray(0.0, dtype=jnp.float32)
        bottleneck_pre_ln_total = jnp.asarray(0.0, dtype=jnp.float32)
        bottleneck_pre_ln_std = jnp.asarray(0.0, dtype=jnp.float32)
        actor_cond_total = jnp.asarray(0.0, dtype=jnp.float32)
        actor_cond_mu_std = jnp.asarray(0.0, dtype=jnp.float32)
        actor_noise_total = jnp.asarray(0.0, dtype=jnp.float32)
        actor_noise_std_avg = jnp.asarray(0.0, dtype=jnp.float32)
        dense_pre_ln = None
        actor_hidden = None
        critic_hidden = None
        actor_dense_pre_ln = None
        critic_dense_pre_ln = None

        if encoder_type == "sigreg_cnn":
            encoder_debug = encode_with_intermediates(params["network"], x)
            actor_hidden = encoder_debug["hidden"]
            critic_hidden = actor_hidden
            dense_pre_ln = encoder_debug["dense_pre_ln"]
            if sigreg_mode == "projected":
                aux_key, sigreg_key = jax.random.split(aux_key)
                sigreg_total, sigreg_re, sigreg_im = sigreg_loss(
                    encoder_debug["u"],
                    sigreg_key,
                    num_slices=args.sigreg_num_slices,
                    num_t=args.sigreg_num_t,
                    t_max=args.sigreg_t_max,
                )
        elif encoder_type in {"innovation_cnn", "innovation_direct_cnn"}:
            encoder_debug = encode_with_intermediates(params["network"], x)
            actor_hidden = encoder_debug["policy_hidden"]
            critic_hidden = actor_hidden
            dense_pre_ln = encoder_debug["dense_pre_ln"]
            aux_key, innovation_key = jax.random.split(aux_key)
            next_encoder_debug = encode_with_intermediates(params["network"], next_x)
            predicted_next_u = innovation_model.apply(
                params["innovation"],
                encoder_debug["u"],
                a,
            )
            residual = next_encoder_debug["u"] - predicted_next_u
            innovation_mask = 1.0 - next_done.astype(jnp.float32)
            innovation_total, innovation_re, innovation_im = sigreg_loss_masked(
                residual,
                innovation_mask,
                innovation_key,
                num_slices=args.innovation_num_slices,
                num_t=args.innovation_num_t,
                t_max=args.innovation_t_max,
            )
        elif encoder_type in {"split_cnn", "separate_cnn"}:
            if args.bottleneck_pre_ln_coef > 0.0:
                encoder_debug = encode_with_intermediates(params["network"], x)
                actor_hidden = encoder_debug["actor_hidden"]
                critic_hidden = encoder_debug["critic_hidden"]
                actor_dense_pre_ln = encoder_debug["actor_dense_pre_ln"]
                critic_dense_pre_ln = encoder_debug["critic_dense_pre_ln"]
            else:
                actor_hidden, critic_hidden = split_actor_critic_hiddens(
                    network.apply(params["network"], x)
                )
        elif encoder_type in {"resnet", "cnn", "cnn_swish", "cnn_swish_ta", "cnn_swish_tb", "cnn_swish_tc", "crate_cnn", "crate_cnn_tanh", "crate_cnn_tanh_resid"} and args.bottleneck_pre_ln_coef > 0.0:
            encoder_debug = encode_with_intermediates(params["network"], x)
            actor_hidden = encoder_debug["hidden"]
            critic_hidden = actor_hidden
            dense_pre_ln = encoder_debug["dense_pre_ln"]
        else:
            actor_hidden, critic_hidden = split_actor_critic_hiddens(
                network.apply(params["network"], x)
            )

        if sigreg_mode == "shared_hidden":
            aux_key, sigreg_key = jax.random.split(aux_key)
            sigreg_total, sigreg_re, sigreg_im = shared_hidden_sigreg(
                actor_hidden,
                critic_hidden,
                sigreg_key,
            )

        if encoder_type in {"split_cnn", "separate_cnn"}:
            (
                actor_vicreg_total,
                actor_vicreg_unit_std_avg,
                actor_vicreg_below_target_frac,
            ) = vicreg_variance_penalty(actor_hidden, args.vicreg_var_target)
            (
                critic_vicreg_total,
                critic_vicreg_unit_std_avg,
                critic_vicreg_below_target_frac,
            ) = vicreg_variance_penalty(critic_hidden, args.vicreg_var_target)
            vicreg_var_total = 0.5 * (actor_vicreg_total + critic_vicreg_total)
            vicreg_var_unit_std_avg = 0.5 * (
                actor_vicreg_unit_std_avg + critic_vicreg_unit_std_avg
            )
            vicreg_var_below_target_frac = 0.5 * (
                actor_vicreg_below_target_frac + critic_vicreg_below_target_frac
            )

            (
                actor_bottleneck_var_total,
                actor_bottom_std_avg,
                actor_bottom_below_target_frac,
            ) = lower_tail_variance_floor_penalty(
                actor_hidden,
                args.bottleneck_var_target,
                args.bottleneck_var_bottom_frac,
            )
            (
                critic_bottleneck_var_total,
                critic_bottom_std_avg,
                critic_bottom_below_target_frac,
            ) = lower_tail_variance_floor_penalty(
                critic_hidden,
                args.bottleneck_var_target,
                args.bottleneck_var_bottom_frac,
            )
            bottleneck_var_total = 0.5 * (
                actor_bottleneck_var_total + critic_bottleneck_var_total
            )
            bottleneck_bottom_std_avg = 0.5 * (
                actor_bottom_std_avg + critic_bottom_std_avg
            )
            bottleneck_bottom_below_target_frac = 0.5 * (
                actor_bottom_below_target_frac + critic_bottom_below_target_frac
            )
            if actor_dense_pre_ln is not None and critic_dense_pre_ln is not None:
                actor_pre_ln_total, actor_pre_ln_std = pre_ln_std_cap_penalty(
                    actor_dense_pre_ln,
                    args.bottleneck_pre_ln_max_std,
                )
                critic_pre_ln_total, critic_pre_ln_std = pre_ln_std_cap_penalty(
                    critic_dense_pre_ln,
                    args.bottleneck_pre_ln_max_std,
                )
                bottleneck_pre_ln_total = 0.5 * (
                    actor_pre_ln_total + critic_pre_ln_total
                )
                bottleneck_pre_ln_std = 0.5 * (
                    actor_pre_ln_std + critic_pre_ln_std
                )
        else:
            (
                vicreg_var_total,
                vicreg_var_unit_std_avg,
                vicreg_var_below_target_frac,
            ) = vicreg_variance_penalty(actor_hidden, args.vicreg_var_target)
            (
                bottleneck_var_total,
                bottleneck_bottom_std_avg,
                bottleneck_bottom_below_target_frac,
            ) = lower_tail_variance_floor_penalty(
                actor_hidden,
                args.bottleneck_var_target,
                args.bottleneck_var_bottom_frac,
            )
            if dense_pre_ln is not None:
                bottleneck_pre_ln_total, bottleneck_pre_ln_std = pre_ln_std_cap_penalty(
                    dense_pre_ln,
                    args.bottleneck_pre_ln_max_std,
                )

        actor_cond_mean, _ = actor.apply(
            params["actor"],
            jax.lax.stop_gradient(actor_hidden),
        )
        actor_cond_total, actor_cond_mu_std = actor_conditionality_penalty(
            actor_cond_mean,
            mb_advantages,
            args.actor_cond_target,
        )

        actor_mean, actor_logstd = actor.apply(params["actor"], actor_hidden)
        actor_noise_total, actor_noise_std_avg = actor_noise_std_cap_penalty(
            actor_logstd,
            args.actor_noise_max_std,
        )
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logstd))
        newlogprob = pi.log_prob(a)
        entropy = pi.entropy()
        newvalue = critic.apply(params["critic"], critic_hidden).squeeze(-1)

        logratio = newlogprob - logp
        ratio = jnp.exp(logratio)
        approx_kl = ((ratio - 1) - logratio).mean()

        if args.norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * jnp.clip(ratio, 1 - args.clip_eps, 1 + args.clip_eps)
        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

        if args.clip_vloss:
            v_loss_unclipped = (newvalue - mb_returns) ** 2
            v_clipped = mb_values + jnp.clip(newvalue - mb_values, -args.clip_eps, args.clip_eps)
            v_loss_clipped = (v_clipped - mb_returns) ** 2
            v_loss = 0.5 * jnp.maximum(v_loss_unclipped, v_loss_clipped).mean()
        else:
            v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

        entropy_loss = entropy.mean()
        total_loss = (
            pg_loss
            - args.ent_coef * entropy_loss
            + v_loss * args.vf_coef
            + sigreg_coef_current * sigreg_total
            + innovation_coef_current * innovation_total
            + vicreg_var_coef_current * vicreg_var_total
            + bottleneck_var_coef_current * bottleneck_var_total
            + bottleneck_pre_ln_coef_current * bottleneck_pre_ln_total
            + actor_cond_coef_current * actor_cond_total
            + actor_noise_coef_current * actor_noise_total
        )

        return total_loss, (
            pg_loss,
            v_loss,
            entropy_loss,
            jax.lax.stop_gradient(approx_kl),
            jax.lax.stop_gradient(sigreg_total),
            jax.lax.stop_gradient(sigreg_re),
            jax.lax.stop_gradient(sigreg_im),
            jax.lax.stop_gradient(innovation_total),
            jax.lax.stop_gradient(innovation_re),
            jax.lax.stop_gradient(innovation_im),
            jax.lax.stop_gradient(vicreg_var_total),
            jax.lax.stop_gradient(vicreg_var_unit_std_avg),
            jax.lax.stop_gradient(vicreg_var_below_target_frac),
            jax.lax.stop_gradient(bottleneck_var_total),
            jax.lax.stop_gradient(bottleneck_bottom_std_avg),
            jax.lax.stop_gradient(bottleneck_bottom_below_target_frac),
            jax.lax.stop_gradient(bottleneck_pre_ln_total),
            jax.lax.stop_gradient(bottleneck_pre_ln_std),
            jax.lax.stop_gradient(actor_cond_total),
            jax.lax.stop_gradient(actor_cond_mu_std),
            jax.lax.stop_gradient(actor_noise_total),
            jax.lax.stop_gradient(actor_noise_std_avg),
        )

    ppo_loss_grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)
    spectrum_batch_size = args.spectrum_batch_size or args.minibatch_size

    def collect_spectrum_diagnostics(
        params,
        storage: Storage,
        key: jax.random.PRNGKey,
        prefix: str,
        task_index: int,
        global_step_value: int,
        sigreg_coef_value: float,
        innovation_coef_value: float,
        vicreg_var_coef_value: float,
        bottleneck_var_coef_value: float,
        bottleneck_pre_ln_coef_value: float,
        actor_cond_coef_value: float,
        actor_noise_coef_value: float,
        phase: Optional[int] = None,
        closest_rollout: bool = False,
    ) -> dict:
        batch = flatten_batch_tree(storage, spectrum_batch_size)

        def spectrum_loss(loss_params, loss_batch):
            loss_value, _ = ppo_loss(
                loss_params,
                loss_batch.obs,
                loss_batch.next_obs,
                loss_batch.next_dones,
                loss_batch.actions,
                loss_batch.logprobs,
                loss_batch.advantages,
                loss_batch.returns,
                loss_batch.values,
                key,
                sigreg_coef_value,
                innovation_coef_value,
                vicreg_var_coef_value,
                bottleneck_var_coef_value,
                bottleneck_pre_ln_coef_value,
                actor_cond_coef_value,
                actor_noise_coef_value,
            )
            return loss_value

        logs = {
            f"{prefix}/task_index": float(task_index),
            f"{prefix}/global_step": float(global_step_value),
            f"{prefix}/lanczos_order": float(args.spectrum_lanczos_order),
            f"{prefix}/lanczos_draws": float(args.spectrum_lanczos_draws),
            f"{prefix}/closest_rollout": float(closest_rollout),
        }
        if phase is not None:
            logs[f"{prefix}/phase"] = float(phase)

        eigvals = hessian_eigenvalues(
            spectrum_loss,
            params,
            batch,
            key,
            order=args.spectrum_lanczos_order,
            draws=args.spectrum_lanczos_draws,
        )
        add_hessian_histogram(
            logs,
            f"{prefix}/hessian",
            eigvals,
            bins=args.spectrum_hist_bins,
        )

        def get_param(path):
            node = params
            for part in path:
                if part not in node:
                    return None
                node = node[part]
            return node

        def add_weight_if_present(name, path):
            matrix = get_param(path)
            if matrix is not None and getattr(matrix, "ndim", 0) >= 2:
                add_matrix_singular_histogram(
                    logs,
                    f"{prefix}/weights/{name}",
                    matrix,
                    bins=args.spectrum_hist_bins,
                )

        for path in encoder_final_stiefel_kernel_paths(encoder_type):
            if path[0] == "network" and path[1] == "params":
                add_weight_if_present(
                    "/".join(("network",) + path[2:-1]),
                    path,
                )
        add_weight_if_present("actor/Dense_0", ("actor", "params", "Dense_0", "kernel"))
        add_weight_if_present("actor/Dense_1", ("actor", "params", "Dense_1", "kernel"))
        add_weight_if_present(
            "actor/CRATEFeedForward_0",
            ("actor", "params", "CRATEFeedForward_0", "weight"),
        )
        add_weight_if_present("critic/Dense_0", ("critic", "params", "Dense_0", "kernel"))
        add_weight_if_present("critic/Dense_1", ("critic", "params", "Dense_1", "kernel"))
        add_weight_if_present(
            "critic/CRATEFeedForward_0",
            ("critic", "params", "CRATEFeedForward_0", "weight"),
        )

        encoder_debug = encode_with_intermediates(params["network"], batch.obs)
        if encoder_type in {"split_cnn", "separate_cnn"}:
            actor_hidden = encoder_debug["actor_hidden"]
            critic_hidden = encoder_debug["critic_hidden"]
        else:
            if "policy_hidden" in encoder_debug:
                actor_hidden = encoder_debug["policy_hidden"]
            else:
                actor_hidden = encoder_debug["hidden"]
            critic_hidden = actor_hidden
        actor_debug = actor_debug_apply(
            params["actor"],
            actor_hidden,
            return_intermediates=True,
        )
        critic_debug = critic_debug_apply(
            params["critic"],
            critic_hidden,
            return_intermediates=True,
        )

        def add_activation_if_present(name, values, key_name):
            if key_name in values:
                add_histogram(
                    logs,
                    f"{prefix}/activations/{name}",
                    values[key_name],
                    bins=args.spectrum_hist_bins,
                )

        if encoder_type in {"split_cnn", "separate_cnn"}:
            add_activation_if_present("network/actor_hidden", encoder_debug, "actor_hidden")
            add_activation_if_present("network/critic_hidden", encoder_debug, "critic_hidden")
        else:
            add_activation_if_present("network/hidden", encoder_debug, "hidden")
            add_activation_if_present("network/policy_hidden", encoder_debug, "policy_hidden")
        add_activation_if_present("actor/Dense_1", actor_debug, "Dense_1")
        add_activation_if_present("actor/CRATEFeedForward_0", actor_debug, "CRATEFeedForward_0")
        add_activation_if_present("actor/Dense_2", actor_debug, "Dense_2")
        add_activation_if_present("critic/Dense_1", critic_debug, "Dense_1")
        add_activation_if_present("critic/CRATEFeedForward_0", critic_debug, "CRATEFeedForward_0")
        return logs

    @jax.jit
    def update_ppo(
        agent_state: TrainState,
        storage: Storage,
        key: jax.random.PRNGKey,
        sigreg_coef_current: jnp.ndarray,
        innovation_coef_current: jnp.ndarray,
        vicreg_var_coef_current: jnp.ndarray,
        bottleneck_var_coef_current: jnp.ndarray,
        bottleneck_pre_ln_coef_current: jnp.ndarray,
        actor_cond_coef_current: jnp.ndarray,
        actor_noise_coef_current: jnp.ndarray,
    ):
        """PPO update."""
        def update_epoch(carry, unused_inp):
            agent_state, key, last_grads = carry
            key, subkey, aug_key = jax.random.split(key, 3)

            def flatten(x):
                return x.reshape((-1,) + x.shape[2:])

            def convert_data(x: jnp.ndarray):
                x = jax.random.permutation(subkey, x)
                x = jnp.reshape(x, (args.num_minibatches, -1) + x.shape[1:])
                return x

            flatten_storage = jax.tree_util.tree_map(flatten, storage)
            shuffled_storage = jax.tree_util.tree_map(convert_data, flatten_storage)
            aug_keys = jax.random.split(aug_key, args.num_minibatches)

            def update_minibatch(carry, inputs):
                agent_state, _ = carry
                minibatch, mb_aug_key = inputs

                (
                    loss,
                    (
                        pg_loss,
                        v_loss,
                        entropy_loss,
                        approx_kl,
                        sigreg_total,
                        sigreg_re,
                        sigreg_im,
                        innovation_total,
                        innovation_re,
                        innovation_im,
                        vicreg_var_total,
                        vicreg_var_unit_std_avg,
                        vicreg_var_below_target_frac,
                        bottleneck_var_total,
                        bottleneck_bottom_std_avg,
                        bottleneck_bottom_below_target_frac,
                        bottleneck_pre_ln_total,
                        bottleneck_pre_ln_std,
                        actor_cond_total,
                        actor_cond_mu_std,
                        actor_noise_total,
                        actor_noise_std_avg,
                    ),
                ), grads = ppo_loss_grad_fn(
                    agent_state.params,
                    minibatch.obs,
                    minibatch.next_obs,
                    minibatch.next_dones,
                    minibatch.actions,
                    minibatch.logprobs,
                    minibatch.advantages,
                    minibatch.returns,
                    minibatch.values,
                    mb_aug_key,
                    sigreg_coef_current,
                    innovation_coef_current,
                    vicreg_var_coef_current,
                    bottleneck_var_coef_current,
                    bottleneck_pre_ln_coef_current,
                    actor_cond_coef_current,
                    actor_noise_coef_current,
                )

                grad_norm = optax.global_norm(grads)
                agent_state = agent_state.apply_gradients(grads=grads)

                return (agent_state, grads), (
                    loss,
                    pg_loss,
                    v_loss,
                    entropy_loss,
                    approx_kl,
                    sigreg_total,
                    sigreg_re,
                    sigreg_im,
                    innovation_total,
                    innovation_re,
                    innovation_im,
                    vicreg_var_total,
                    vicreg_var_unit_std_avg,
                    vicreg_var_below_target_frac,
                    bottleneck_var_total,
                    bottleneck_bottom_std_avg,
                    bottleneck_bottom_below_target_frac,
                    bottleneck_pre_ln_total,
                    bottleneck_pre_ln_std,
                    actor_cond_total,
                    actor_cond_mu_std,
                    actor_noise_total,
                    actor_noise_std_avg,
                    grad_norm,
                )

            (
                (agent_state, last_grads),
                (
                    loss,
                    pg_loss,
                    v_loss,
                    entropy_loss,
                    approx_kl,
                    sigreg_total,
                    sigreg_re,
                    sigreg_im,
                    innovation_total,
                    innovation_re,
                    innovation_im,
                    vicreg_var_total,
                    vicreg_var_unit_std_avg,
                    vicreg_var_below_target_frac,
                    bottleneck_var_total,
                    bottleneck_bottom_std_avg,
                    bottleneck_bottom_below_target_frac,
                    bottleneck_pre_ln_total,
                    bottleneck_pre_ln_std,
                    actor_cond_total,
                    actor_cond_mu_std,
                    actor_noise_total,
                    actor_noise_std_avg,
                    grad_norm,
                ),
            ) = jax.lax.scan(
                update_minibatch,
                (agent_state, last_grads),
                (shuffled_storage, aug_keys),
            )
            return (agent_state, key, last_grads), (
                loss,
                pg_loss,
                v_loss,
                entropy_loss,
                approx_kl,
                sigreg_total,
                sigreg_re,
                sigreg_im,
                innovation_total,
                innovation_re,
                innovation_im,
                vicreg_var_total,
                vicreg_var_unit_std_avg,
                vicreg_var_below_target_frac,
                bottleneck_var_total,
                bottleneck_bottom_std_avg,
                bottleneck_bottom_below_target_frac,
                bottleneck_pre_ln_total,
                bottleneck_pre_ln_std,
                actor_cond_total,
                actor_cond_mu_std,
                actor_noise_total,
                actor_noise_std_avg,
                grad_norm,
            )

        init_grads = jax.tree_util.tree_map(jnp.zeros_like, agent_state.params)
        (
            (agent_state, key, final_grads),
            (
                loss,
                pg_loss,
                v_loss,
                entropy_loss,
                approx_kl,
                sigreg_total,
                sigreg_re,
                sigreg_im,
                innovation_total,
                innovation_re,
                innovation_im,
                vicreg_var_total,
                vicreg_var_unit_std_avg,
                vicreg_var_below_target_frac,
                bottleneck_var_total,
                bottleneck_bottom_std_avg,
                bottleneck_bottom_below_target_frac,
                bottleneck_pre_ln_total,
                bottleneck_pre_ln_std,
                actor_cond_total,
                actor_cond_mu_std,
                actor_noise_total,
                actor_noise_std_avg,
                grad_norm,
            ),
        ) = jax.lax.scan(
            update_epoch,
            (agent_state, key, init_grads),
            (),
            length=args.update_epochs,
        )
        return (
            agent_state,
            loss,
            pg_loss,
            v_loss,
            entropy_loss,
            approx_kl,
            sigreg_total,
            sigreg_re,
            sigreg_im,
            innovation_total,
            innovation_re,
            innovation_im,
            vicreg_var_total,
            vicreg_var_unit_std_avg,
            vicreg_var_below_target_frac,
            bottleneck_var_total,
            bottleneck_bottom_std_avg,
            bottleneck_bottom_below_target_frac,
            bottleneck_pre_ln_total,
            bottleneck_pre_ln_std,
            actor_cond_total,
            actor_cond_mu_std,
            actor_noise_total,
            actor_noise_std_avg,
            grad_norm,
            final_grads,
            key,
        )

    # Start the game
    global_step = 0
    start_time = time.time()

    # Reset environment
    key, reset_key = jax.random.split(key)
    reset_rngs = jax.random.split(reset_key, args.n_envs)
    env_state = envs.reset_with_sys(current_sys, reset_rngs)

    # Initialize frame stack with initial observation
    frame_stack = FrameStack.create(args.n_envs, args.frame_stack, raw_obs_shape)
    frame_stack = frame_stack.reset(env_state.pixels)
    next_obs = frame_stack.get_stacked()
    next_done = jnp.zeros(args.n_envs, dtype=jnp.bool_)

    # Initialize reward normalizer
    reward_normalizer = RewardNormalizer.create(n_envs=args.n_envs, gamma=args.gamma)

    if args.slippery_ant:
        slippery_friction_schedule_jnp = jnp.asarray(slippery_friction_schedule, dtype=jnp.float32)
        slippery_default_geom_friction_jnp = jnp.asarray(
            slippery_default_geom_friction,
            dtype=jnp.float32,
        )

        def sys_for_rollout_step(current_sys, rollout_start_policy_step, step):
            physical_timestep = (rollout_start_policy_step + step) * args.action_repeat
            phase = physical_timestep // args.slippery_change_every
            schedule_phase = jnp.minimum(
                jnp.maximum(phase - 1, 0),
                slippery_friction_schedule_jnp.shape[0] - 1,
            )
            friction = slippery_friction_schedule_jnp[schedule_phase]
            scheduled_geom_friction = slippery_default_geom_friction_jnp.at[:, 0].set(friction)
            new_geom_friction = jax.lax.select(
                phase > 0,
                scheduled_geom_friction,
                slippery_default_geom_friction_jnp,
            )
            return current_sys.replace(geom_friction=new_geom_friction)
    else:
        def sys_for_rollout_step(current_sys, rollout_start_policy_step, step):
            return current_sys

    def step_once(current_sys, rollout_start_policy_step, carry, step):
        agent_state, episode_stats, reward_norm, fs, env_state, obs, done, key = carry
        action, logprob, value, key = get_action_and_value(agent_state, obs, key)

        step_sys = sys_for_rollout_step(current_sys, rollout_start_policy_step, step)
        env_state = envs.step_with_sys(step_sys, env_state, action)
        raw_obs = env_state.pixels
        raw_reward = env_state.reward
        next_done_local = env_state.done.astype(jnp.bool_)

        fs = fs.push(raw_obs)
        fs = fs.reset(raw_obs, next_done_local)
        next_obs_local = fs.get_stacked()

        reward_norm = reward_norm.update(raw_reward, next_done_local.astype(jnp.float32))
        reward = reward_norm.normalize(raw_reward)

        new_episode_return = episode_stats.episode_returns + raw_reward
        new_episode_length = episode_stats.episode_lengths + 1
        episode_stats = episode_stats.replace(
            episode_returns=jnp.where(next_done_local, 0.0, new_episode_return),
            episode_lengths=jnp.where(next_done_local, 0, new_episode_length).astype(jnp.int32),
            returned_episode_returns=jnp.where(
                next_done_local,
                new_episode_return,
                episode_stats.returned_episode_returns,
            ),
            returned_episode_lengths=jnp.where(
                next_done_local,
                new_episode_length,
                episode_stats.returned_episode_lengths,
            ).astype(jnp.int32),
        )

        storage = Storage(
            obs=obs,
            next_obs=next_obs_local,
            actions=action,
            logprobs=logprob,
            dones=done,
            next_dones=next_done_local,
            values=value,
            rewards=reward,
            raw_rewards=raw_reward,
            returns=jnp.zeros_like(reward),
            advantages=jnp.zeros_like(reward),
        )
        return (
            agent_state,
            episode_stats,
            reward_norm,
            fs,
            env_state,
            next_obs_local,
            next_done_local,
            key,
        ), storage

    def rollout(
        current_sys,
        rollout_start_policy_step,
        agent_state,
        episode_stats,
        reward_norm,
        fs,
        env_state,
        next_obs,
        next_done,
        key,
        max_steps,
    ):
        (
            agent_state,
            episode_stats,
            reward_norm,
            fs,
            env_state,
            next_obs,
            next_done,
            key,
        ), storage = jax.lax.scan(
            partial(step_once, current_sys, rollout_start_policy_step),
            (agent_state, episode_stats, reward_norm, fs, env_state, next_obs, next_done, key),
            jnp.arange(max_steps),
        )
        return agent_state, episode_stats, reward_norm, fs, env_state, next_obs, next_done, storage, key

    rollout = partial(rollout, max_steps=args.num_steps)
    rollout = jax.jit(rollout)

    print("\nStarting training...")
    cumulative_episodic_return = 0.0

    def collect_probe_plot_images(output_dir: str) -> dict:
        """Load probe plot PNGs as wandb.Image for logging, if tracking is enabled."""
        if not args.track or not args.probe_save_plots or output_dir is None:
            return {}

        plot_files = {
            "probe/plots/r2_vs_k": f"{args.env_name}_r2_vs_k.png",
            "probe/plots/scree": f"{args.env_name}_scree.png",
            "probe/plots/corr_heatmap": f"{args.env_name}_corr_heatmap.png",
            "additive/plots/parity": f"{args.env_name}_additive_parity.png",
            "additive/plots/cosine_gram": f"{args.env_name}_cosine_gram.png",
            "additive/plots/raw_gram": f"{args.env_name}_raw_gram.png",
            "additive/plots/component_norms": f"{args.env_name}_component_norms.png",
        }

        image_logs = {}
        for log_key, file_name in plot_files.items():
            file_path = os.path.join(output_dir, file_name)
            if os.path.exists(file_path):
                image_logs[log_key] = wandb.Image(file_path)
        return image_logs

    # Track next probe threshold as an absolute step count so LCM cadence issues
    # don't arise when global_step jumps by (num_steps * n_envs) each iteration.
    next_probe_step = args.probe_interval if args.probe_interval > 0 else float("inf")
    # Probe metrics are buffered here and flushed into the main wandb.log call
    # so we never call wandb.log twice at the same step.
    pending_probe_metrics: dict = {}

    # Probe at initialisation (step 0) — establishes random-encoder baseline.
    if args.probe_interval > 0:
        from linear_probe import run_online_probe
        import pickle
        probe_init_output = os.path.join(args.probe_output_dir, "step_0") if args.probe_output_dir else None
        probe_init_metrics = run_online_probe(
            network=network,
            network_params=agent_state.params["network"],
            envs=envs,
            args_dict=vars(args),
            n_eval_envs=args.n_envs,
            n_eval_steps=args.probe_n_eval_steps,
            seed=args.data_seed,
            output_dir=probe_init_output if args.probe_save_plots else None,
        )
        print(f"[probe] step=0 (init) metrics={probe_init_metrics}")
        if args.save_checkpoint:
            ckpt_dir = os.path.join(args.checkpoint_dir, run_name)
            os.makedirs(ckpt_dir, exist_ok=True)
            params_bytes = flax.serialization.to_bytes(agent_state.params)
            ckpt_path = os.path.join(ckpt_dir, "params_step0.pkl")
            with open(ckpt_path, "wb") as f:
                pickle.dump({"params_bytes": params_bytes, "args": vars(args),
                             "global_step": 0}, f)
            print(f"Checkpoint saved to {ckpt_path}")
        if args.track:
            probe_init_media = collect_probe_plot_images(probe_init_output)
            wandb.log({**probe_init_metrics, **probe_init_media}, step=0)

    for iteration in range(1, args.num_updates + 1):
        if (
            continual_config is not None
            and iteration > 1
            and (iteration - 1) % updates_per_task == 0
        ):
            task_index = (iteration - 1) // updates_per_task
            current_task = make_dynamics_task(
                continual_config,
                base_sys,
                task_index=task_index,
                schedule_seed=schedule_seed,
            )
            current_task_index = task_index
            current_sys = apply_dynamics_task(base_sys, current_task)
            task_start_step = global_step

            key, reset_key = jax.random.split(key)
            reset_rngs = jax.random.split(reset_key, args.n_envs)
            env_state = envs.reset_with_sys(current_sys, reset_rngs)
            frame_stack = frame_stack.reset(env_state.pixels)
            next_obs = frame_stack.get_stacked()
            next_done = jnp.zeros(args.n_envs, dtype=jnp.bool_)
            episode_stats = EpisodeStatistics(
                episode_returns=jnp.zeros(args.n_envs, dtype=jnp.float32),
                episode_lengths=jnp.zeros(args.n_envs, dtype=jnp.int32),
                returned_episode_returns=jnp.zeros(args.n_envs, dtype=jnp.float32),
                returned_episode_lengths=jnp.zeros(args.n_envs, dtype=jnp.int32),
            )
            reward_normalizer = reward_normalizer.replace(
                discounted_return=jnp.zeros(args.n_envs),
            )

            print(
                f"continual_switch task={task_index} step={global_step} "
                f"default={current_task.is_default} values="
                f"{json.dumps(current_task.values, sort_keys=True)}"
            )

            if args.track:
                import wandb
                wandb.log(
                    continual_log_metrics(
                        current_task,
                        switch_every_env_steps=continual_config.switch_every_env_steps,
                        task_start_step=task_start_step,
                        global_step=global_step,
                    ),
                    step=global_step,
                )

        iteration_time_start = time.time()
        (
            agent_state,
            episode_stats,
            reward_normalizer,
            frame_stack,
            env_state,
            next_obs,
            next_done,
            storage,
            key,
        ) = rollout(
            current_sys,
            global_step // args.n_envs,
            agent_state,
            episode_stats,
            reward_normalizer,
            frame_stack,
            env_state,
            next_obs,
            next_done,
            key,
        )
        global_step += args.num_steps * args.n_envs
        storage = compute_gae(agent_state, next_obs, next_done, storage)


        sigreg_coef_current = sigreg_coef_from_update(
            args.sigreg_coef,
            iteration - 1,
            args.sigreg_warmup_updates,
            args.sigreg_ramp_updates,
        )
        innovation_coef_current = sigreg_coef_from_update(
            args.innovation_coef,
            iteration - 1,
            args.innovation_warmup_updates,
            args.innovation_ramp_updates,
        )
        vicreg_var_coef_current = sigreg_coef_from_update(
            args.vicreg_var_coef,
            iteration - 1,
            args.vicreg_var_warmup_updates,
            args.vicreg_var_ramp_updates,
        )
        bottleneck_var_coef_current = early_window_coef(
            args.bottleneck_var_coef,
            iteration - 1,
            args.bottleneck_reg_stop_updates,
        )
        bottleneck_pre_ln_coef_current = early_window_coef(
            args.bottleneck_pre_ln_coef,
            iteration - 1,
            args.bottleneck_reg_stop_updates,
        )
        actor_cond_coef_current = early_window_coef(
            args.actor_cond_coef,
            iteration - 1,
            args.actor_cond_stop_updates,
        )
        actor_noise_coef_current = early_window_coef(
            args.actor_noise_coef,
            iteration - 1,
            args.actor_noise_stop_updates,
        )
        if (
            args.track
            and continual_config is not None
            and (iteration - 1) % updates_per_task == 0
        ):
            import wandb
            key, spectrum_key = jax.random.split(key)
            wandb.log(
                collect_spectrum_diagnostics(
                    agent_state.params,
                    storage,
                    spectrum_key,
                    "spectrum/task_start",
                    current_task_index,
                    global_step,
                    sigreg_coef_current,
                    innovation_coef_current,
                    vicreg_var_coef_current,
                    bottleneck_var_coef_current,
                    bottleneck_pre_ln_coef_current,
                    actor_cond_coef_current,
                    actor_noise_coef_current,
                ),
                step=global_step,
            )
        (
            agent_state,
            loss,
            pg_loss,
            v_loss,
            entropy_loss,
            approx_kl,
            sigreg_total,
            sigreg_re,
            sigreg_im,
            innovation_total,
            innovation_re,
            innovation_im,
            vicreg_var_total,
            vicreg_var_unit_std_avg,
            vicreg_var_below_target_frac,
            bottleneck_var_total,
            bottleneck_bottom_std_avg,
            bottleneck_bottom_below_target_frac,
            bottleneck_pre_ln_total,
            bottleneck_pre_ln_std,
            actor_cond_total,
            actor_cond_mu_std,
            actor_noise_total,
            actor_noise_std_avg,
            grad_norm,
            final_grads,
            key,
        ) = update_ppo(
            agent_state,
            storage,
            key,
            sigreg_coef_current,
            innovation_coef_current,
            vicreg_var_coef_current,
            bottleneck_var_coef_current,
            bottleneck_pre_ln_coef_current,
            actor_cond_coef_current,
            actor_noise_coef_current,
        )
        if (
            args.track
            and continual_config is not None
            and (iteration % updates_per_task == 0 or iteration == args.num_updates)
        ):
            import wandb
            key, spectrum_key = jax.random.split(key)
            wandb.log(
                collect_spectrum_diagnostics(
                    agent_state.params,
                    storage,
                    spectrum_key,
                    "spectrum/task_end",
                    current_task_index,
                    global_step,
                    sigreg_coef_current,
                    innovation_coef_current,
                    vicreg_var_coef_current,
                    bottleneck_var_coef_current,
                    bottleneck_pre_ln_coef_current,
                    actor_cond_coef_current,
                    actor_noise_coef_current,
                ),
                step=global_step,
            )

        if args.track and args.slippery_ant:
            import wandb
            slippery_phase = slippery_phase_from_timestep(
                timestep=int((global_step // args.n_envs) * args.action_repeat),
                change_every=args.slippery_change_every,
                num_seeded_phases=int(slippery_friction_schedule.shape[0]),
            )
            if last_slippery_phase_logged is None:
                key, spectrum_key = jax.random.split(key)
                wandb.log(
                    collect_spectrum_diagnostics(
                        agent_state.params,
                        storage,
                        spectrum_key,
                        "spectrum/task_start",
                        slippery_phase,
                        global_step,
                        sigreg_coef_current,
                        innovation_coef_current,
                        vicreg_var_coef_current,
                        bottleneck_var_coef_current,
                        bottleneck_pre_ln_coef_current,
                        actor_cond_coef_current,
                        actor_noise_coef_current,
                        phase=slippery_phase,
                        closest_rollout=True,
                    ),
                    step=global_step,
                )
                last_slippery_phase_logged = slippery_phase
            elif slippery_phase != last_slippery_phase_logged:
                key, end_key, start_key = jax.random.split(key, 3)
                wandb.log(
                    collect_spectrum_diagnostics(
                        agent_state.params,
                        storage,
                        end_key,
                        "spectrum/task_end",
                        last_slippery_phase_logged,
                        global_step,
                        sigreg_coef_current,
                        innovation_coef_current,
                        vicreg_var_coef_current,
                        bottleneck_var_coef_current,
                        bottleneck_pre_ln_coef_current,
                        actor_cond_coef_current,
                        actor_noise_coef_current,
                        phase=last_slippery_phase_logged,
                        closest_rollout=True,
                    ),
                    step=global_step,
                )
                wandb.log(
                    collect_spectrum_diagnostics(
                        agent_state.params,
                        storage,
                        start_key,
                        "spectrum/task_start",
                        slippery_phase,
                        global_step,
                        sigreg_coef_current,
                        innovation_coef_current,
                        vicreg_var_coef_current,
                        bottleneck_var_coef_current,
                        bottleneck_pre_ln_coef_current,
                        actor_cond_coef_current,
                        actor_noise_coef_current,
                        phase=slippery_phase,
                        closest_rollout=True,
                    ),
                    step=global_step,
                )
                last_slippery_phase_logged = slippery_phase
            if iteration == args.num_updates:
                key, spectrum_key = jax.random.split(key)
                wandb.log(
                    collect_spectrum_diagnostics(
                        agent_state.params,
                        storage,
                        spectrum_key,
                        "spectrum/task_end",
                        last_slippery_phase_logged,
                        global_step,
                        sigreg_coef_current,
                        innovation_coef_current,
                        vicreg_var_coef_current,
                        bottleneck_var_coef_current,
                        bottleneck_pre_ln_coef_current,
                        actor_cond_coef_current,
                        actor_noise_coef_current,
                        phase=last_slippery_phase_logged,
                        closest_rollout=True,
                    ),
                    step=global_step,
                )

        # Online linear probe — fire when global_step crosses the next threshold.
        # Using >= avoids the LCM cadence problem that % would cause when
        # global_step jumps by (num_steps * n_envs = 1280) each iteration.
        if global_step >= next_probe_step:
            from linear_probe import run_online_probe
            probe_output = os.path.join(args.probe_output_dir, f"step_{global_step}")
            probe_metrics = run_online_probe(
                network=network,
                network_params=agent_state.params["network"],
                envs=envs,
                args_dict=vars(args),
                n_eval_envs=args.n_envs,
                n_eval_steps=args.probe_n_eval_steps,
                seed=args.data_seed + iteration,
                output_dir=probe_output if args.probe_save_plots else None,
            )
            print(f"[probe] step={global_step} metrics={probe_metrics}")
            # Buffer into pending_probe_metrics; flushed in the main wandb.log
            # call below so we never call wandb.log twice at the same step.
            pending_probe_metrics.update(probe_metrics)
            if args.track:
                pending_probe_metrics.update(collect_probe_plot_images(probe_output))

            # Save a checkpoint at each probe point so the probe can be re-run
            # offline on any training snapshot without retraining.
            if args.save_checkpoint:
                import pickle
                ckpt_dir = os.path.join(args.checkpoint_dir, run_name)
                os.makedirs(ckpt_dir, exist_ok=True)
                params_bytes = flax.serialization.to_bytes(agent_state.params)
                ckpt_path = os.path.join(ckpt_dir, f"params_step{global_step}.pkl")
                with open(ckpt_path, "wb") as f:
                    pickle.dump({"params_bytes": params_bytes, "args": vars(args),
                                 "global_step": global_step}, f)
                print(f"Checkpoint saved to {ckpt_path}")

            next_probe_step += args.probe_interval

        if iteration % args.log_interval == 0:
            avg_episodic_return = np.mean(jax.device_get(episode_stats.returned_episode_returns))
            avg_episodic_length = np.mean(jax.device_get(episode_stats.returned_episode_lengths))
            cumulative_episodic_return += avg_episodic_return
            sps = int(global_step / (time.time() - start_time))
            sps_update = int(args.n_envs * args.num_steps / (time.time() - iteration_time_start))
            slippery_metrics = None
            if args.slippery_ant:
                slippery_metrics = latest_slippery_pixel_metrics(
                    global_step,
                    args,
                    slippery_default_geom_friction,
                    slippery_friction_schedule,
                )

            base_msg = (
                f"update={iteration} step={global_step} "
                f"ep_return={avg_episodic_return:.1f} "
                f"ep_len={avg_episodic_length * args.action_repeat:.0f} "
                f"loss={loss[-1, -1].item():.4f} "
                f"SPS={sps}"
            )
            if slippery_metrics is not None:
                base_msg += (
                    f" friction={slippery_metrics['slippery/friction_slide_first_env']:.4g}"
                    f" phase={slippery_metrics['slippery/phase_first_env']}"
                )
            print(base_msg)

            if args.track:
                opt_step = iteration * args.update_epochs * args.num_minibatches
                if args.anneal_lr:
                    update_idx = opt_step // steps_per_update
                    encoder_lr_current = lr_from_update(
                        args.encoder_lr,
                        update_idx,
                        args.encoder_warmup_updates,
                    )
                    heads_adam_lr_current = lr_from_update(
                        args.heads_adam_lr,
                        update_idx,
                        warmup_updates=0,
                    )
                else:
                    encoder_lr_current = args.encoder_lr
                    heads_adam_lr_current = args.heads_adam_lr
                log_dict = {
                    "global_step": global_step,
                    "charts/avg_episodic_return": avg_episodic_return,
                    "charts/cumulative_episodic_return": cumulative_episodic_return,
                    "charts/avg_episodic_length": avg_episodic_length * args.action_repeat,
                    "charts/encoder_lr": float(encoder_lr_current),
                    "charts/heads_adam_lr": float(heads_adam_lr_current),
                    "charts/heads_stiefel_lr": args.heads_stiefel_lr,
                    "charts/heads_muon_lr": args.heads_muon_lr,
                    "charts/head_optimizer_is_adam": float(args.heads_optimizer == "adam"),
                    "charts/head_optimizer_is_stiefel": float(args.heads_optimizer == "stiefel"),
                    "charts/head_optimizer_is_stiefel_admm": float(args.heads_optimizer == "stiefel_admm"),
                    "charts/head_optimizer_is_muon": float(args.heads_optimizer == "muon"),
                    "charts/head_optimizer_is_aurora": float(args.heads_optimizer == "aurora"),
                    "charts/SPS": sps,
                    "charts/SPS_update": sps_update,
                    "losses/value_loss": v_loss[-1, -1].item(),
                    "losses/policy_loss": pg_loss[-1, -1].item(),
                    "losses/entropy": entropy_loss[-1, -1].item(),
                    "losses/approx_kl": approx_kl[-1, -1].item(),
                    "losses/grad_norm": grad_norm[-1, -1].item(),
                    "losses/loss": loss[-1, -1].item(),
                    "losses/sigreg_total": sigreg_total[-1, -1].item(),
                    "losses/sigreg_re": sigreg_re[-1, -1].item(),
                    "losses/sigreg_im": sigreg_im[-1, -1].item(),
                    "losses/sigreg_coef": float(sigreg_coef_current),
                    "losses/innovation_total": innovation_total[-1, -1].item(),
                    "losses/innovation_re": innovation_re[-1, -1].item(),
                    "losses/innovation_im": innovation_im[-1, -1].item(),
                    "losses/innovation_coef": float(innovation_coef_current),
                    "losses/vicreg_var_total": vicreg_var_total[-1, -1].item(),
                    "losses/vicreg_var_coef": float(vicreg_var_coef_current),
                    "vicreg_var/unit_std_avg": vicreg_var_unit_std_avg[-1, -1].item(),
                    "vicreg_var/below_target_frac": vicreg_var_below_target_frac[-1, -1].item(),
                    "vicreg_var/target": args.vicreg_var_target,
                    "losses/bottleneck_var_total": bottleneck_var_total[-1, -1].item(),
                    "losses/bottleneck_var_coef": float(bottleneck_var_coef_current),
                    "losses/bottleneck_pre_ln_total": bottleneck_pre_ln_total[-1, -1].item(),
                    "losses/bottleneck_pre_ln_coef": float(bottleneck_pre_ln_coef_current),
                    "bottleneck/bottom_std_avg": bottleneck_bottom_std_avg[-1, -1].item(),
                    "bottleneck/bottom_below_target_frac": bottleneck_bottom_below_target_frac[-1, -1].item(),
                    "bottleneck/var_target": args.bottleneck_var_target,
                    "bottleneck/var_bottom_frac": args.bottleneck_var_bottom_frac,
                    "bottleneck/pre_ln_std_train": bottleneck_pre_ln_std[-1, -1].item(),
                    "bottleneck/pre_ln_max_std": args.bottleneck_pre_ln_max_std,
                    "bottleneck/reg_stop_updates": args.bottleneck_reg_stop_updates,
                    "losses/actor_cond_total": actor_cond_total[-1, -1].item(),
                    "losses/actor_cond_coef": float(actor_cond_coef_current),
                    "actor_cond/mu_std_advw": actor_cond_mu_std[-1, -1].item(),
                    "actor_cond/target": args.actor_cond_target,
                    "actor_cond/stop_updates": args.actor_cond_stop_updates,
                    "losses/actor_noise_total": actor_noise_total[-1, -1].item(),
                    "losses/actor_noise_coef": float(actor_noise_coef_current),
                    "actor_noise/std_avg": actor_noise_std_avg[-1, -1].item(),
                    "actor_noise/max_std": args.actor_noise_max_std,
                    "actor_noise/stop_updates": args.actor_noise_stop_updates,
                }

                if continual_config is not None:
                    log_dict.update(
                        continual_log_metrics(
                            current_task,
                            switch_every_env_steps=continual_config.switch_every_env_steps,
                            task_start_step=task_start_step,
                            global_step=global_step,
                        )
                    )
                if slippery_metrics is not None:
                    log_dict.update(slippery_metrics)

                traj_metrics = rollout_trajectory_metrics(
                    storage,
                    frame_stack=args.frame_stack,
                    obs_channels=raw_obs_shape[2],
                    max_action=args.max_action,
                )
                for k, v in traj_metrics.items():
                    log_dict[k] = float(v)

                # Debug metrics for encoder representations
                if args.debug_repr:
                    sample_obs = storage.obs[0, :256]
                    sample_next_obs = storage.next_obs[0, :256]
                    sample_actions = storage.actions[0, :256]
                    sample_next_done = storage.next_dones[0, :256]
                    if encoder_type in {"resnet", "cnn", "cnn_swish", "cnn_swish_ta", "cnn_swish_tb", "cnn_swish_tc", "crate_cnn", "crate_cnn_tanh", "crate_cnn_tanh_resid", "cnn_swish_tanh", "cnn_swish_tanh_resid", "cnn_swish_tanh_resid_learned", "split_cnn", "separate_cnn", "sigreg_cnn", "innovation_cnn", "innovation_direct_cnn"}:
                        cnn_debug = encode_with_intermediates(
                            agent_state.params["network"],
                            sample_obs,
                        )
                        if encoder_type in {"split_cnn", "separate_cnn"}:
                            hidden = cnn_debug["actor_hidden"]
                            critic_hidden_debug = cnn_debug["critic_hidden"]
                            actor_mean_debug, actor_logstd_debug = actor.apply(
                                agent_state.params["actor"],
                                hidden,
                            )
                            values_debug = critic.apply(
                                agent_state.params["critic"],
                                critic_hidden_debug,
                            ).squeeze(-1)
                            if encoder_type == "separate_cnn":
                                actor_dense_kernel = agent_state.params["network"]["params"]["actor_encoder"]["Dense_0"]["kernel"]
                                actor_dense_grad = final_grads["network"]["params"]["actor_encoder"]["Dense_0"]["kernel"]
                                critic_dense_kernel = agent_state.params["network"]["params"]["critic_encoder"]["Dense_0"]["kernel"]
                                critic_dense_grad = final_grads["network"]["params"]["critic_encoder"]["Dense_0"]["kernel"]
                            else:
                                actor_dense_kernel = agent_state.params["network"]["params"]["actor_dense"]["kernel"]
                                actor_dense_grad = final_grads["network"]["params"]["actor_dense"]["kernel"]
                                critic_dense_kernel = agent_state.params["network"]["params"]["critic_dense"]["kernel"]
                                critic_dense_grad = final_grads["network"]["params"]["critic_dense"]["kernel"]
                            dense_metrics = cnn_dense_metrics(
                                cnn_debug["actor_dense_pre_ln"],
                                hidden,
                                actor_dense_kernel,
                                actor_dense_grad,
                            )
                            for k, v in dense_metrics.items():
                                log_dict[k] = float(v)
                            critic_dense_metrics = cnn_dense_metrics(
                                cnn_debug["critic_dense_pre_ln"],
                                critic_hidden_debug,
                                critic_dense_kernel,
                                critic_dense_grad,
                                prefix="critic_cnn_dense",
                            )
                            for k, v in critic_dense_metrics.items():
                                log_dict[k] = float(v)
                            critic_repr_metrics = encoder_repr_metrics(critic_hidden_debug)
                            for k, v in critic_repr_metrics.items():
                                log_dict[k.replace("repr/", "critic_repr/")] = float(v)
                        else:
                            policy_hidden_debug = cnn_debug.get("policy_hidden", cnn_debug["hidden"])
                            hidden = policy_hidden_debug
                            actor_mean_debug, actor_logstd_debug = actor.apply(
                                agent_state.params["actor"],
                                policy_hidden_debug,
                            )
                            values_debug = critic.apply(
                                agent_state.params["critic"],
                                policy_hidden_debug,
                            ).squeeze(-1)
                            if encoder_type == "innovation_direct_cnn":
                                dense_metrics = cnn_dense_metrics(
                                    cnn_debug["projector_pre_ln"],
                                    policy_hidden_debug,
                                    agent_state.params["network"]["params"]["innovation_projector"]["kernel"],
                                    final_grads["network"]["params"]["innovation_projector"]["kernel"],
                                )
                                preproj_dense_metrics = cnn_dense_metrics(
                                    cnn_debug["dense_pre_ln"],
                                    cnn_debug["hidden"],
                                    agent_state.params["network"]["params"]["Dense_0"]["kernel"],
                                    final_grads["network"]["params"]["Dense_0"]["kernel"],
                                    prefix="encoder_hidden",
                                )
                                for k, v in preproj_dense_metrics.items():
                                    log_dict[k] = float(v)
                                preproj_repr_metrics = encoder_repr_metrics(cnn_debug["hidden"])
                                for k, v in preproj_repr_metrics.items():
                                    log_dict[k.replace("repr/", "encoder_hidden_repr/")] = float(v)
                            else:
                                dense_kernel_name = "dense_stage2" if encoder_type in {"cnn_swish_tanh", "cnn_swish_tanh_resid", "cnn_swish_tanh_resid_learned"} else "Dense_0"
                                dense_metrics = cnn_dense_metrics(
                                    cnn_debug["dense_pre_ln"],
                                    cnn_debug["hidden"],
                                    agent_state.params["network"]["params"][dense_kernel_name]["kernel"],
                                    final_grads["network"]["params"][dense_kernel_name]["kernel"],
                                )
                                if encoder_type in {"cnn_swish_tanh", "cnn_swish_tanh_resid", "cnn_swish_tanh_resid_learned"}:
                                    stage1_dense_metrics = cnn_dense_metrics(
                                        cnn_debug["stage1_dense_pre_ln"],
                                        cnn_debug["stage1_hidden"],
                                        agent_state.params["network"]["params"]["dense_stage1"]["kernel"],
                                        final_grads["network"]["params"]["dense_stage1"]["kernel"],
                                        prefix="encoder_stage1",
                                    )
                                    for k, v in stage1_dense_metrics.items():
                                        log_dict[k] = float(v)
                                    stage1_repr_metrics = encoder_repr_metrics(cnn_debug["stage1_hidden"])
                                    for k, v in stage1_repr_metrics.items():
                                        log_dict[k.replace("repr/", "encoder_stage1_repr/")] = float(v)
                                    if "residual_gate" in cnn_debug:
                                        log_dict["encoder/residual_gate"] = float(cnn_debug["residual_gate"])
                            for k, v in dense_metrics.items():
                                log_dict[k] = float(v)
                        if encoder_type == "sigreg_cnn":
                            proj_metrics = projected_latent_metrics(cnn_debug["u"], prefix="sigreg_proj")
                            for k, v in proj_metrics.items():
                                log_dict[k] = float(v)
                        if encoder_type in {"innovation_cnn", "innovation_direct_cnn"}:
                            proj_metrics = projected_latent_metrics(cnn_debug["u"], prefix="innovation_proj")
                            for k, v in proj_metrics.items():
                                log_dict[k] = float(v)
                            next_cnn_debug = encode_with_intermediates(
                                agent_state.params["network"],
                                sample_next_obs,
                            )
                            innovation_debug = innovation_with_intermediates(
                                agent_state.params["innovation"],
                                cnn_debug["u"],
                                sample_actions,
                            )
                            innovation_metrics = innovation_dynamics_metrics(
                                cnn_debug["u"],
                                next_cnn_debug["u"],
                                innovation_debug["predicted_next_u"],
                                innovation_debug["state_term"],
                                innovation_debug["action_term"],
                                sample_next_done,
                            )
                            for k, v in innovation_metrics.items():
                                log_dict[k] = float(v)
                    else:
                        hidden, critic_hidden_debug = split_actor_critic_hiddens(
                            network.apply(agent_state.params["network"], sample_obs)
                        )
                        actor_mean_debug, actor_logstd_debug = actor.apply(
                            agent_state.params["actor"],
                            hidden,
                        )
                        values_debug = critic.apply(
                            agent_state.params["critic"],
                            critic_hidden_debug,
                        ).squeeze(-1)

                    repr_metrics = encoder_repr_metrics(hidden)
                    for k, v in repr_metrics.items():
                        log_dict[k] = float(v)

                    policy_metrics = policy_output_metrics(
                        actor_mean_debug,
                        actor_logstd_debug,
                        values_debug,
                    )
                    for k, v in policy_metrics.items():
                        log_dict[k] = float(v)

                    grad_metrics = compute_grad_norms(final_grads)
                    for k, v in grad_metrics.items():
                        log_dict[k] = float(v)

                # Flush any buffered probe metrics into the same wandb step.
                if pending_probe_metrics:
                    log_dict.update(pending_probe_metrics)
                    pending_probe_metrics = {}

                wandb.log(log_dict, step=global_step)

    elapsed = time.time() - start_time
    print(f"\nTraining finished in {elapsed:.1f}s")
    print(f"Average SPS: {args.total_timesteps / elapsed:.0f}")

    if args.save_checkpoint:
        import pickle
        ckpt_dir = os.path.join(args.checkpoint_dir, run_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        params_bytes = flax.serialization.to_bytes(agent_state.params)
        ckpt_path = os.path.join(ckpt_dir, "params.pkl")
        with open(ckpt_path, "wb") as f:
            pickle.dump({"params_bytes": params_bytes, "args": vars(args)}, f)
        print(f"Checkpoint saved to {ckpt_path}")

    if args.track:
        wandb.finish()
