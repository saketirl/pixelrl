#!/usr/bin/env python
"""
PPO for Brax environments with state observations - CleanRL style adaptation.
Adapted from CleanRL's PPO implementation for continuous control with state observations.
This serves as a baseline to verify the algorithm works before testing pixel observations.
#To continue this session, run codex resume 019dac51-812f-7023-bfdd-553feaa73809
"""
import os
import random
import time
from dataclasses import dataclass
from functools import partial
from typing import Optional

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
from optimizers import (
    OnlineStiefelState,
    aurora,
    manifold_stiefel,
    manifold_stiefel_admm,
    online_stiefel,
)

# Import Brax from local source (pixelbrax/brax/brax)
import sys
sys.path.insert(0, "/users/apraka15/arjun/pixelrl/pixelbrax/brax")
from brax import envs as brax_envs
from configs.continual.slippery_ant_wrapper import (
    DEFAULT_SLIPPERY_FRICTIONS_CSV,
    NonstationaryFrictionBraxWrapper,
    VecEnv,
    load_seeded_friction_schedule_table,
)

SUPPORTED_SLIPPERY_ENVS = {"ant", "humanoid"}

# Fix weird OOM https://github.com/google/jax/discussions/6332#discussioncomment-1279991
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"
# Fix CUDNN non-determinism
os.environ["TF_XLA_FLAGS"] = "--xla_gpu_autotune_level=2 --xla_gpu_deterministic_reductions"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 0
    """seed of the experiment"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "benchmark"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""

    # Environment arguments
    env_name: str = "halfcheetah"
    """the name of the environment"""
    backend: str = "spring"
    """the physics backend (spring, generalized, positional)"""
    n_envs: int = 512
    """the number of parallel game environments"""

    # Algorithm specific arguments
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    adam_eps: float = 1e-5
    """epsilon parameter for Adam"""
    base_optimizer: str = "adam"
    """Optimizer for Adam-managed params: adam or adamw"""
    weight_decay: float = 0.0
    """Weight decay for AdamW-managed params"""
    heads_optimizer: str = "adam"
    """Optimizer for actor/critic matrix params: adam, stiefel, stiefel_admm, stiefel_online, or aurora"""
    heads_stiefel_lr: float = 0.001
    """Learning rate for actor/critic matrix params when using Stiefel or Aurora"""
    stiefel_dual_lr: float = 0.01
    """Dual learning rate for manifold Stiefel"""
    stiefel_dual_steps: int = 5
    """Number of dual optimization steps for manifold Stiefel"""
    stiefel_msign_steps: int = 5
    """Number of matrix-sign iterations for manifold Stiefel"""
    actor_stiefel_max_grad_norm: float = 100.0
    """Maximum gradient norm for actor Stiefel params"""
    critic_stiefel_max_grad_norm: float = 1.0
    """Maximum gradient norm for critic Stiefel params"""
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
    """the maximum norm for the gradient clipping"""
    max_action: float = 1.0
    """maximum action value for clipping"""
    log_interval: int = 10
    """logging interval (in updates)"""
    actor_critic_activation: str = "swish"
    """Activation for all MLP hidden layers: swish or relu"""
    network_arch: str = "ppo"
    """Network architecture: ppo for shared trunk + deeper heads, lop for lop-jax style shallow independent heads."""
    reward_normalize: bool = True
    """Normalize rewards using discounted-return RMS statistics"""
    obs_normalize: bool = False
    """Normalize state observations with running mean and variance."""
    obs_norm_clip: float = 10.0
    """Clip normalized observations to this absolute value."""

    # Policy std/mean parameterization
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
    use_crate_head: bool = False
    """If true, use CRATE-style FeedForward as final hidden layer in actor/critic."""
    crate_step_size: float = 0.1
    """Step size for CRATE FeedForward ISTA update."""
    use_crate_network: bool = False
    """If true, use a CRATE-style FeedForward block in the state-observation trunk."""
    network_crate_step_size: float = 0.1
    """Step size for the state trunk CRATE FeedForward ISTA update."""

    # Action repeat
    action_repeat: int = 1
    """Number of times to repeat each action (frame skip)"""

    # Slippery friction schedule
    slippery: bool = False
    """Use the CSV-backed slippery friction schedule."""
    slippery_ant: bool = False
    """Deprecated alias for --slippery."""
    slippery_change_every: int = 100_000
    """Number of underlying per-env timesteps between slippery friction changes."""
    slippery_schedule_seed: Optional[int] = None
    """CSV row used for slippery friction phases; defaults to the training seed."""
    slippery_probe_steps: int = 0
    """Run a zero-action schedule probe for this many steps before training."""
    slippery_probe_only: bool = False
    """Exit after the slippery schedule probe."""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_updates: int = 0
    """the number of updates (computed in runtime)"""


class Network(nn.Module):
    """MLP encoder for state observations."""
    activation: str = "swish"
    
    @nn.compact
    def __call__(self, x, return_intermediates: bool = False):
        # x: (B, obs_dim) - 1D state observations
        activation = actor_critic_activation_fn(self.activation)
        dense_0 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        hidden_0 = activation(dense_0)
        dense_1 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(hidden_0)
        x = activation(dense_1)
        if return_intermediates:
            return {
                "hidden": x,
                "Dense_0": hidden_0,
                "Dense_1": x,
                "Dense_0_pre": dense_0,
                "Dense_1_pre": dense_1,
            }
        return x


class IdentityNetwork(nn.Module):
    """Identity state trunk for lop-jax style independent actor/critic MLPs."""

    @nn.compact
    def __call__(self, x, return_intermediates: bool = False):
        self.param("dummy", constant(0.0), ())
        if return_intermediates:
            return {"hidden": x}
        return x


def actor_critic_activation_fn(name: str):
    if name == "swish":
        return nn.swish
    if name == "relu":
        return nn.relu
    raise ValueError(
        f"Unsupported actor_critic_activation={name!r}. Expected 'swish' or 'relu'."
    )


def resolve_heads_optimizer(heads_optimizer: str) -> str:
    optimizer = heads_optimizer.lower()
    if optimizer not in {"adam", "stiefel", "stiefel_admm", "stiefel_online", "aurora"}:
        raise ValueError(
            f"Unsupported heads_optimizer={heads_optimizer!r}. "
            "Expected one of: adam, stiefel, stiefel_admm, stiefel_online, aurora."
        )
    return optimizer


def resolve_base_optimizer(base_optimizer: str) -> str:
    optimizer = base_optimizer.lower()
    if optimizer not in {"adam", "adamw"}:
        raise ValueError(
            f"Unsupported base_optimizer={base_optimizer!r}. Expected one of: adam, adamw."
        )
    return optimizer


def is_stiefel_matrix_param(param) -> bool:
    return param.ndim >= 2 and min(param.shape) > 1


def is_aurora_matrix_param(param) -> bool:
    return param.ndim == 2 and min(param.shape) > 1


def optimizer_label_for_param(path, param, heads_optimizer: str) -> str:
    top_level = path[0]
    if top_level == "network":
        return "network_adam"
    if top_level == "actor":
        if heads_optimizer == "stiefel" and is_stiefel_matrix_param(param):
            return "actor_stiefel"
        if heads_optimizer == "stiefel_admm" and is_stiefel_matrix_param(param):
            return "actor_stiefel_admm"
        if heads_optimizer == "stiefel_online" and is_stiefel_matrix_param(param):
            return "actor_stiefel_online"
        if heads_optimizer == "aurora" and is_aurora_matrix_param(param):
            return "actor_aurora"
        return "heads_adam"
    if top_level == "critic":
        if heads_optimizer == "stiefel" and is_stiefel_matrix_param(param):
            return "critic_stiefel"
        if heads_optimizer == "stiefel_admm" and is_stiefel_matrix_param(param):
            return "critic_stiefel_admm"
        if heads_optimizer == "stiefel_online" and is_stiefel_matrix_param(param):
            return "critic_stiefel_online"
        if heads_optimizer == "aurora" and is_aurora_matrix_param(param):
            return "critic_aurora"
        return "heads_adam"
    return "heads_adam"


def create_optimizer(
    learning_rate,
    *,
    adam_eps: float,
    base_optimizer: str,
    weight_decay: float,
    max_grad_norm: float,
    heads_optimizer: str,
    heads_stiefel_lr: float,
    stiefel_dual_lr: float,
    stiefel_dual_steps: int,
    stiefel_msign_steps: int,
    actor_stiefel_max_grad_norm: float,
    critic_stiefel_max_grad_norm: float,
):
    base_optimizer = resolve_base_optimizer(base_optimizer)
    heads_optimizer = resolve_heads_optimizer(heads_optimizer)

    def make_base_tx():
        optimizer = optax.adam if base_optimizer == "adam" else optax.adamw
        kwargs = {"learning_rate": learning_rate, "eps": adam_eps}
        if base_optimizer == "adamw":
            kwargs["weight_decay"] = weight_decay
        return optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.inject_hyperparams(optimizer)(**kwargs),
        )

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
            msign_steps=stiefel_msign_steps,
            min_ndim=2,
        ),
    )
    critic_stiefel_admm_tx = optax.chain(
        optax.clip_by_global_norm(critic_stiefel_max_grad_norm),
        manifold_stiefel_admm(
            learning_rate=heads_stiefel_lr,
            msign_steps=stiefel_msign_steps,
            min_ndim=2,
        ),
    )
    actor_stiefel_online_tx = optax.chain(
        optax.clip_by_global_norm(actor_stiefel_max_grad_norm),
        online_stiefel(
            learning_rate=heads_stiefel_lr,
            dual_lr=stiefel_dual_lr,
            msign_steps=stiefel_msign_steps,
            min_ndim=2,
        ),
    )
    critic_stiefel_online_tx = optax.chain(
        optax.clip_by_global_norm(critic_stiefel_max_grad_norm),
        online_stiefel(
            learning_rate=heads_stiefel_lr,
            dual_lr=stiefel_dual_lr,
            msign_steps=stiefel_msign_steps,
            min_ndim=2,
        ),
    )
    actor_aurora_tx = optax.chain(
        optax.clip_by_global_norm(actor_stiefel_max_grad_norm),
        aurora(
            learning_rate=heads_stiefel_lr,
            adam_eps=adam_eps,
        ),
    )
    critic_aurora_tx = optax.chain(
        optax.clip_by_global_norm(critic_stiefel_max_grad_norm),
        aurora(
            learning_rate=heads_stiefel_lr,
            adam_eps=adam_eps,
        ),
    )

    transforms = {
        "network_adam": make_base_tx(),
        "heads_adam": make_base_tx(),
        "actor_stiefel": actor_stiefel_tx,
        "critic_stiefel": critic_stiefel_tx,
        "actor_stiefel_admm": actor_stiefel_admm_tx,
        "critic_stiefel_admm": critic_stiefel_admm_tx,
        "actor_stiefel_online": actor_stiefel_online_tx,
        "critic_stiefel_online": critic_stiefel_online_tx,
        "actor_aurora": actor_aurora_tx,
        "critic_aurora": critic_aurora_tx,
    }

    def label_fn(params):
        flat = flax.traverse_util.flatten_dict(params)
        labels = {
            path: optimizer_label_for_param(path, param, heads_optimizer)
            for path, param in flat.items()
        }
        return flax.core.freeze(flax.traverse_util.unflatten_dict(labels))

    return optax.multi_transform(transforms=transforms, param_labels=label_fn)


def count_optimizer_params(params, heads_optimizer: str) -> dict:
    flat = flax.traverse_util.flatten_dict(params)
    counts = {
        "network_adam": 0,
        "heads_adam": 0,
        "actor_stiefel": 0,
        "critic_stiefel": 0,
        "actor_stiefel_admm": 0,
        "critic_stiefel_admm": 0,
        "actor_stiefel_online": 0,
        "critic_stiefel_online": 0,
        "actor_aurora": 0,
        "critic_aurora": 0,
    }
    for path, param in flat.items():
        counts[optimizer_label_for_param(path, param, heads_optimizer)] += int(param.size)
    return counts


def online_stiefel_constraint_metrics(opt_state) -> dict:
    states = [
        state
        for state in jax.tree_util.tree_leaves(
            opt_state,
            is_leaf=lambda x: isinstance(x, OnlineStiefelState),
        )
        if isinstance(state, OnlineStiefelState)
    ]
    residuals = []
    normalized_residuals = []
    for state in states:
        residual_leaves = jax.tree_util.tree_leaves(state.last_constraint_residual)
        dual_leaves = jax.tree_util.tree_leaves(state.dual_lambda)
        for residual, dual_lambda in zip(residual_leaves, dual_leaves):
            if residual.shape != () or dual_lambda.ndim != 2:
                continue
            residuals.append(residual)
            normalized_residuals.append(residual / jnp.sqrt(dual_lambda.size))
    if not residuals:
        return {}
    residuals = jnp.asarray(residuals)
    normalized_residuals = jnp.asarray(normalized_residuals)
    return {
        "optim/stiefel_online_constraint_mean": float(jnp.mean(residuals)),
        "optim/stiefel_online_constraint_max": float(jnp.max(residuals)),
        "optim/stiefel_online_constraint_rms_mean": float(jnp.mean(normalized_residuals)),
        "optim/stiefel_online_constraint_rms_max": float(jnp.max(normalized_residuals)),
    }


def matrix_constraint_metrics(params) -> dict:
    flat = flax.traverse_util.flatten_dict(params)
    metrics = {}
    all_fro = []
    all_rms = []

    for module_name in ("actor", "critic"):
        fro_residuals = []
        rms_residuals = []
        for path, param in flat.items():
            if not path or path[0] != module_name or not is_stiefel_matrix_param(param):
                continue
            matrix = param.T if param.shape[-2] < param.shape[-1] else param
            gram = matrix.T @ matrix
            residual = gram - jnp.eye(gram.shape[-1], dtype=gram.dtype)
            fro_residual = jnp.linalg.norm(residual, ord="fro")
            rms_residual = fro_residual / jnp.sqrt(jnp.asarray(residual.size, dtype=gram.dtype))
            fro_residuals.append(fro_residual)
            rms_residuals.append(rms_residual)

        if not fro_residuals:
            continue
        fro_residuals = jnp.asarray(fro_residuals)
        rms_residuals = jnp.asarray(rms_residuals)
        metrics.update({
            f"optim/matrix_constraint_{module_name}_mean": float(jnp.mean(fro_residuals)),
            f"optim/matrix_constraint_{module_name}_max": float(jnp.max(fro_residuals)),
            f"optim/matrix_constraint_{module_name}_rms_mean": float(jnp.mean(rms_residuals)),
            f"optim/matrix_constraint_{module_name}_rms_max": float(jnp.max(rms_residuals)),
        })
        all_fro.extend(fro_residuals)
        all_rms.extend(rms_residuals)

    if all_fro:
        all_fro = jnp.asarray(all_fro)
        all_rms = jnp.asarray(all_rms)
        metrics.update({
            "optim/matrix_constraint_mean": float(jnp.mean(all_fro)),
            "optim/matrix_constraint_max": float(jnp.max(all_fro)),
            "optim/matrix_constraint_rms_mean": float(jnp.mean(all_rms)),
            "optim/matrix_constraint_rms_max": float(jnp.max(all_rms)),
        })
    return metrics


class Critic(nn.Module):
    """Value network with 2 hidden layers for sufficient capacity."""
    activation: str = "swish"

    @nn.compact
    def __call__(self, x, return_intermediates: bool = False):
        activation = actor_critic_activation_fn(self.activation)
        dense_0 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        hidden_0 = activation(dense_0)
        dense_1 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(hidden_0)
        hidden_1 = activation(dense_1)
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(hidden_1)
        if return_intermediates:
            return {
                "value": value,
                "Dense_0": hidden_0,
                "Dense_1": hidden_1,
                "Dense_2": value,
            }
        return value


class LopCritic(nn.Module):
    """lop-jax style critic: one hidden layer directly on observations."""
    activation: str = "relu"

    @nn.compact
    def __call__(self, x, return_intermediates: bool = False):
        activation = actor_critic_activation_fn(self.activation)
        dense_0 = nn.Dense(256, kernel_init=nn.initializers.lecun_uniform(), bias_init=constant(0.0))(x)
        hidden_0 = activation(dense_0)
        value = nn.Dense(1, kernel_init=nn.initializers.lecun_uniform(), bias_init=constant(0.0))(hidden_0)
        if return_intermediates:
            return {
                "value": value,
                "Dense_0": hidden_0,
                "Dense_1": value,
            }
        return value


class CRATEFeedForward(nn.Module):
    """CRATE-style FeedForward layer implementing an ISTA step."""
    dim: int
    step_size: float = 0.1

    @nn.compact
    def __call__(self, x):
        weight = self.param(
            "weight",
            nn.initializers.kaiming_uniform(),
            (self.dim, self.dim),
        )
        grad_1 = (x @ weight.T) @ weight
        grad_2 = x @ weight
        grad_update = self.step_size * (grad_2 - grad_1)
        return nn.swish(x + grad_update)


class CRATENetwork(nn.Module):
    """State-observation trunk with a CRATE FeedForward final block."""
    activation: str = "swish"
    crate_step_size: float = 0.1

    @nn.compact
    def __call__(self, x, return_intermediates: bool = False):
        activation = actor_critic_activation_fn(self.activation)
        dense_0 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        hidden_0 = activation(dense_0)
        crate_hidden = CRATEFeedForward(dim=256, step_size=self.crate_step_size)(hidden_0)
        if return_intermediates:
            return {
                "hidden": crate_hidden,
                "Dense_0": hidden_0,
                "CRATEFeedForward_0": crate_hidden,
                "Dense_0_pre": dense_0,
            }
        return crate_hidden


class CRATECritic(nn.Module):
    """Value network with CRATE FeedForward as final hidden layer."""
    crate_step_size: float = 0.1
    activation: str = "swish"

    @nn.compact
    def __call__(self, x, return_intermediates: bool = False):
        activation = actor_critic_activation_fn(self.activation)
        dense_0 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        hidden_0 = activation(dense_0)
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


class Actor(nn.Module):
    """Continuous action actor with 2 hidden layers using Gaussian distribution."""
    action_dim: int
    activation: str = "swish"
    logstd_min: float = -5.0
    logstd_max: float = 2.0
    clip_global_logstd: bool = False
    bounded_global_logstd: bool = False
    actor_logstd_init: float = 0.0
    actor_mean_tanh: bool = False
    actor_mean_scale: float = 1.0

    @nn.compact
    def __call__(self, x, return_intermediates: bool = False):
        activation = actor_critic_activation_fn(self.activation)
        dense_0 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        hidden_0 = activation(dense_0)
        dense_1 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(hidden_0)
        hidden_1 = activation(dense_1)
        actor_mean = nn.Dense(
            self.action_dim, 
            kernel_init=orthogonal(0.01), 
            bias_init=constant(0.0)
        )(hidden_1)
        if self.actor_mean_tanh:
            actor_mean = self.actor_mean_scale * jnp.tanh(actor_mean)

        if self.bounded_global_logstd:
            eps = 1e-6
            init = jnp.clip(
                self.actor_logstd_init,
                self.logstd_min + eps,
                self.logstd_max - eps,
            )
            frac = (init - self.logstd_min) / (self.logstd_max - self.logstd_min)
            raw_init = jnp.log(frac) - jnp.log1p(-frac)
            actor_logstd_raw = self.param(
                "log_std_raw",
                constant(raw_init),
                (self.action_dim,),
            )
            actor_logstd = self.logstd_min + (
                self.logstd_max - self.logstd_min
            ) * jax.nn.sigmoid(actor_logstd_raw)
        else:
            actor_logstd = self.param(
                "log_std",
                constant(self.actor_logstd_init),
                (self.action_dim,),
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


class LopActor(nn.Module):
    """lop-jax style actor: one hidden layer directly on observations."""
    action_dim: int
    activation: str = "relu"
    logstd_min: float = -5.0
    logstd_max: float = 2.0
    clip_global_logstd: bool = False
    bounded_global_logstd: bool = False
    actor_logstd_init: float = 0.0
    actor_mean_tanh: bool = False
    actor_mean_scale: float = 1.0

    @nn.compact
    def __call__(self, x, return_intermediates: bool = False):
        activation = actor_critic_activation_fn(self.activation)
        dense_0 = nn.Dense(256, kernel_init=nn.initializers.lecun_uniform(), bias_init=constant(0.0))(x)
        hidden_0 = activation(dense_0)
        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.lecun_uniform(),
            bias_init=constant(0.0),
        )(hidden_0)
        if self.actor_mean_tanh:
            actor_mean = self.actor_mean_scale * jnp.tanh(actor_mean)

        if self.bounded_global_logstd:
            eps = 1e-6
            init = jnp.clip(
                self.actor_logstd_init,
                self.logstd_min + eps,
                self.logstd_max - eps,
            )
            frac = (init - self.logstd_min) / (self.logstd_max - self.logstd_min)
            raw_init = jnp.log(frac) - jnp.log1p(-frac)
            actor_logstd_raw = self.param(
                "log_std_raw",
                constant(raw_init),
                (self.action_dim,),
            )
            actor_logstd = self.logstd_min + (
                self.logstd_max - self.logstd_min
            ) * jax.nn.sigmoid(actor_logstd_raw)
        else:
            actor_logstd = self.param(
                "log_std",
                constant(self.actor_logstd_init),
                (self.action_dim,),
            )
            if self.clip_global_logstd:
                actor_logstd = jnp.clip(actor_logstd, self.logstd_min, self.logstd_max)

        if return_intermediates:
            return {
                "actor_mean": actor_mean,
                "actor_logstd": actor_logstd,
                "Dense_0": hidden_0,
                "Dense_1": actor_mean,
            }
        return actor_mean, actor_logstd


class CRATEActor(nn.Module):
    """Continuous action actor with CRATE FeedForward as final hidden layer."""
    action_dim: int
    crate_step_size: float = 0.1
    activation: str = "swish"
    logstd_min: float = -5.0
    logstd_max: float = 2.0
    clip_global_logstd: bool = False
    bounded_global_logstd: bool = False
    actor_logstd_init: float = 0.0
    actor_mean_tanh: bool = False
    actor_mean_scale: float = 1.0

    @nn.compact
    def __call__(self, x, return_intermediates: bool = False):
        activation = actor_critic_activation_fn(self.activation)
        dense_0 = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        hidden_0 = activation(dense_0)
        crate_hidden = CRATEFeedForward(dim=256, step_size=self.crate_step_size)(hidden_0)
        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(crate_hidden)
        if self.actor_mean_tanh:
            actor_mean = self.actor_mean_scale * jnp.tanh(actor_mean)

        if self.bounded_global_logstd:
            eps = 1e-6
            init = jnp.clip(
                self.actor_logstd_init,
                self.logstd_min + eps,
                self.logstd_max - eps,
            )
            frac = (init - self.logstd_min) / (self.logstd_max - self.logstd_min)
            raw_init = jnp.log(frac) - jnp.log1p(-frac)
            actor_logstd_raw = self.param(
                "log_std_raw",
                constant(raw_init),
                (self.action_dim,),
            )
            actor_logstd = self.logstd_min + (
                self.logstd_max - self.logstd_min
            ) * jax.nn.sigmoid(actor_logstd_raw)
        else:
            actor_logstd = self.param(
                "log_std",
                constant(self.actor_logstd_init),
                (self.action_dim,),
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


# Using FrozenDict for params instead of a dataclass for Flax compatibility
# Params structure: {'network': ..., 'actor': ..., 'critic': ...}


@flax.struct.dataclass
class Storage:
    obs: jnp.array
    actions: jnp.array
    logprobs: jnp.array
    dones: jnp.array
    values: jnp.array
    advantages: jnp.array
    returns: jnp.array
    rewards: jnp.array


@flax.struct.dataclass
class SlipperyRolloutInfo:
    friction_slide: jnp.array
    timestep: jnp.array


@flax.struct.dataclass
class EpisodeStatistics:
    episode_returns: jnp.array
    episode_lengths: jnp.array
    returned_episode_returns: jnp.array
    returned_episode_lengths: jnp.array


@flax.struct.dataclass
class RewardNormalizer:
    """
    Reward normalization using discounted returns (CleanRL style).
    Normalizes rewards by the standard deviation of discounted returns,
    which is more stable for continuous control.
    """
    return_rms_mean: jnp.array  # Running mean of returns (unused but tracked)
    return_rms_var: jnp.array   # Running variance of returns
    return_rms_count: jnp.array
    discounted_return: jnp.array  # Track discounted returns per environment
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
        """
        Update running statistics with a batch of rewards.
        Uses discounted returns for variance estimation (like gym.wrappers.NormalizeReward).
        """
        # Update discounted returns: R_t = r_t + gamma * R_{t+1} * (1 - done)
        new_discounted_return = rewards + self.gamma * self.discounted_return * (1.0 - dones)
        
        # Update running statistics of returns
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
        
        # Reset discounted return where episodes ended
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


@flax.struct.dataclass
class ObsNormalizer:
    """Running mean/std normalizer for vector state observations."""
    mean: jnp.array
    var: jnp.array
    count: jnp.array
    enabled: bool = flax.struct.field(pytree_node=False, default=False)
    clip: float = flax.struct.field(pytree_node=False, default=10.0)

    @classmethod
    def create(cls, obs_shape, enabled=False, clip=10.0):
        return cls(
            mean=jnp.zeros(obs_shape, dtype=jnp.float32),
            var=jnp.ones(obs_shape, dtype=jnp.float32),
            count=jnp.array(1e-4, dtype=jnp.float32),
            enabled=enabled,
            clip=clip,
        )

    def update(self, obs):
        if not self.enabled:
            return self
        obs = obs.astype(jnp.float32)
        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = jnp.asarray(obs.shape[0], dtype=jnp.float32)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + jnp.square(delta) * self.count * batch_count / total_count
        new_var = m2 / total_count
        return self.replace(mean=new_mean, var=new_var, count=total_count)

    def normalize(self, obs, epsilon=1e-8):
        if not self.enabled:
            return obs
        normalized = (obs.astype(jnp.float32) - self.mean) / jnp.sqrt(self.var + epsilon)
        return jnp.clip(normalized, -self.clip, self.clip)


def make_brax_envs(args):
    """Create Brax environments with state observations."""
    if args.slippery:
        env = NonstationaryFrictionBraxWrapper(
            env_name=args.env_name,
            backend=args.backend,
            change_every=args.slippery_change_every,
            schedule_seed=args.slippery_schedule_seed,
            action_repeat=args.action_repeat,
        )
        action_dim = env.action_size
        obs_dim = env.observation_size[0]
        return VecEnv(env), action_dim, obs_dim

    env = brax_envs.create(
        env_name=args.env_name,
        backend=args.backend,
        episode_length=1000,
        action_repeat=args.action_repeat,
        auto_reset=True,
        batch_size=args.n_envs,
    )
    
    action_dim = env.action_size
    obs_dim = env.observation_size
    
    return env, action_dim, obs_dim


def slippery_num_seeded_phases() -> int:
    table = load_seeded_friction_schedule_table(str(DEFAULT_SLIPPERY_FRICTIONS_CSV))
    return int(table.shape[1])


def slippery_phase_from_timestep(timestep: int, change_every: int, num_seeded_phases: int) -> int:
    """Map a 1-indexed post-step timestep to the default-plus-CSV regime index."""
    zero_based_timestep = max(int(timestep) - 1, 0)
    phase = zero_based_timestep // change_every
    total_phases = num_seeded_phases + 1
    return min(phase, total_phases - 1)


def latest_slippery_metrics(rollout_info: SlipperyRolloutInfo, args) -> dict[str, float]:
    friction = jax.device_get(rollout_info.friction_slide[-1])
    timestep = jax.device_get(rollout_info.timestep[-1])
    first_timestep = int(np.asarray(timestep)[0])
    num_seeded_phases = slippery_num_seeded_phases()
    return {
        "slippery/friction_slide_mean": float(np.mean(friction)),
        "slippery/friction_slide_first_env": float(np.asarray(friction)[0]),
        "slippery/timestep_first_env": first_timestep,
        "slippery/phase_first_env": slippery_phase_from_timestep(
            timestep=first_timestep,
            change_every=args.slippery_change_every,
            num_seeded_phases=num_seeded_phases,
        ),
        "slippery/change_every": int(args.slippery_change_every),
    }


def run_slippery_probe(envs, args, key):
    """Prints the wrapper's friction schedule under zero actions."""
    if args.slippery_probe_steps <= 0:
        return key

    key, reset_key = jax.random.split(key)
    reset_rngs = jax.random.split(reset_key, args.n_envs)
    obs, state = envs.reset(reset_rngs, None)
    del obs

    zero_action = jnp.zeros((args.n_envs, envs.action_size), dtype=jnp.float32)

    @jax.jit
    def probe_rollout(state, key):
        def step(carry, _):
            state, key = carry
            key, step_key = jax.random.split(key)
            step_rngs = jax.random.split(step_key, args.n_envs)
            _, next_state, _, _, info = envs.step(step_rngs, state, zero_action, None)
            friction = info["friction"][:, 0, 0]
            timestep = next_state.info["timestep"]
            return (next_state, key), (friction, timestep)

        return jax.lax.scan(
            step,
            (state, key),
            None,
            length=args.slippery_probe_steps,
        )

    (state, key), (friction, timestep) = probe_rollout(state, key)
    del state
    friction = np.asarray(jax.device_get(friction))
    timestep = np.asarray(jax.device_get(timestep))

    print("\nSlippery wrapper probe:")
    print(
        "  source="
        f"{DEFAULT_SLIPPERY_FRICTIONS_CSV} "
        f"change_every={args.slippery_change_every} "
        f"seed={args.slippery_schedule_seed}"
    )
    print("  step timestep phase friction_first_env")
    num_seeded_phases = slippery_num_seeded_phases()
    for i in range(args.slippery_probe_steps):
        phase = slippery_phase_from_timestep(
            timestep=int(timestep[i, 0]),
            change_every=args.slippery_change_every,
            num_seeded_phases=num_seeded_phases,
        )
        print(
            f"  {i + 1:4d} {int(timestep[i, 0]):8d} "
            f"{phase:5d} {float(friction[i, 0]):.6g}"
        )

    return key


if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.actor_critic_activation not in {"swish", "relu"}:
        raise ValueError(
            "actor_critic_activation must be one of: swish, relu; "
            f"got {args.actor_critic_activation!r}"
        )
    if args.action_repeat <= 0:
        raise ValueError("--action-repeat must be positive.")
    if args.slippery_ant:
        args.slippery = True
        if args.env_name != "ant":
            print("--slippery-ant is deprecated; use --slippery for non-Ant slippery envs.")
    if args.slippery:
        if args.env_name not in SUPPORTED_SLIPPERY_ENVS:
            supported = ", ".join(sorted(SUPPORTED_SLIPPERY_ENVS))
            raise ValueError(f"--slippery supports only envs in {{{supported}}}; got {args.env_name!r}.")
        if args.slippery_change_every <= 0:
            raise ValueError("--slippery-change-every must be positive.")
        if args.slippery_schedule_seed is None:
            args.slippery_schedule_seed = args.seed
        if args.slippery_schedule_seed < 0:
            raise ValueError("--slippery-schedule-seed must be non-negative.")
        if args.slippery_probe_only and args.slippery_probe_steps <= 0:
            raise ValueError("--slippery-probe-only requires --slippery-probe-steps > 0.")
    args.base_optimizer = resolve_base_optimizer(args.base_optimizer)
    args.heads_optimizer = resolve_heads_optimizer(args.heads_optimizer)
    args.batch_size = int(args.n_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_updates = args.total_timesteps // args.batch_size
    run_name = f"{args.env_name}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
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
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, network_key, actor_key, critic_key = jax.random.split(key, 4)

    # Environment setup
    print("JAX devices:", jax.devices())
    print(f"env name: {args.env_name}")
    print(f"total timesteps: {args.total_timesteps}")
    print(f"num steps per rollout: {args.num_steps}")
    print(f"num envs: {args.n_envs}")
    print(f"learning rate: {args.learning_rate}")
    print(f"adam_eps: {args.adam_eps}")
    print(f"base_optimizer: {args.base_optimizer}")
    print(f"weight_decay: {args.weight_decay}")
    print(f"seed: {args.seed}")
    print(f"num_updates: {args.num_updates}")
    print(f"minibatch_size: {args.minibatch_size}")
    print(f"actor_critic_activation: {args.actor_critic_activation}")
    print(f"network_arch: {args.network_arch}")
    print(f"reward_normalize: {args.reward_normalize}")
    print(f"obs_normalize: {args.obs_normalize}")
    print(f"obs_norm_clip: {args.obs_norm_clip}")
    print(f"anneal_lr: {args.anneal_lr}")
    print(f"actor_logstd_min: {args.actor_logstd_min}")
    print(f"actor_logstd_max: {args.actor_logstd_max}")
    print(f"clip_global_logstd: {args.clip_global_logstd}")
    print(f"bounded_global_logstd: {args.bounded_global_logstd}")
    print(f"actor_logstd_init: {args.actor_logstd_init}")
    print(f"actor_mean_tanh: {args.actor_mean_tanh}")
    print(f"actor_mean_scale: {args.actor_mean_scale}")
    print(f"use_crate_network: {args.use_crate_network}")
    print(f"network_crate_step_size: {args.network_crate_step_size}")
    print(f"use_crate_head: {args.use_crate_head}")
    print(f"crate_step_size: {args.crate_step_size}")
    print(f"heads_optimizer: {args.heads_optimizer}")
    print(f"heads_stiefel_lr: {args.heads_stiefel_lr}")
    print(f"stiefel_dual_lr: {args.stiefel_dual_lr}")
    print(f"stiefel_dual_steps: {args.stiefel_dual_steps}")
    print(f"stiefel_msign_steps: {args.stiefel_msign_steps}")
    print(f"actor_stiefel_max_grad_norm: {args.actor_stiefel_max_grad_norm}")
    print(f"critic_stiefel_max_grad_norm: {args.critic_stiefel_max_grad_norm}")
    if args.slippery:
        print(f"slippery_change_every: {args.slippery_change_every}")
        print(f"slippery_friction_csv: {DEFAULT_SLIPPERY_FRICTIONS_CSV}")
        print(f"slippery_schedule_seed: {args.slippery_schedule_seed}")
    
    envs, action_dim, obs_dim = make_brax_envs(args)
    print(f"action_dim: {action_dim}")
    print(f"obs_dim: {obs_dim}")
    print(f"action_repeat: {args.action_repeat}")

    if args.slippery:
        key = run_slippery_probe(envs, args, key)
    if args.slippery_probe_only:
        raise SystemExit(0)

    def reset_env(rng):
        if args.slippery:
            reset_rngs = jax.random.split(rng, args.n_envs)
            obs, state = envs.reset(reset_rngs, None)
            return obs, state
        state = envs.reset(rng)
        return state.obs, state

    def step_env(key, state, action):
        if args.slippery:
            key, env_key = jax.random.split(key)
            step_rngs = jax.random.split(env_key, args.n_envs)
            obs, next_state, reward, done, info = envs.step(step_rngs, state, action, None)
            rollout_info = SlipperyRolloutInfo(
                friction_slide=info["friction"][:, 0, 0],
                timestep=next_state.info["timestep"],
            )
            return key, obs, next_state, reward, done.astype(jnp.bool_), rollout_info
        next_state = envs.step(state, action)
        rollout_info = SlipperyRolloutInfo(
            friction_slide=jnp.zeros(args.n_envs, dtype=jnp.float32),
            timestep=jnp.zeros(args.n_envs, dtype=jnp.int32),
        )
        return key, next_state.obs, next_state, next_state.reward, next_state.done.astype(jnp.bool_), rollout_info
    
    # Observation shape is 1D for state-based observations
    obs_shape = (obs_dim,)
    print(f"obs_shape: {obs_shape}")
    
    episode_stats = EpisodeStatistics(
        episode_returns=jnp.zeros(args.n_envs, dtype=jnp.float32),
        episode_lengths=jnp.zeros(args.n_envs, dtype=jnp.int32),
        returned_episode_returns=jnp.zeros(args.n_envs, dtype=jnp.float32),
        returned_episode_lengths=jnp.zeros(args.n_envs, dtype=jnp.int32),
    )

    def linear_schedule(count):
        frac = 1.0 - (count // (args.num_minibatches * args.update_epochs)) / args.num_updates
        return args.learning_rate * frac

    # Initialize networks
    if args.network_arch == "lop":
        if args.use_crate_network or args.use_crate_head:
            raise ValueError("--network-arch=lop does not support CRATE trunk/head options.")
        network = IdentityNetwork()
        print("Using lop-jax style identity state trunk")
    elif args.network_arch == "ppo" and args.use_crate_network:
        network = CRATENetwork(
            activation=args.actor_critic_activation,
            crate_step_size=args.network_crate_step_size,
        )
        print(f"Using CRATE state trunk with step_size={args.network_crate_step_size}")
    elif args.network_arch == "ppo":
        network = Network(activation=args.actor_critic_activation)
    else:
        raise ValueError(f"Unsupported --network-arch={args.network_arch!r}; expected 'ppo' or 'lop'.")
    actor_kwargs = dict(
        action_dim=action_dim,
        activation=args.actor_critic_activation,
        logstd_min=args.actor_logstd_min,
        logstd_max=args.actor_logstd_max,
        clip_global_logstd=args.clip_global_logstd,
        bounded_global_logstd=args.bounded_global_logstd,
        actor_logstd_init=args.actor_logstd_init,
        actor_mean_tanh=args.actor_mean_tanh,
        actor_mean_scale=args.actor_mean_scale,
    )
    if args.network_arch == "lop":
        actor = LopActor(**actor_kwargs)
        critic = LopCritic(activation=args.actor_critic_activation)
        print("Using lop-jax style shallow actor/critic heads")
    elif args.use_crate_head:
        actor = CRATEActor(crate_step_size=args.crate_step_size, **actor_kwargs)
        critic = CRATECritic(
            crate_step_size=args.crate_step_size,
            activation=args.actor_critic_activation,
        )
        print(f"Using CRATE heads with step_size={args.crate_step_size}")
    else:
        actor = Actor(**actor_kwargs)
        critic = Critic(activation=args.actor_critic_activation)
    
    dummy_obs = jnp.zeros((1,) + obs_shape)
    network_params = network.init(network_key, dummy_obs)
    dummy_hidden = network.apply(network_params, dummy_obs)
    all_params = flax.core.freeze({
        "network": network_params,
        "actor": actor.init(actor_key, dummy_hidden),
        "critic": critic.init(critic_key, dummy_hidden),
    })
    optimizer_counts = count_optimizer_params(all_params, args.heads_optimizer)
    print("\nParameter breakdown by optimizer:")
    print(f"  network_adam: {optimizer_counts['network_adam']:,}")
    print(f"  heads_adam: {optimizer_counts['heads_adam']:,}")
    print(f"  actor_stiefel: {optimizer_counts['actor_stiefel']:,}")
    print(f"  critic_stiefel: {optimizer_counts['critic_stiefel']:,}")
    print(f"  actor_stiefel_admm: {optimizer_counts['actor_stiefel_admm']:,}")
    print(f"  critic_stiefel_admm: {optimizer_counts['critic_stiefel_admm']:,}")
    print(f"  actor_stiefel_online: {optimizer_counts['actor_stiefel_online']:,}")
    print(f"  critic_stiefel_online: {optimizer_counts['critic_stiefel_online']:,}")
    print(f"  actor_aurora: {optimizer_counts['actor_aurora']:,}")
    print(f"  critic_aurora: {optimizer_counts['critic_aurora']:,}")
    
    agent_state = TrainState.create(
        apply_fn=None,
        params=all_params,
        tx=create_optimizer(
            linear_schedule if args.anneal_lr else args.learning_rate,
            adam_eps=args.adam_eps,
            base_optimizer=args.base_optimizer,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            heads_optimizer=args.heads_optimizer,
            heads_stiefel_lr=args.heads_stiefel_lr,
            stiefel_dual_lr=args.stiefel_dual_lr,
            stiefel_dual_steps=args.stiefel_dual_steps,
            stiefel_msign_steps=args.stiefel_msign_steps,
            actor_stiefel_max_grad_norm=args.actor_stiefel_max_grad_norm,
            critic_stiefel_max_grad_norm=args.critic_stiefel_max_grad_norm,
        ),
    )
    network.apply = jax.jit(network.apply)
    actor.apply = jax.jit(actor.apply)
    critic.apply = jax.jit(critic.apply)

    @jax.jit
    def get_action_and_value(
        agent_state: TrainState,
        next_obs: np.ndarray,
        key: jax.random.PRNGKey,
    ):
        """Sample action, calculate value, logprob, and return updated key."""
        hidden = network.apply(agent_state.params['network'], next_obs)
        actor_mean, actor_logstd = actor.apply(agent_state.params['actor'], hidden)
        
        # Create Gaussian distribution
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logstd))
        
        key, subkey = jax.random.split(key)
        action = pi.sample(seed=subkey)
        logprob = pi.log_prob(action)
        value = critic.apply(agent_state.params['critic'], hidden)
        
        # Clip action
        action = jnp.clip(action, -args.max_action, args.max_action)
        
        return action, logprob, value.squeeze(-1), key

    @jax.jit
    def get_action_and_value2(
        params: flax.core.FrozenDict,
        x: np.ndarray,
        action: np.ndarray,
    ):
        """Calculate value, logprob of supplied action, and entropy."""
        hidden = network.apply(params['network'], x)
        actor_mean, actor_logstd = actor.apply(params['actor'], hidden)
        
        # Create Gaussian distribution
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logstd))
        
        logprob = pi.log_prob(action)
        entropy = pi.entropy()
        value = critic.apply(params['critic'], hidden).squeeze(-1)
        
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
            agent_state.params['critic'], 
            network.apply(agent_state.params['network'], next_obs)
        ).squeeze(-1)

        advantages = jnp.zeros((args.n_envs,))
        dones = jnp.concatenate([storage.dones, next_done[None, :]], axis=0)
        values = jnp.concatenate([storage.values, next_value[None, :]], axis=0)
        _, advantages = jax.lax.scan(
            compute_gae_once, advantages, (dones[1:], values[1:], values[:-1], storage.rewards), reverse=True
        )
        storage = storage.replace(
            advantages=advantages,
            returns=advantages + storage.values,
        )
        return storage

    def ppo_loss(params, x, a, logp, mb_advantages, mb_returns, mb_values):
        newlogprob, entropy, newvalue = get_action_and_value2(params, x, a)
        logratio = newlogprob - logp
        ratio = jnp.exp(logratio)
        approx_kl = ((ratio - 1) - logratio).mean()

        if args.norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * jnp.clip(ratio, 1 - args.clip_eps, 1 + args.clip_eps)
        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

        # Value loss with clipping
        if args.clip_vloss:
            v_loss_unclipped = (newvalue - mb_returns) ** 2
            v_clipped = mb_values + jnp.clip(
                newvalue - mb_values, -args.clip_eps, args.clip_eps
            )
            v_loss_clipped = (v_clipped - mb_returns) ** 2
            v_loss = 0.5 * jnp.maximum(v_loss_unclipped, v_loss_clipped).mean()
        else:
            v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
        return loss, (pg_loss, v_loss, entropy_loss, jax.lax.stop_gradient(approx_kl))

    ppo_loss_grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)

    @jax.jit
    def update_ppo(
        agent_state: TrainState,
        storage: Storage,
        key: jax.random.PRNGKey,
    ):
        def update_epoch(carry, unused_inp):
            agent_state, key = carry
            key, subkey = jax.random.split(key)

            def flatten(x):
                return x.reshape((-1,) + x.shape[2:])

            def convert_data(x: jnp.ndarray):
                x = jax.random.permutation(subkey, x)
                x = jnp.reshape(x, (args.num_minibatches, -1) + x.shape[1:])
                return x

            flatten_storage = jax.tree_util.tree_map(flatten, storage)
            shuffled_storage = jax.tree_util.tree_map(convert_data, flatten_storage)

            def update_minibatch(carry, minibatch):
                agent_state = carry
                (loss, (pg_loss, v_loss, entropy_loss, approx_kl)), grads = ppo_loss_grad_fn(
                    agent_state.params,
                    minibatch.obs,
                    minibatch.actions,
                    minibatch.logprobs,
                    minibatch.advantages,
                    minibatch.returns,
                    minibatch.values,
                )
                agent_state = agent_state.apply_gradients(grads=grads)
                return agent_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads)

            agent_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads) = jax.lax.scan(
                update_minibatch, agent_state, shuffled_storage
            )
            return (agent_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads)

        (agent_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl, grads) = jax.lax.scan(
            update_epoch, (agent_state, key), (), length=args.update_epochs
        )
        return agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key

    # Start the game
    global_step = 0
    start_time = time.time()

    # Reset environment
    key, reset_key = jax.random.split(key)
    next_obs, env_state = reset_env(reset_key)
    obs_normalizer = ObsNormalizer.create(
        obs_shape,
        enabled=args.obs_normalize,
        clip=args.obs_norm_clip,
    )
    obs_normalizer = obs_normalizer.update(next_obs)
    next_obs = obs_normalizer.normalize(next_obs)
    next_done = jnp.zeros(args.n_envs, dtype=jnp.bool_)

    # Initialize reward normalizer (discounted return-based, like CleanRL)
    reward_normalizer = RewardNormalizer.create(n_envs=args.n_envs, gamma=args.gamma)

    def step_once(carry, step):
        agent_state, episode_stats, reward_norm, obs_norm, env_state, obs, done, key = carry
        action, logprob, value, key = get_action_and_value(agent_state, obs, key)

        key, next_obs_raw, env_state, raw_reward, next_done, rollout_info = step_env(
            key,
            env_state,
            action,
        )
        obs_norm = obs_norm.update(next_obs_raw)
        next_obs = obs_norm.normalize(next_obs_raw)

        # Update reward statistics only when normalization is enabled.
        if args.reward_normalize:
            reward_norm = reward_norm.update(raw_reward, next_done.astype(jnp.float32))
            reward = reward_norm.normalize(raw_reward)
        else:
            reward = raw_reward

        # Update episode statistics (use raw reward for tracking true returns)
        new_episode_return = episode_stats.episode_returns + raw_reward
        new_episode_length = episode_stats.episode_lengths + 1
        episode_stats = episode_stats.replace(
            # Use jnp.where to preserve dtypes
            episode_returns=jnp.where(next_done, 0.0, new_episode_return),
            episode_lengths=jnp.where(next_done, 0, new_episode_length).astype(jnp.int32),
            returned_episode_returns=jnp.where(
                next_done, new_episode_return, episode_stats.returned_episode_returns
            ),
            returned_episode_lengths=jnp.where(
                next_done, new_episode_length, episode_stats.returned_episode_lengths
            ).astype(jnp.int32),
        )

        storage = Storage(
            obs=obs,
            actions=action,
            logprobs=logprob,
            dones=done,
            values=value,
            rewards=reward,  # Use normalized reward for training
            returns=jnp.zeros_like(reward),
            advantages=jnp.zeros_like(reward),
        )
        return (
            agent_state,
            episode_stats,
            reward_norm,
            obs_norm,
            env_state,
            next_obs,
            next_done,
            key,
        ), (storage, rollout_info)

    def rollout(
        agent_state,
        episode_stats,
        reward_norm,
        obs_norm,
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
            obs_norm,
            env_state,
            next_obs,
            next_done,
            key,
        ), (storage, rollout_info) = jax.lax.scan(
            step_once,
            (agent_state, episode_stats, reward_norm, obs_norm, env_state, next_obs, next_done, key),
            jnp.arange(max_steps),
        )
        return (
            agent_state,
            episode_stats,
            reward_norm,
            obs_norm,
            env_state,
            next_obs,
            next_done,
            storage,
            rollout_info,
            key,
        )

    rollout = partial(rollout, max_steps=args.num_steps)
    rollout = jax.jit(rollout)

    print("Starting training...")
    for iteration in range(1, args.num_updates + 1):
        iteration_time_start = time.time()
        (
            agent_state,
            episode_stats,
            reward_normalizer,
            obs_normalizer,
            env_state,
            next_obs,
            next_done,
            storage,
            rollout_info,
            key,
        ) = rollout(
            agent_state,
            episode_stats,
            reward_normalizer,
            obs_normalizer,
            env_state,
            next_obs,
            next_done,
            key,
        )
        global_step += args.num_steps * args.n_envs
        storage = compute_gae(agent_state, next_obs, next_done, storage)
        agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key = update_ppo(
            agent_state,
            storage,
            key,
        )
        
        if iteration % args.log_interval == 0:
            avg_episodic_return = np.mean(jax.device_get(episode_stats.returned_episode_returns))
            avg_episodic_length = np.mean(jax.device_get(episode_stats.returned_episode_lengths))
            sps = int(global_step / (time.time() - start_time))
            sps_update = int(args.n_envs * args.num_steps / (time.time() - iteration_time_start))
            slippery_metrics = latest_slippery_metrics(rollout_info, args) if args.slippery else None
            stiefel_online_metrics = (
                online_stiefel_constraint_metrics(agent_state.opt_state)
                if args.heads_optimizer == "stiefel_online"
                else {}
            )
            matrix_metrics = matrix_constraint_metrics(agent_state.params)
            
            base_msg = (
                f"update={iteration} step={global_step} "
                f"ep_return={avg_episodic_return:.1f} "
                f"ep_len={avg_episodic_length * args.action_repeat:.0f} "  # Actual env steps
                f"loss={loss[-1, -1].item():.4f} "
                f"SPS={sps}"
            )
            if slippery_metrics is not None:
                base_msg += (
                    f" friction={slippery_metrics['slippery/friction_slide_first_env']:.4g}"
                    f" phase={slippery_metrics['slippery/phase_first_env']}"
                )
            if stiefel_online_metrics:
                base_msg += (
                    " stiefel_online_constraint="
                    f"{stiefel_online_metrics['optim/stiefel_online_constraint_max']:.4g}"
                    " stiefel_online_constraint_rms="
                    f"{stiefel_online_metrics['optim/stiefel_online_constraint_rms_max']:.4g}"
                )
            if matrix_metrics:
                base_msg += (
                    " matrix_constraint="
                    f"{matrix_metrics['optim/matrix_constraint_max']:.4g}"
                    " matrix_constraint_rms="
                    f"{matrix_metrics['optim/matrix_constraint_rms_max']:.4g}"
                )
            print(base_msg)
            
            if args.track:
                lr = float(linear_schedule(iteration * args.num_minibatches * args.update_epochs)) if args.anneal_lr else args.learning_rate
                log_dict = {
                    "global_step": global_step,
                    "charts/avg_episodic_return": avg_episodic_return,
                    "charts/avg_episodic_length": avg_episodic_length * args.action_repeat,
                    "charts/learning_rate": lr,
                    "charts/SPS": sps,
                    "charts/SPS_update": sps_update,
                    "losses/value_loss": v_loss[-1, -1].item(),
                    "losses/policy_loss": pg_loss[-1, -1].item(),
                    "losses/entropy": entropy_loss[-1, -1].item(),
                    "losses/approx_kl": approx_kl[-1, -1].item(),
                    "losses/loss": loss[-1, -1].item(),
                }

                if slippery_metrics is not None:
                    log_dict.update(slippery_metrics)
                log_dict.update(matrix_metrics)
                log_dict.update(stiefel_online_metrics)

                wandb.log(log_dict, step=global_step)

    elapsed = time.time() - start_time
    print(f"\nTraining finished in {elapsed:.1f}s")
    print(f"Average SPS: {args.total_timesteps / elapsed:.0f}")
    
    if args.track:
        wandb.finish()
