"""
Matched upstairs/downstairs CT-DDPG with linear Stiefel representations.

This is a self-contained experiment for the equivariance statement

    o = M s,  M.T M = I,
    X_up,0 = M X_down,0

where each actor/critic branch first computes z = X.T x and then applies an
arbitrary differentiable head.  The implementation uses three representations
(policy, value, and advantage) to preserve the separation in CT-DDPG.  The
same argument applies independently to all three.

Two head families are supported:

* structured:
    pi(z) = K z
    V(z) = z.T P z + b.T z + c
    Psi(z, a) = z.T Z a + r.T a - a.T R a
* mlp:
    independent tanh MLPs for pi, V, and learned Psi residual, with the
    same known -a.T R a control-cost curvature

The centered advantage rate is always

    psi(z, a) = Psi(z, a) - Psi(z, pi(z)).

Value and advantage parameters receive separate partial gradients of the same
multistep [r-psi] martingale residual. The code does not regress an
instantaneous advantage rate onto an accumulated residual. Stiefel encoders
use Cayley updates, head parameters use matched Adam states, and actor updates
begin after a critic warm-up.

Training uses one shared downstairs-behavior batch for the two models.  This
is the coupling required by the theorem.  Separate on-policy evaluation
rollouts use common random numbers and are only performance diagnostics.

The script also computes a counterfactual one-step probe at every iteration:
the current downstairs model is lifted exactly via X_up = M X_down, its heads
are copied, and one noisy upstairs update is compared with the downstairs
update without committing the probe parameters.  This isolates the local
O(eta * epsilon) discrepancy from accumulated closed-loop divergence. In the
clean condition, the independently measured one-step update is recorded before
removing roundoff drift from the invariant relation; disable that numerical
synchronization with --no-synchronize-clean.

Example smoke run:

    JAX_PLATFORMS=cpu uv run python \
      lqr_ac_compare/compare_linear_rep_nonlinear_heads.py \
      --iters 10 --T 1 --head-types structured mlp

Full default sweep:

    uv run python lqr_ac_compare/compare_linear_rep_nonlinear_heads.py
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib
import numpy as np
from scipy.linalg import solve_continuous_are

matplotlib.use("Agg")
import matplotlib.pyplot as plt

Array = np.ndarray
JaxArray = jax.Array
Params = Dict[str, Any]
ENCODER_KEYS = ("x_pi", "x_v", "x_a")


@dataclass(frozen=True)
class LQRProblem:
    d: int
    ds: int
    da: int
    M: Array
    G: Array
    H: Array
    Q: Array
    R: Array
    K_opt: Array


def stiefel_init(n: int, k: int, rng: np.random.Generator) -> Array:
    if n < k:
        raise ValueError(f"Stiefel matrix requires n >= k, got {n=} and {k=}")
    q, r = np.linalg.qr(rng.standard_normal((n, k)), mode="reduced")
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    return q * signs


def construct_problem(
    d: int,
    ds: int,
    da: int,
    alpha: float,
    seed: int,
) -> LQRProblem:
    """Construct the compatible LQR used by the original comparison."""
    if d < ds:
        raise ValueError(f"Need d >= ds, got {d=} and {ds=}")
    if ds < da:
        raise ValueError(f"Need ds >= da, got {ds=} and {da=}")

    rng = np.random.default_rng(seed)
    M = stiefel_init(d, ds, rng)
    H = stiefel_init(ds, da, rng)
    G = -alpha * np.eye(ds)
    R = np.eye(da)
    Q = H @ H.T + 2.0 * alpha * np.eye(ds)

    # P = I solves the CARE, hence K_opt = R^-1 H.T P = H.T.
    return LQRProblem(
        d=d,
        ds=ds,
        da=da,
        M=M,
        G=G,
        H=H,
        Q=Q,
        R=R,
        K_opt=H.T,
    )


def init_mlp(
    dims: Sequence[int],
    rng: np.random.Generator,
    output_scale: float = 0.1,
) -> Tuple[Dict[str, JaxArray], ...]:
    layers: List[Dict[str, JaxArray]] = []
    for index, (fan_in, fan_out) in enumerate(zip(dims[:-1], dims[1:])):
        scale = np.sqrt(2.0 / fan_in)
        if index == len(dims) - 2:
            scale *= output_scale
        layers.append(
            {
                "w": jnp.asarray(scale * rng.standard_normal((fan_in, fan_out))),
                "b": jnp.zeros((fan_out,), dtype=jnp.float64),
            }
        )
    return tuple(layers)


def apply_mlp(layers: Sequence[Mapping[str, JaxArray]], x: JaxArray) -> JaxArray:
    for layer in layers[:-1]:
        x = jnp.tanh(x @ layer["w"] + layer["b"])
    return x @ layers[-1]["w"] + layers[-1]["b"]


def init_heads(
    head_type: str,
    k: int,
    da: int,
    hidden_dim: int,
    seed: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    if head_type == "structured":
        p_raw = rng.standard_normal((k, k))
        p = -0.05 * (p_raw + p_raw.T)
        return {
            "pi": {
                "K": jnp.asarray(0.05 * rng.standard_normal((da, k))),
            },
            "v": {
                "P": jnp.asarray(p),
                "b": jnp.asarray(0.01 * rng.standard_normal((k,))),
                "c": jnp.asarray(0.0),
            },
            "a": {
                # Z and r learn the state-action part of the Hamiltonian.
                # The known -a.T R a curvature is added in make_head_ops.
                "Z": jnp.zeros((k, da), dtype=jnp.float64),
                "r": jnp.zeros((da,), dtype=jnp.float64),
            },
        }
    if head_type == "mlp":
        return {
            "pi": init_mlp((k, hidden_dim, hidden_dim, da), rng),
            "v": init_mlp((k, hidden_dim, hidden_dim, 1), rng),
            "a": init_mlp((k + da, hidden_dim, hidden_dim, 1), rng),
        }
    raise ValueError(f"Unknown head type: {head_type}")


def tree_copy(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda x: jnp.array(x, copy=True), tree)


def tree_zeros_like(tree: Any) -> Any:
    return jax.tree_util.tree_map(jnp.zeros_like, tree)


def init_matched_models(
    problem: LQRProblem,
    k: int,
    head_type: str,
    hidden_dim: int,
    seed: int,
) -> Tuple[Params, Params]:
    rng = np.random.default_rng(seed)
    x_pi = stiefel_init(problem.ds, k, rng)
    x_v = stiefel_init(problem.ds, k, rng)
    x_a = stiefel_init(problem.ds, k, rng)
    heads = init_heads(head_type, k, problem.da, hidden_dim, seed + 1000)

    down = {
        "x_pi": jnp.asarray(x_pi),
        "x_v": jnp.asarray(x_v),
        "x_a": jnp.asarray(x_a),
        "h_pi": tree_copy(heads["pi"]),
        "h_v": tree_copy(heads["v"]),
        "h_a": tree_copy(heads["a"]),
    }
    for branch in ("pi", "v", "a"):
        head = down[f"h_{branch}"]
        down[f"adam_m_{branch}"] = tree_zeros_like(head)
        down[f"adam_v_{branch}"] = tree_zeros_like(head)
        down[f"adam_step_{branch}"] = jnp.asarray(0, dtype=jnp.int32)
    up = lift_downstairs_model(down, problem.M)
    return down, up


def lift_downstairs_model(down: Params, M: Array) -> Params:
    M_jax = jnp.asarray(M)
    lifted = {
        "x_pi": M_jax @ down["x_pi"],
        "x_v": M_jax @ down["x_v"],
        "x_a": M_jax @ down["x_a"],
        "h_pi": tree_copy(down["h_pi"]),
        "h_v": tree_copy(down["h_v"]),
        "h_a": tree_copy(down["h_a"]),
    }
    for key, value in down.items():
        if key not in lifted and key not in ENCODER_KEYS:
            lifted[key] = tree_copy(value)
    return lifted


def make_head_ops(
    head_type: str,
    action_cost: Array,
) -> Dict[str, Callable[..., Any]]:
    action_cost_jax = jnp.asarray(action_cost)
    if head_type == "structured":

        def policy(head: Mapping[str, JaxArray], z: JaxArray) -> JaxArray:
            return head["K"] @ z

        def value(head: Mapping[str, JaxArray], z: JaxArray) -> JaxArray:
            p = 0.5 * (head["P"] + head["P"].T)
            return z @ p @ z + head["b"] @ z + head["c"]

        def raw_advantage(
            head: Mapping[str, JaxArray],
            z: JaxArray,
            action: JaxArray,
        ) -> JaxArray:
            return (
                z @ head["Z"] @ action
                + head["r"] @ action
                - action @ action_cost_jax @ action
            )

    elif head_type == "mlp":

        def policy(head: Any, z: JaxArray) -> JaxArray:
            return apply_mlp(head, z)

        def value(head: Any, z: JaxArray) -> JaxArray:
            return jnp.squeeze(apply_mlp(head, z), axis=-1)

        def raw_advantage(head: Any, z: JaxArray, action: JaxArray) -> JaxArray:
            learned = jnp.squeeze(
                apply_mlp(head, jnp.concatenate((z, action), axis=-1)),
                axis=-1,
            )
            return learned - action @ action_cost_jax @ action

    else:
        raise ValueError(f"Unknown head type: {head_type}")

    policy_batch = jax.vmap(policy, in_axes=(None, 0))
    value_batch = jax.vmap(value, in_axes=(None, 0))
    raw_advantage_batch = jax.vmap(raw_advantage, in_axes=(None, 0, 0))
    action_gradient_batch = jax.vmap(
        jax.grad(raw_advantage, argnums=2),
        in_axes=(None, 0, 0),
    )

    def predict(
        x_pi: JaxArray,
        h_pi: Any,
        x_v: JaxArray,
        h_v: Any,
        x_a: JaxArray,
        h_a: Any,
        observations: JaxArray,
        actions: JaxArray,
    ) -> Tuple[JaxArray, JaxArray, JaxArray]:
        z_pi = observations @ x_pi
        z_v = observations @ x_v
        z_a = observations @ x_a
        pi_actions = policy_batch(h_pi, z_pi)
        values = value_batch(h_v, z_v)
        raw_actions = raw_advantage_batch(h_a, z_a[:-1], actions)
        raw_policy = raw_advantage_batch(h_a, z_a[:-1], pi_actions[:-1])
        return values, pi_actions, raw_actions - raw_policy

    def value_loss(
        x_v: JaxArray,
        h_v: Any,
        observations: JaxArray,
        targets: JaxArray,
    ) -> JaxArray:
        values = value_batch(h_v, observations @ x_v)
        return 0.5 * jnp.mean(jnp.square(values - targets))

    def advantage_loss(
        x_a: JaxArray,
        h_a: Any,
        observations: JaxArray,
        actions: JaxArray,
        policy_actions: JaxArray,
        base_residual: JaxArray,
        weights: JaxArray,
        dt: JaxArray,
    ) -> JaxArray:
        z_a = observations[:-1] @ x_a
        psi = raw_advantage_batch(h_a, z_a, actions) - raw_advantage_batch(
            h_a, z_a, policy_actions
        )
        # Differentiate the same [r-q] martingale residual used by the value
        # loss. Regressing q toward a residual that already contains q has a
        # spurious q -> 0 fixed point.
        integrated_psi = jnp.convolve(psi, weights[::-1], mode="valid") * dt
        residual = base_residual + integrated_psi
        return 0.5 * jnp.mean(jnp.square(residual))

    def actor_objective(
        x_pi: JaxArray,
        h_pi: Any,
        observations: JaxArray,
        x_a: JaxArray,
        h_a: Any,
        dt: JaxArray,
    ) -> JaxArray:
        # x_a and h_a are critic constants because gradients are requested only
        # with respect to x_pi and h_pi.  This is the DDPG semigradient through
        # d Psi / d action, not the derivative of centered psi(z, pi(z)) == 0.
        z_pi = observations @ x_pi
        z_a = observations @ x_a
        actions = policy_batch(h_pi, z_pi)
        return dt * jnp.mean(raw_advantage_batch(h_a, z_a, actions))

    return {
        "policy_batch": jax.jit(policy_batch),
        "value_batch": jax.jit(value_batch),
        "raw_advantage_batch": jax.jit(raw_advantage_batch),
        "action_gradient_batch": jax.jit(action_gradient_batch),
        "predict": jax.jit(predict),
        "value_grad": jax.jit(jax.value_and_grad(value_loss, argnums=(0, 1))),
        "advantage_grad": jax.jit(jax.value_and_grad(advantage_loss, argnums=(0, 1))),
        "actor_grad": jax.jit(jax.value_and_grad(actor_objective, argnums=(0, 1))),
    }


def cayley_ascent(X: JaxArray, G: JaxArray, eta: float) -> JaxArray:
    """Move along G while preserving X.T X = I up to solve precision."""
    x = np.asarray(X)
    g = np.asarray(G)
    omega = g @ x.T - x @ g.T
    identity = np.eye(x.shape[0])
    # Expansion: X_plus = X + eta * (G - X G.T X) + O(eta^2).
    x_plus = np.linalg.solve(
        identity - 0.5 * eta * omega,
        (identity + 0.5 * eta * omega) @ x,
    )
    return jnp.asarray(x_plus)


def tree_step(tree: Any, grad: Any, scale: float) -> Any:
    return jax.tree_util.tree_map(lambda x, g: x + scale * g, tree, grad)


def tree_adam_step(
    tree: Any,
    grad: Any,
    first_moment: Any,
    second_moment: Any,
    step: JaxArray,
    learning_rate: float,
    ascent: bool = False,
) -> Tuple[Any, Any, Any, JaxArray]:
    """Adam for head parameters; identical head gradients give exact matching."""
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-6
    step_int = int(np.asarray(step)) + 1
    first_moment = jax.tree_util.tree_map(
        lambda m, g: beta1 * m + (1.0 - beta1) * g,
        first_moment,
        grad,
    )
    second_moment = jax.tree_util.tree_map(
        lambda v, g: beta2 * v + (1.0 - beta2) * jnp.square(g),
        second_moment,
        grad,
    )
    first_correction = 1.0 - beta1**step_int
    second_correction = 1.0 - beta2**step_int
    direction = 1.0 if ascent else -1.0
    tree = jax.tree_util.tree_map(
        lambda x, m, v: x
        + direction
        * learning_rate
        * (m / first_correction)
        / (jnp.sqrt(v / second_correction) + epsilon),
        tree,
        first_moment,
        second_moment,
    )
    return (
        tree,
        first_moment,
        second_moment,
        jnp.asarray(step_int, dtype=jnp.int32),
    )


def compute_martingale_terms(
    params: Params,
    observations: Array,
    actions: Array,
    rewards: Array,
    ops: Mapping[str, Callable[..., Any]],
    beta: float,
    dt: float,
    horizon: int,
) -> Tuple[Array, Array, Array, Array, Array, Array]:
    """Build detached terms for the two partial martingale gradients."""
    values_jax, pi_actions_jax, psi_jax = ops["predict"](
        params["x_pi"],
        params["h_pi"],
        params["x_v"],
        params["h_v"],
        params["x_a"],
        params["h_a"],
        jnp.asarray(observations),
        jnp.asarray(actions),
    )
    values = np.asarray(values_jax)
    pi_actions = np.asarray(pi_actions_jax)
    psi = np.asarray(psi_jax)

    n_steps = rewards.shape[0]
    horizon = min(horizon, n_steps)
    gamma = np.exp(-beta * dt)
    weights = gamma ** np.arange(horizon)
    reward_integral = np.correlate(rewards, weights, mode="valid") * dt
    psi_integral = np.correlate(psi, weights, mode="valid") * dt
    n_samples = reward_integral.shape[0]
    discounted_future = (gamma**horizon) * values[horizon:]

    # e = V_t - integral(r-q) - gamma^L V_{t+L}.  The value update
    # treats its target as fixed, while the q update differentiates the
    # integrated-q term in this same residual.
    base_residual = values[:n_samples] - reward_integral - discounted_future
    residual = base_residual + psi_integral
    targets = reward_integral - psi_integral + discounted_future
    return targets, base_residual, residual, values, pi_actions, weights


def ctddpg_update(
    params: Params,
    observations: Array,
    actions: Array,
    rewards: Array,
    ops: Mapping[str, Callable[..., Any]],
    beta: float,
    dt: float,
    horizon: int,
    eta_actor: float,
    eta_critic: float,
    eta_advantage: float,
    update_actor: bool = True,
) -> Tuple[Params, Dict[str, float]]:
    (
        targets,
        base_residual,
        residual,
        _,
        pi_actions,
        weights,
    ) = compute_martingale_terms(
        params,
        observations,
        actions,
        rewards,
        ops,
        beta,
        dt,
        horizon,
    )
    n_samples = targets.shape[0]
    obs_fit = jnp.asarray(observations[:n_samples])

    value_loss, (grad_x_v, grad_h_v) = ops["value_grad"](
        params["x_v"],
        params["h_v"],
        obs_fit,
        jnp.asarray(targets),
    )
    advantage_loss, (grad_x_a, grad_h_a) = ops["advantage_grad"](
        params["x_a"],
        params["h_a"],
        jnp.asarray(observations),
        jnp.asarray(actions),
        jnp.asarray(pi_actions[:-1]),
        jnp.asarray(base_residual),
        jnp.asarray(weights),
        jnp.asarray(dt),
    )

    updated = dict(params)
    updated["x_v"] = cayley_ascent(params["x_v"], -grad_x_v, eta_critic)
    (
        updated["h_v"],
        updated["adam_m_v"],
        updated["adam_v_v"],
        updated["adam_step_v"],
    ) = tree_adam_step(
        params["h_v"],
        grad_h_v,
        params["adam_m_v"],
        params["adam_v_v"],
        params["adam_step_v"],
        eta_critic,
    )
    updated["x_a"] = cayley_ascent(params["x_a"], -grad_x_a, eta_advantage)
    (
        updated["h_a"],
        updated["adam_m_a"],
        updated["adam_v_a"],
        updated["adam_step_a"],
    ) = tree_adam_step(
        params["h_a"],
        grad_h_a,
        params["adam_m_a"],
        params["adam_v_a"],
        params["adam_step_a"],
        eta_advantage,
    )

    actor_objective, (grad_x_pi, grad_h_pi) = ops["actor_grad"](
        params["x_pi"],
        params["h_pi"],
        obs_fit,
        updated["x_a"],
        updated["h_a"],
        jnp.asarray(dt),
    )
    if update_actor:
        updated["x_pi"] = cayley_ascent(params["x_pi"], grad_x_pi, eta_actor)
        (
            updated["h_pi"],
            updated["adam_m_pi"],
            updated["adam_v_pi"],
            updated["adam_step_pi"],
        ) = tree_adam_step(
            params["h_pi"],
            grad_h_pi,
            params["adam_m_pi"],
            params["adam_v_pi"],
            params["adam_step_pi"],
            eta_actor,
            ascent=True,
        )

    return updated, {
        "value_loss": float(value_loss),
        "advantage_loss": float(advantage_loss),
        "actor_objective": float(actor_objective),
        "martingale_residual_rms": float(np.sqrt(np.mean(np.square(residual)))),
        "grad_pi_norm": float(jnp.linalg.norm(grad_x_pi)),
        "grad_v_norm": float(jnp.linalg.norm(grad_x_v)),
        "grad_a_norm": float(jnp.linalg.norm(grad_x_a)),
        "actor_updated": float(update_actor),
    }


def policy_numpy(
    params: Params,
    observations: Array,
    ops: Mapping[str, Callable[..., Any]],
) -> Array:
    z = jnp.asarray(observations) @ params["x_pi"]
    return np.asarray(ops["policy_batch"](params["h_pi"], z))


def collect_shared_batch(
    down: Params,
    problem: LQRProblem,
    ops: Mapping[str, Callable[..., Any]],
    epsilon: float,
    dt: float,
    n_steps: int,
    exploration_std: float,
    transition_noise: float,
    action_clip: float,
    process_rng: np.random.Generator,
    exploration_rng: np.random.Generator,
    observation_rng: np.random.Generator,
) -> Tuple[Array, Array, Array, Array]:
    """Collect one common batch using the downstairs actor as behavior policy."""
    states = np.zeros((n_steps + 1, problem.ds))
    upstairs_observations = np.zeros((n_steps + 1, problem.d))
    actions = np.zeros((n_steps, problem.da))
    rewards = np.zeros((n_steps,))
    states[0] = 0.5

    for t in range(n_steps):
        s = states[t]
        upstairs_observations[t] = (
            problem.M @ s + epsilon * observation_rng.standard_normal(problem.d)
        )
        action = policy_numpy(down, s[None, :], ops)[0]
        action = action + exploration_std * exploration_rng.standard_normal(problem.da)
        action = np.clip(action, -action_clip, action_clip)

        cost = s @ problem.Q @ s + action @ problem.R @ action
        rewards[t] = -float(cost)
        drift = problem.G @ s + problem.H @ action
        states[t + 1] = (
            s
            + drift * dt
            + transition_noise * np.sqrt(dt) * process_rng.standard_normal(problem.ds)
        )
        actions[t] = action

    upstairs_observations[-1] = problem.M @ states[
        -1
    ] + epsilon * observation_rng.standard_normal(problem.d)
    return states, upstairs_observations, actions, rewards


def discounted_return(rewards: Array, beta: float, dt: float) -> float:
    weights = np.exp(-beta * dt * np.arange(rewards.shape[0]))
    return float(np.sum(weights * rewards) * dt)


def discounted_returns(rewards: Array, beta: float, dt: float) -> Array:
    """Discount a batch of reward trajectories along the final axis."""
    weights = np.exp(-beta * dt * np.arange(rewards.shape[-1]))
    return np.sum(rewards * weights[None, :], axis=-1) * dt


def discounted_lqr_gain(problem: LQRProblem, beta: float) -> Array:
    """Infinite-horizon gain for the beta-discounted continuous-time LQR."""
    discounted_drift = problem.G - 0.5 * beta * np.eye(problem.ds)
    p = solve_continuous_are(
        discounted_drift,
        problem.H,
        problem.Q,
        problem.R,
    )
    return np.linalg.solve(problem.R, problem.H.T @ p)


def mean_and_standard_error(values: Array) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if values.size < 2:
        return mean, 0.0
    return mean, float(np.std(values, ddof=1) / np.sqrt(values.size))


def evaluate_return_batch(
    params: Params,
    problem: LQRProblem,
    ops: Mapping[str, Callable[..., Any]],
    upstairs: bool,
    epsilon: float,
    dt: float,
    beta: float,
    process_noise: Array,
    observation_noise: Array,
    exploration_noise: Array,
    transition_noise: float,
    exploration_std: float,
    action_clip: float,
) -> Array:
    """Evaluate one policy on a fixed, vectorized bank of noise sequences."""
    n_episodes, n_steps, _ = process_noise.shape
    states = 0.5 * np.ones((n_episodes, problem.ds))
    rewards = np.zeros((n_episodes, n_steps))
    for t in range(n_steps):
        if upstairs:
            observation = states @ problem.M.T + epsilon * observation_noise[:, t]
        else:
            observation = states
        action = policy_numpy(params, observation, ops)
        action = action + exploration_std * exploration_noise[:, t]
        action = np.clip(action, -action_clip, action_clip)
        state_cost = np.einsum("bi,ij,bj->b", states, problem.Q, states)
        action_cost = np.einsum("bi,ij,bj->b", action, problem.R, action)
        rewards[:, t] = -(state_cost + action_cost)
        drift = states @ problem.G.T + action @ problem.H.T
        states = (
            states + drift * dt + transition_noise * np.sqrt(dt) * process_noise[:, t]
        )
    return discounted_returns(rewards, beta, dt)


def evaluate_linear_reference_batch(
    gain: Array,
    problem: LQRProblem,
    dt: float,
    beta: float,
    process_noise: Array,
    exploration_noise: Array,
    transition_noise: float,
    exploration_std: float,
    action_clip: float,
) -> Array:
    """Evaluate a = -K s on the same fixed bank used for learned policies."""
    n_episodes, n_steps, _ = process_noise.shape
    states = 0.5 * np.ones((n_episodes, problem.ds))
    rewards = np.zeros((n_episodes, n_steps))
    for t in range(n_steps):
        action = -states @ gain.T
        action = action + exploration_std * exploration_noise[:, t]
        action = np.clip(action, -action_clip, action_clip)
        state_cost = np.einsum("bi,ij,bj->b", states, problem.Q, states)
        action_cost = np.einsum("bi,ij,bj->b", action, problem.R, action)
        rewards[:, t] = -(state_cost + action_cost)
        drift = states @ problem.G.T + action @ problem.H.T
        states = (
            states + drift * dt + transition_noise * np.sqrt(dt) * process_noise[:, t]
        )
    return discounted_returns(rewards, beta, dt)


def tree_squared_norm(tree: Any) -> float:
    return float(
        sum(
            np.sum(np.square(np.asarray(leaf)))
            for leaf in jax.tree_util.tree_leaves(tree)
        )
    )


def tree_squared_difference(left: Any, right: Any) -> float:
    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    return float(
        sum(
            np.sum(np.square(np.asarray(a) - np.asarray(b)))
            for a, b in zip(left_leaves, right_leaves)
        )
    )


def head_discrepancy(down: Params, up: Params) -> float:
    numerator = 0.0
    denominator = 0.0
    for key in ("h_pi", "h_v", "h_a"):
        numerator += tree_squared_difference(up[key], down[key])
        denominator += tree_squared_norm(down[key])
    return float(np.sqrt(numerator) / (np.sqrt(denominator) + 1e-15))


def representation_metrics(
    down: Params,
    up: Params,
    M: Array,
) -> Dict[str, float]:
    projected_sq = 0.0
    full_sq = 0.0
    leakage_sq = 0.0
    orthogonality_sq = 0.0
    projector_perp = np.eye(M.shape[0]) - M @ M.T
    total_columns = 0

    for key in ENCODER_KEYS:
        xd = np.asarray(down[key])
        xu = np.asarray(up[key])
        projected_sq += np.linalg.norm(xu.T @ M - xd.T) ** 2
        full_sq += np.linalg.norm(xu - M @ xd) ** 2
        leakage_sq += np.linalg.norm(projector_perp @ xu) ** 2
        orthogonality_sq += np.linalg.norm(xd.T @ xd - np.eye(xd.shape[1])) ** 2
        orthogonality_sq += np.linalg.norm(xu.T @ xu - np.eye(xu.shape[1])) ** 2
        total_columns += xd.shape[1]

    scale = np.sqrt(total_columns)
    return {
        "rep_projected": float(np.sqrt(projected_sq) / scale),
        "rep_full": float(np.sqrt(full_sq) / scale),
        "rep_leakage": float(np.sqrt(leakage_sq) / scale),
        "orthogonality": float(
            np.sqrt(orthogonality_sq) / np.sqrt(2 * len(ENCODER_KEYS))
        ),
    }


def update_discrepancy(
    down_before: Params,
    down_after: Params,
    up_before: Params,
    up_after: Params,
    M: Array,
) -> Dict[str, float]:
    full_sq = 0.0
    projected_sq = 0.0
    total_columns = 0
    for key in ENCODER_KEYS:
        delta_down = np.asarray(down_after[key] - down_before[key])
        delta_up = np.asarray(up_after[key] - up_before[key])
        full_sq += np.linalg.norm(delta_up - M @ delta_down) ** 2
        projected_sq += np.linalg.norm(delta_up.T @ M - delta_down.T) ** 2
        total_columns += delta_down.shape[1]
    scale = np.sqrt(total_columns)
    return {
        "update_full": float(np.sqrt(full_sq) / scale),
        "update_projected": float(np.sqrt(projected_sq) / scale),
    }


def output_discrepancies(
    down: Params,
    up: Params,
    problem: LQRProblem,
    ops: Mapping[str, Callable[..., Any]],
    states: Array,
    action_offsets: Array,
    reference_gain: Array,
) -> Dict[str, float]:
    clean_upstairs = states @ problem.M.T
    down_values, down_pi, _ = ops["predict"](
        down["x_pi"],
        down["h_pi"],
        down["x_v"],
        down["h_v"],
        down["x_a"],
        down["h_a"],
        jnp.asarray(states),
        jnp.zeros((states.shape[0] - 1, problem.da)),
    )
    up_values, up_pi, _ = ops["predict"](
        up["x_pi"],
        up["h_pi"],
        up["x_v"],
        up["h_v"],
        up["x_a"],
        up["h_a"],
        jnp.asarray(clean_upstairs),
        jnp.zeros((states.shape[0] - 1, problem.da)),
    )
    down_values_np = np.asarray(down_values)
    up_values_np = np.asarray(up_values)
    down_pi_np = np.asarray(down_pi)
    up_pi_np = np.asarray(up_pi)

    # Evaluate centered advantage away from a = pi, where it is not
    # identically zero.  The same physical actions are used on both sides.
    actions = down_pi_np[:-1] + action_offsets[: states.shape[0] - 1]
    z_a_down = jnp.asarray(states[:-1]) @ down["x_a"]
    z_a_up = jnp.asarray(clean_upstairs[:-1]) @ up["x_a"]
    raw_down = ops["raw_advantage_batch"](down["h_a"], z_a_down, jnp.asarray(actions))
    raw_down_pi = ops["raw_advantage_batch"](
        down["h_a"], z_a_down, jnp.asarray(down_pi_np[:-1])
    )
    raw_up = ops["raw_advantage_batch"](up["h_a"], z_a_up, jnp.asarray(actions))
    raw_up_pi = ops["raw_advantage_batch"](
        up["h_a"], z_a_up, jnp.asarray(up_pi_np[:-1])
    )
    psi_down = np.asarray(raw_down - raw_down_pi)
    psi_up = np.asarray(raw_up - raw_up_pi)

    reference_actions = -states @ reference_gain.T

    return {
        "policy_mse": float(np.mean(np.square(up_pi_np - down_pi_np))),
        "down_policy_reference_mse": float(
            np.mean(np.square(down_pi_np - reference_actions))
        ),
        "up_policy_reference_mse": float(
            np.mean(np.square(up_pi_np - reference_actions))
        ),
        "value_mse": float(np.mean(np.square(up_values_np - down_values_np))),
        "advantage_mse": float(np.mean(np.square(psi_up - psi_down))),
    }


def critic_gradient_diagnostics(
    params: Params,
    problem: LQRProblem,
    ops: Mapping[str, Callable[..., Any]],
    states: Array,
    reference_gain: Array,
) -> Dict[str, float]:
    """Compare the learned actor signal with the optimal LQR Hamiltonian."""
    actions = policy_numpy(params, states, ops)
    z_a = jnp.asarray(states) @ params["x_a"]
    learned = np.asarray(
        ops["action_gradient_batch"](
            params["h_a"],
            z_a,
            jnp.asarray(actions),
        )
    )
    optimal = -2.0 * (actions @ problem.R + states @ reference_gain.T @ problem.R)
    learned_norm = float(np.linalg.norm(learned))
    optimal_norm = float(np.linalg.norm(optimal))
    denominator = learned_norm * optimal_norm
    cosine = (
        float(np.sum(learned * optimal) / denominator)
        if denominator > 1e-15
        else np.nan
    )
    return {
        "critic_gradient_cosine": cosine,
        "critic_gradient_norm_ratio": learned_norm / (optimal_norm + 1e-15),
        "critic_gradient_mse": float(np.mean(np.square(learned - optimal))),
    }


def run_condition(
    problem: LQRProblem,
    head_type: str,
    epsilon: float,
    args: argparse.Namespace,
    ops: Mapping[str, Callable[..., Any]],
    eval_states: Array,
    eval_action_offsets: Array,
) -> List[Dict[str, float]]:
    down, up = init_matched_models(
        problem,
        args.k,
        head_type,
        args.hidden_dim,
        args.init_seed,
    )
    n_steps = int(round(args.T / args.dt))
    condition_seed = args.noise_seed
    process_rng = np.random.default_rng(condition_seed)
    exploration_rng = np.random.default_rng(condition_seed + 1)
    observation_rng = np.random.default_rng(condition_seed + 2)
    rows: List[Dict[str, float]] = []
    eta_actor = args.eta_actor
    if head_type == "mlp":
        eta_actor *= args.mlp_actor_lr_scale

    # Reuse this exact evaluation bank at every checkpoint so return changes
    # reflect policy changes rather than a new draw of environmental luck.
    eval_rng = np.random.default_rng(args.eval_seed)
    eval_process = eval_rng.standard_normal((args.eval_episodes, n_steps, problem.ds))
    eval_observation = eval_rng.standard_normal(
        (args.eval_episodes, n_steps, problem.d)
    )
    eval_exploration = eval_rng.standard_normal(
        (args.eval_episodes, n_steps, problem.da)
    )
    reference_gain = discounted_lqr_gain(problem, args.beta)
    lqr_returns = evaluate_linear_reference_batch(
        reference_gain,
        problem,
        args.dt,
        args.beta,
        eval_process,
        eval_exploration,
        args.transition_noise,
        args.eval_exploration_std,
        args.action_clip,
    )
    zero_returns = evaluate_linear_reference_batch(
        np.zeros_like(reference_gain),
        problem,
        args.dt,
        args.beta,
        eval_process,
        eval_exploration,
        args.transition_noise,
        args.eval_exploration_std,
        args.action_clip,
    )
    eval_return_lqr, eval_return_lqr_se = mean_and_standard_error(lqr_returns)
    eval_return_zero, eval_return_zero_se = mean_and_standard_error(zero_returns)

    for iteration in range(args.iters):
        states, upstairs_obs, actions, rewards = collect_shared_batch(
            down,
            problem,
            ops,
            epsilon,
            args.dt,
            n_steps,
            args.exploration_std,
            args.transition_noise,
            args.action_clip,
            process_rng,
            exploration_rng,
            observation_rng,
        )

        update_actor = iteration >= args.actor_warmup_iters
        if head_type == "mlp" and update_actor:
            update_actor = (
                iteration - args.actor_warmup_iters
            ) % args.mlp_actor_update_every == 0
        down_before = down
        up_before = up
        down_after, down_stats = ctddpg_update(
            down_before,
            states,
            actions,
            rewards,
            ops,
            args.beta,
            args.dt,
            args.n_steps,
            eta_actor,
            args.eta_critic,
            args.eta_advantage,
            update_actor,
        )
        up_after, up_stats = ctddpg_update(
            up_before,
            upstairs_obs,
            actions,
            rewards,
            ops,
            args.beta,
            args.dt,
            args.n_steps,
            eta_actor,
            args.eta_critic,
            args.eta_advantage,
            update_actor,
        )

        # Counterfactual local probe: lift the same downstairs starting point
        # before applying the noisy upstairs update.
        probe_up_before = lift_downstairs_model(down_before, problem.M)
        probe_up_after, _ = ctddpg_update(
            probe_up_before,
            upstairs_obs,
            actions,
            rewards,
            ops,
            args.beta,
            args.dt,
            args.n_steps,
            eta_actor,
            args.eta_critic,
            args.eta_advantage,
            update_actor,
        )
        probe_update = update_discrepancy(
            down_before,
            down_after,
            probe_up_before,
            probe_up_after,
            problem.M,
        )
        actual_update = update_discrepancy(
            down_before,
            down_after,
            up_before,
            up_after,
            problem.M,
        )

        # Long nonlinear runs can amplify roundoff from the random lift even
        # when every one-step update is equivariant. Project back to the exact
        # clean invariant relation after measuring the independent update.
        if epsilon == 0.0 and args.synchronize_clean:
            up_after = lift_downstairs_model(down_after, problem.M)

        down, up = down_after, up_after
        rep = representation_metrics(down, up, problem.M)
        outputs = output_discrepancies(
            down,
            up,
            problem,
            ops,
            eval_states,
            eval_action_offsets,
            reference_gain,
        )
        gradient_diagnostics = critic_gradient_diagnostics(
            down,
            problem,
            ops,
            eval_states,
            reference_gain,
        )

        eval_return_down = np.nan
        eval_return_down_se = np.nan
        eval_return_up = np.nan
        eval_return_up_se = np.nan
        eval_return_gap = np.nan
        eval_return_gap_se = np.nan
        if iteration % args.eval_every == 0 or iteration == args.iters - 1:
            down_returns = evaluate_return_batch(
                down,
                problem,
                ops,
                False,
                epsilon,
                args.dt,
                args.beta,
                eval_process,
                eval_observation,
                eval_exploration,
                args.transition_noise,
                args.eval_exploration_std,
                args.action_clip,
            )
            up_returns = evaluate_return_batch(
                up,
                problem,
                ops,
                True,
                epsilon,
                args.dt,
                args.beta,
                eval_process,
                eval_observation,
                eval_exploration,
                args.transition_noise,
                args.eval_exploration_std,
                args.action_clip,
            )
            eval_return_down, eval_return_down_se = mean_and_standard_error(
                down_returns
            )
            eval_return_up, eval_return_up_se = mean_and_standard_error(up_returns)
            eval_return_gap, eval_return_gap_se = mean_and_standard_error(
                up_returns - down_returns
            )

        row = {
            "head_type": head_type,
            "epsilon": float(epsilon),
            "iteration": iteration,
            "train_return": discounted_return(rewards, args.beta, args.dt),
            "eval_return_down": float(eval_return_down),
            "eval_return_down_se": float(eval_return_down_se),
            "eval_return_up": float(eval_return_up),
            "eval_return_up_se": float(eval_return_up_se),
            "eval_return_gap": float(eval_return_gap),
            "eval_return_gap_se": float(eval_return_gap_se),
            "eval_return_zero": eval_return_zero,
            "eval_return_zero_se": eval_return_zero_se,
            "eval_return_lqr": eval_return_lqr,
            "eval_return_lqr_se": eval_return_lqr_se,
            "head_discrepancy": head_discrepancy(down, up),
            "down_value_loss": down_stats["value_loss"],
            "up_value_loss": up_stats["value_loss"],
            "down_advantage_loss": down_stats["advantage_loss"],
            "up_advantage_loss": up_stats["advantage_loss"],
            "down_martingale_residual_rms": down_stats["martingale_residual_rms"],
            "up_martingale_residual_rms": up_stats["martingale_residual_rms"],
            "actor_updated": down_stats["actor_updated"],
            # DDPG maximizes actor_objective, so expose its negative using the
            # conventional minimization-oriented "policy loss" name.
            "down_policy_loss": -down_stats["actor_objective"],
            "up_policy_loss": -up_stats["actor_objective"],
            "actual_update_full": actual_update["update_full"],
            "actual_update_projected": actual_update["update_projected"],
            "probe_update_full": probe_update["update_full"],
            "probe_update_projected": probe_update["update_projected"],
            **rep,
            **outputs,
            **gradient_diagnostics,
        }
        rows.append(row)

        if epsilon == 0.0 and args.assert_clean_tol > 0.0:
            clean_error = max(
                row["rep_full"],
                row["head_discrepancy"],
                np.sqrt(row["policy_mse"]),
                np.sqrt(row["value_mse"]),
                np.sqrt(row["advantage_mse"]),
            )
            if clean_error > args.assert_clean_tol:
                raise AssertionError(
                    "Clean equivariance exceeded tolerance at "
                    f"iteration {iteration}: {clean_error:.3e} > "
                    f"{args.assert_clean_tol:.3e}"
                )

        if iteration % args.log_every == 0 or iteration == args.iters - 1:
            print(
                f"[{head_type:10s} eps={epsilon:0.3f} "
                f"iter={iteration:4d}] "
                f"E_full={row['rep_full']:.3e} "
                f"E_pi={row['policy_mse']:.3e} "
                f"D_probe={row['probe_update_full']:.3e}"
            )

    return rows


CSV_FIELDS = (
    "head_type",
    "epsilon",
    "iteration",
    "train_return",
    "eval_return_down",
    "eval_return_down_se",
    "eval_return_up",
    "eval_return_up_se",
    "eval_return_gap",
    "eval_return_gap_se",
    "eval_return_zero",
    "eval_return_zero_se",
    "eval_return_lqr",
    "eval_return_lqr_se",
    "rep_projected",
    "rep_full",
    "rep_leakage",
    "policy_mse",
    "down_policy_reference_mse",
    "up_policy_reference_mse",
    "value_mse",
    "advantage_mse",
    "head_discrepancy",
    "orthogonality",
    "actual_update_full",
    "actual_update_projected",
    "probe_update_full",
    "probe_update_projected",
    "down_value_loss",
    "up_value_loss",
    "down_advantage_loss",
    "up_advantage_loss",
    "down_martingale_residual_rms",
    "up_martingale_residual_rms",
    "actor_updated",
    "critic_gradient_cosine",
    "critic_gradient_norm_ratio",
    "critic_gradient_mse",
    "down_policy_loss",
    "up_policy_loss",
)


def write_csv(rows: Iterable[Mapping[str, Any]], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def clean_control_summary(
    rows: Sequence[Mapping[str, float]],
    head_types: Sequence[str],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for head_type in head_types:
        evaluated = [
            row
            for row in rows
            if str(row["head_type"]) == head_type
            and float(row["epsilon"]) == 0.0
            and np.isfinite(float(row["eval_return_down"]))
        ]
        if not evaluated:
            continue
        best = max(evaluated, key=lambda row: float(row["eval_return_down"]))
        initial = evaluated[0]
        final = evaluated[-1]
        summary[head_type] = {
            "initial_return": float(initial["eval_return_down"]),
            "final_return": float(final["eval_return_down"]),
            "best_return": float(best["eval_return_down"]),
            "best_iteration": int(best["iteration"]),
            "zero_action_return": float(initial["eval_return_zero"]),
            "discounted_lqr_return": float(initial["eval_return_lqr"]),
            "final_policy_reference_mse": float(final["down_policy_reference_mse"]),
            "max_upstairs_downstairs_return_gap": max(
                abs(float(row["eval_return_gap"])) for row in evaluated
            ),
        }
    return summary


def noise_scaling_summary(
    rows: Sequence[Mapping[str, float]],
    head_types: Sequence[str],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for head_type in head_types:
        head_rows = [r for r in rows if r["head_type"] == head_type]
        epsilons = sorted({float(r["epsilon"]) for r in head_rows})
        means = []
        for epsilon in epsilons:
            values = [
                float(r["probe_update_full"])
                for r in head_rows
                if float(r["epsilon"]) == epsilon
            ]
            means.append(float(np.mean(values)))

        positive = [
            (epsilon, error)
            for epsilon, error in zip(epsilons, means)
            if epsilon > 0.0 and error > 0.0
        ]
        log_log_slope = np.nan
        linear_coefficient = np.nan
        linear_r2 = np.nan
        if len(positive) >= 2:
            x = np.asarray([item[0] for item in positive])
            y = np.asarray([item[1] for item in positive])
            log_log_slope = float(np.polyfit(np.log(x), np.log(y), deg=1)[0])
            linear_coefficient = float(np.dot(x, y) / np.dot(x, x))
            prediction = linear_coefficient * x
            denominator = np.sum(np.square(y - np.mean(y)))
            linear_r2 = (
                float(1.0 - np.sum(np.square(y - prediction)) / denominator)
                if denominator > 0.0
                else np.nan
            )

        summary[head_type] = {
            "epsilon": epsilons,
            "mean_probe_update_full": means,
            "log_log_slope": log_log_slope,
            "linear_coefficient_through_origin": linear_coefficient,
            "linear_r2_through_origin": linear_r2,
        }
    return summary


def plot_training(
    rows: Sequence[Mapping[str, float]],
    head_type: str,
    output_path: Path,
) -> None:
    head_rows = [r for r in rows if r["head_type"] == head_type]
    epsilons = sorted({float(r["epsilon"]) for r in head_rows})
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(epsilons)))
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=False)
    axes_flat = axes.ravel()

    panels = (
        ("rep_projected", r"$E_{\mathrm{rep}}$", True),
        ("rep_full", r"$E_{\mathrm{full}}$", True),
        ("rep_leakage", r"$E_{\perp}$", True),
        ("policy_mse", "Actor output MSE", True),
        ("value_mse", "Value output MSE", True),
        ("advantage_mse", "Advantage-rate MSE", True),
        ("head_discrepancy", "Head parameter discrepancy", True),
        ("probe_update_full", "One-step probe discrepancy", True),
    )
    for axis, (metric, title, use_log) in zip(axes_flat[:8], panels):
        for color, epsilon in zip(colors, epsilons):
            condition = [r for r in head_rows if float(r["epsilon"]) == epsilon]
            x = [int(r["iteration"]) for r in condition]
            y = np.maximum(
                [float(r[metric]) for r in condition],
                np.finfo(float).tiny,
            )
            axis.plot(x, y, color=color, label=f"{epsilon:g}")
        if use_log:
            axis.set_yscale("log")
        axis.set_title(title)
        axis.set_xlabel("Iteration")
        axis.grid(True, alpha=0.25)

    return_axis = axes_flat[8]
    upstairs_return_colors = colors if len(epsilons) > 1 else np.asarray(["#D55E00"])
    for color, epsilon in zip(upstairs_return_colors, epsilons):
        condition = [
            r
            for r in head_rows
            if float(r["epsilon"]) == epsilon
            and np.isfinite(float(r["eval_return_up"]))
        ]
        x = np.asarray([int(r["iteration"]) for r in condition])
        y = np.asarray([float(r["eval_return_up"]) for r in condition])
        se = np.asarray([float(r["eval_return_up_se"]) for r in condition])
        return_axis.plot(
            x,
            y,
            color=color,
            linewidth=4.5,
            alpha=0.65,
            marker="x",
            markevery=(2, 10),
            label=f"upstairs eps={epsilon:g}",
        )
        return_axis.fill_between(x, y - se, y + se, color=color, alpha=0.06)
    clean_down = [
        r
        for r in head_rows
        if float(r["epsilon"]) == epsilons[0]
        and np.isfinite(float(r["eval_return_down"]))
    ]
    x_down = np.asarray([int(r["iteration"]) for r in clean_down])
    y_down = np.asarray([float(r["eval_return_down"]) for r in clean_down])
    se_down = np.asarray([float(r["eval_return_down_se"]) for r in clean_down])
    return_axis.plot(
        x_down,
        y_down,
        color="#0072B2",
        linestyle="-",
        linewidth=1.8,
        marker="o",
        markerfacecolor="white",
        markevery=(0, 10),
        label="downstairs",
    )
    return_axis.fill_between(
        x_down,
        y_down - se_down,
        y_down + se_down,
        color="#0072B2",
        alpha=0.08,
    )
    if clean_down:
        baseline = clean_down[0]
        return_axis.axhline(
            float(baseline["eval_return_zero"]),
            color="0.45",
            linestyle=":",
            label="zero action",
        )
        return_axis.axhline(
            float(baseline["eval_return_lqr"]),
            color="#009E73",
            linestyle="-.",
            label="discounted LQR",
        )
    return_axis.set_title("Fixed-bank on-policy evaluation return")
    return_axis.set_xlabel("Iteration")
    return_axis.grid(True, alpha=0.25)
    return_axis.legend(fontsize=8)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title=r"Observation noise $\epsilon$",
        loc="upper center",
        ncol=max(1, len(epsilons)),
    )
    fig.suptitle(
        f"Linear Stiefel representation with {head_type} heads",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_clean_control_comparison(
    rows: Sequence[Mapping[str, float]],
    output_path: Path,
) -> None:
    clean_rows = [row for row in rows if float(row["epsilon"]) == 0.0]
    if not clean_rows:
        return

    head_types = sorted({str(row["head_type"]) for row in clean_rows})
    head_styles = {"structured": "-", "mlp": "--"}
    head_markers = {"structured": "o", "mlp": "s"}
    downstairs_color = "#0072B2"
    upstairs_color = "#D55E00"
    diagnostic_colors = plt.cm.tab10(np.arange(len(head_types)))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    return_axis, policy_axis, reference_axis, gradient_axis = axes.ravel()

    for index, (diagnostic_color, head_type) in enumerate(
        zip(diagnostic_colors, head_types)
    ):
        condition = [row for row in clean_rows if str(row["head_type"]) == head_type]
        iterations = np.asarray([int(row["iteration"]) for row in condition])
        evaluated = [
            row for row in condition if np.isfinite(float(row["eval_return_down"]))
        ]
        eval_iterations = np.asarray([int(row["iteration"]) for row in evaluated])
        down_return = np.asarray([float(row["eval_return_down"]) for row in evaluated])
        up_return = np.asarray([float(row["eval_return_up"]) for row in evaluated])
        return_se = np.asarray([float(row["eval_return_down_se"]) for row in evaluated])
        line_style = head_styles.get(head_type, "-")
        marker = head_markers.get(head_type, "o")

        # Draw the wider upstairs trace first. The narrow downstairs trace and
        # staggered markers remain visible even when both curves coincide.
        return_axis.plot(
            eval_iterations,
            up_return,
            color=upstairs_color,
            linestyle=line_style,
            linewidth=4.5,
            alpha=0.65,
            marker="x",
            markevery=(2 + index, 10),
            label=f"{head_type} upstairs",
        )
        return_axis.plot(
            eval_iterations,
            down_return,
            color=downstairs_color,
            linestyle=line_style,
            linewidth=1.8,
            marker=marker,
            markerfacecolor="white",
            markevery=(index, 10),
            label=f"{head_type} downstairs",
        )
        return_axis.fill_between(
            eval_iterations,
            down_return - return_se,
            down_return + return_se,
            color=downstairs_color,
            alpha=0.08,
        )

        down_policy_loss = [float(row["down_policy_loss"]) for row in condition]
        up_policy_loss = [float(row["up_policy_loss"]) for row in condition]
        policy_axis.plot(
            iterations,
            up_policy_loss,
            color=upstairs_color,
            linestyle=line_style,
            linewidth=4.0,
            alpha=0.65,
            marker="x",
            markevery=(40 + 10 * index, 100),
            label=f"{head_type} upstairs",
        )
        policy_axis.plot(
            iterations,
            down_policy_loss,
            color=downstairs_color,
            linestyle=line_style,
            linewidth=1.6,
            marker=marker,
            markerfacecolor="white",
            markevery=(10 * index, 100),
            label=f"{head_type} downstairs",
        )

        down_reference = [float(row["down_policy_reference_mse"]) for row in condition]
        up_reference = [float(row["up_policy_reference_mse"]) for row in condition]
        reference_axis.plot(
            iterations,
            up_reference,
            color=upstairs_color,
            linestyle=line_style,
            linewidth=4.0,
            alpha=0.65,
            marker="x",
            markevery=(40 + 10 * index, 100),
            label=f"{head_type} upstairs",
        )
        reference_axis.plot(
            iterations,
            down_reference,
            color=downstairs_color,
            linestyle=line_style,
            linewidth=1.6,
            marker=marker,
            markerfacecolor="white",
            markevery=(10 * index, 100),
            label=f"{head_type} downstairs",
        )
        gradient_axis.plot(
            iterations,
            [float(row["critic_gradient_cosine"]) for row in condition],
            color=diagnostic_color,
            linestyle=line_style,
            label=head_type,
        )

    baseline = clean_rows[0]
    return_axis.axhline(
        float(baseline["eval_return_zero"]),
        color="0.45",
        linestyle=":",
        label="zero action",
    )
    return_axis.axhline(
        float(baseline["eval_return_lqr"]),
        color="#009E73",
        linestyle="-.",
        label="discounted LQR",
    )
    gradient_axis.axhline(0.0, color="0.45", linewidth=1)

    return_axis.set_title("Fixed-bank discounted return")
    policy_axis.set_title("Policy loss (negative actor objective)")
    reference_axis.set_title("Policy error versus discounted LQR")
    gradient_axis.set_title("Critic action-gradient cosine versus LQR")
    reference_axis.set_yscale("log")
    for axis in axes.ravel():
        axis.set_xlabel("Iteration")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8, ncol=2)
    fig.suptitle("Clean observation experiment: downstairs blue, upstairs orange")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_noise_scaling(
    summary: Mapping[str, Any],
    output_path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(7, 5))
    for head_type, values in summary.items():
        epsilon = np.asarray(values["epsilon"])
        error = np.asarray(values["mean_probe_update_full"])
        mask = (epsilon > 0.0) & (error > 0.0)
        axis.loglog(
            epsilon[mask],
            error[mask],
            marker="o",
            label=(f"{head_type} " f"(slope={values['log_log_slope']:.2f})"),
        )
    axis.set_xlabel(r"Observation noise $\epsilon$")
    axis.set_ylabel("Mean one-step full update discrepancy")
    axis.set_title(r"Local equivariance error versus $\epsilon$")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Matched Stiefel representations with structured or nonlinear "
            "heads under the CT-DDPG martingale loss."
        )
    )
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--ds", type=int, default=4)
    parser.add_argument("--da", type=int, default=1)
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Representation dimension; defaults to ds.",
    )
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument(
        "--head-types",
        choices=("structured", "mlp"),
        nargs="+",
        default=("structured", "mlp"),
    )
    parser.add_argument(
        "--observation-noises",
        type=float,
        nargs="+",
        default=(0.0, 0.01, 0.025, 0.05, 0.1),
    )
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--T", type=float, default=10.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument(
        "--n-steps",
        type=int,
        default=30,
        help="Bootstrapping horizon in environment steps.",
    )
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--eta-actor", type=float, default=0.01)
    parser.add_argument(
        "--mlp-actor-lr-scale",
        type=float,
        default=0.05,
        help="Multiplier on eta-actor for the more sensitive MLP policy.",
    )
    parser.add_argument("--eta-critic", type=float, default=0.01)
    parser.add_argument(
        "--eta-advantage",
        type=float,
        default=0.01,
        help="Learning rate for the integrated advantage martingale gradient.",
    )
    parser.add_argument(
        "--mlp-actor-update-every",
        type=int,
        default=8,
        help="Delayed-policy-update interval for the MLP actor.",
    )
    parser.add_argument(
        "--actor-warmup-iters",
        type=int,
        default=100,
        help="Critic-only iterations before deterministic policy updates.",
    )
    parser.add_argument("--exploration-std", type=float, default=0.1)
    parser.add_argument("--eval-exploration-std", type=float, default=0.0)
    parser.add_argument("--transition-noise", type=float, default=0.1)
    parser.add_argument("--action-clip", type=float, default=20.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-seed", type=int, default=1)
    parser.add_argument("--noise-seed", type=int, default=2)
    parser.add_argument("--eval-seed", type=int, default=3)
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=32,
        help="Fixed process-noise trajectories reused at every checkpoint.",
    )
    parser.add_argument("--eval-size", type=int, default=256)
    parser.add_argument("--eval-action-std", type=float, default=0.2)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--no-synchronize-clean",
        action="store_false",
        dest="synchronize_clean",
        help=(
            "Do not remove floating-point drift from the epsilon=0 invariant "
            "relation after recording each independent one-step update."
        ),
    )
    parser.set_defaults(synchronize_clean=True)
    parser.add_argument(
        "--assert-clean-tol",
        type=float,
        default=1e-8,
        help="Fail if clean equivariance exceeds this; set <=0 to disable.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lqr_ac_compare/linear_rep_head_results"),
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.k is None:
        args.k = args.ds
    if not 1 <= args.k <= args.ds:
        raise ValueError(f"k must satisfy 1 <= k <= ds, got {args.k}")
    if args.dt <= 0.0 or args.T <= 0.0:
        raise ValueError("dt and T must be positive")
    if min(args.eta_actor, args.eta_critic, args.eta_advantage) <= 0.0:
        raise ValueError("all learning rates must be positive")
    if args.mlp_actor_lr_scale <= 0.0:
        raise ValueError("mlp-actor-lr-scale must be positive")
    if args.mlp_actor_update_every < 1:
        raise ValueError("mlp-actor-update-every must be positive")
    if int(round(args.T / args.dt)) < args.n_steps:
        raise ValueError("Rollout length T/dt must be at least n_steps")
    if args.iters < 1:
        raise ValueError("iters must be positive")
    if args.actor_warmup_iters < 0:
        raise ValueError("actor-warmup-iters must be nonnegative")
    if args.eval_every < 1 or args.log_every < 1:
        raise ValueError("eval-every and log-every must be positive")
    if args.eval_episodes < 1:
        raise ValueError("eval-episodes must be positive")
    if any(epsilon < 0.0 for epsilon in args.observation_noises):
        raise ValueError("Observation-noise values must be nonnegative")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    problem = construct_problem(
        args.d,
        args.ds,
        args.da,
        args.alpha,
        args.seed,
    )
    eval_rng = np.random.default_rng(args.eval_seed)
    eval_states = eval_rng.standard_normal((args.eval_size + 1, args.ds))
    eval_action_offsets = args.eval_action_std * eval_rng.standard_normal(
        (args.eval_size, args.da)
    )

    print("Linear-representation upstairs/downstairs CT-DDPG")
    print(
        f"d/ds/da/k={args.d}/{args.ds}/{args.da}/{args.k}, "
        f"heads={list(args.head_types)}, eps={list(args.observation_noises)}"
    )
    print(
        f"||M.T M - I||_F="
        f"{np.linalg.norm(problem.M.T @ problem.M - np.eye(args.ds)):.3e}"
    )

    all_rows: List[Dict[str, float]] = []
    for head_type in args.head_types:
        ops = make_head_ops(head_type, problem.R)
        for epsilon in args.observation_noises:
            all_rows.extend(
                run_condition(
                    problem,
                    head_type,
                    float(epsilon),
                    args,
                    ops,
                    eval_states,
                    eval_action_offsets,
                )
            )

    csv_path = args.output_dir / "metrics.csv"
    write_csv(all_rows, csv_path)
    scaling = noise_scaling_summary(all_rows, args.head_types)
    summary = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "M_orthogonality_residual": float(
            np.linalg.norm(problem.M.T @ problem.M - np.eye(args.ds))
        ),
        "noise_scaling": scaling,
        "clean_control": clean_control_summary(all_rows, args.head_types),
    }
    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)

    if not args.no_plots:
        for head_type in args.head_types:
            plot_training(
                all_rows,
                head_type,
                args.output_dir / f"{head_type}_training.png",
            )
        plot_clean_control_comparison(
            all_rows,
            args.output_dir / "epsilon0_control_comparison.png",
        )
        plot_noise_scaling(
            scaling,
            args.output_dir / "noise_scaling.png",
        )

    print(f"Saved metrics: {csv_path}")
    print(f"Saved summary: {summary_path}")
    for head_type, values in scaling.items():
        print(f"{head_type} local noise slope: " f"{values['log_log_slope']:.3f}")


if __name__ == "__main__":
    main()
