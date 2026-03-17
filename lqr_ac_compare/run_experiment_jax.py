
"""JAX version of the LQR actor-critic experiment."""

import argparse
import os
import time

import jax
import jax.numpy as jnp
from jax import random
import numpy as np  # only for scipy compatibility in env init

from lqr_env_jax import LQRParams, LQRUpDownEnv
from models_jax import (
    init_actor_up, init_critic_up, actor_up_act,
    init_actor_dn, init_critic_dn, actor_dn_act,
)
from learning_jax import (
    upstairs_actor_critic_update,
    downstairs_actor_critic_update,
)


def truncate_traj(traj: dict, steps: int):
    steps = int(steps)
    return {
        "s": traj["s"][: steps + 1],
        "o": traj["o"][: steps + 1],
        "a": traj["a"][: steps],
        "r": traj["r"][: steps],
    }


def compute_discounted_return(rewards: jnp.ndarray, beta: float, dt: float) -> float:
    N = len(rewards)
    discount = jnp.exp(-beta * jnp.arange(N) * dt)
    return float(jnp.sum(discount * rewards) * dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--ds", type=int, default=4)
    ap.add_argument("--da", type=int, default=1)

    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--T", type=float, default=2.0)
    ap.add_argument("--horizon_steps", type=int, default=200)

    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--exploration_std", type=float, default=0.05)
    ap.add_argument("--eps_obs", type=float, default=0.05)
    ap.add_argument("--transition_noise", type=float, default=1.0)
    ap.add_argument("--reward_state_scale", type=float, default=1.0)
    ap.add_argument("--reward_action_scale", type=float, default=1.0)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--shared_trajectory", action="store_true")

    # Learning rates
    ap.add_argument("--eta_actor_up", type=float, default=2e-3)
    ap.add_argument("--eta_value_up", type=float, default=2e-3)
    ap.add_argument("--eta_adv_up", type=float, default=2e-3)

    ap.add_argument("--eta_down", type=float, default=1e-3)

    ap.add_argument("--seed", type=int, default=0)

    # Wandb
    ap.add_argument("--track", action="store_true")
    ap.add_argument("--wandb_project", type=str, default="benchmark")
    ap.add_argument("--wandb_entity", type=str, default=None)
    ap.add_argument("--exp_name", type=str, default="lqr_ac_compare_jax")

    args = ap.parse_args()

    # Initialize JAX random key
    key = random.PRNGKey(args.seed)

    # Initialize wandb
    if args.track:
        import wandb
        run_name = f"{args.exp_name}_d{args.d}_ds{args.ds}_seed{args.seed}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
        )

    # Create environment (use jnp arrays for GPU)
    params = LQRParams(
        d=args.d, ds=args.ds, da=args.da,
        dt=args.dt, T=args.T,
        sigma_explore=0.05, eps_obs=args.eps_obs,
        seed=args.seed,
        Sigma_base=args.transition_noise * jnp.eye(args.ds),
        Q=args.reward_state_scale * jnp.eye(args.ds),
        R=args.reward_action_scale * jnp.eye(args.da),
    )
    env = LQRUpDownEnv(params)

    # Initialize models
    key, k1, k2, k3, k4 = random.split(key, 5)
    actor_up = init_actor_up(k1, args.d, args.da)
    critic_up = init_critic_up(k2, args.d, args.da)
    actor_dn = init_actor_dn(k3, args.ds, args.da)
    critic_dn = init_critic_dn(k4, args.ds, args.da)

    # Verify arrays are on GPU
    print(f"Actor W1 device: {actor_up.W1.devices()}")

    print("\n=== Running JAX upstairs vs downstairs learning ===")
    print(f"JAX devices: {jax.devices()}")
    print(f"d={args.d}, ds={args.ds}, da={args.da}, dt={args.dt}, T={args.T}, steps={env.n_steps}")
    print(f"iters={args.iters}, beta={args.beta}, exploration_std={args.exploration_std}")
    print(f"eps_obs={args.eps_obs}, transition_noise={args.transition_noise}")
    print(f"reward scales: state={args.reward_state_scale}, action={args.reward_action_scale}")
    print(f"shared_trajectory: {args.shared_trajectory}")
    print(f"upstairs etas: actor/value/adv = {args.eta_actor_up}/{args.eta_value_up}/{args.eta_adv_up}")
    print(f"downstairs eta = {args.eta_down}")

    # Print optimal LQR solution
    opt_info = env.optimal_average_reward(exploration_std=args.exploration_std)
    print(f"\n=== Optimal LQR Solution ===")
    print(f"K_opt shape: {opt_info['K_opt'].shape}")
    print(f"Initial reward (at s0): {opt_info['initial_reward']:.4f}")
    print(f"Steady-state avg reward (with noise): {opt_info['steady_state_avg_reward']:.4f}")
    print(f"V(s0) total cost-to-go: {opt_info['V_s0']:.4f}\n")

    # JIT compile the rollout
    print("JIT compiling... ", end="", flush=True)
    start_jit = time.time()

    # Warm-up JIT compilation
    key, subkey = random.split(key)
    _ = env.rollout(subkey, lambda obs: actor_dn_act(actor_dn, obs["downstairs"]),
                    exploration_std=args.exploration_std)
    jit_time = time.time() - start_jit
    print(f"done ({jit_time:.2f}s)")

    print("\nStarting training...")
    start_time = time.time()

    for it in range(args.iters):
        iter_start = time.time()

        if args.shared_trajectory:
            # Use downstairs policy to generate shared trajectory
            key, subkey = random.split(key)
            traj = env.rollout(
                subkey,
                lambda obs: actor_dn_act(actor_dn, obs["downstairs"]),
                exploration_std=args.exploration_std
            )
            if args.horizon_steps > 0:
                Nkeep = min(args.horizon_steps, traj["a"].shape[0])
                traj = truncate_traj(traj, Nkeep)
            traj_up = traj
            traj_dn = traj
        else:
            # Separate trajectories
            key, k1, k2 = random.split(key, 3)

            traj_up = env.rollout(
                k1,
                lambda obs: actor_up_act(actor_up, obs["upstairs"]),
                exploration_std=args.exploration_std
            )
            if args.horizon_steps > 0:
                Nkeep = min(args.horizon_steps, traj_up["a"].shape[0])
                traj_up = truncate_traj(traj_up, Nkeep)

            traj_dn = env.rollout(
                k2,
                lambda obs: actor_dn_act(actor_dn, obs["downstairs"]),
                exploration_std=args.exploration_std
            )
            if args.horizon_steps > 0:
                Nkeep = min(args.horizon_steps, traj_dn["a"].shape[0])
                traj_dn = truncate_traj(traj_dn, Nkeep)

        # Upstairs update
        actor_up, critic_up, stats_up = upstairs_actor_critic_update(
            actor_up, critic_up,
            traj_up["o"], traj_up["a"], traj_up["r"],
            beta=args.beta, dt=args.dt,
            eta_actor=args.eta_actor_up,
            eta_value=args.eta_value_up,
            eta_adv=args.eta_adv_up,
        )

        # Downstairs update
        actor_dn, critic_dn, stats_dn = downstairs_actor_critic_update(
            actor_dn, critic_dn,
            traj_dn["s"], traj_dn["a"], traj_dn["r"],
            beta=args.beta, dt=args.dt,
            L_eff=3,
            eta_actor=args.eta_down,
            eta_value=args.eta_down,
            eta_adv=args.eta_down,
        )

        J_up = float(traj_up["r"].mean())
        J_dn = float(traj_dn["r"].mean())

        J_up_discounted = compute_discounted_return(traj_up["r"], args.beta, args.dt)
        J_dn_discounted = compute_discounted_return(traj_dn["r"], args.beta, args.dt)

        iter_time = time.time() - iter_start

        print(f"[iter {it+1:03d}] up: r={J_up:+.4f}, td={stats_up['td_mse_batch']:.2e} | "
              f"dn: r={J_dn:+.4f}, td={stats_dn['td_mse']:.2e} | {iter_time*1000:.1f}ms")

        if args.track:
            log_dict = {
                "iteration": it + 1,
                "upstairs/mean_reward": J_up,
                "upstairs/discounted_return": J_up_discounted,
                "upstairs/td_mse_batch": stats_up["td_mse_batch"],
                "upstairs/ortho_err_W1": stats_up["ortho_err_W1"],
                "upstairs/ortho_err_U1": stats_up["ortho_err_U1"],
                "downstairs/mean_reward": J_dn,
                "downstairs/discounted_return": J_dn_discounted,
                "downstairs/td_mse": stats_dn["td_mse"],
                "downstairs/ortho_err_Wc": stats_dn["ortho_err_Wc"],
                "downstairs/ortho_err_Zb": stats_dn["ortho_err_Zb"],
                "optimal/steady_state_avg_reward": opt_info["steady_state_avg_reward"],
                "optimal/initial_reward": opt_info["initial_reward"],
                "timing/iter_ms": iter_time * 1000,
            }
            wandb.log(log_dict, step=it + 1)

    total_time = time.time() - start_time
    print(f"\nDone. Total time: {total_time:.2f}s ({args.iters / total_time:.1f} iter/s)")

    if args.track:
        wandb.finish()


if __name__ == "__main__":
    main()
