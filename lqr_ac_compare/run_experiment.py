
import argparse
import os
import numpy as np

from lqr_env import LQRParams, LQRUpDownEnv
from upstairs_models import ActorDLN_L3, CriticDLN_L3
from upstairs_learning import upstairs_actor_critic_update_L3
from downstairs_models import DownstairsActorShallowT2, DownstairsCriticShallowT2
from downstairs_learning import downstairs_actor_critic_update_T2


def truncate_traj(traj: dict, steps: int):
    # steps is number of transitions N to keep; arrays are (N+1, ...) for s/o and (N, ...) for a/r
    steps = int(steps)
    traj2 = {
        "s": traj["s"][: steps + 1].copy(),
        "o": traj["o"][: steps + 1].copy(),
        "a": traj["a"][: steps].copy(),
        "r": traj["r"][: steps].copy(),
    }
    return traj2


def compute_discounted_return(rewards: np.ndarray, beta: float, dt: float) -> float:
    """Compute cumulative discounted return: sum_t e^{-beta*t} r_t dt."""
    N = len(rewards)
    discount = np.exp(-beta * np.arange(N) * dt)
    return float(np.sum(discount * rewards) * dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--ds", type=int, default=4)
    ap.add_argument("--da", type=int, default=1)

    # Keep defaults runnable; you can still set --T 10 for full horizon.
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--T", type=float, default=2.0)
    ap.add_argument("--horizon_steps", type=int, default=200, help="Optional truncation of the rollout (number of transitions).")

    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--exploration_std", type=float, default=0.05)
    ap.add_argument("--eps_obs", type=float, default=0.05, help="Observation noise scale")
    ap.add_argument("--transition_noise", type=float, default=1.0, help="Scale factor for Sigma_base transition noise")
    ap.add_argument("--reward_state_scale", type=float, default=1.0, help="Scale for state cost in reward (Q matrix)")
    ap.add_argument("--reward_action_scale", type=float, default=1.0, help="Scale for action cost in reward (R matrix)")
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--shared_trajectory", action="store_true", help="Use same trajectory for both upstairs and downstairs")

    # Upstairs learning rates
    ap.add_argument("--eta_actor_up", type=float, default=2e-3)
    ap.add_argument("--eta_value_up", type=float, default=2e-3)
    ap.add_argument("--eta_adv_up", type=float, default=2e-3)
    ap.add_argument("--batch_size_up", type=int, default=16)

    # Downstairs learning rate (requested small)
    ap.add_argument("--eta_down", type=float, default=1e-3)

    ap.add_argument("--seed", type=int, default=0)

    # Wandb tracking
    ap.add_argument("--track", action="store_true", help="Track with wandb")
    ap.add_argument("--wandb_project", type=str, default="benchmark", help="Wandb project name")
    ap.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity")
    ap.add_argument("--exp_name", type=str, default="lqr_ac_compare", help="Experiment name for wandb")

    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

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

    params = LQRParams(
        d=args.d, ds=args.ds, da=args.da,
        dt=args.dt, T=args.T,
        sigma_explore=0.05, eps_obs=args.eps_obs,
        seed=args.seed,
        Sigma_base=args.transition_noise * np.eye(args.ds),
        Q=args.reward_state_scale * np.eye(args.ds),
        R=args.reward_action_scale * np.eye(args.da),
    )
    env = LQRUpDownEnv(params)

    actor_up = ActorDLN_L3(d=args.d, da=args.da, seed=args.seed + 1)
    critic_up = CriticDLN_L3(d=args.d, da=args.da, seed=args.seed + 2)

    actor_dn = DownstairsActorShallowT2(ds=args.ds, da=args.da, seed=args.seed + 3)
    critic_dn = DownstairsCriticShallowT2(ds=args.ds, da=args.da, seed=args.seed + 4)

    print("\n=== Running upstairs vs downstairs learning ===")
    print(f"d={args.d}, ds={args.ds}, da={args.da}, dt={args.dt}, T={args.T}, steps={env.n_steps} (trunc to {args.horizon_steps})")
    print(f"iters={args.iters}, beta={args.beta}, exploration_std={args.exploration_std}, eps_obs={args.eps_obs}, transition_noise={args.transition_noise}")
    print(f"reward scales: state={args.reward_state_scale}, action={args.reward_action_scale}")
    print(f"upstairs (Cayley) etas: actor/value/adv = {args.eta_actor_up}/{args.eta_value_up}/{args.eta_adv_up}, batch={args.batch_size_up}")
    print(f"downstairs (T2-style) eta = {args.eta_down}")
    print(f"shared_trajectory: {args.shared_trajectory}")

    # Print optimal LQR solution
    opt_info = env.optimal_average_reward(exploration_std=args.exploration_std)
    print(f"\n=== Optimal LQR Solution ===")
    print(f"K_opt shape: {opt_info['K_opt'].shape}")
    print(f"Initial reward (at s0): {opt_info['initial_reward']:.4f}")
    print(f"Steady-state avg reward (with noise): {opt_info['steady_state_avg_reward']:.4f}")
    print(f"V(s0) total cost-to-go: {opt_info['V_s0']:.4f}\n")

    for it in range(args.iters):
        if args.shared_trajectory:
            # Use downstairs policy to generate a single shared trajectory
            def dn_policy(obs_dict):
                return actor_dn.act(obs_dict["downstairs"])

            traj = env.rollout(dn_policy, exploration_std=args.exploration_std, use_upstairs=False)
            if args.horizon_steps > 0:
                Nkeep = min(args.horizon_steps, traj["a"].shape[0])
                traj = truncate_traj(traj, Nkeep)

            # Both use the same trajectory
            traj_up = traj
            traj_dn = traj
        else:
            # Separate trajectories (original behavior)
            # Upstairs
            def up_policy(obs_dict):
                return actor_up.act_upstairs(obs_dict["upstairs"])

            traj_up = env.rollout(up_policy, exploration_std=args.exploration_std, use_upstairs=True)
            if args.horizon_steps > 0:
                Nkeep = min(args.horizon_steps, traj_up["a"].shape[0])
                traj_up = truncate_traj(traj_up, Nkeep)

            # Downstairs
            def dn_policy(obs_dict):
                return actor_dn.act(obs_dict["downstairs"])

            traj_dn = env.rollout(dn_policy, exploration_std=args.exploration_std, use_upstairs=False)
            if args.horizon_steps > 0:
                Nkeep = min(args.horizon_steps, traj_dn["a"].shape[0])
                traj_dn = truncate_traj(traj_dn, Nkeep)

        # Upstairs update (uses observations o)
        stats_up = upstairs_actor_critic_update_L3(
            actor_up, critic_up,
            traj_up["o"], traj_up["a"], traj_up["r"],
            beta=args.beta, dt=args.dt,
            eta_actor=args.eta_actor_up,
            eta_value=args.eta_value_up,
            eta_adv=args.eta_adv_up,
            batch_size=args.batch_size_up,
            rng=rng,
        )

        # Downstairs update (uses states s)
        stats_dn = downstairs_actor_critic_update_T2(
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

        # Compute cumulative discounted returns
        J_up_discounted = compute_discounted_return(traj_up["r"], args.beta, args.dt)
        J_dn_discounted = compute_discounted_return(traj_dn["r"], args.beta, args.dt)

        print(f"[iter {it+1:02d}]  upstairs: mean r={J_up:+.4f}, td_mse_batch={stats_up['td_mse_batch']:.4e}, orthoW1={stats_up['ortho_err_W1']:.2e}")
        print(f"           downstairs: mean r={J_dn:+.4f}, td_mse={stats_dn['td_mse']:.4e}, orthoWc={stats_dn['ortho_err_Wc']:.2e}")

        # Log to wandb
        if args.track:
            log_dict = {
                "iteration": it + 1,
                # Upstairs metrics
                "upstairs/mean_reward": J_up,
                "upstairs/discounted_return": J_up_discounted,
                "upstairs/td_mse_batch": stats_up["td_mse_batch"],
                "upstairs/ortho_err_W1": stats_up["ortho_err_W1"],
                "upstairs/ortho_err_U1": stats_up["ortho_err_U1"],
                "upstairs/batch_size": stats_up["batch_size"],
                # Downstairs metrics
                "downstairs/mean_reward": J_dn,
                "downstairs/discounted_return": J_dn_discounted,
                "downstairs/td_mse": stats_dn["td_mse"],
                "downstairs/ortho_err_Wc": stats_dn["ortho_err_Wc"],
                "downstairs/ortho_err_Zb": stats_dn["ortho_err_Zb"],
                # Optimal reference (constant line)
                "optimal/steady_state_avg_reward": opt_info["steady_state_avg_reward"],
                "optimal/initial_reward": opt_info["initial_reward"],
            }
            wandb.log(log_dict, step=it + 1)

    print("\nDone.")

    if args.track:
        wandb.finish()


if __name__ == "__main__":
    main()
