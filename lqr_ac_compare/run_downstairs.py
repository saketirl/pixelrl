
import argparse
import numpy as np

from lqr_env import LQRParams, LQRUpDownEnv
from downstairs_models import DownstairsActorShallowT2, DownstairsCriticShallowT2
from downstairs_learning import downstairs_actor_critic_update_T2


def truncate_traj(traj: dict, steps: int):
    """Truncate trajectory to given number of transitions."""
    steps = int(steps)
    return {
        "s": traj["s"][: steps + 1].copy(),
        "a": traj["a"][: steps].copy(),
        "r": traj["r"][: steps].copy(),
    }


def compute_discounted_return(rewards: np.ndarray, beta: float, dt: float) -> float:
    """Compute cumulative discounted return: sum_t e^{-beta*t} r_t dt."""
    N = len(rewards)
    discount = np.exp(-beta * np.arange(N) * dt)
    return float(np.sum(discount * rewards) * dt)


def construct_compatible_lqr(ds, da, alpha=0.5, seed=42):
    """
    Construct LQR parameters where optimal solution is Stiefel-compatible.

    Returns G, H, Q, R such that:
    - P_opt = I (orthogonal)
    - ||K_opt|| = 1 (unit norm)
    """
    rng = np.random.default_rng(seed)

    # G = -alpha * I (stable diagonal)
    G = -alpha * np.eye(ds)

    # H: random direction with unit Frobenius norm
    H_raw = rng.standard_normal((ds, da))
    H = H_raw / np.linalg.norm(H_raw)

    # R = I
    R = np.eye(da)

    # Q = H H^T + 2*alpha * I (ensures P = I solves Riccati)
    Q = H @ H.T + 2 * alpha * np.eye(ds)

    return G, H, Q, R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", type=int, default=4, help="Latent state dimension")
    ap.add_argument("--da", type=int, default=1, help="Action dimension")

    ap.add_argument("--dt", type=float, default=0.02, help="Time step")
    ap.add_argument("--T", type=float, default=2.0, help="Horizon length")
    ap.add_argument("--horizon_steps", type=int, default=200,
                    help="Optional truncation of the rollout (number of transitions)")

    ap.add_argument("--beta", type=float, default=0.1, help="Discount rate")
    ap.add_argument("--exploration_std", type=float, default=0.05, help="Exploration noise std")
    ap.add_argument("--transition_noise", type=float, default=1.0,
                    help="Scale factor for Sigma_base transition noise")

    ap.add_argument("--iters", type=int, default=100, help="Number of training iterations")
    ap.add_argument("--L_eff", type=int, default=3, help="Effective depth for Stiefel coefficients")

    # Use Stiefel-compatible LQR construction
    ap.add_argument("--compatible", action="store_true",
                    help="Use Stiefel-compatible LQR (P=I, ||K||=1)")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="Stability margin for compatible LQR (G = -alpha * I)")

    # Downstairs learning rates
    ap.add_argument("--eta_actor", type=float, default=1e-3, help="Actor learning rate")
    ap.add_argument("--eta_value", type=float, default=1e-3, help="Value function learning rate")
    ap.add_argument("--eta_adv", type=float, default=1e-3, help="Advantage function learning rate")

    ap.add_argument("--seed", type=int, default=0)

    # Wandb tracking
    ap.add_argument("--track", action="store_true", help="Track with wandb")
    ap.add_argument("--wandb_project", type=str, default="lqr-downstairs", help="Wandb project name")
    ap.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity")
    ap.add_argument("--exp_name", type=str, default="lqr_downstairs", help="Experiment name for wandb")

    args = ap.parse_args()

    # Initialize wandb
    if args.track:
        import wandb
        run_name = f"{args.exp_name}_ds{args.ds}_da{args.da}_seed{args.seed}"
        if args.compatible:
            run_name += "_compatible"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
        )

    # Create LQR environment
    if args.compatible:
        # Use Stiefel-compatible construction: P=I, ||K||=1
        G, H, Q, R = construct_compatible_lqr(args.ds, args.da, alpha=args.alpha, seed=args.seed)
        params = LQRParams(
            d=16,
            ds=args.ds,
            da=args.da,
            dt=args.dt,
            T=args.T,
            G=G,
            H=H,
            Q=Q,
            R=R,
            sigma_explore=0.05,
            eps_obs=0.0,
            seed=args.seed,
            Sigma_base=args.transition_noise * np.eye(args.ds),
        )
    else:
        # Random LQR (original behavior)
        params = LQRParams(
            d=16,
            ds=args.ds,
            da=args.da,
            dt=args.dt,
            T=args.T,
            sigma_explore=0.05,
            eps_obs=0.0,
            seed=args.seed,
            Sigma_base=args.transition_noise * np.eye(args.ds),
            Q=np.eye(args.ds),
            R=np.eye(args.da),
        )
    env = LQRUpDownEnv(params)

    # Initialize downstairs actor and critic
    actor = DownstairsActorShallowT2(ds=args.ds, da=args.da, seed=args.seed + 1)
    critic = DownstairsCriticShallowT2(ds=args.ds, da=args.da, seed=args.seed + 2)

    print("\n=== Downstairs-only LQR Learning ===")
    if args.compatible:
        print(f"MODE: Stiefel-compatible (P=I, ||K||=1), alpha={args.alpha}")
    else:
        print("MODE: Random LQR")
    print(f"ds={args.ds}, da={args.da}, dt={args.dt}, T={args.T}, steps={env.n_steps}")
    if args.horizon_steps > 0:
        print(f"Truncating rollouts to {args.horizon_steps} steps")
    print(f"iters={args.iters}, beta={args.beta}, exploration_std={args.exploration_std}")
    print(f"transition_noise={args.transition_noise}")
    print(f"L_eff={args.L_eff}, etas: actor={args.eta_actor}, value={args.eta_value}, adv={args.eta_adv}")

    # Print optimal LQR solution
    opt_info = env.optimal_average_reward(exploration_std=args.exploration_std)
    print(f"\n=== Optimal LQR Solution ===")
    print(f"K_opt shape: {opt_info['K_opt'].shape}")
    print(f"K_opt:\n{opt_info['K_opt']}")
    print(f"Initial reward (at s0): {opt_info['initial_reward']:.4f}")
    print(f"Steady-state avg reward (with noise): {opt_info['steady_state_avg_reward']:.4f}")
    print(f"V(s0) total cost-to-go: {opt_info['V_s0']:.4f}\n")

    for it in range(args.iters):
        # Generate trajectory using current policy
        def policy(obs_dict):
            return actor.act(obs_dict["downstairs"])

        traj = env.rollout(policy, exploration_std=args.exploration_std, use_upstairs=False)

        if args.horizon_steps > 0:
            Nkeep = min(args.horizon_steps, traj["a"].shape[0])
            traj = truncate_traj(traj, Nkeep)

        # Downstairs actor-critic update
        stats = downstairs_actor_critic_update_T2(
            actor, critic,
            traj["s"], traj["a"], traj["r"],
            beta=args.beta,
            dt=args.dt,
            L_eff=args.L_eff,
            eta_actor=args.eta_actor,
            eta_value=args.eta_value,
            eta_adv=args.eta_adv,
        )

        mean_reward = float(traj["r"].mean())
        discounted_return = compute_discounted_return(traj["r"], args.beta, args.dt)

        # Print progress
        if (it + 1) % 10 == 0 or it == 0:
            print(f"[iter {it+1:04d}]  mean_r={mean_reward:+.4f}  "
                  f"td_mse={stats['td_mse']:.4e}  "
                  f"ortho_Wc={stats['ortho_err_Wc']:.2e}  "
                  f"ortho_Zb={stats['ortho_err_Zb']:.2e}")

        # Log to wandb
        if args.track:
            log_dict = {
                "iteration": it + 1,
                "mean_reward": mean_reward,
                "discounted_return": discounted_return,
                "td_mse": stats["td_mse"],
                "ortho_err_Wc": stats["ortho_err_Wc"],
                "ortho_err_Zb": stats["ortho_err_Zb"],
                "optimal/steady_state_avg_reward": opt_info["steady_state_avg_reward"],
                "optimal/initial_reward": opt_info["initial_reward"],
            }
            wandb.log(log_dict, step=it + 1)

    # Final summary
    print(f"\n=== Training Complete ===")
    print(f"Final mean reward: {mean_reward:+.4f}")
    print(f"Optimal steady-state reward: {opt_info['steady_state_avg_reward']:.4f}")
    print(f"Gap to optimal: {opt_info['steady_state_avg_reward'] - mean_reward:.4f}")

    # Print learned vs optimal gain
    print(f"\nLearned actor Wc (gain matrix):")
    print(actor.Wc)
    print(f"\nOptimal K* (negated for comparison, since a = Wc @ s vs a = -K @ s):")
    print(-opt_info['K_opt'])

    if args.track:
        wandb.finish()


if __name__ == "__main__":
    main()
