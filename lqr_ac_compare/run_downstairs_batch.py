"""
Run downstairs learning with batch multi-step updates (CT-DDPG style).
"""
import argparse
import numpy as np

from lqr_env import LQRParams, LQRUpDownEnv
from downstairs_models import DownstairsActorShallowT2, DownstairsCriticShallowT2
from downstairs_learning_batch import batch_multistep_update


def construct_compatible_lqr(ds, da, alpha=0.5, seed=42):
    """Construct LQR with P=I, ||K||=1."""
    rng = np.random.default_rng(seed)
    G = -alpha * np.eye(ds)
    H_raw = rng.standard_normal((ds, da))
    H = H_raw / np.linalg.norm(H_raw)
    R = np.eye(da)
    Q = H @ H.T + 2 * alpha * np.eye(ds)
    return G, H, Q, R


def main():
    np.set_printoptions(precision=4, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=int, default=4)
    parser.add_argument("--da", type=int, default=1)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--T", type=float, default=10.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--L", type=int, default=10, help="Multi-step horizon")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--eta_actor", type=float, default=0.01)
    parser.add_argument("--eta_critic", type=float, default=0.05)
    parser.add_argument("--exploration_std", type=float, default=0.1)
    parser.add_argument("--transition_noise", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compatible", action="store_true", help="Use Stiefel-compatible LQR")
    args = parser.parse_args()

    ds, da = args.ds, args.da

    # Create LQR environment
    if args.compatible:
        G, H, Q, R = construct_compatible_lqr(ds, da, alpha=0.5, seed=args.seed)
        params = LQRParams(
            d=16, ds=ds, da=da, dt=args.dt, T=args.T,
            G=G, H=H, Q=Q, R=R,
            sigma_explore=0.05, eps_obs=0.0, seed=args.seed,
            Sigma_base=args.transition_noise * np.eye(ds),
        )
    else:
        params = LQRParams(
            d=16, ds=ds, da=da, dt=args.dt, T=args.T,
            sigma_explore=0.05, eps_obs=0.0, seed=args.seed,
            Sigma_base=args.transition_noise * np.eye(ds),
            Q=np.eye(ds), R=np.eye(da),
        )

    env = LQRUpDownEnv(params)
    K_opt = env.K_opt
    opt_info = env.optimal_average_reward(exploration_std=args.exploration_std)

    print("=== Batch Multi-step Downstairs Learning ===")
    print(f"Mode: {'Stiefel-compatible' if args.compatible else 'Random LQR'}")
    print(f"ds={ds}, da={da}, dt={args.dt}, T={args.T}, L={args.L}")
    print(f"beta={args.beta}, exploration_std={args.exploration_std}")
    print(f"eta_actor={args.eta_actor}, eta_critic={args.eta_critic}")
    print(f"\nOptimal K: {K_opt}")
    print(f"Optimal steady-state reward: {opt_info['steady_state_avg_reward']:.4f}")

    # Initialize actor and critic
    actor = DownstairsActorShallowT2(ds=ds, da=da, seed=args.seed + 1)
    critic = DownstairsCriticShallowT2(ds=ds, da=da, seed=args.seed + 2)

    # For LQR, the advantage gradient dPsi/da should point towards optimal action
    # dA/da = 2R(a - a*) where a* = -Ks
    # So dPsi/da = 2R*a + 2R*K*s = 2R*a + 2*H^T*P*s (for K = R^{-1}H^T P)
    #
    # For compatible LQR: P=I, R=I, so dPsi/da = 2a + 2H^T s
    # At a=0: dPsi/da = 2H^T s, so Zb^T s should equal 2H^T s
    # Therefore Zb = 2H, but we need unit norm for Stiefel.
    #
    # Actually, the direction matters more than scale. Initialize Zb ∝ H
    if args.compatible:
        # H is stored in env, get it and normalize
        H = env.H  # (ds, da)
        Zb_init = H / np.linalg.norm(H)  # unit norm
        critic.Zb = Zb_init.copy()
        print(f"Initialized Zb to H direction: {critic.Zb.flatten()}")

    print(f"\nInitial Wc: {actor.Wc}")
    print(f"Target -K:  {-K_opt}")

    for it in range(args.iters):
        # Rollout
        def policy(obs):
            return actor.act(obs["downstairs"])

        traj = env.rollout(policy, exploration_std=args.exploration_std, use_upstairs=False)

        # Batch update
        stats = batch_multistep_update(
            actor, critic,
            traj["s"], traj["a"], traj["r"],
            beta=args.beta,
            dt=args.dt,
            L=args.L,
            eta_actor=args.eta_actor,
            eta_critic=args.eta_critic,
        )

        mean_r = traj["r"].mean()

        # Compute alignment with optimal
        W_flat = actor.Wc.flatten()
        K_opt_flat = (-K_opt).flatten()
        cos_sim = np.dot(W_flat, K_opt_flat) / (np.linalg.norm(W_flat) * np.linalg.norm(K_opt_flat) + 1e-10)

        if (it + 1) % 20 == 0 or it == 0:
            print(f"[iter {it+1:3d}] r={mean_r:.4f}, mse={stats['mse']:.4f}, "
                  f"cos={cos_sim:.4f}, ||G_Wc||={stats['G_Wc_norm']:.2e}, ||G_Ub||={stats['G_Ub_norm']:.2e}")

    print(f"\n=== Final Results ===")
    print(f"Learned Wc: {actor.Wc}")
    print(f"Target -K:  {-K_opt}")
    print(f"Cosine similarity: {cos_sim:.4f}")

    # Evaluate final policy
    def final_policy(obs):
        return actor.act(obs["downstairs"])

    rewards = []
    for _ in range(10):
        traj = env.rollout(final_policy, exploration_std=args.exploration_std)
        rewards.append(traj["r"].mean())
    print(f"\nFinal mean reward: {np.mean(rewards):.4f} ± {np.std(rewards):.4f}")

    # Compare to optimal
    def opt_policy(obs):
        return (-K_opt @ obs["downstairs"]).flatten()

    rewards = []
    for _ in range(10):
        traj = env.rollout(opt_policy, exploration_std=args.exploration_std)
        rewards.append(traj["r"].mean())
    print(f"Optimal mean reward: {np.mean(rewards):.4f} ± {np.std(rewards):.4f}")


if __name__ == "__main__":
    main()
