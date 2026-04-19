"""
Run upstairs learning with batch multi-step updates (CT-DDPG style).

Uses high-dimensional observations o = M @ s where M has orthonormal columns.
"""
import argparse
import numpy as np

from lqr_env import LQRParams, LQRUpDownEnv
from upstairs_models import ActorDLN_L3, CriticDLN_L3
from upstairs_learning_batch import upstairs_batch_multistep_update


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
    parser.add_argument("--d", type=int, default=64, help="Observation dimension")
    parser.add_argument("--ds", type=int, default=4, help="Latent state dimension")
    parser.add_argument("--da", type=int, default=1, help="Action dimension")
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--T", type=float, default=10.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--L", type=int, default=10, help="Multi-step horizon")
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--eta_actor", type=float, default=0.05)
    parser.add_argument("--eta_critic", type=float, default=0.1)
    parser.add_argument("--exploration_std", type=float, default=0.1)
    parser.add_argument("--transition_noise", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compatible", action="store_true", help="Use Stiefel-compatible LQR")
    parser.add_argument("--optimal_init", action="store_true", help="Initialize actor/critic to optimal")
    args = parser.parse_args()

    d, ds, da = args.d, args.ds, args.da

    # Create LQR environment
    if args.compatible:
        G, H, Q, R = construct_compatible_lqr(ds, da, alpha=0.5, seed=args.seed)
        params = LQRParams(
            d=d, ds=ds, da=da, dt=args.dt, T=args.T,
            G=G, H=H, Q=Q, R=R,
            sigma_explore=0.05, eps_obs=0.05, seed=args.seed,
            Sigma_base=args.transition_noise * np.eye(ds),
        )
    else:
        params = LQRParams(
            d=d, ds=ds, da=da, dt=args.dt, T=args.T,
            sigma_explore=0.05, eps_obs=0.05, seed=args.seed,
            Sigma_base=args.transition_noise * np.eye(ds),
            Q=np.eye(ds), R=np.eye(da),
        )

    env = LQRUpDownEnv(params)
    K_opt = env.K_opt
    M = env.M  # observation matrix (d, ds)
    opt_info = env.optimal_average_reward(exploration_std=args.exploration_std)

    print("=== Batch Multi-step Upstairs Learning (L=3 DLN) ===")
    print(f"Mode: {'Stiefel-compatible' if args.compatible else 'Random LQR'}")
    print(f"d={d}, ds={ds}, da={da}, dt={args.dt}, T={args.T}, L={args.L}")
    print(f"beta={args.beta}, exploration_std={args.exploration_std}")
    print(f"eta_actor={args.eta_actor}, eta_critic={args.eta_critic}")
    print(f"\nOptimal K (latent): {K_opt}")
    print(f"Optimal steady-state reward: {opt_info['steady_state_avg_reward']:.4f}")

    # The optimal upstairs policy is: a = -K @ s = -K @ M^+ @ o
    # where M^+ = M^T (since M has orthonormal columns)
    # So optimal Weff = -K @ M^T  (da × d)
    Weff_opt = -K_opt @ M.T
    print(f"Optimal Weff = -K @ M^T, ||Weff_opt|| = {np.linalg.norm(Weff_opt):.4f}")

    # Initialize actor and critic
    actor = ActorDLN_L3(d=d, da=da, seed=args.seed + 1)
    critic = CriticDLN_L3(d=d, da=da, seed=args.seed + 2)

    # For LQR, advantage A(s,a) = -(a - a*)^T R (a - a*) where a* = -Ks
    # Gradient: dA/da = -2R(a - a*) = -2(a + Ks) at R=I
    # At current policy a = Weff @ o = Weff @ M @ s:
    #   dA/da = -2(Weff @ M @ s + K @ s) = -2(Weff @ M + K) @ s = -2(Weff + K @ M.T) @ M @ s
    #         = -2(Weff + K @ M.T) @ o
    # So Zeff @ o should equal -2(Weff + K @ M.T) @ o, meaning Zeff = -2(Weff + K @ M.T)
    if args.compatible:
        # Get initial Weff
        Weff_init_raw = actor.effective_matrix()  # before any modification

        if args.optimal_init:
            # Initialize actor to optimal: Weff = -K @ M.T
            actor.W3 = Weff_opt / (np.linalg.norm(Weff_opt, axis=1, keepdims=True) + 1e-12)
            actor.W1 = np.eye(d)
            actor.W2 = np.eye(d)
            Weff_current = actor.effective_matrix()
            print("Using optimal initialization for actor W3")

            # At optimal policy, gradient dA/da = -2(-K @ M.T + K @ M.T) @ o = 0
            # So Zeff should be ~0, but we need some direction for learning
            # Use small perturbation from optimal
            target_Zeff = -2 * (Weff_current + K_opt @ M.T)
            if np.linalg.norm(target_Zeff) < 1e-6:
                # At optimum, use random direction
                target_Zeff = np.random.randn(da, d)
        else:
            # Random W3: compute gradient at initial (random) policy
            Weff_current = actor.effective_matrix()
            # dA/da|_{current policy} = -2(Weff + K @ M.T) @ o
            target_Zeff = -2 * (Weff_current + K_opt @ M.T)
            print(f"Using random actor W3, initial Weff")

        print(f"Target Zeff = -2(Weff + K@M.T), ||target|| = {np.linalg.norm(target_Zeff):.4f}")

        # Normalize and set Z3
        critic.Z3 = target_Zeff / (np.linalg.norm(target_Zeff, axis=1, keepdims=True) + 1e-12)
        critic.Z1 = np.eye(d)
        critic.Z2 = np.eye(d)

        # Verify initialization
        Zeff_init = critic.Z3 @ critic.Z2 @ critic.Z1
        cos_Zeff = np.sum(Zeff_init * target_Zeff) / (np.linalg.norm(Zeff_init) * np.linalg.norm(target_Zeff) + 1e-10)
        print(f"After init: cos(Zeff, target) = {cos_Zeff:.4f}")

    print(f"\nInitial Weff = W3 @ W2 @ W1:")
    Weff_init = actor.effective_matrix()
    print(f"  ||Weff|| = {np.linalg.norm(Weff_init):.4f}")

    # Cosine similarity between Weff and Weff_opt
    def cos_sim(A, B):
        return np.sum(A * B) / (np.linalg.norm(A) * np.linalg.norm(B) + 1e-10)

    cos_init = cos_sim(Weff_init, Weff_opt)
    print(f"  cos(Weff, Weff_opt) = {cos_init:.4f}")

    for it in range(args.iters):
        # Rollout using upstairs policy
        def policy(obs):
            return actor.act_upstairs(obs["upstairs"])

        traj = env.rollout(policy, exploration_std=args.exploration_std, use_upstairs=True)

        # Batch update
        stats = upstairs_batch_multistep_update(
            actor, critic,
            traj["o"], traj["a"], traj["r"],
            beta=args.beta,
            dt=args.dt,
            L=args.L,
            eta_actor=args.eta_actor,
            eta_critic=args.eta_critic,
        )

        mean_r = traj["r"].mean()
        Weff = actor.effective_matrix()
        cos = cos_sim(Weff, Weff_opt)

        if (it + 1) % 50 == 0 or it == 0:
            print(f"[iter {it+1:4d}] r={mean_r:.4f}, mse={stats['mse']:.4f}, "
                  f"cos={cos:.4f}, ||G_W1||={stats['G_W1_norm']:.2e}")

    print(f"\n=== Final Results ===")
    Weff_final = actor.effective_matrix()
    cos_final = cos_sim(Weff_final, Weff_opt)
    print(f"Final cos(Weff, Weff_opt) = {cos_final:.4f}")
    print(f"||Weff_final|| = {np.linalg.norm(Weff_final):.4f}")
    print(f"||Weff_opt|| = {np.linalg.norm(Weff_opt):.4f}")

    # Evaluate final policy
    def final_policy(obs):
        return actor.act_upstairs(obs["upstairs"])

    rewards = []
    for _ in range(10):
        traj = env.rollout(final_policy, exploration_std=args.exploration_std)
        rewards.append(traj["r"].mean())
    print(f"\nFinal mean reward: {np.mean(rewards):.4f} ± {np.std(rewards):.4f}")

    # Compare to optimal
    def opt_policy(obs):
        # Optimal: a = -K @ s, but we only have o
        # Since o = M @ s + noise, and M^T @ M = I, we use s ≈ M^T @ o
        s_est = M.T @ obs["upstairs"]
        return (-K_opt @ s_est).flatten()

    rewards = []
    for _ in range(10):
        traj = env.rollout(opt_policy, exploration_std=args.exploration_std)
        rewards.append(traj["r"].mean())
    print(f"Optimal mean reward: {np.mean(rewards):.4f} ± {np.std(rewards):.4f}")


if __name__ == "__main__":
    main()
