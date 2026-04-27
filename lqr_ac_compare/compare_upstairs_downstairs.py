"""
Compare upstairs vs downstairs learning to verify Stiefel_Ascent_RLC.pdf Theorem 2.

The paper claims that learning dynamics in the high-dimensional "upstairs" space
should match the low-dimensional "downstairs" latent space when using Stiefel
manifold optimization with deep linear networks.

This script runs both approaches on the same LQR problem and compares:
1. Learning curves (reward over iterations)
2. Policy convergence (cosine similarity to optimal)
3. Critic MSE convergence
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from lqr_env import LQRParams, LQRUpDownEnv
from downstairs_models import DownstairsActorShallowT2, DownstairsCriticShallowT2
from upstairs_models import ActorDLN_L3, CriticDLN_L3
from downstairs_learning_batch import batch_multistep_update
from upstairs_learning_batch import upstairs_batch_multistep_update


def construct_compatible_lqr(ds: int, da: int, alpha: float = 0.5, seed: int = 42):
    """Construct Stiefel-compatible LQR with P=I, K on Stiefel manifold (K K^T = I_da)."""
    rng = np.random.default_rng(seed)
    G = -alpha * np.eye(ds)
    # H must have orthonormal columns (H^T H = I_da) so that K = H^T is on Stiefel
    H_raw = rng.standard_normal((ds, da))
    H, _ = np.linalg.qr(H_raw, mode='reduced')  # H is (ds, da) with H^T H = I_da
    R = np.eye(da)
    Q = H @ H.T + 2 * alpha * np.eye(ds)
    return G, H, Q, R


def cos_sim(A: np.ndarray, B: np.ndarray) -> float:
    """Cosine similarity between two matrices."""
    return float(np.sum(A * B) / (np.linalg.norm(A) * np.linalg.norm(B) + 1e-10))


def run_downstairs(
    env: LQRUpDownEnv,
    K_opt: np.ndarray,
    iters: int,
    L: int,
    beta: float,
    dt: float,
    eta_actor: float,
    eta_critic: float,
    exploration_std: float,
    init_seed: int,
    exploration_seed: int,
    return_init: bool = False,  # Return init params for upstairs matching
    use_generator: bool = False,  # Use generator-based TD error
    use_sgd: bool = False,  # Use standard SGD instead of Cayley retraction
    csv_path: str = None,  # Path to write intermediate results
    log_every: int = 1,  # Write to CSV every N iterations
) -> Dict[str, List[float]]:
    """Run downstairs learning and return metrics."""
    import csv
    ds, da = env.ds, env.da

    # Use init_seed for initialization (same across runs with same LQR)
    actor = DownstairsActorShallowT2(ds=ds, da=da, seed=init_seed)
    critic = DownstairsCriticShallowT2(ds=ds, da=da, seed=init_seed + 100)

    # Reseed env RNG for exploration (varies across runs)
    env.rng = np.random.default_rng(exploration_seed)

    # Initialize Zb to H direction (compatible with Stiefel LQR)
    # H has orthonormal columns, so it's already on Stiefel
    H = env.H
    critic.Zb = H.copy()

    # Capture initial params for matched upstairs init
    init_params = {
        "Wc_col": actor.Wc_col.copy(),  # (ds, da) Stiefel
        "Ub": critic.Ub.copy(),          # (ds, ds) orthogonal
        "Zb": critic.Zb.copy(),          # (ds, da) Stiefel
    }

    # Print initial cos similarity
    Wc = actor.Wc_col.T  # (da, ds)
    init_cos = cos_sim(-Wc, K_opt)
    print(f"  Initial cos(policy, optimal): {init_cos:.4f}")

    metrics = {
        "rewards": [],
        "cos_to_opt": [],
        "mse": [],
    }

    # Open CSV file for incremental writing
    csv_file = None
    csv_writer = None
    if csv_path:
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['iter', 'down_reward', 'down_cos', 'down_mse'])

    buffer = []  # Buffer for batch writing

    for it in range(iters):
        # Rollout using downstairs policy
        def policy(obs):
            return actor.act(obs["downstairs"])

        traj = env.rollout(policy, exploration_std=exploration_std, use_upstairs=False)

        # Update
        stats = batch_multistep_update(
            actor, critic,
            traj["s"], traj["a"], traj["r"],
            beta=beta,
            dt=dt,
            L=L,
            eta_actor=eta_actor,
            eta_critic=eta_critic,
            use_generator=use_generator,
            use_sgd=use_sgd,
        )

        # Metrics
        mean_r = float(traj["r"].mean())
        Wc = actor.Wc_col.T  # (da, ds)
        cos = cos_sim(-Wc, K_opt)  # policy is a = Wc.T @ s, optimal is a = -K @ s

        metrics["rewards"].append(mean_r)
        metrics["cos_to_opt"].append(cos)
        metrics["mse"].append(stats["mse"])

        # Buffer row for CSV
        if csv_writer:
            buffer.append([it, mean_r, cos, stats["mse"]])
            # Write every log_every iterations
            if (it + 1) % log_every == 0 or it == iters - 1:
                for row in buffer:
                    csv_writer.writerow(row)
                csv_file.flush()
                buffer = []

    if csv_file:
        csv_file.close()

    if return_init:
        return metrics, init_params
    return metrics


def run_upstairs(
    env: LQRUpDownEnv,
    K_opt: np.ndarray,
    M: np.ndarray,
    iters: int,
    L: int,
    beta: float,
    dt: float,
    eta_actor: float,
    eta_critic: float,
    exploration_std: float,
    init_seed: int,
    exploration_seed: int,
    init_from_downstairs: dict = None,  # For matched initialization
    use_generator: bool = False,  # Use generator-based TD error
    use_cayley: bool = True,  # Use Cayley retraction (False = standard gradient descent)
    balanced: bool = False,  # Use balanced DLN initialization
) -> Dict[str, List[float]]:
    """Run upstairs learning and return metrics."""
    d, da = env.d, env.da
    ds = env.ds

    # Use init_seed for initialization (same across runs with same LQR)
    actor = ActorDLN_L3(d=d, da=da, seed=init_seed, balanced=balanced)
    critic = CriticDLN_L3(d=d, da=da, seed=init_seed + 100, balanced=balanced)

    # Reseed env RNG for exploration (varies across runs)
    env.rng = np.random.default_rng(exploration_seed)

    # Optimal Weff = -K @ M.T for upstairs
    Weff_opt = -K_opt @ M.T

    if init_from_downstairs is not None:
        # Match effective parameters from downstairs initialization
        # Actor: Weff @ M should equal Wc_col.T
        # Set W1 = W2 = I, W3 = Wc_col.T @ M.T (with row normalization)
        Wc_col = init_from_downstairs["Wc_col"]  # (ds, da)
        Wc = Wc_col.T  # (da, ds)
        W3_target = Wc @ M.T  # (da, d)
        # Normalize rows for row_unit constraint
        row_norms = np.linalg.norm(W3_target, axis=1, keepdims=True) + 1e-12
        actor.W3 = W3_target / row_norms
        actor.W1 = np.eye(d)
        actor.W2 = np.eye(d)

        # Critic value: M.T @ Ueff @ M should equal Ub
        # Set U1 = U2 = U3 = I, then Ueff = I, M.T @ I @ M = M.T @ M = I_ds (close to Ub)
        # Better: embed Ub into d×d space
        Ub = init_from_downstairs["Ub"]  # (ds, ds) orthogonal
        # Ueff should satisfy M.T @ Ueff @ M = Ub
        # One solution: Ueff = M @ Ub @ M.T + (I - M @ M.T) (orthogonal complement)
        # Simpler: set U1 = U2 = I, U3 = M @ Ub @ M.T + orthogonal complement
        Ueff_target = M @ Ub @ M.T
        # Add orthogonal complement to make it full rank orthogonal
        P_M = M @ M.T  # projection onto column space of M
        P_perp = np.eye(d) - P_M
        Ueff_full = Ueff_target + P_perp
        # Project to nearest orthogonal matrix
        U, S, Vt = np.linalg.svd(Ueff_full)
        critic.U3 = U @ Vt
        critic.U1 = np.eye(d)
        critic.U2 = np.eye(d)

        # Critic advantage: Zeff @ M should give equivalent to Zb.T
        # Set Z1 = Z2 = I, Z3 = Zb.T @ M.T (with row normalization)
        Zb = init_from_downstairs["Zb"]  # (ds, da)
        Z3_target = Zb.T @ M.T  # (da, d)
        row_norms = np.linalg.norm(Z3_target, axis=1, keepdims=True) + 1e-12
        critic.Z3 = Z3_target / row_norms
        critic.Z1 = np.eye(d)
        critic.Z2 = np.eye(d)

        print(f"  Matched from downstairs: W3 rows normalized, U3 projected to orthogonal")
    else:
        # Initialize advantage gradient direction for DPG
        Weff_current = actor.effective_matrix()
        target_Zeff = -2 * (Weff_current + K_opt @ M.T)
        if np.linalg.norm(target_Zeff) > 1e-6:
            critic.Z3 = target_Zeff / (np.linalg.norm(target_Zeff, axis=1, keepdims=True) + 1e-12)
            critic.Z1 = np.eye(d)
            critic.Z2 = np.eye(d)

    # Print initial cos similarity
    Weff = actor.effective_matrix()
    init_cos = cos_sim(Weff, Weff_opt)
    print(f"  Initial cos(policy, optimal): {init_cos:.4f}")

    metrics = {
        "rewards": [],
        "cos_to_opt": [],
        "mse": [],
    }

    for it in range(iters):
        # Rollout using upstairs policy
        def policy(obs):
            return actor.act_upstairs(obs["upstairs"])

        traj = env.rollout(policy, exploration_std=exploration_std, use_upstairs=True)

        # Update
        stats = upstairs_batch_multistep_update(
            actor, critic,
            traj["o"], traj["a"], traj["r"],
            beta=beta,
            dt=dt,
            L=L,
            eta_actor=eta_actor,
            eta_critic=eta_critic,
            use_generator=use_generator,
            use_cayley=use_cayley,
        )

        # Metrics
        mean_r = float(traj["r"].mean())
        Weff = actor.effective_matrix()
        cos = cos_sim(Weff, Weff_opt)

        metrics["rewards"].append(mean_r)
        metrics["cos_to_opt"].append(cos)
        metrics["mse"].append(stats["mse"])

    return metrics


def plot_comparison(
    down_metrics: Dict[str, List[float]],
    up_metrics: Dict[str, List[float]],
    save_path: str = None,
):
    """Plot comparison of upstairs vs downstairs learning."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    iters = len(down_metrics["rewards"])
    x = np.arange(iters)

    # Rewards
    axes[0].plot(x, down_metrics["rewards"], label="Downstairs", alpha=0.7)
    axes[0].plot(x, up_metrics["rewards"], label="Upstairs", alpha=0.7)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Mean Reward")
    axes[0].set_title("Learning Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Cosine similarity to optimal
    axes[1].plot(x, down_metrics["cos_to_opt"], label="Downstairs", alpha=0.7)
    axes[1].plot(x, up_metrics["cos_to_opt"], label="Upstairs", alpha=0.7)
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Cosine Similarity")
    axes[1].set_title("Policy Convergence to Optimal")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([-1.1, 1.1])

    # Critic MSE
    axes[2].semilogy(x, down_metrics["mse"], label="Downstairs", alpha=0.7)
    axes[2].semilogy(x, up_metrics["mse"], label="Upstairs", alpha=0.7)
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("MSE (log scale)")
    axes[2].set_title("Critic Value MSE")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    plt.close()


def main():
    np.set_printoptions(precision=4, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=64, help="Observation dimension")
    parser.add_argument("--ds", type=int, default=4, help="Latent state dimension")
    parser.add_argument("--da", type=int, default=1, help="Action dimension")
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--T", type=float, default=10.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--n_steps", type=int, default=30, help="Multi-step horizon for bootstrapping")
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--eta_actor", type=float, default=0.01)
    parser.add_argument("--eta_critic", type=float, default=0.05)
    parser.add_argument("--exploration_std", type=float, default=0.1)
    parser.add_argument("--transition_noise", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.5, help="Stability parameter (smaller = harder)")
    parser.add_argument("--seed", type=int, default=0, help="Seed for LQR problem construction")
    parser.add_argument("--init_seed", type=int, default=None, help="Seed for policy initialization (default: seed+1)")
    parser.add_argument("--noise_seed", type=int, default=None, help="Seed for exploration/env noise (default: seed+2)")
    parser.add_argument("--save_plot", type=str, default="comparison_plot.png", help="Path to save plot")
    parser.add_argument("--save_csv", type=str, default="comparison_results.csv", help="Path to save CSV")
    parser.add_argument("--use_generator", action="store_true", help="Use generator-based TD error instead of multi-step")
    parser.add_argument("--use_sgd", action="store_true", help="Use standard SGD instead of Cayley retraction (for both upstairs and downstairs)")
    parser.add_argument("--balanced", action="store_true", help="Use balanced DLN initialization (arXiv:2411.09004)")
    parser.add_argument("--downstairs_only", action="store_true", help="Only run downstairs learning (skip upstairs)")
    args = parser.parse_args()

    # Separate seeds for: LQR problem, policy init, exploration noise
    init_seed = args.init_seed if args.init_seed is not None else args.seed + 1
    noise_seed = args.noise_seed if args.noise_seed is not None else args.seed + 2

    d, ds, da = args.d, args.ds, args.da

    # Create Stiefel-compatible LQR
    G, H, Q, R = construct_compatible_lqr(ds, da, alpha=args.alpha, seed=args.seed)
    params = LQRParams(
        d=d, ds=ds, da=da, dt=args.dt, T=args.T,
        G=G, H=H, Q=Q, R=R,
        sigma_explore=0.05, eps_obs=0.05, seed=args.seed,
        Sigma_base=args.transition_noise * np.eye(ds),
    )
    env = LQRUpDownEnv(params)
    K_opt = env.K_opt
    M = env.M

    print("=" * 60)
    print("Upstairs vs Downstairs Comparison (Stiefel_Ascent_RLC.pdf)")
    print("=" * 60)
    print(f"d={d}, ds={ds}, da={da}, dt={args.dt}, T={args.T}, n_steps={args.n_steps}")
    print(f"beta={args.beta}, exploration_std={args.exploration_std}")
    print(f"eta_actor={args.eta_actor}, eta_critic={args.eta_critic}")
    print(f"\nOptimal K (latent): {K_opt.flatten()}")
    print(f"||K_opt|| = {np.linalg.norm(K_opt):.4f}")
    # Verify Stiefel constraint: K K^T = I_da
    KKT = K_opt @ K_opt.T
    print(f"K K^T = {KKT.flatten()} (should be I_{da})")

    # Run downstairs FIRST to get Stiefel-valid initialization
    print("\n--- Running Downstairs Learning ---")
    if args.use_generator:
        print("  (Using generator-based TD error)")
    if args.use_sgd:
        print("  (Using standard SGD, no Cayley retraction)")
    # Pass csv_path for incremental logging when downstairs_only
    down_csv = args.save_csv if args.downstairs_only else None
    down_metrics, init_params = run_downstairs(
        env, K_opt,
        iters=args.iters,
        L=args.n_steps,
        beta=args.beta,
        dt=args.dt,
        eta_actor=args.eta_actor,
        eta_critic=args.eta_critic,
        exploration_std=args.exploration_std,
        init_seed=init_seed,  # Different from LQR seed to avoid correlation
        exploration_seed=noise_seed,  # Varies for different exploration noise
        return_init=True,
        use_generator=args.use_generator,
        use_sgd=args.use_sgd,
        csv_path=down_csv,
        log_every=1,
    )
    print(f"Final reward: {down_metrics['rewards'][-1]:.4f}")
    print(f"Final cos(policy, optimal): {down_metrics['cos_to_opt'][-1]:.4f}")

    if args.downstairs_only:
        # Summary already printed, CSV already written incrementally
        print("\n" + "=" * 60)
        print("DOWNSTAIRS ONLY SUMMARY")
        print("=" * 60)
        down_final_r = np.mean(down_metrics["rewards"][-50:])
        down_final_cos = np.mean(down_metrics["cos_to_opt"][-50:])
        print(f"Final 50-iter average reward: {down_final_r:.4f}")
        print(f"Final 50-iter average cos: {down_final_cos:.4f}")
        if args.save_csv:
            print(f"Results saved incrementally to {args.save_csv}")
        return

    # Run upstairs with MATCHED initialization from downstairs
    print("\n--- Running Upstairs Learning (matched init) ---")
    if args.use_generator:
        print("  (Using generator-based TD error)")
    if args.use_sgd:
        print("  (Using standard SGD, no Cayley retraction)")
    if args.balanced:
        print("  (Using balanced DLN initialization)")
    up_metrics = run_upstairs(
        env, K_opt, M,
        iters=args.iters,
        L=args.n_steps,
        beta=args.beta,
        dt=args.dt,
        eta_actor=args.eta_actor,
        eta_critic=args.eta_critic,
        exploration_std=args.exploration_std,
        init_seed=init_seed,  # Different from LQR seed to avoid correlation
        exploration_seed=noise_seed,  # Varies for different exploration noise
        init_from_downstairs=init_params if not args.balanced else None,  # balanced uses its own init
        use_generator=args.use_generator,
        use_cayley=not args.use_sgd,
        balanced=args.balanced,
    )
    print(f"Final reward: {up_metrics['rewards'][-1]:.4f}")
    print(f"Final cos(policy, optimal): {up_metrics['cos_to_opt'][-1]:.4f}")

    # Evaluate optimal policy
    print("\n--- Optimal Policy Performance ---")
    opt_rewards = []
    for _ in range(10):
        def opt_policy(obs):
            s_est = M.T @ obs["upstairs"]
            return (-K_opt @ s_est).flatten()
        traj = env.rollout(opt_policy, exploration_std=args.exploration_std, use_upstairs=True)
        opt_rewards.append(float(traj["r"].mean()))
    opt_mean = float(np.mean(opt_rewards))
    opt_std = float(np.std(opt_rewards))
    print(f"Optimal policy reward: {opt_mean:.4f} ± {opt_std:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Compute correlation between learning curves
    r_corr = np.corrcoef(down_metrics["rewards"], up_metrics["rewards"])[0, 1]
    cos_corr = np.corrcoef(down_metrics["cos_to_opt"], up_metrics["cos_to_opt"])[0, 1]

    print(f"Reward curve correlation: {r_corr:.4f}")
    print(f"Policy convergence correlation: {cos_corr:.4f}")

    # Final performance comparison
    down_final_r = np.mean(down_metrics["rewards"][-50:])
    up_final_r = np.mean(up_metrics["rewards"][-50:])
    down_final_cos = np.mean(down_metrics["cos_to_opt"][-50:])
    up_final_cos = np.mean(up_metrics["cos_to_opt"][-50:])

    print(f"\nFinal 50-iter average reward: Down={down_final_r:.4f}, Up={up_final_r:.4f}, Optimal={opt_mean:.4f}")
    print(f"Final 50-iter average cos: Down={down_final_cos:.4f}, Up={up_final_cos:.4f}")

    if abs(cos_corr) > 0.8:
        print("\n✓ POLICY CONVERGENCE MATCHES between upstairs and downstairs")
        print(f"  (Policy correlation = {cos_corr:.4f})")
        if abs(down_final_cos - up_final_cos) < 0.1:
            print(f"  Final policies converged to similar quality (cos diff = {abs(down_final_cos - up_final_cos):.4f})")
    else:
        print("\n✗ Policy convergence differs between upstairs and downstairs")
        print(f"  (Policy correlation = {cos_corr:.4f} - may need hyperparameter tuning)")

    # Plot
    plot_comparison(down_metrics, up_metrics, save_path=args.save_plot)

    # Save CSV (every 10 steps + final)
    if args.save_csv:
        import csv
        with open(args.save_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['iter', 'down_reward', 'up_reward', 'down_cos', 'up_cos', 'down_mse', 'up_mse'])
            for i in range(len(down_metrics["rewards"])):
                writer.writerow([
                    i,
                    down_metrics["rewards"][i],
                    up_metrics["rewards"][i],
                    down_metrics["cos_to_opt"][i],
                    up_metrics["cos_to_opt"][i],
                    down_metrics["mse"][i],
                    up_metrics["mse"][i],
                ])
        print(f"Saved results to {args.save_csv}")


if __name__ == "__main__":
    main()
