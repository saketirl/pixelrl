"""
Debug version to understand why learning isn't converging.
"""
import argparse
import numpy as np

from lqr_env import LQRParams, LQRUpDownEnv
from downstairs_models import DownstairsActorShallowT2, DownstairsCriticShallowT2


def construct_compatible_lqr(ds, da, alpha=0.5, seed=42):
    """Construct LQR with P=I, ||K||=1."""
    rng = np.random.default_rng(seed)
    G = -alpha * np.eye(ds)
    H_raw = rng.standard_normal((ds, da))
    H = H_raw / np.linalg.norm(H_raw)
    R = np.eye(da)
    Q = H @ H.T + 2 * alpha * np.eye(ds)
    return G, H, Q, R


def cayley_retract_stiefel(X, G, eta):
    """Cayley retraction on Stiefel manifold."""
    n, _ = X.shape
    I = np.eye(n, dtype=X.dtype)
    A = G @ X.T - X @ G.T
    B = (I - 0.5 * eta * A) @ X
    return np.linalg.solve(I + 0.5 * eta * A, B)


def main():
    np.set_printoptions(precision=4, suppress=True)

    ds, da = 4, 1
    alpha = 0.5
    seed = 0

    # Create compatible LQR
    G, H, Q, R = construct_compatible_lqr(ds, da, alpha, seed)

    params = LQRParams(
        d=16, ds=ds, da=da, dt=0.02, T=2.0,
        G=G, H=H, Q=Q, R=R,
        sigma_explore=0.05, eps_obs=0.0, seed=seed,
        Sigma_base=0.1 * np.eye(ds),
    )
    env = LQRUpDownEnv(params)

    # Get optimal solution
    K_opt = env.K_opt  # optimal gain: a* = -K @ s
    P_opt = env.P_opt  # optimal value: V*(s) = s^T P s

    print("=== Optimal Solution ===")
    print(f"K_opt: {K_opt}")
    print(f"Optimal Wc = -K_opt: {-K_opt}")
    print(f"P_opt:\n{P_opt}")
    print(f"||K_opt|| = {np.linalg.norm(K_opt):.4f}")

    # Initialize actor/critic
    actor = DownstairsActorShallowT2(ds=ds, da=da, seed=seed+1)
    critic = DownstairsCriticShallowT2(ds=ds, da=da, seed=seed+2)

    print(f"\n=== Initial Parameters ===")
    print(f"Actor Wc: {actor.Wc}")
    print(f"Critic Ub:\n{critic.Ub}")

    # Learning parameters
    beta = 0.1
    dt = 0.02
    eta = 0.01

    print(f"\n=== Learning (eta={eta}) ===")

    for it in range(100):
        # Rollout
        def policy(obs_dict):
            return actor.act(obs_dict["downstairs"])

        traj = env.rollout(policy, exploration_std=0.05, use_upstairs=False)
        s_traj = traj["s"][:101]  # truncate
        a_traj = traj["a"][:100]
        r_traj = traj["r"][:100]

        mean_r = r_traj.mean()

        # Compute gradients manually to debug
        N = len(r_traj)
        G_Wc = np.zeros_like(actor.Wc_col)
        G_Ub = np.zeros_like(critic.Ub)
        G_Uc = np.zeros_like(critic.Uc_col)
        G_c = 0.0
        G_Zb = np.zeros_like(critic.Zb)
        G_Zc = np.zeros_like(critic.Zc_row)

        total_delta = 0
        total_g = np.zeros(da)

        for k in range(N):
            t = k * dt
            w = np.exp(-beta * t)

            s_t = s_traj[k]
            s_tp1 = s_traj[k + 1]
            a_t = a_traj[k]
            r_t = r_traj[k]

            phi_t = critic.phi(s_t)
            phi_tp1 = critic.phi(s_tp1)
            psi_t = critic.psi(s_t, a_t, actor)

            # TD error
            delta = r_t - psi_t + (np.exp(-beta * dt) * phi_tp1 - phi_t) / dt
            total_delta += delta

            a_pi = actor.act(s_t)
            a_bar = (a_t - a_pi).flatten()

            # Critic gradients
            G_Ub += (w * delta * dt) * np.outer(s_t, s_t)
            G_Uc += (w * delta * dt) * s_t.reshape(-1, 1)
            G_c += (w * delta * dt)
            G_Zb += (w * delta * dt) * np.outer(s_t, a_bar)
            G_Zc += (w * delta * dt) * a_bar.reshape(1, -1)

            # Actor gradient direction
            g_t = (critic.Zb.T @ s_t).flatten() + critic.Zc_row.flatten()
            total_g += g_t

            G_Wc += (w * dt) * (s_t.reshape(ds, 1) @ g_t.reshape(1, da))

        # What direction should actor move?
        # Current Wc
        current_Wc = actor.Wc.flatten()
        optimal_Wc = (-K_opt).flatten()
        diff_to_optimal = optimal_Wc - current_Wc

        # Gradient direction (normalized)
        G_Wc_flat = (G_Wc.T).flatten()  # Wc = Wc_col.T, so gradient for Wc

        # Cosine similarity between gradient and direction to optimal
        if np.linalg.norm(G_Wc_flat) > 1e-10:
            cos_sim = np.dot(G_Wc_flat, diff_to_optimal) / (np.linalg.norm(G_Wc_flat) * np.linalg.norm(diff_to_optimal) + 1e-10)
        else:
            cos_sim = 0

        if it % 10 == 0:
            print(f"\n[iter {it}] mean_r={mean_r:.4f}, td_mse={total_delta**2/N:.4f}")
            print(f"  Wc:      {current_Wc}")
            print(f"  Optimal: {optimal_Wc}")
            print(f"  cos(G_Wc, diff) = {cos_sim:.4f}, avg_g={total_g/N}")
            print(f"  Zb: {critic.Zb.flatten()}")

        # Apply critic updates first
        L_eff = 3
        critic.Ub = cayley_retract_stiefel(critic.Ub, L_eff * G_Ub, eta)
        critic.Uc_col = cayley_retract_stiefel(critic.Uc_col, (L_eff - 1) * G_Uc, eta)
        critic.c = critic.c + eta * G_c
        critic.Zb = cayley_retract_stiefel(critic.Zb, (L_eff - 1) * G_Zb, eta)
        critic.Zc_row = critic.Zc_row + eta * max(L_eff - 2, 0) * G_Zc

        # Apply actor update
        # Try NEGATIVE gradient - flip sign to descend in cost (ascend in reward)
        actor.Wc_col = cayley_retract_stiefel(actor.Wc_col, -(L_eff - 1) * G_Wc, eta)

    print(f"\n=== After Learning ===")
    print(f"Final Wc: {actor.Wc}")
    print(f"Optimal:  {-K_opt}")


if __name__ == "__main__":
    main()
