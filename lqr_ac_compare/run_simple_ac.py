"""
Simple actor-critic without Stiefel constraints to verify basic algorithm works.
Uses unconstrained linear actor and quadratic critic.
"""
import argparse
import numpy as np

from lqr_env import LQRParams, LQRUpDownEnv


def construct_compatible_lqr(ds, da, alpha=0.5, seed=42):
    rng = np.random.default_rng(seed)
    G = -alpha * np.eye(ds)
    H_raw = rng.standard_normal((ds, da))
    H = H_raw / np.linalg.norm(H_raw)
    R = np.eye(da)
    Q = H @ H.T + 2 * alpha * np.eye(ds)
    return G, H, Q, R


class SimpleActor:
    """Linear policy: a = W @ s"""
    def __init__(self, ds, da, seed=0):
        rng = np.random.default_rng(seed)
        self.W = 0.1 * rng.standard_normal((da, ds))

    def act(self, s):
        return (self.W @ s).flatten()


class SimpleCritic:
    """
    Value: V(s) = s^T P s + p^T s + c
    Advantage: A(s, a) = (a - pi(s))^T M (a - pi(s)) for some M

    Simplified: just learn V(s) = s^T P s (symmetric P)
    """
    def __init__(self, ds, seed=0):
        rng = np.random.default_rng(seed)
        self.ds = ds
        # Initialize P as small symmetric matrix
        P_raw = 0.1 * rng.standard_normal((ds, ds))
        self.P = 0.5 * (P_raw + P_raw.T)

    def value(self, s):
        return float(s.T @ self.P @ s)


def main():
    np.set_printoptions(precision=4, suppress=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=int, default=4)
    parser.add_argument("--da", type=int, default=1)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--eta_actor", type=float, default=0.001)
    parser.add_argument("--eta_critic", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ds, da = args.ds, args.da

    # Create compatible LQR
    G, H, Q, R = construct_compatible_lqr(ds, da, alpha=0.5, seed=args.seed)

    params = LQRParams(
        d=16, ds=ds, da=da, dt=0.02, T=10.0,
        G=G, H=H, Q=Q, R=R,
        sigma_explore=0.05, eps_obs=0.0, seed=args.seed,
        Sigma_base=0.1 * np.eye(ds),
    )
    env = LQRUpDownEnv(params)

    K_opt = env.K_opt
    opt_info = env.optimal_average_reward(exploration_std=0.05)

    print("=== Simple Actor-Critic (no Stiefel constraints) ===")
    print(f"Optimal K: {K_opt}")
    print(f"Optimal steady-state reward: {opt_info['steady_state_avg_reward']:.4f}")

    actor = SimpleActor(ds, da, seed=args.seed + 1)
    critic = SimpleCritic(ds, seed=args.seed + 2)

    print(f"\nInitial actor W: {actor.W}")
    print(f"Target -K_opt:   {-K_opt}")

    for it in range(args.iters):
        # Rollout
        def policy(obs):
            return actor.act(obs["downstairs"])

        traj = env.rollout(policy, exploration_std=0.05, use_upstairs=False)
        s_traj = traj["s"]
        a_traj = traj["a"]
        r_traj = traj["r"]
        N = len(r_traj)

        mean_r = r_traj.mean()

        # Compute returns (simple MC return)
        returns = np.zeros(N)
        G_t = 0
        for t in reversed(range(N)):
            G_t = r_traj[t] + args.gamma * G_t
            returns[t] = G_t

        # Update critic (MSE loss on returns)
        grad_P = np.zeros_like(critic.P)
        for t in range(N):
            s = s_traj[t]
            V = critic.value(s)
            td_error = returns[t] - V
            # Gradient of V = s^T P s w.r.t. P is s @ s^T
            grad_P += td_error * np.outer(s, s)
        grad_P /= N
        # Make gradient symmetric
        grad_P = 0.5 * (grad_P + grad_P.T)
        critic.P += args.eta_critic * grad_P

        # Update actor (policy gradient)
        # For deterministic policy, use ∂V/∂s * ∂s/∂a type gradient
        # Simplified: use advantage * action gradient
        grad_W = np.zeros_like(actor.W)
        for t in range(N):
            s = s_traj[t]
            a = a_traj[t]
            V = critic.value(s)
            advantage = returns[t] - V

            # Policy gradient: ∂J/∂W ≈ advantage * ∂log π/∂W
            # For deterministic policy with exploration noise:
            # a = W @ s + noise, so ∂a/∂W = s (feature)
            # Gradient direction: advantage * outer(a - W@s, s)
            # But a - W@s is the noise, which we don't know

            # Alternative: use REINFORCE-style gradient
            # ∂J/∂W ≈ advantage * s (assuming linear policy)
            grad_W += advantage * np.outer(a, s)

        grad_W /= N
        actor.W += args.eta_actor * grad_W

        if (it + 1) % 20 == 0 or it == 0:
            # Compute cosine similarity to optimal
            W_flat = actor.W.flatten()
            K_opt_flat = (-K_opt).flatten()
            cos_sim = np.dot(W_flat, K_opt_flat) / (np.linalg.norm(W_flat) * np.linalg.norm(K_opt_flat) + 1e-10)

            print(f"[iter {it+1:3d}] mean_r={mean_r:.4f}, cos(W, -K*)={cos_sim:.4f}, ||W||={np.linalg.norm(actor.W):.4f}")

    print(f"\n=== Final Results ===")
    print(f"Learned W: {actor.W}")
    print(f"Target -K: {-K_opt}")

    # Test final policy
    def final_policy(obs):
        return actor.act(obs["downstairs"])

    rewards = []
    for _ in range(10):
        traj = env.rollout(final_policy, exploration_std=0.05)
        rewards.append(traj["r"].mean())
    print(f"Final mean reward: {np.mean(rewards):.4f} ± {np.std(rewards):.4f}")

    # Test optimal
    def opt_policy(obs):
        return (-K_opt @ obs["downstairs"]).flatten()

    rewards = []
    for _ in range(10):
        traj = env.rollout(opt_policy, exploration_std=0.05)
        rewards.append(traj["r"].mean())
    print(f"Optimal mean reward: {np.mean(rewards):.4f} ± {np.std(rewards):.4f}")


if __name__ == "__main__":
    main()
