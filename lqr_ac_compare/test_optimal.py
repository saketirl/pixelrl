"""
Test that the optimal policy achieves optimal reward.
"""
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


def main():
    np.set_printoptions(precision=4, suppress=True)

    ds, da = 4, 1
    seed = 0

    G, H, Q, R = construct_compatible_lqr(ds, da, alpha=0.5, seed=seed)

    params = LQRParams(
        d=16, ds=ds, da=da, dt=0.02, T=20.0,  # longer horizon
        G=G, H=H, Q=Q, R=R,
        sigma_explore=0.05, eps_obs=0.0, seed=seed,
        Sigma_base=0.1 * np.eye(ds),
    )
    env = LQRUpDownEnv(params)

    K_opt = env.K_opt
    opt_info = env.optimal_average_reward(exploration_std=0.05)

    print("=== Optimal LQR Solution ===")
    print(f"K_opt: {K_opt}")
    print(f"Optimal steady-state reward: {opt_info['steady_state_avg_reward']:.4f}")

    # Test random policy
    print("\n=== Random Policy ===")
    def random_policy(obs):
        return np.random.randn(da) * 0.1

    rewards = []
    for _ in range(10):
        traj = env.rollout(random_policy, exploration_std=0.05)
        rewards.append(traj["r"].mean())
    print(f"Mean reward over 10 rollouts: {np.mean(rewards):.4f} ± {np.std(rewards):.4f}")

    # Test optimal policy
    print("\n=== Optimal Policy (a = -K @ s) ===")
    def optimal_policy(obs):
        s = obs["downstairs"]
        return (-K_opt @ s).flatten()

    rewards = []
    for _ in range(10):
        traj = env.rollout(optimal_policy, exploration_std=0.05)
        rewards.append(traj["r"].mean())
    print(f"Mean reward over 10 rollouts: {np.mean(rewards):.4f} ± {np.std(rewards):.4f}")

    # Show reward trajectory
    traj = env.rollout(optimal_policy, exploration_std=0.05)
    print(f"  Early rewards (t=0-10):   {traj['r'][:10].mean():.4f}")
    print(f"  Mid rewards (t=100-200):  {traj['r'][100:200].mean():.4f}")
    print(f"  Late rewards (t=500-end): {traj['r'][500:].mean():.4f}")

    # Test zero policy
    print("\n=== Zero Policy (a = 0) ===")
    def zero_policy(obs):
        return np.zeros(da)

    rewards = []
    for _ in range(10):
        traj = env.rollout(zero_policy, exploration_std=0.05)
        rewards.append(traj["r"].mean())
    print(f"Mean reward over 10 rollouts: {np.mean(rewards):.4f} ± {np.std(rewards):.4f}")

    # Test negative optimal
    print("\n=== Negative Optimal Policy (a = +K @ s, wrong sign) ===")
    def wrong_policy(obs):
        s = obs["downstairs"]
        return (K_opt @ s).flatten()  # wrong sign

    rewards = []
    for _ in range(10):
        traj = env.rollout(wrong_policy, exploration_std=0.05)
        rewards.append(traj["r"].mean())
    print(f"Mean reward over 10 rollouts: {np.mean(rewards):.4f} ± {np.std(rewards):.4f}")


if __name__ == "__main__":
    main()
