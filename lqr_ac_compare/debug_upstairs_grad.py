"""
Debug upstairs actor gradient direction.
"""
import numpy as np

from lqr_env import LQRParams, LQRUpDownEnv
from upstairs_models import ActorDLN_L3, CriticDLN_L3
from upstairs_learning_batch import grad_a_Psi, grad_actor_W, cayley_retract


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

    d, ds, da = 64, 4, 1
    seed = 0

    G, H, Q, R = construct_compatible_lqr(ds, da, alpha=0.5, seed=seed)
    params = LQRParams(
        d=d, ds=ds, da=da, dt=0.02, T=10.0,
        G=G, H=H, Q=Q, R=R,
        sigma_explore=0.05, eps_obs=0.05, seed=seed,
        Sigma_base=0.1 * np.eye(ds),
    )
    env = LQRUpDownEnv(params)
    K_opt = env.K_opt
    M = env.M

    Weff_opt = -K_opt @ M.T  # (da, d)
    target_Zeff = -env.H.T @ M.T  # (da, d) - correct advantage gradient

    print(f"K_opt = {K_opt}")
    print(f"||Weff_opt|| = {np.linalg.norm(Weff_opt):.4f}")
    print(f"||target_Zeff|| = {np.linalg.norm(target_Zeff):.4f}")

    # Initialize actor with random W3
    actor = ActorDLN_L3(d=d, da=da, seed=seed + 1)

    # Initialize critic with optimal Z3
    critic = CriticDLN_L3(d=d, da=da, seed=seed + 2)
    critic.Z3 = target_Zeff / (np.linalg.norm(target_Zeff) + 1e-12)
    critic.Z1 = np.eye(d)
    critic.Z2 = np.eye(d)

    Weff_init = actor.effective_matrix()
    cos_init = np.sum(Weff_init * Weff_opt) / (np.linalg.norm(Weff_init) * np.linalg.norm(Weff_opt))

    print(f"\nInitial cos(Weff, Weff_opt) = {cos_init:.4f}")

    # Sample some observations
    s = np.random.randn(ds)
    o = M @ s  # observation

    # Compute advantage gradient at o
    g_a = grad_a_Psi(o, critic.Z1, critic.Z2, critic.Z3,
                     critic.Zp1, critic.Zp2, critic.Zp3)

    print(f"\nAt state s = {s}")
    print(f"Observation o norm = {np.linalg.norm(o):.4f}")
    print(f"Advantage gradient g_a = {g_a}")

    # What's the optimal action?
    a_opt = -K_opt @ s
    print(f"Optimal action a* = {a_opt}")

    # Current policy action
    a_curr = actor.act_upstairs(o)
    print(f"Current action a = {a_curr}")

    # g_a should point from current action toward optimal action
    # For LQR: dA/da = -2(a - a*), so at a_curr, dA/da = -2(a_curr - a_opt) = -2(a_curr + K@s)
    expected_g_a = -2 * (a_curr - a_opt)
    print(f"Expected g_a (from LQR) = {expected_g_a}")

    # Compute actor gradients
    dW = grad_actor_W(o, g_a, actor.W1, actor.W2, actor.W3)
    print(f"\nActor gradients:")
    print(f"||dW['W1']|| = {np.linalg.norm(dW['W1']):.4e}")
    print(f"||dW['W2']|| = {np.linalg.norm(dW['W2']):.4e}")

    # Apply a small update and see which direction Weff moves
    eta = 0.01
    W1_new = cayley_retract(actor.W1, dW['W1'], eta)
    W2_new = cayley_retract(actor.W2, dW['W2'], eta)

    Weff_new = actor.W3 @ W2_new @ W1_new
    cos_new = np.sum(Weff_new * Weff_opt) / (np.linalg.norm(Weff_new) * np.linalg.norm(Weff_opt))

    print(f"\nAfter update:")
    print(f"cos(Weff_new, Weff_opt) = {cos_new:.4f}")
    print(f"Change in cos: {cos_new - cos_init:+.4f}")

    # Let's also check what direction we WANT to move
    # We want to move Weff toward Weff_opt
    # The Euclidean gradient of ||Weff - Weff_opt||^2 w.r.t. Weff is 2(Weff - Weff_opt)
    # So we want to move in direction -(Weff - Weff_opt) = Weff_opt - Weff
    desired_direction = Weff_opt - Weff_init
    print(f"\nDesired Weff direction (toward opt): {desired_direction[:, :4]}...")

    # Actual direction from gradient
    actual_Weff_change = Weff_new - Weff_init
    print(f"Actual Weff change: {actual_Weff_change[:, :4]}...")

    # Alignment between desired and actual
    cos_alignment = np.sum(desired_direction * actual_Weff_change) / (
        np.linalg.norm(desired_direction) * np.linalg.norm(actual_Weff_change) + 1e-10)
    print(f"Alignment cos(desired, actual): {cos_alignment:.4f}")

    # ============================================
    # Alternative: Direct Weff gradient approach
    # ============================================
    print("\n=== Alternative: Direct Weff gradient ===")

    # Gradient at Weff level: ∂J/∂Weff = g_a @ o.T (outer product)
    # For J = g_a.T @ a = g_a.T @ Weff @ o
    G_Weff = np.outer(g_a, o)  # (da, d) = (1, 64)
    print(f"||G_Weff|| = {np.linalg.norm(G_Weff):.4f}")

    # Project onto tangent space of unit-norm row vectors
    # For Weff with ||Weff|| = 1, the tangent space is perpendicular to Weff
    # Projection: G_proj = G_Weff - (Weff @ G_Weff.T) * Weff
    proj_coef = float(Weff_init @ G_Weff.T)
    G_proj = G_Weff - proj_coef * Weff_init
    print(f"||G_proj|| = {np.linalg.norm(G_proj):.4f}")

    # Move Weff directly on the sphere
    Weff_direct = Weff_init + eta * G_proj
    Weff_direct = Weff_direct / np.linalg.norm(Weff_direct)  # re-normalize

    cos_direct = np.sum(Weff_direct * Weff_opt) / (np.linalg.norm(Weff_direct) * np.linalg.norm(Weff_opt))
    print(f"Direct update: cos(Weff_direct, Weff_opt) = {cos_direct:.4f}")
    print(f"Change in cos: {cos_direct - cos_init:+.4f}")

    direct_change = Weff_direct - Weff_init
    cos_direct_align = np.sum(desired_direction * direct_change) / (
        np.linalg.norm(desired_direction) * np.linalg.norm(direct_change) + 1e-10)
    print(f"Direct alignment cos(desired, actual): {cos_direct_align:.4f}")

    # ============================================
    # Alternative: Update Q = W2 @ W1 directly
    # ============================================
    print("\n=== Alternative: Update Q = W2 @ W1 directly ===")

    Q = actor.W2 @ actor.W1  # (d, d) orthogonal
    W3 = actor.W3  # (da, d)

    # Weff = W3 @ Q, we want to update Q to increase J = g_a.T @ W3 @ Q @ o
    # ∂J/∂Q = W3.T @ g_a @ o.T  (chain rule)
    G_Q = W3.T @ np.outer(g_a, o)  # (d, d)
    print(f"||G_Q|| = {np.linalg.norm(G_Q):.4f}")

    # Cayley retraction on Q
    Q_new = cayley_retract(Q, G_Q, eta)

    Weff_Q = W3 @ Q_new
    cos_Q = np.sum(Weff_Q * Weff_opt) / (np.linalg.norm(Weff_Q) * np.linalg.norm(Weff_opt))
    print(f"Update Q directly: cos(Weff_Q, Weff_opt) = {cos_Q:.4f}")
    print(f"Change in cos: {cos_Q - cos_init:+.4f}")


if __name__ == "__main__":
    main()
