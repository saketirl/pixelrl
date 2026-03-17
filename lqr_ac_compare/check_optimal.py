"""
Check the optimal LQR solution and compare to the critic parametrization.

For continuous-time LQR with dynamics: ds = (G s + H a) dt + noise
and cost rate: c(s,a) = s^T Q s + a^T R a

The optimal value function is: V*(s) = s^T P s
where P solves the continuous-time algebraic Riccati equation (CARE):
    G^T P + P G - P H R^{-1} H^T P + Q = 0

The optimal policy is: a* = -K s, where K = R^{-1} H^T P
"""

import numpy as np
from lqr_env import LQRParams, LQRUpDownEnv

# Use same defaults as run_downstairs.py
params = LQRParams(
    d=16,
    ds=4,
    da=1,
    dt=0.02,
    T=2.0,
    sigma_explore=0.05,
    eps_obs=0.0,
    seed=0,
    Sigma_base=1.0 * np.eye(4),  # transition_noise=1.0
    Q=1.0 * np.eye(4),           # reward_state_scale=1.0
    R=1.0 * np.eye(1),           # reward_action_scale=1.0
)

env = LQRUpDownEnv(params)

print("=== LQR System Matrices (randomly generated with seed=0) ===")
print(f"\nG (drift matrix, ds x ds):\n{env.G}")
print(f"\nH (control matrix, ds x da):\n{env.H}")
print(f"\nQ (state cost, ds x ds):\n{env.Q}")
print(f"\nR (action cost, da x da):\n{env.R}")

print("\n=== Optimal LQR Solution ===")
print(f"\nP_opt (Riccati solution, ds x ds):\n{env.P_opt}")
print(f"\nEigenvalues of P_opt: {np.linalg.eigvalsh(env.P_opt)}")
print(f"\nK_opt (optimal gain, da x ds):\n{env.K_opt}")

print("\n=== Mapping to Critic Parametrization ===")
print("""
The downstairs critic models:
    phi(s) = s^T Ub s + Uc^T s + c

The optimal value function is:
    V*(s) = s^T P_opt s

So ideally:
    Ub = P_opt
    Uc = 0
    c = 0

HOWEVER: The code constrains Ub to be ORTHOGONAL (Ub^T Ub = I).
This is a significant constraint! Orthogonal matrices have singular values = 1,
so s^T Ub s can only represent quadratic forms with a limited range.

For the quadratic form s^T Ub s with orthogonal Ub:
    s^T Ub s = s^T ((Ub + Ub^T)/2) s  (antisymmetric part vanishes)

The symmetric part (Ub + Ub^T)/2 has eigenvalues in [-1, 1].
""")

# Check what symmetric part of a random orthogonal matrix looks like
from downstairs_models import orthogonal_init
rng = np.random.default_rng(42)
Ub_rand = orthogonal_init(4, rng)
Ub_sym = (Ub_rand + Ub_rand.T) / 2
print(f"Example orthogonal Ub:\n{Ub_rand}")
print(f"\nSymmetric part (Ub + Ub^T)/2:\n{Ub_sym}")
print(f"Eigenvalues of symmetric part: {np.linalg.eigvalsh(Ub_sym)}")

print(f"\nP_opt eigenvalues for comparison: {np.linalg.eigvalsh(env.P_opt)}")

print("\n=== Implication ===")
print("""
The orthogonal constraint on Ub means the critic CANNOT exactly represent
the optimal value function V*(s) = s^T P_opt s unless P_opt happens to
have eigenvalues near 1 (which is unlikely for general LQR problems).

This is a representation gap in the current model architecture.
Consider either:
1. Relaxing Ub to be symmetric PSD instead of orthogonal
2. Adding a scalar multiplier: phi(s) = alpha * s^T Ub s + ...
3. Using a different factorization
""")

# What the optimal actor should be
print("\n=== Optimal Actor ===")
print(f"""
The downstairs actor models: a = Wc @ s  (where Wc = Wc_col^T)
The optimal policy is: a* = -K_opt @ s

So the optimal Wc = -K_opt:
{-env.K_opt}

The actor constrains Wc_col to be Stiefel (orthonormal columns).
Since Wc_col is (ds x da) = (4 x 1), this means Wc_col is a unit vector.
So Wc = Wc_col^T is a (1 x 4) row vector with unit norm.

Optimal -K_opt has norm: {np.linalg.norm(-env.K_opt):.4f}

If norm(-K_opt) != 1, the actor cannot exactly represent the optimal policy
under the Stiefel constraint.
""")
