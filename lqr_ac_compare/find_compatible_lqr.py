"""
Find LQR parameters where the optimal solution is compatible with the
Stiefel/orthogonal constraints in the downstairs model.

Requirements:
1. Actor: ||K_opt|| = 1 (unit norm optimal gain)
2. Critic value: P_opt is orthogonal (P^T P = I)
3. Critic advantage: Zb should have orthonormal columns

Mathematical derivation:
-----------------------
For P_opt to be both symmetric (from Riccati) and orthogonal, we need P^2 = I.
Combined with P being PSD, this forces P = I (identity).

Substituting P = I into the continuous-time ARE:
    G^T P + P G - P H R^{-1} H^T P + Q = 0
    G^T + G - H R^{-1} H^T + Q = 0

Solving for Q:
    Q = H R^{-1} H^T - G^T - G

For Q to be PSD, we need: H R^{-1} H^T - G^T - G >= 0

Simple construction:
- Let G = -alpha * I  (stable diagonal system)
- Then -G^T - G = 2*alpha * I
- Q = H R^{-1} H^T + 2*alpha * I  (always PSD for alpha > 0)

For ||K_opt|| = 1:
- K_opt = R^{-1} H^T P = R^{-1} H^T  (since P = I)
- Need ||R^{-1} H^T|| = 1
- If R = I, need ||H|| = 1
"""

import numpy as np
from scipy.linalg import solve_continuous_are, solve_continuous_lyapunov

def verify_riccati(G, H, Q, R, P_expected):
    """Check if P satisfies the ARE."""
    lhs = G.T @ P_expected + P_expected @ G - P_expected @ H @ np.linalg.solve(R, H.T @ P_expected) + Q
    return np.linalg.norm(lhs)

def construct_unit_norm_lqr(ds, da, alpha=0.5, seed=42):
    """
    Construct LQR with P_opt = I and ||K_opt|| = 1.

    Args:
        ds: state dimension
        da: action dimension
        alpha: stability margin (G = -alpha * I)
        seed: random seed for H direction
    """
    rng = np.random.default_rng(seed)

    # G = -alpha * I (stable)
    G = -alpha * np.eye(ds)

    # H: random direction, unit Frobenius norm
    H_raw = rng.standard_normal((ds, da))
    H = H_raw / np.linalg.norm(H_raw)  # ||H||_F = 1

    # R = I
    R = np.eye(da)

    # Q = H H^T + 2*alpha * I  (ensures P = I solves ARE)
    Q = H @ H.T + 2 * alpha * np.eye(ds)

    return G, H, Q, R

def analyze_lqr(G, H, Q, R, name=""):
    """Analyze LQR solution and check constraints."""
    ds, da = H.shape

    print(f"\n{'='*60}")
    print(f"LQR Analysis: {name}")
    print(f"{'='*60}")

    # Solve Riccati
    try:
        P = solve_continuous_are(G, H, Q, R)
    except Exception as e:
        print(f"Failed to solve Riccati: {e}")
        return None

    K = np.linalg.solve(R, H.T @ P)

    print(f"\nSystem dimensions: ds={ds}, da={da}")
    print(f"\nG:\n{G}")
    print(f"\nH:\n{H}")
    print(f"\nQ:\n{Q}")
    print(f"\nR:\n{R}")

    print(f"\n--- Optimal Solution ---")
    print(f"P_opt:\n{P}")
    print(f"K_opt:\n{K}")

    # Check constraints
    print(f"\n--- Constraint Checks ---")

    # 1. P orthogonality: P^T P = I?
    P_ortho_err = np.linalg.norm(P.T @ P - np.eye(ds))
    print(f"||P^T P - I|| = {P_ortho_err:.6f}  (want 0 for orthogonal)")

    # P symmetry (should always hold)
    P_sym_err = np.linalg.norm(P - P.T)
    print(f"||P - P^T|| = {P_sym_err:.6f}  (symmetry check)")

    # P eigenvalues
    P_eigs = np.linalg.eigvalsh(P)
    print(f"eigenvalues(P) = {P_eigs}")

    # 2. K norm
    K_norm = np.linalg.norm(K)
    print(f"||K_opt|| = {K_norm:.6f}  (want 1 for unit norm)")

    # Riccati residual
    riccati_res = verify_riccati(G, H, Q, R, P)
    print(f"Riccati residual = {riccati_res:.2e}")

    # 3. For advantage: what should Zb be?
    # The advantage gradient w.r.t. action at optimal is related to (R + H^T P H)
    # For the model psi(s,a) - psi(s,a*) where psi = s^T Zb a + Zc a
    # dpsi/da = Zb^T s + Zc^T
    # This should relate to H^T * (something) for the dynamics coupling

    print(f"\n--- Advantage Structure ---")
    # H^T is (da, ds), for the Stiefel constraint we need ||H|| on each column
    print(f"H^T (transpose, da x ds):\n{H.T}")
    H_col_norms = np.linalg.norm(H, axis=0)  # norm of each column of H
    print(f"Column norms of H: {H_col_norms}")
    print(f"||H||_F = {np.linalg.norm(H):.6f}")

    # For Zb to have orthonormal columns (ds x da), we need H/||H|| if da=1
    if da == 1:
        Zb_optimal_direction = H / np.linalg.norm(H)
        print(f"\nOptimal Zb direction (unit vector in H direction):\n{Zb_optimal_direction}")

    return {"P": P, "K": K, "G": G, "H": H, "Q": Q, "R": R}


print("="*60)
print("CASE 1: Constructed LQR with P=I, ||K||=1")
print("="*60)

# Construct the special case
G, H, Q, R = construct_unit_norm_lqr(ds=4, da=1, alpha=0.5, seed=42)
result1 = analyze_lqr(G, H, Q, R, "Constructed P=I case")

print("\n" + "="*60)
print("CASE 2: Verify with different alpha")
print("="*60)

G2, H2, Q2, R2 = construct_unit_norm_lqr(ds=4, da=1, alpha=1.0, seed=42)
result2 = analyze_lqr(G2, H2, Q2, R2, "alpha=1.0")

print("\n" + "="*60)
print("CASE 3: Multi-action case (da=2)")
print("="*60)

G3, H3, Q3, R3 = construct_unit_norm_lqr(ds=4, da=2, alpha=0.5, seed=42)
result3 = analyze_lqr(G3, H3, Q3, R3, "da=2 case")


print("\n" + "="*60)
print("SUMMARY: Parameters for run_downstairs.py")
print("="*60)
print("""
To use these compatible parameters, you'll need to modify the LQR env
to accept custom G, H matrices instead of random ones.

For the simplest case (ds=4, da=1, alpha=0.5):
- G = -0.5 * I_4
- H = unit vector (random direction)
- Q = H @ H^T + I_4
- R = I_1

This gives:
- P_opt = I (orthogonal and symmetric)
- K_opt = H^T with ||K|| = 1 (unit norm)
- Optimal Wc = -K^T = -H (unit vector, Stiefel compatible!)
- Optimal Ub = P = I (orthogonal!)
""")

# Save the parameters for use
if result1:
    print("\n--- Exact parameters for Case 1 ---")
    print(f"G = np.array({result1['G'].tolist()})")
    print(f"H = np.array({result1['H'].tolist()})")
    print(f"Q = np.array({result1['Q'].tolist()})")
    print(f"R = np.array({result1['R'].tolist()})")
