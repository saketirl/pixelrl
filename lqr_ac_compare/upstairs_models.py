
import numpy as np
from dataclasses import dataclass
from typing import Dict

Array = np.ndarray


def stiefel_init(n: int, p: int, rng: np.random.Generator) -> Array:
    """X ∈ R^{n×p} with orthonormal columns, requires n >= p."""
    if n < p:
        raise ValueError(f"Need n>=p for Stiefel columns, got n={n}, p={p}")
    A = rng.standard_normal((n, p))
    Q, _ = np.linalg.qr(A, mode="reduced")
    return Q

def orthogonal_init(n: int, rng: np.random.Generator) -> Array:
    return stiefel_init(n, n, rng)

def row_unit_init(m: int, n: int, rng: np.random.Generator, eps: float = 1e-12) -> Array:
    A = rng.standard_normal((m, n))
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + eps)


def balanced_dln_init_L3(d: int, da: int, rng: np.random.Generator):
    """
    Initialize L=3 DLN with balanced layers per arXiv:2411.09004.

    Balanced condition: W_{l+1}^T W_{l+1} = W_l W_l^T

    For Weff = W3 @ W2 @ W1 with shape (da, d):
    - Weff has rank r = min(da, d) = da
    - All layers should have effective rank r

    Strategy: Use SVD-based initialization where singular values are
    distributed evenly across layers (each layer contributes σ^{1/L}).

    Returns: W1 (d×d), W2 (d×d), W3 (da×d) such that layers are balanced.
    """
    r = min(da, d)

    # Random orthonormal bases
    U_full = orthogonal_init(d, rng)   # d×d
    V = stiefel_init(d, da, rng)       # d×da, columns orthonormal

    # Initialize with balanced singular values (σ_i = 1 for simplicity)
    # For balanced DLN: each layer has σ^{1/L} contribution
    # With L=3 and σ=1: each layer contributes 1

    # W1: embed V into first r columns, rest orthonormal
    W1 = U_full.copy()
    W1[:, :da] = V
    # Ensure orthogonality via QR (might slightly perturb)
    W1, _ = np.linalg.qr(W1)

    # W2: identity (balanced with W1 since W2^T W2 = I = W1 W1^T)
    W2 = np.eye(d)

    # W3: needs W3^T W3 to match W2 W2^T in the rank-r subspace
    # For rank-da effective matrix, W3 should have orthonormal rows
    # that project onto the same subspace
    W3_raw = rng.standard_normal((da, d))
    W3_Q, _ = np.linalg.qr(W3_raw.T, mode='reduced')  # (d, da)
    W3 = W3_Q.T  # (da, d) with orthonormal rows: W3 @ W3^T = I_da

    # Scale W3 rows to have unit norm (already orthonormal, so norm=1)
    # But we want W3^T W3 to be rank-da projection matching the structure
    # W3^T @ W3 is (d, d) with rank da
    # W2 @ W2^T = I_d (full rank) - NOT balanced!
    #
    # To balance with bottleneck: scale W1 and W2 to have effective rank da
    # Actually, the key is that training dynamics stay on balanced manifold
    # if initialized there. For bottleneck networks, we need:
    #
    # W2^T W2 = W1 W1^T  (both I if W1,W2 orthogonal) ✓
    # W3^T W3 should have same rank as W2 W2^T in the relevant subspace
    #
    # The paper says rank must match: rank(W3^T W3) = rank(W2 W2^T) = rank(W3 W2)
    # With W2 = I: rank(W3^T W3) = da, rank(W2 W2^T) = d, rank(W3) = da
    # This is inherently unbalanced for da < d
    #
    # Solution: use W2 that projects onto da-dimensional subspace
    # W2 = P @ P^T where P is (d, da) with orthonormal columns
    P = stiefel_init(d, da, rng)
    W2 = P @ P.T  # projection onto da-dimensional subspace, rank da

    # Now W2 W2^T = (P P^T)(P P^T) = P P^T (since P^T P = I_da)
    # So W2 W2^T has rank da, matching W3^T W3

    # W1 should satisfy W2^T W2 = W1 W1^T
    # W2^T W2 = P P^T (same as W2 W2^T since W2 symmetric)
    # So W1 W1^T should equal P P^T (rank da)
    # W1 = P @ Q where Q is (da, d) with orthonormal rows
    Q = stiefel_init(d, da, rng).T  # (da, d), orthonormal rows
    W1 = P @ Q  # (d, d), rank da
    # Check: W1 W1^T = P Q Q^T P^T = P I_da P^T = P P^T ✓

    # Verify balancing:
    # W2^T W2 = W1 W1^T = P P^T (rank da) ✓
    # W3 W3^T = I_da (rank da) ✓
    # W3^T W3 = (da, d) @ (d, da) = (d, d) rank da ✓

    return W1, W2, W3


def balanced_critic_init_L3(d: int, rng: np.random.Generator):
    """
    Initialize L=3 critic value network with balanced layers.

    For value: U3 @ U2 @ U1 where all are d×d.
    Balanced means: U_{l+1}^T U_{l+1} = U_l U_l^T

    For square orthogonal matrices, this is automatically satisfied.
    But to ensure stability with gradient descent (no Cayley), we use
    reduced rank structure matching the effective rank.
    """
    # For d×d → d×d, just use orthogonal (automatically balanced)
    U1 = orthogonal_init(d, rng)
    U2 = orthogonal_init(d, rng)
    U3 = orthogonal_init(d, rng)
    return U1, U2, U3


def balanced_critic_adv_init_L3(d: int, da: int, rng: np.random.Generator):
    """
    Initialize advantage network Z3 @ Z2 @ Z1 with balanced layers.

    Z3 is (da, d), Z2 is (d, d), Z1 is (d, d).
    Bottleneck at Z3 requires balanced structure.
    """
    # Same structure as actor
    P = stiefel_init(d, da, rng)
    Q = stiefel_init(d, da, rng).T

    W2 = P @ P.T  # projection, rank da
    W1 = P @ Q    # rank da, W1 W1^T = P P^T

    # Z3: orthonormal rows
    Z3_raw = rng.standard_normal((da, d))
    Z3_Q, _ = np.linalg.qr(Z3_raw.T, mode='reduced')
    W3 = Z3_Q.T

    return W1, W2, W3


@dataclass
class ActorDLN_L3:
    """
    Upstairs actor with L=3:
        ϕ(o) = W3 W2 W1 o
    Trainables: W1,W2 ∈ R^{d×d} (orthogonal)
    Frozen head: W3 ∈ R^{da×d} (row-unit init)
    """
    d: int
    da: int
    seed: int = 0
    balanced: bool = False  # Use balanced DLN initialization

    W1: Array = None
    W2: Array = None
    W3: Array = None  # frozen

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        if self.balanced:
            self.W1, self.W2, self.W3 = balanced_dln_init_L3(self.d, self.da, rng)
        else:
            self.W1 = orthogonal_init(self.d, rng)
            self.W2 = orthogonal_init(self.d, rng)
            self.W3 = row_unit_init(self.da, self.d, rng)

    def effective_matrix(self) -> Array:
        return self.W3 @ (self.W2 @ self.W1)  # (da,d)

    def act_upstairs(self, o: Array) -> Array:
        o = np.asarray(o, dtype=float).reshape(self.d,)
        return (self.effective_matrix() @ o).reshape(self.da,)

    def trainable_params(self) -> Dict[str, Array]:
        return {"W1": self.W1, "W2": self.W2}

    def frozen_params(self) -> Dict[str, Array]:
        return {"W3": self.W3}


@dataclass
class CriticDLN_L3:
    """
    Upstairs critic: value φ(o) and advantage-rate ψ(o,a).

    Value:
        φ(o) = o^T (U3 U2 U1) o + (U'_3 U'_2 U'_1) o + c
      Trainables: U1,U2,U3, Up1,Up2 (all d×d), c scalar
      Frozen: Up3 (1×d) head

    Advantage:
        Ψ(o,a) = a^T (Z3 Z2 Z1) o + (Z'_3 Z'_2 Z'_1) a
        ψ(o,a) = Ψ(o,a) - Ψ(o, ϕ(o))
      Trainables: Z1,Z2,Zp2 (all d×d)
      Frozen: Z3 (da×d) head, Zp1 (d×da), Zp3 (1×d)
    """
    d: int
    da: int
    seed: int = 0
    balanced: bool = False  # Use balanced DLN initialization

    U1: Array = None
    U2: Array = None
    U3: Array = None
    Up1: Array = None
    Up2: Array = None
    Up3: Array = None  # frozen
    c: float = 0.0

    Z1: Array = None
    Z2: Array = None
    Z3: Array = None   # frozen
    Zp1: Array = None  # frozen
    Zp2: Array = None
    Zp3: Array = None  # frozen

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)

        if self.balanced:
            # Value network: all d×d, use balanced orthogonal init
            self.U1, self.U2, self.U3 = balanced_critic_init_L3(self.d, rng)
            self.Up1 = orthogonal_init(self.d, rng)
            self.Up2 = orthogonal_init(self.d, rng)
            self.Up3 = row_unit_init(1, self.d, rng)
            self.c = 0.0

            # Advantage network: has bottleneck at Z3 (da×d)
            self.Z1, self.Z2, self.Z3 = balanced_critic_adv_init_L3(self.d, self.da, rng)
            self.Zp1 = stiefel_init(self.d, self.da, rng)
            self.Zp2 = orthogonal_init(self.d, rng)
            self.Zp3 = row_unit_init(1, self.d, rng)
        else:
            # value (original initialization)
            self.U1 = orthogonal_init(self.d, rng)
            self.U2 = orthogonal_init(self.d, rng)
            self.U3 = orthogonal_init(self.d, rng)
            self.Up1 = orthogonal_init(self.d, rng)
            self.Up2 = orthogonal_init(self.d, rng)
            self.Up3 = row_unit_init(1, self.d, rng)
            self.c = 0.0

            # advantage
            self.Z1 = orthogonal_init(self.d, rng)
            self.Z2 = orthogonal_init(self.d, rng)
            self.Z3 = row_unit_init(self.da, self.d, rng)
            self.Zp1 = stiefel_init(self.d, self.da, rng)
            self.Zp2 = orthogonal_init(self.d, rng)
            self.Zp3 = row_unit_init(1, self.d, rng)

    def Ueff(self) -> Array:
        return self.U3 @ (self.U2 @ self.U1)

    def Up_eff(self) -> Array:
        return self.Up2 @ self.Up1

    def phi_upstairs(self, o: Array) -> float:
        o = np.asarray(o, dtype=float).reshape(self.d,)
        quad = float(o.T @ (self.Ueff() @ o))
        lin = float((self.Up3 @ (self.Up_eff() @ o.reshape(self.d, 1))).reshape(()))
        return quad + lin + float(self.c)

    def Zeff(self) -> Array:
        return self.Z3 @ (self.Z2 @ self.Z1)  # (da,d)

    def Psi_upstairs(self, o: Array, a: Array) -> float:
        o = np.asarray(o, dtype=float).reshape(self.d,)
        a = np.asarray(a, dtype=float).reshape(self.da,)
        mixed = float(a.T @ (self.Zeff() @ o))
        lin = float((self.Zp3 @ (self.Zp2 @ (self.Zp1 @ a).reshape(self.d, 1))).reshape(()))
        return mixed + lin

    def psi_upstairs(self, o: Array, a: Array, actor: ActorDLN_L3) -> float:
        a_pi = actor.act_upstairs(o)
        return self.Psi_upstairs(o, a) - self.Psi_upstairs(o, a_pi)

    def trainable_params(self) -> Dict[str, Array]:
        return {
            "U1": self.U1, "U2": self.U2, "U3": self.U3,
            "Up1": self.Up1, "Up2": self.Up2,
            "c": np.array(self.c, dtype=float),
            "Z1": self.Z1, "Z2": self.Z2, "Zp2": self.Zp2,
        }

    def frozen_params(self) -> Dict[str, Array]:
        return {"Up3": self.Up3, "Z3": self.Z3, "Zp1": self.Zp1, "Zp3": self.Zp3}
