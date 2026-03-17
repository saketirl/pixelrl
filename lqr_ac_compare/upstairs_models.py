
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

    W1: Array = None
    W2: Array = None
    W3: Array = None  # frozen

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
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
        # value
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
