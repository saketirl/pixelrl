
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


@dataclass
class DownstairsActorShallowT2:
    """
    Shallow downstairs actor, stored as Stiefel columns:
        Wc_col ∈ R^{ds×da}, with action a = (Wc_col^T) s
    """
    ds: int
    da: int
    seed: int = 0
    Wc_col: Array = None

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.Wc_col = stiefel_init(self.ds, self.da, rng)

    @property
    def Wc(self) -> Array:
        return self.Wc_col.T  # (da, ds)

    def act(self, s: Array) -> Array:
        s = np.asarray(s, dtype=float).reshape(self.ds,)
        return (self.Wc @ s).reshape(self.da,)

    def trainable_params(self) -> Dict[str, Array]:
        return {"Wc_col": self.Wc_col}


@dataclass
class DownstairsCriticShallowT2:
    """
    Shallow downstairs critic (low-dim analogue).

    Value:
        phi(s) = s^T Ub s + Uc_row s + c
      Ub ∈ R^{ds×ds} orthogonal
      Uc_col ∈ R^{ds×1} Stiefel columns (Uc_row = Uc_col^T)
      c scalar

    Advantage-rate:
        Psi(s,a) = s^T Zb a + Zc_row a
        psi(s,a) = Psi(s,a) - Psi(s, pi(s))
      Zb ∈ R^{ds×da} Stiefel columns
      Zc_row ∈ R^{1×da} unconstrained
    """
    ds: int
    da: int
    seed: int = 0

    Ub: Array = None
    Uc_col: Array = None
    c: float = 0.0

    Zb: Array = None
    Zc_row: Array = None

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.Ub = orthogonal_init(self.ds, rng)
        self.Uc_col = stiefel_init(self.ds, 1, rng)
        self.c = 0.0

        self.Zb = stiefel_init(self.ds, self.da, rng)
        self.Zc_row = 0.01 * rng.standard_normal((1, self.da))

    @property
    def Uc_row(self) -> Array:
        return self.Uc_col.T

    def phi(self, s: Array) -> float:
        s = np.asarray(s, dtype=float).reshape(self.ds,)
        quad = float(s.T @ (self.Ub @ s))
        lin = float((self.Uc_row @ s.reshape(self.ds, 1)).reshape(()))
        return quad + lin + float(self.c)

    def Psi(self, s: Array, a: Array) -> float:
        s = np.asarray(s, dtype=float).reshape(self.ds,)
        a = np.asarray(a, dtype=float).reshape(self.da,)
        mixed = float(s.T @ (self.Zb @ a.reshape(self.da, 1)))
        lin_a = float((self.Zc_row @ a.reshape(self.da, 1)).reshape(()))
        return mixed + lin_a

    def psi(self, s: Array, a: Array, actor: DownstairsActorShallowT2) -> float:
        a_pi = actor.act(s)
        return self.Psi(s, a) - self.Psi(s, a_pi)

    def trainable_params(self) -> Dict[str, Array]:
        return {
            "Ub": self.Ub,
            "Uc_col": self.Uc_col,
            "c": np.array(self.c, dtype=float),
            "Zb": self.Zb,
            "Zc_row": self.Zc_row,
        }
