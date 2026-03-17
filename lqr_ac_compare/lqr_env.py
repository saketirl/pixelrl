
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from scipy.linalg import solve_continuous_are, solve_continuous_lyapunov

Array = np.ndarray


def principal_sqrt_psd(A: Array, eps: float = 1e-12) -> Array:
    """Principal symmetric square root of PSD matrix."""
    A = 0.5 * (A + A.T)
    w, V = np.linalg.eigh(A)
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w + eps)) @ V.T


def stiefel_init(n: int, p: int, rng: np.random.Generator) -> Array:
    """X ∈ R^{n×p} with orthonormal columns, requires n >= p."""
    if n < p:
        raise ValueError(f"Need n>=p for Stiefel columns, got n={n}, p={p}")
    A = rng.standard_normal((n, p))
    Q, _ = np.linalg.qr(A, mode="reduced")
    return Q


@dataclass
class LQRParams:
    d: int = 512
    ds: int = 4
    da: int = 1
    dt: float = 0.01
    T: float = 10.0

    # Drift matrices
    G: Optional[Array] = None   # (ds, ds)
    H: Optional[Array] = None   # (ds, da)

    # Cost matrices (reward = -cost)
    Q: Optional[Array] = None   # (ds, ds)
    R: Optional[Array] = None   # (da, da)

    # Base transition noise Σ (requested identity by default)
    Sigma_base: Optional[Array] = None  # (ds, ds)

    # Noise scales for simplified downstairs diffusion
    sigma_explore: float = 0.1
    eps_obs: float = 0.05

    seed: int = 0


class LQRUpDownEnv:
    """
    Latent state dynamics (Euler–Maruyama discretization of simplified downstairs SDE):
        s_{t+dt} = s_t + (G s_t + H a_t) dt + Sigma_eff(K) sqrt(dt) η_t
    with Σ_base = I by default, and
        Sigma_eff(K) Sigma_eff(K)^T
        = sigma_explore^2 H H^T + Σ_base Σ_base^T + eps_obs^2 (H K)(H K)^T

    Observations:
        downstairs: s_t
        upstairs:   o_t = M s_t + eps_obs ξ_t
    """
    def __init__(self, params: LQRParams, M: Optional[Array] = None):
        self.p = params
        self.rng = np.random.default_rng(self.p.seed)

        self.ds, self.da, self.d = self.p.ds, self.p.da, self.p.d
        self.dt = float(self.p.dt)
        self.n_steps = int(round(self.p.T / self.p.dt))

        # Drift matrices
        if self.p.G is None:
            A = self.rng.standard_normal((self.ds, self.ds))
            self.G = A - 0.6 * np.eye(self.ds)
        else:
            self.G = np.array(self.p.G, dtype=float)

        if self.p.H is None:
            self.H = self.rng.standard_normal((self.ds, self.da))
        else:
            self.H = np.array(self.p.H, dtype=float)

        # Cost
        self.Q = np.eye(self.ds) if self.p.Q is None else np.array(self.p.Q, dtype=float)
        self.R = np.eye(self.da) if self.p.R is None else np.array(self.p.R, dtype=float)

        # Σ_base
        if self.p.Sigma_base is None:
            self.Sigma_base = np.eye(self.ds)
        else:
            self.Sigma_base = np.array(self.p.Sigma_base, dtype=float)

        # Observation map M
        if M is None:
            self.M = stiefel_init(self.d, self.ds, self.rng)  # orthonormal columns
        else:
            self.M = np.array(M, dtype=float)

        self.t = 0
        self.s = np.zeros((self.ds,), dtype=float)

        # Compute optimal LQR solution
        self._compute_optimal_lqr()

    def _compute_optimal_lqr(self):
        """Compute optimal LQR gain and value function via Riccati equation."""
        try:
            # Solve continuous-time algebraic Riccati equation:
            # G^T P + P G - P H R^{-1} H^T P + Q = 0
            self.P_opt = solve_continuous_are(self.G, self.H, self.Q, self.R)
            # Optimal gain: K* = R^{-1} H^T P
            self.K_opt = np.linalg.solve(self.R, self.H.T @ self.P_opt)
            # Closed-loop dynamics: A_cl = G - H K*
            self.A_cl = self.G - self.H @ self.K_opt
            self._lqr_solved = True
        except Exception as e:
            print(f"Warning: Could not solve Riccati equation: {e}")
            self.P_opt = None
            self.K_opt = None
            self.A_cl = None
            self._lqr_solved = False

    def optimal_gain(self) -> Optional[Array]:
        """Return optimal gain matrix K* such that a* = -K* s."""
        return self.K_opt if self._lqr_solved else None

    def optimal_average_reward(self, s0: Optional[Array] = None,
                                exploration_std: float = 0.0) -> dict:
        """
        Compute expected average reward under optimal policy.

        Returns dict with:
        - 'avg_reward_no_noise': avg reward if no transition/exploration noise
        - 'avg_reward_with_noise': avg reward accounting for noise (steady-state)
        - 'initial_cost': cost at initial state s0
        """
        if not self._lqr_solved:
            return {"error": "Riccati equation not solved"}

        if s0 is None:
            s0 = 0.5 * np.ones(self.ds)
        s0 = np.array(s0, dtype=float).reshape(self.ds,)

        # Initial value V(s0) = s0^T P s0 (total cost-to-go from s0, no noise)
        V_s0 = float(s0.T @ self.P_opt @ s0)

        # Initial instantaneous cost
        a0_opt = -self.K_opt @ s0
        initial_cost = float(s0.T @ self.Q @ s0 + a0_opt.T @ self.R @ a0_opt)

        # Effective noise covariance (Σ_eff Σ_eff^T)
        Sigma_cov = self.Sigma_base @ self.Sigma_base.T
        if self.p.sigma_explore != 0.0:
            Sigma_cov = Sigma_cov + (self.p.sigma_explore ** 2) * (self.H @ self.H.T)
        if exploration_std > 0.0:
            # Exploration noise adds to action: a = -Ks + ε, ε ~ N(0, exploration_std²I)
            Sigma_cov = Sigma_cov + (exploration_std ** 2) * (self.H @ self.H.T)

        # Steady-state covariance under optimal policy (solve Lyapunov equation):
        # A_cl Σ_s + Σ_s A_cl^T + Σ_cov = 0
        try:
            Sigma_s = solve_continuous_lyapunov(self.A_cl, -Sigma_cov)
            # Steady-state average cost = tr((Q + K^T R K) Σ_s)
            cost_matrix = self.Q + self.K_opt.T @ self.R @ self.K_opt
            steady_state_cost = float(np.trace(cost_matrix @ Sigma_s))
        except Exception:
            Sigma_s = None
            steady_state_cost = None

        return {
            "V_s0": V_s0,  # Total discounted cost from s0 (beta->0 limit)
            "initial_cost": initial_cost,  # -reward at t=0
            "initial_reward": -initial_cost,
            "steady_state_avg_cost": steady_state_cost,
            "steady_state_avg_reward": -steady_state_cost if steady_state_cost else None,
            "K_opt": self.K_opt,
        }

    def reset(self, s0: Optional[Array] = None) -> Dict[str, Array]:
        self.t = 0
        if s0 is None:
            self.s = 0.5 * np.ones(self.ds)
        else:
            s0 = np.array(s0, dtype=float).reshape(self.ds,)
            self.s = s0.copy()
        return self._obs()

    def _obs(self) -> Dict[str, Array]:
        s_obs = self.s.copy()
        if self.p.eps_obs > 0.0:
            o = self.M @ self.s + self.p.eps_obs * self.rng.standard_normal(self.d)
        else:
            o = self.M @ self.s
        return {"downstairs": s_obs, "upstairs": o}

    def _sigma_eff(self, K: Optional[Array]) -> Array:
        cov = self.Sigma_base @ self.Sigma_base.T
        if self.p.sigma_explore != 0.0:
            cov = cov + (self.p.sigma_explore ** 2) * (self.H @ self.H.T)
        if K is not None and self.p.eps_obs != 0.0:
            K = np.array(K, dtype=float).reshape(self.da, self.ds)
            HK = self.H @ K
            cov = cov + (self.p.eps_obs ** 2) * (HK @ HK.T)
        return principal_sqrt_psd(cov)

    def step(self, action: Array, *, K_for_diffusion: Optional[Array] = None) -> Tuple[Dict[str, Array], float, bool, Dict]:
        a = np.array(action, dtype=float).reshape(self.da,)
        Sigma_eff = self._sigma_eff(K_for_diffusion)

        drift = (self.G @ self.s) + (self.H @ a)
        noise = Sigma_eff @ self.rng.standard_normal(self.ds) * np.sqrt(self.dt)
        s_next = self.s + drift * self.dt + noise

        # cost uses current state/action (standard)
        cost = float(self.s.T @ self.Q @ self.s + a.T @ self.R @ a)
        reward = -cost

        self.s = s_next
        self.t += 1
        done = self.t >= self.n_steps
        info = {"t": self.t, "state": self.s.copy(), "action": a.copy()}
        return self._obs(), reward, done, info

    def rollout(self, policy_fn, *, exploration_std: float = 0.1, use_upstairs: bool = True,
                use_K_for_diffusion: Optional[Array] = None) -> Dict[str, Array]:
        """
        Roll out for full horizon. policy_fn takes (obs_dict) and returns deterministic action (da,).
        """
        obs = self.reset()
        o_dim = self.d
        traj = {
            "s": np.zeros((self.n_steps + 1, self.ds)),
            "o": np.zeros((self.n_steps + 1, self.d)),
            "a": np.zeros((self.n_steps, self.da)),
            "r": np.zeros((self.n_steps,)),
        }
        traj["s"][0] = obs["downstairs"]
        traj["o"][0] = obs["upstairs"]

        for k in range(self.n_steps):
            a_det = np.array(policy_fn(obs), dtype=float).reshape(self.da,)
            a_exec = a_det + exploration_std * self.rng.standard_normal(self.da)

            obs, r, done, _ = self.step(a_exec, K_for_diffusion=use_K_for_diffusion)
            traj["a"][k] = a_exec
            traj["r"][k] = r
            traj["s"][k + 1] = obs["downstairs"]
            traj["o"][k + 1] = obs["upstairs"]
            if done:
                break
        return traj
