
import jax
import jax.numpy as jnp
from jax import random
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, NamedTuple
from functools import partial

# Keep scipy for one-time Riccati solve (CPU is fine for this)
import numpy as np
from scipy.linalg import solve_continuous_are, solve_continuous_lyapunov

Array = jnp.ndarray


@jax.jit
def principal_sqrt_psd(A: Array, eps: float = 1e-12) -> Array:
    """Principal symmetric square root of PSD matrix."""
    A = 0.5 * (A + A.T)
    w, V = jnp.linalg.eigh(A)
    w = jnp.clip(w, 0.0, None)
    return (V * jnp.sqrt(w + eps)) @ V.T


def stiefel_init(key: random.PRNGKey, n: int, p: int) -> Array:
    """X in R^{n x p} with orthonormal columns, requires n >= p."""
    A = random.normal(key, (n, p))
    Q, _ = jnp.linalg.qr(A, mode="reduced")
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

    # Base transition noise
    Sigma_base: Optional[Array] = None  # (ds, ds)

    # Noise scales
    sigma_explore: float = 0.1
    eps_obs: float = 0.05

    seed: int = 0


class EnvState(NamedTuple):
    """Immutable environment state for JAX."""
    s: Array          # current state (ds,)
    t: int            # current timestep
    key: random.PRNGKey


class LQRUpDownEnv:
    """
    JAX-compatible LQR environment.

    Latent state dynamics (Euler-Maruyama discretization):
        s_{t+dt} = s_t + (G s_t + H a_t) dt + Sigma_eff sqrt(dt) eta_t

    Observations:
        downstairs: s_t
        upstairs:   o_t = M s_t + eps_obs * xi_t
    """
    def __init__(self, params: LQRParams, M: Optional[Array] = None):
        self.p = params
        key = random.PRNGKey(self.p.seed)

        self.ds, self.da, self.d = self.p.ds, self.p.da, self.p.d
        self.dt = float(self.p.dt)
        self.n_steps = int(round(self.p.T / self.p.dt))

        # Initialize matrices
        key, k1, k2, k3 = random.split(key, 4)

        # Drift matrices
        if self.p.G is None:
            A = random.normal(k1, (self.ds, self.ds))
            self.G = A - 0.6 * jnp.eye(self.ds)
        else:
            self.G = jnp.array(self.p.G)

        if self.p.H is None:
            self.H = random.normal(k2, (self.ds, self.da))
        else:
            self.H = jnp.array(self.p.H)

        # Cost matrices
        self.Q = jnp.eye(self.ds) if self.p.Q is None else jnp.array(self.p.Q)
        self.R = jnp.eye(self.da) if self.p.R is None else jnp.array(self.p.R)

        # Sigma_base
        if self.p.Sigma_base is None:
            self.Sigma_base = jnp.eye(self.ds)
        else:
            self.Sigma_base = jnp.array(self.p.Sigma_base)

        # Observation map M
        if M is None:
            self.M = stiefel_init(k3, self.d, self.ds)
        else:
            self.M = jnp.array(M)

        # Precompute effective noise covariance (without K-dependent term)
        self.Sigma_cov_base = self.Sigma_base @ self.Sigma_base.T
        if self.p.sigma_explore != 0.0:
            self.Sigma_cov_base = self.Sigma_cov_base + (self.p.sigma_explore ** 2) * (self.H @ self.H.T)
        self.Sigma_eff = principal_sqrt_psd(self.Sigma_cov_base)

        self._key = key

        # Compute optimal LQR solution (using numpy/scipy on CPU)
        self._compute_optimal_lqr()

    def _compute_optimal_lqr(self):
        """Compute optimal LQR gain via Riccati equation (CPU)."""
        try:
            G_np = np.array(self.G)
            H_np = np.array(self.H)
            Q_np = np.array(self.Q)
            R_np = np.array(self.R)

            self.P_opt = jnp.array(solve_continuous_are(G_np, H_np, Q_np, R_np))
            self.K_opt = jnp.linalg.solve(self.R, self.H.T @ self.P_opt)
            self.A_cl = self.G - self.H @ self.K_opt
            self._lqr_solved = True
        except Exception as e:
            print(f"Warning: Could not solve Riccati equation: {e}")
            self.P_opt = None
            self.K_opt = None
            self.A_cl = None
            self._lqr_solved = False

    def optimal_gain(self) -> Optional[Array]:
        return self.K_opt if self._lqr_solved else None

    def optimal_average_reward(self, s0: Optional[Array] = None,
                                exploration_std: float = 0.0) -> dict:
        """Compute expected average reward under optimal policy."""
        if not self._lqr_solved:
            return {"error": "Riccati equation not solved"}

        if s0 is None:
            s0 = 0.5 * jnp.ones(self.ds)
        s0 = jnp.array(s0).reshape(self.ds,)

        V_s0 = float(s0.T @ self.P_opt @ s0)

        a0_opt = -self.K_opt @ s0
        initial_cost = float(s0.T @ self.Q @ s0 + a0_opt.T @ self.R @ a0_opt)

        Sigma_cov = np.array(self.Sigma_cov_base)
        if exploration_std > 0.0:
            Sigma_cov = Sigma_cov + (exploration_std ** 2) * np.array(self.H @ self.H.T)

        try:
            A_cl_np = np.array(self.A_cl)
            Sigma_s = solve_continuous_lyapunov(A_cl_np, -Sigma_cov)
            cost_matrix = np.array(self.Q + self.K_opt.T @ self.R @ self.K_opt)
            steady_state_cost = float(np.trace(cost_matrix @ Sigma_s))
        except Exception:
            steady_state_cost = None

        return {
            "V_s0": V_s0,
            "initial_cost": initial_cost,
            "initial_reward": -initial_cost,
            "steady_state_avg_cost": steady_state_cost,
            "steady_state_avg_reward": -steady_state_cost if steady_state_cost else None,
            "K_opt": self.K_opt,
        }

    def reset(self, key: random.PRNGKey, s0: Optional[Array] = None) -> Tuple[EnvState, Dict[str, Array]]:
        """Reset environment and return initial state."""
        if s0 is None:
            s = 0.5 * jnp.ones(self.ds)
        else:
            s = jnp.array(s0).reshape(self.ds,)

        state = EnvState(s=s, t=0, key=key)
        obs = self._obs(state)
        return state, obs

    def _obs(self, state: EnvState) -> Dict[str, Array]:
        """Generate observations from state."""
        key, subkey = random.split(state.key)
        o = self.M @ state.s
        if self.p.eps_obs > 0.0:
            o = o + self.p.eps_obs * random.normal(subkey, (self.d,))
        return {"downstairs": state.s, "upstairs": o}

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: EnvState, action: Array) -> Tuple[EnvState, Dict[str, Array], float, bool]:
        """Take a step in the environment."""
        a = action.reshape(self.da,)

        key, k1, k2 = random.split(state.key, 3)

        drift = (self.G @ state.s) + (self.H @ a)
        noise = self.Sigma_eff @ random.normal(k1, (self.ds,)) * jnp.sqrt(self.dt)
        s_next = state.s + drift * self.dt + noise

        cost = state.s.T @ self.Q @ state.s + a.T @ self.R @ a
        reward = -cost

        new_state = EnvState(s=s_next, t=state.t + 1, key=k2)
        done = new_state.t >= self.n_steps

        obs = self._obs(new_state)
        return new_state, obs, reward, done

    def rollout(self, key: random.PRNGKey, policy_fn, *,
                exploration_std: float = 0.1) -> Dict[str, Array]:
        """
        Roll out for full horizon using jax.lax.scan for efficiency.
        policy_fn takes (state s, observation o) and returns action.
        """
        key, reset_key = random.split(key)
        state, obs = self.reset(reset_key)

        def scan_step(carry, step_key):
            state, obs = carry
            k1, k2 = random.split(step_key)

            # Get action from policy
            a_det = policy_fn(obs)
            a_exec = a_det + exploration_std * random.normal(k1, (self.da,))

            # Store current state/obs before step
            s_t = state.s
            o_t = obs["upstairs"]

            # Take step
            new_state, new_obs, reward, done = self.step(state, a_exec)

            return (new_state, new_obs), (s_t, o_t, a_exec, reward)

        # Generate keys for each step
        step_keys = random.split(key, self.n_steps)

        # Run scan
        (final_state, final_obs), (s_traj, o_traj, a_traj, r_traj) = jax.lax.scan(
            scan_step, (state, obs), step_keys
        )

        # Append final state/obs
        s_traj = jnp.concatenate([s_traj, final_state.s[None, :]], axis=0)
        o_traj = jnp.concatenate([o_traj, final_obs["upstairs"][None, :]], axis=0)

        return {
            "s": s_traj,  # (N+1, ds)
            "o": o_traj,  # (N+1, d)
            "a": a_traj,  # (N, da)
            "r": r_traj,  # (N,)
        }
