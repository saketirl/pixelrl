"""Implement a random rollout of HalfCheetah environment with state, action, rewards in a CSV file."""

import csv
import jax
import jax.numpy as jnp
from functools import partial
from clean_rl_reference.pure_jax_wrapper import BraxGymnaxWrapper


def random_rollout(
    env_name: str = "halfcheetah",
    backend: str = "spring",
    num_steps: int = 1000,
    seed: int = 42,
    output_file: str = "rollout_data.csv",
):
    """Run a random rollout and save data to CSV.
    
    Args:
        env_name: Name of the Brax environment.
        backend: Brax backend to use ("spring", "mjx", etc.).
        num_steps: Number of environment steps to run.
        seed: Random seed for reproducibility.
        output_file: Path to output CSV file.
    """
    # Create environment
    env = BraxGymnaxWrapper(env_name=env_name, backend=backend)
    
    # Initialize random key
    key = jax.random.PRNGKey(seed)
    
    # Reset environment
    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key)
    
    # Pre-generate all random keys for the rollout
    step_keys = jax.random.split(key, num_steps * 2).reshape(num_steps, 2, -1)
    
    @jax.jit
    def rollout_step(carry, keys):
        """Single step of the rollout, designed for jax.lax.scan."""
        obs, state = carry
        action_key, step_key = keys[0], keys[1]
        
        # Sample random action in [-1, 1]
        action = jax.random.uniform(
            action_key, 
            shape=(env.action_size,), 
            minval=-1.0, 
            maxval=1.0
        )
        
        # Step environment
        next_obs, next_state, reward, done, info = env.step(step_key, state, action)
        
        # Return new carry and outputs to collect
        return (next_obs, next_state), (obs, action, reward, done)
    
    # Run the entire rollout as a single compiled operation
    print("Running rollout (first run includes JIT compilation)...")
    _, (observations, actions, rewards, dones) = jax.lax.scan(
        rollout_step,
        (obs, state),
        step_keys,
    )
    
    # Block until computation is done
    observations = jax.device_get(observations)
    actions = jax.device_get(actions)
    rewards = jax.device_get(rewards)
    dones = jax.device_get(dones)
    
    # Write to CSV
    obs_dim = env.observation_size[0]
    action_dim = env.action_size
    
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        
        # Write header
        header = ["step"]
        header += [f"obs_{i}" for i in range(obs_dim)]
        header += [f"action_{i}" for i in range(action_dim)]
        header += ["reward", "done"]
        writer.writerow(header)
        
        # Write data rows
        for step in range(num_steps):
            data = [step]
            data += observations[step].tolist()
            data += actions[step].tolist()
            data += [float(rewards[step]), bool(dones[step])]
            writer.writerow(data)
    
    print(f"Rollout data saved to {output_file}")
    print(f"Total steps: {num_steps}")
    print(f"Observation dim: {obs_dim}")
    print(f"Action dim: {action_dim}")
    
    # Compute some statistics
    print(f"Total reward: {float(rewards.sum()):.2f}")
    print(f"Mean reward: {float(rewards.mean()):.4f}")
    
    return observations, actions, rewards, dones


if __name__ == "__main__":
    rollout_data = random_rollout(
        env_name="halfcheetah",
        backend="spring",
        num_steps=1000,
        seed=42,
        output_file="halfcheetah_rollout.csv",
    )