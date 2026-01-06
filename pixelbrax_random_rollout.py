"""Implement a random rollout of PixelBrax environment with state, action, rewards in a CSV file and rendered frames as a GIF."""

import csv
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
from functools import partial
from pixelbrax.env_utils import make_pixel_brax


def random_rollout(
    env_name: str = "halfcheetah",
    backend: str = "spring",
    num_steps: int = 500,
    seed: int = 42,
    output_csv: str = "pixelbrax_rollout_data.csv",
    output_gif: str = "pixelbrax_rollout.gif",
    hw: int = 84,
    fps: int = 30,
):
    """Run a random rollout of PixelBrax environment and save data.

    Args:
        env_name: Name of the Brax environment.
        backend: Brax backend to use ("spring", "generalized", "positional").
        num_steps: Number of environment steps to run.
        seed: Random seed for reproducibility.
        output_csv: Path to output CSV file with state data.
        output_gif: Path to output GIF file with rendered frames.
        hw: Height and width of rendered frames.
        fps: Frames per second for the output GIF.
    """
    # Create environment with a single env for visualization
    n_envs = 1
    print(f"Creating PixelBrax environment: {env_name} with backend: {backend}")

    env, initial_state, reset_keys = make_pixel_brax(
        backend=backend,
        env_name=env_name,
        n_envs=n_envs,
        seed=seed,
        hw=hw,
        distractor=None,
        return_float32=True,
    )

    # Initialize random key
    key = jax.random.PRNGKey(seed)

    # Get initial state
    key, reset_key = jax.random.split(key)
    reset_keys = jax.random.split(reset_key, n_envs)
    state = env.reset(reset_keys)

    print(f"Observation size (brax state): {env.observation_size}")
    print(f"Action size: {env.action_size}")
    print(f"Pixel observation shape: {state.pixels.shape}")

    # Storage for rollout data
    observations = []  # underlying brax state observations
    actions_list = []
    rewards = []
    dones = []
    frames = []  # pixel observations for GIF

    print(f"Running rollout for {num_steps} steps...")

    for step in range(num_steps):
        # Sample random action in [-1, 1]
        key, action_key = jax.random.split(key)
        action = jax.random.uniform(
            action_key,
            shape=(n_envs, env.action_size),
            minval=-1.0,
            maxval=1.0
        )

        # Store current state data
        obs = jax.device_get(state.obs[0])  # Get first env's observation
        observations.append(obs)
        actions_list.append(jax.device_get(action[0]))

        # Extract a single frame for GIF (last 3 channels = most recent frame)
        # pixels shape is (n_envs, hw, hw, 9) - 3 stacked frames of 3 channels each
        pixel_frame = jax.device_get(state.pixels[0, :, :, -3:])  # Get last RGB frame
        # Convert from float [0,1] to uint8 [0,255]
        pixel_frame = (pixel_frame * 255).astype(np.uint8)
        frames.append(pixel_frame)

        # Step environment
        next_state = env.step(state, action)

        # Store reward and done
        rewards.append(float(jax.device_get(next_state.reward[0])))
        dones.append(bool(jax.device_get(next_state.done[0])))

        state = next_state

        if (step + 1) % 100 == 0:
            print(f"  Step {step + 1}/{num_steps}")

    # Convert lists to arrays
    observations = np.array(observations)
    actions_arr = np.array(actions_list)
    rewards = np.array(rewards)
    dones = np.array(dones)

    # Write CSV with underlying state data
    obs_dim = observations.shape[1]
    action_dim = actions_arr.shape[1]

    with open(output_csv, "w", newline="") as f:
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
            data += actions_arr[step].tolist()
            data += [float(rewards[step]), bool(dones[step])]
            writer.writerow(data)

    print(f"State data saved to {output_csv}")

    # Create GIF from frames
    print(f"Creating GIF with {len(frames)} frames...")
    pil_frames = [Image.fromarray(frame) for frame in frames]

    # Save as GIF
    duration = int(1000 / fps)  # Duration per frame in milliseconds
    pil_frames[0].save(
        output_gif,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0,
    )
    print(f"GIF saved to {output_gif}")

    # Also save as mp4 using available libraries
    output_mp4 = output_gif.replace('.gif', '.mp4')
    mp4_saved = False

    # Option 1: Try OpenCV
    if not mp4_saved:
        try:
            import cv2
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_mp4, fourcc, fps, (w, h))
            for frame in frames:
                # OpenCV uses BGR, so convert from RGB
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            out.release()
            print(f"MP4 saved to {output_mp4} (using OpenCV)")
            mp4_saved = True
        except ImportError:
            pass

    # Option 2: Try ffmpeg via subprocess
    if not mp4_saved:
        try:
            import subprocess
            import shutil
            if shutil.which('ffmpeg'):
                h, w = frames[0].shape[:2]
                cmd = [
                    'ffmpeg', '-y',
                    '-f', 'rawvideo',
                    '-vcodec', 'rawvideo',
                    '-s', f'{w}x{h}',
                    '-pix_fmt', 'rgb24',
                    '-r', str(fps),
                    '-i', '-',
                    '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p',
                    output_mp4
                ]
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
                for frame in frames:
                    proc.stdin.write(frame.tobytes())
                proc.stdin.close()
                proc.wait()
                print(f"MP4 saved to {output_mp4} (using ffmpeg)")
                mp4_saved = True
        except Exception:
            pass

    # Option 3: Try imageio as fallback
    if not mp4_saved:
        try:
            import imageio
            imageio.mimsave(output_mp4, frames, fps=fps)
            print(f"MP4 saved to {output_mp4} (using imageio)")
            mp4_saved = True
        except Exception:
            pass

    if not mp4_saved:
        print("(No MP4 encoder available - install opencv-python or ffmpeg)")

    # Print statistics
    print(f"\nRollout Statistics:")
    print(f"  Total steps: {num_steps}")
    print(f"  Observation dim: {obs_dim}")
    print(f"  Action dim: {action_dim}")
    print(f"  Total reward: {float(rewards.sum()):.2f}")
    print(f"  Mean reward: {float(rewards.mean()):.4f}")
    print(f"  Episodes completed: {int(dones.sum())}")

    return observations, actions_arr, rewards, dones, frames


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Random rollout of PixelBrax environment")
    parser.add_argument("--env-name", type=str, default="halfcheetah",
                        help="Environment name (halfcheetah, walker2d, ant, etc.)")
    parser.add_argument("--backend", type=str, default="spring",
                        help="Brax backend (spring, generalized, positional)")
    parser.add_argument("--num-steps", type=int, default=500,
                        help="Number of steps to run")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--hw", type=int, default=84,
                        help="Height/width of rendered frames")
    parser.add_argument("--fps", type=int, default=30,
                        help="FPS for output GIF/video")
    parser.add_argument("--output-csv", type=str, default=None,
                        help="Output CSV file path")
    parser.add_argument("--output-gif", type=str, default=None,
                        help="Output GIF file path")

    args = parser.parse_args()

    # Set default output paths based on env name
    output_csv = args.output_csv or f"pixelbrax_{args.env_name}_rollout.csv"
    output_gif = args.output_gif or f"pixelbrax_{args.env_name}_rollout.gif"

    random_rollout(
        env_name=args.env_name,
        backend=args.backend,
        num_steps=args.num_steps,
        seed=args.seed,
        output_csv=output_csv,
        output_gif=output_gif,
        hw=args.hw,
        fps=args.fps,
    )
