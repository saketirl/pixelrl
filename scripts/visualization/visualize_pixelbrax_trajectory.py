"""Render a short PixelBrax trajectory for visual inspection."""

import argparse
from pathlib import Path
import sys

import cv2
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pixelbrax.env_utils import make_pixel_brax


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-name",
        choices=["ant_goal", "humanoid_goal", "ant_u_maze", "humanoid_u_maze"],
        required=True,
    )
    parser.add_argument("--backend", default="generalized")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy", choices=["random", "zero"], default="random")
    parser.add_argument("--hw", type=int, default=128)
    parser.add_argument("--out", type=Path, default=Path("trajectory.mp4"))
    parser.add_argument("--first-frame-out", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=30.0)
    return parser.parse_args()


def write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    if not frames:
        raise ValueError("no frames to write")
    path.parent.mkdir(parents=True, exist_ok=True)

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer for {path}")

    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def main() -> None:
    args = parse_args()
    env, state, _ = make_pixel_brax(
        backend=args.backend,
        env_name=args.env_name,
        n_envs=1,
        seed=args.seed,
        hw=args.hw,
        distractor=None,
        return_float32=False,
        frame_stack=1,
    )

    key = jax.random.PRNGKey(args.seed + 1)
    frames = [np.asarray(jax.device_get(state.pixels[0, :, :, -3:]))]
    if args.first_frame_out is not None:
        args.first_frame_out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frames[0]).save(args.first_frame_out)
    returns = 0.0
    min_dist = float("inf")
    success_any = False

    for _ in range(args.steps):
        if args.policy == "zero":
            action = jnp.zeros((1, env.action_size))
        else:
            key, action_key = jax.random.split(key)
            action = jax.random.uniform(
                action_key, (1, env.action_size), minval=-1.0, maxval=1.0
            )

        state = env.step(state, action)
        frames.append(np.asarray(jax.device_get(state.pixels[0, :, :, -3:])))

        reward = float(jax.device_get(state.reward[0]))
        returns += reward
        metrics = jax.device_get(state.metrics)
        if "dist" in metrics:
            dist = float(metrics["dist"][0])
            min_dist = min(min_dist, dist)
        if "success" in metrics:
            success_any = success_any or bool(metrics["success"][0])

    write_video(args.out, frames, args.fps)
    final_metrics = jax.device_get(state.metrics)
    final_dist = float(final_metrics["dist"][0]) if "dist" in final_metrics else float("nan")
    print(f"wrote: {args.out}")
    if args.first_frame_out is not None:
        print(f"wrote_first_frame: {args.first_frame_out}")
    print(f"return: {returns:.3f}")
    print(f"final_dist: {final_dist:.3f}")
    print(f"min_dist: {min_dist:.3f}")
    print(f"success_any: {success_any}")


if __name__ == "__main__":
    main()
