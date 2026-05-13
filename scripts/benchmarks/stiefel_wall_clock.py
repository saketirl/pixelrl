#!/usr/bin/env python
"""Wall-clock benchmark for original and ADMM Stiefel updates."""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from optimizers.manifold_stiefel_admm_optax import manifold_stiefel_admm_update
from optimizers.manifold_stiefel_optax import manifold_stiefel_update


def orthonormal_matrix(key, shape):
    rows, cols = shape
    if rows >= cols:
        q, _ = jnp.linalg.qr(jax.random.normal(key, shape))
        return q[:, :cols]

    q, _ = jnp.linalg.qr(jax.random.normal(key, (cols, rows)))
    return q[:, :rows].T


def parse_shape(value):
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("shape must look like 256x256")
    return int(parts[0]), int(parts[1])


def benchmark_shape(shape, args):
    w_key, g_key = jax.random.split(jax.random.PRNGKey(sum(shape)))
    weights = orthonormal_matrix(w_key, shape)
    grads = jax.random.normal(g_key, shape)

    dual_update = jax.jit(
        lambda w, g: manifold_stiefel_update(
            w,
            g,
            eta=args.learning_rate,
            alpha=args.dual_lr,
            steps=args.dual_steps,
            msign_steps=args.msign_steps,
        )
    )
    admm_update = jax.jit(
        lambda w, g: manifold_stiefel_admm_update(
            w,
            g,
            eta=args.learning_rate,
            steps=args.admm_steps,
            rho=args.admm_rho,
            msign_steps=args.msign_steps,
        )
    )

    timings = {}
    for name, update_fn in (("dual", dual_update), ("admm", admm_update)):
        compiled = update_fn.lower(weights, grads).compile()

        current = weights
        for _ in range(args.warmup):
            current = compiled(current, grads)
        current.block_until_ready()

        start = time.perf_counter()
        for _ in range(args.iters):
            current = compiled(current, grads)
        current.block_until_ready()
        elapsed = time.perf_counter() - start
        timings[name] = elapsed / args.iters

    ratio = timings["admm"] / timings["dual"]
    print(
        f"{shape[0]}x{shape[1]} "
        f"dual_ms={timings['dual'] * 1000:.4f} "
        f"admm_ms={timings['admm'] * 1000:.4f} "
        f"admm_over_dual={ratio:.2f}x"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shape",
        action="append",
        type=parse_shape,
        default=None,
        help="Matrix shape to benchmark, e.g. 256x256. May be repeated.",
    )
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--dual-lr", type=float, default=0.01)
    parser.add_argument("--dual-steps", type=int, default=5)
    parser.add_argument("--admm-steps", type=int, default=10)
    parser.add_argument("--admm-rho", type=float, default=4.0)
    parser.add_argument("--msign-steps", type=int, default=5)
    args = parser.parse_args()

    if args.shape is None:
        args.shape = [(256, 256), (512, 256), (256, 64), (256, 8), (8, 256)]

    print(f"backend={jax.default_backend()} devices={jax.devices()}")
    print(
        "settings "
        f"dual_steps={args.dual_steps} "
        f"admm_steps={args.admm_steps} "
        f"msign_steps={args.msign_steps} "
        f"iters={args.iters} warmup={args.warmup}"
    )
    for shape in args.shape:
        benchmark_shape(shape, args)


if __name__ == "__main__":
    main()
