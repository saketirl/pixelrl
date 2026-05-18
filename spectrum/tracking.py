"""Utilities for WandB spectrum and activation diagnostics."""

from __future__ import annotations

from typing import Callable, Iterable

import flax
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from spectrum.density import tridiag_to_eigv
from spectrum.lanczos import lanczos_alg


def flatten_batch_tree(batch, batch_size: int):
    """Flatten leading rollout/minibatch axes and keep a deterministic prefix."""

    def flatten(x):
        x = jnp.asarray(x)
        if x.ndim >= 2:
            x = x.reshape((-1,) + x.shape[2:])
        return x[:batch_size]

    return jax.tree_util.tree_map(flatten, batch)


def _summary(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {}
    return {
        f"{prefix}/count": float(values.size),
        f"{prefix}/min": float(np.min(values)),
        f"{prefix}/max": float(np.max(values)),
        f"{prefix}/mean": float(np.mean(values)),
        f"{prefix}/std": float(np.std(values)),
        f"{prefix}/p01": float(np.percentile(values, 1)),
        f"{prefix}/p05": float(np.percentile(values, 5)),
        f"{prefix}/p50": float(np.percentile(values, 50)),
        f"{prefix}/p95": float(np.percentile(values, 95)),
        f"{prefix}/p99": float(np.percentile(values, 99)),
    }


def _histogram(values: np.ndarray, bins: int):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    if np.min(values) == np.max(values):
        center = float(values[0])
        delta = max(abs(center) * 1e-3, 1e-6)
        bin_edges = np.linspace(center - delta, center + delta, bins + 1)
    else:
        _, bin_edges = np.histogram(values, bins=bins)
    counts, bin_edges = np.histogram(values, bins=bin_edges)
    return np_histogram_to_wandb(counts, bin_edges)


def np_histogram_to_wandb(counts: np.ndarray, bin_edges: np.ndarray):
    import wandb

    return wandb.Histogram(np_histogram=(counts, bin_edges))


def add_distribution(logs: dict, prefix: str, values, bins: int):
    values = np.asarray(jax.device_get(values)).reshape(-1)
    logs.update(_summary(values, prefix))
    hist = _histogram(values, bins)
    if hist is not None:
        logs[prefix] = hist


def add_histogram(logs: dict, prefix: str, values, bins: int):
    values = np.asarray(jax.device_get(values)).reshape(-1)
    hist = _histogram(values, bins)
    if hist is not None:
        logs[prefix] = hist


def spectrum_rank_metrics(values: np.ndarray, prefix: str, eps: float = 1e-12) -> dict[str, float]:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    vals = np.abs(vals[np.isfinite(vals)])
    if vals.size == 0:
        return {}
    total = np.sum(vals)
    sq_total = np.sum(vals ** 2)
    max_val = np.max(vals)
    probs = vals / max(total, eps)
    entropy = -np.sum(np.where(probs > 0, probs * np.log(probs), 0.0))
    desc_sq = np.sort(vals ** 2)[::-1]
    cumsum_sq = np.cumsum(desc_sq) / max(sq_total, eps)
    approx_rank = int(np.argmax(cumsum_sq >= 0.99) + 1)
    return {
        f"{prefix}/rank_nonzero": float(np.count_nonzero(vals > eps)),
        f"{prefix}/effective_rank": float(np.exp(entropy)),
        f"{prefix}/stable_rank": float(sq_total / max(max_val ** 2, eps)),
        f"{prefix}/top_fraction": float(max_val / max(total, eps)),
        f"{prefix}/approx_rank_99_sq": float(approx_rank),
    }


def add_matrix_spectrum(logs: dict, prefix: str, matrix, bins: int):
    matrix = jnp.asarray(matrix)
    singular_values = jnp.linalg.svd(matrix, compute_uv=False)
    singular_np = np.asarray(jax.device_get(singular_values))
    gram_np = singular_np ** 2
    add_distribution(logs, f"{prefix}/singular_values", singular_np, bins)
    logs.update(spectrum_rank_metrics(singular_np, f"{prefix}/singular_values"))
    add_distribution(logs, f"{prefix}/gram_eigenvalues", gram_np, bins)
    logs.update(spectrum_rank_metrics(gram_np, f"{prefix}/gram_eigenvalues"))


def add_matrix_singular_histogram(logs: dict, prefix: str, matrix, bins: int):
    singular_values = jnp.linalg.svd(jnp.asarray(matrix), compute_uv=False)
    add_histogram(logs, f"{prefix}/singular_values", singular_values, bins)


def iter_matrix_params(params, include_prefixes: Iterable[str] = ("network", "actor", "critic")):
    flat = flax.traverse_util.flatten_dict(params)
    for path, value in flat.items():
        if path[0] not in include_prefixes:
            continue
        if path[-1] not in {"kernel", "weight"}:
            continue
        if getattr(value, "ndim", 0) < 2:
            continue
        name = "/".join(str(part) for part in path if part != "params")
        yield name, value


def hessian_eigenvalues(
    loss_fn: Callable,
    params,
    batch,
    key,
    order: int,
    draws: int,
):
    flat_params, unravel = ravel_pytree(params)
    dim = flat_params.shape[0]

    def flat_loss(flat_x):
        return loss_fn(unravel(flat_x), batch)

    def hvp(v):
        return jax.jvp(jax.grad(flat_loss), (flat_params,), (v,))[1]

    eigvals = []
    for draw in range(draws):
        draw_key = jax.random.fold_in(key, draw)
        tridiag, _ = lanczos_alg(hvp, dim, order, draw_key)
        eigvals.append(jnp.linalg.eigvalsh(tridiag))
    return jnp.concatenate(eigvals, axis=0)


def hessian_eigenvalues_and_tridiags(
    loss_fn: Callable,
    params,
    batch,
    key,
    order: int,
    draws: int,
):
    flat_params, unravel = ravel_pytree(params)
    dim = flat_params.shape[0]

    def flat_loss(flat_x):
        return loss_fn(unravel(flat_x), batch)

    def hvp(v):
        return jax.jvp(jax.grad(flat_loss), (flat_params,), (v,))[1]

    eigvals = []
    tridiags = []
    for draw in range(draws):
        draw_key = jax.random.fold_in(key, draw)
        tridiag, _ = lanczos_alg(hvp, dim, order, draw_key)
        tridiags.append(tridiag)
        eigvals.append(jnp.linalg.eigvalsh(tridiag))
    return jnp.concatenate(eigvals, axis=0), jnp.stack(tridiags, axis=0)


def add_hessian_spectrum(logs: dict, prefix: str, eigvals, bins: int):
    eig_np = np.asarray(jax.device_get(eigvals))
    add_distribution(logs, f"{prefix}/eigenvalues", eig_np, bins)
    logs.update(spectrum_rank_metrics(eig_np, f"{prefix}/eigenvalues"))
    logs[f"{prefix}/negative_fraction"] = float(np.mean(eig_np < 0.0))
    logs[f"{prefix}/positive_fraction"] = float(np.mean(eig_np > 0.0))


def add_hessian_histogram(logs: dict, prefix: str, eigvals, bins: int):
    add_histogram(logs, f"{prefix}/eigenvalues", eigvals, bins)


def add_hessian_density_curve(
    logs: dict,
    prefix: str,
    tridiags,
    sigma_squared: float = 1e-5,
    grid_len: int = 10000,
    image_label: str | None = None,
    image_xlim: tuple[float, float] = (-4000.0, 10000.0),
    image_ylim: tuple[float, float] = (1e-10, 1e2),
):
    eig_vals, weights = tridiag_to_eigv(tridiags)
    lambda_max = jnp.nanmean(jnp.max(eig_vals, axis=1), axis=0) + 1e-2
    lambda_min = jnp.nanmean(jnp.min(eig_vals, axis=1), axis=0) - 1e-2
    grids = jnp.linspace(lambda_min, lambda_max, num=grid_len)
    sigma = sigma_squared * jnp.maximum(1.0, lambda_max - lambda_min)
    deltas = grids[None, :, None] - eig_vals[:, None, :]
    kernels = jnp.exp(-(deltas ** 2) / (2.0 * sigma))
    kernels = kernels / jnp.sqrt(2.0 * np.pi * sigma)
    density_each_draw = jnp.sum(kernels * weights[:, None, :], axis=-1)
    density = jnp.nanmean(density_each_draw, axis=0)
    norm_fact = jnp.sum(density) * (grids[1] - grids[0])
    density = density / norm_fact
    grids_np = np.asarray(jax.device_get(grids), dtype=np.float64)
    density_np = np.asarray(jax.device_get(density), dtype=np.float64)
    finite = np.isfinite(grids_np) & np.isfinite(density_np)
    if not np.any(finite):
        return

    import wandb

    table = wandb.Table(
        data=[
            [float(x), float(y)]
            for x, y in zip(grids_np[finite], density_np[finite])
        ],
        columns=["eigenvalue", "density"],
    )
    logs[f"{prefix}/density_curve"] = wandb.plot.line(
        table,
        "eigenvalue",
        "density",
        title=f"{prefix}/density_curve",
    )
    logs[f"{prefix}/density_image"] = density_curve_image(
        grids_np[finite],
        density_np[finite],
        label=image_label or prefix,
        xlim=image_xlim,
        ylim=image_ylim,
    )
    logs[f"{prefix}/density_sigma_squared"] = float(sigma_squared)
    logs[f"{prefix}/density_grid_len"] = float(grid_len)


def density_curve_image(
    grids: np.ndarray,
    density: np.ndarray,
    label: str,
    xlim: tuple[float, float] = (-4000.0, 10000.0),
    ylim: tuple[float, float] = (1e-10, 1e2),
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    import wandb

    finite = np.isfinite(grids) & np.isfinite(density) & (density > 0.0)
    grids = grids[finite]
    density = density[finite]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.semilogy(grids, density, label=label, color="blue", linewidth=2)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_ylabel("Density")
    ax.set_xlabel("Eigenvalue")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.canvas.draw()
    image = Image.fromarray(np.asarray(fig.canvas.buffer_rgba()))
    plt.close(fig)
    return wandb.Image(image)


def add_weight_spectra(logs: dict, params, prefix: str, bins: int):
    for name, value in iter_matrix_params(params):
        add_matrix_spectrum(logs, f"{prefix}/weights/{name}", value, bins)


def add_activation_distributions(logs: dict, activations: dict, prefix: str, bins: int):
    for name, value in activations.items():
        if value is None:
            continue
        add_distribution(logs, f"{prefix}/activations/{name}", value, bins)
