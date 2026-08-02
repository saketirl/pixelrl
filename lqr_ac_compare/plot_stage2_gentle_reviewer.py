"""Create the focused three-panel reviewer figure for the final Stage-2 run.

The default source experiment is the 20-seed ``final_gentle`` ensemble with
independent upstairs/downstairs trajectories and the pure MLP advantage head

    Psi_theta(z, a) = MLP_theta([z, a]).

The figure contains only discounted return, policy loss, and value loss.  The
discounted-LQR reference is intentionally omitted; the zero-action reference
is retained to show whether training improves control performance.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from plot_overnight_two_stage import (
    ZERO_COLOR,
    Run,
    StageSpec,
    discover_runs,
    draw_branch,
    stack_field,
    summarize_stage,
    validate_stage,
)


DEFAULT_STAGE2_ROOT = (
    Path("lqr_ac_compare/linear_rep_head_results/overnight_two_stage")
    / "stage2"
    / "final_gentle"
)
DEFAULT_OUTPUT_DIR = Path(
    "lqr_ac_compare/linear_rep_head_results/overnight_two_stage/"
    "final_report_gentle/stage2_reviewer"
)


def format_mean_se(statistic: Mapping[str, float], digits: int = 5) -> str:
    """Format an across-seed mean and standard error as mu +/- SE."""
    return (
        f"{float(statistic['mean']):.{digits}f} "
        f"± {float(statistic['se']):.{digits}f}"
    )


def reviewer_metrics_table(
    aggregate: Mapping[str, np.ndarray], interval: int = 100
) -> str:
    """Tabulate the plotted branch metrics at regular evaluation checkpoints.

    Math:
        For metric m and branch b, each entry is

            mu_(m,b)(t) +/- SE_(m,b)(t),
            SE_(m,b)(t) = sd_s[m_(s,b)(t)] / sqrt(S).

        Returns use the fixed-bank evaluation J_(s,b)(t). Losses use the same
        trailing-W per-seed averages plotted in the figure. Consequently, a
        loss entry is unavailable before the first complete smoothing window.

    Code map:
        Keep evaluation checkpoints divisible by ``interval`` and append the
        actual final checkpoint when training ends between intervals. Lookups
        join ``eval_iterations`` to return arrays and ``smooth_iterations`` to
        trailing-loss arrays without interpolation.
    """
    if interval <= 0:
        raise ValueError("table interval must be positive")

    eval_iterations = np.asarray(aggregate["eval_iterations"], dtype=int)
    smooth_iterations = np.asarray(aggregate["smooth_iterations"], dtype=int)

    def iteration_lookup(values: np.ndarray) -> Mapping[int, int]:
        return dict(zip(values.tolist(), range(values.size)))

    eval_lookup = iteration_lookup(eval_iterations)
    smooth_lookup = iteration_lookup(smooth_iterations)
    checkpoints = [
        int(value) for value in eval_iterations if int(value) % interval == 0
    ]
    final_checkpoint = int(eval_iterations[-1])
    if not checkpoints or checkpoints[-1] != final_checkpoint:
        checkpoints.append(final_checkpoint)

    def cell(prefix: str, index: int, digits: int) -> str:
        statistic = {
            "mean": float(aggregate[f"{prefix}_mean"][index]),
            "se": float(aggregate[f"{prefix}_se"][index]),
        }
        return format_mean_se(statistic, digits)

    header = (
        "| Iteration | Return (down) | Return (up) | Policy loss (down) | "
        "Policy loss (up) | Value loss (down) | Value loss (up) |\n"
        "|---:|---:|---:|---:|---:|---:|---:|"
    )
    rows = [header]
    for iteration in checkpoints:
        eval_index = eval_lookup[iteration]
        smooth_index = smooth_lookup.get(iteration)
        if smooth_index is None:
            loss_cells = ["—"] * 4
        else:
            loss_cells = [
                cell("down_policy_loss_smooth", smooth_index, 5),
                cell("up_policy_loss_smooth", smooth_index, 5),
                cell("down_value_loss_smooth", smooth_index, 6),
                cell("up_value_loss_smooth", smooth_index, 6),
            ]
        rows.append(
            "| "
            + " | ".join(
                [
                    str(iteration),
                    cell("eval_return_down", eval_index, 5),
                    cell("eval_return_up", eval_index, 5),
                    *loss_cells,
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def reviewer_method_section(summary: Mapping[str, Any]) -> str:
    """Explain the nonlinear-head extension and exact run configuration.

    Math:
        The retained lift is o=M s with M^T M=I_ds. For q in {pi,V,A},
        z_q^b=(X_q^b)^T x^b and X_(q,0)^up=M X_(q,0)^down, so at
        epsilon=0 the matched representations satisfy z_q^up=z_q^down.

    Code map:
        Numerical values come from the homogeneous saved configuration and
        seed list. MLP depth and optimizer constants are implementation
        constants.
    """
    config = summary["common_config"]
    d = int(config["d"])
    ds = int(config["ds"])
    da = int(config["da"])
    k = int(config["k"])
    hidden = int(config["hidden_dim"])
    dt = float(config["dt"])
    duration = float(config["T"])
    rollout_steps = int(round(duration / dt))
    beta = float(config["beta"])
    gamma = float(np.exp(-beta * dt))
    horizon = int(config["n_steps"])
    epsilon = float(summary["epsilon"])
    if np.isclose(epsilon, 0.0, rtol=0.0, atol=1e-12):
        experiment_regime = "This clean run tests exact correspondence."
    else:
        experiment_regime = (
            f"The present run uses `epsilon={epsilon:g}` as a controlled\n"
            "robustness test outside that exact guarantee."
        )
    curvature = float(summary["curvature_scale"])
    action_input_scale = float(summary["advantage_action_input_scale"])
    actor_base_rate = float(config["eta_actor"])
    actor_rate = actor_base_rate * float(config["mlp_actor_lr_scale"])
    actor_warmup = int(config["actor_warmup_iters"])
    actor_stop = int(config["actor_stop_iters"])
    actor_cadence = int(config["mlp_actor_update_every"])
    last_actor_update = (
        actor_warmup
        + ((actor_stop - 1 - actor_warmup) // actor_cadence) * actor_cadence
    )
    iterations = int(config["iters"])
    eval_every = int(config["eval_every"])
    last_regular_eval = ((iterations - 1) // eval_every) * eval_every

    def compact_span(values: Sequence[int]) -> str:
        ordered = sorted(int(value) for value in values)
        contiguous = ordered == list(range(ordered[0], ordered[-1] + 1))
        if len(ordered) > 1 and contiguous:
            return f"{ordered[0]}–{ordered[-1]}"
        return ", ".join(str(value) for value in ordered)

    seed_parameters = summary["seed_parameters"]
    init_seeds = [int(item["init_seed"]) for item in seed_parameters]
    init_seed_span = compact_span(init_seeds)
    noise_seeds = [int(item["noise_seed"]) for item in seed_parameters]
    noise_seed_span = compact_span(noise_seeds)

    alpha = float(config["alpha"])
    problem_seed = int(config["seed"])
    transition_noise = float(config["transition_noise"])
    exploration_std = float(config["exploration_std"])
    policy_bound = float(config["mlp_policy_action_limit"])
    action_clip = float(config["action_clip"])
    critic_rate = float(config["eta_critic"])
    advantage_rate = float(config["eta_advantage"])
    critic_stop = int(config["critic_stop_iters"])
    eval_episodes = int(config["eval_episodes"])
    eval_seed = int(config["eval_seed"])
    seed_count = int(summary["n_seeds"])

    def setting(*parts: str) -> str:
        return " ".join(parts)

    setup_rows = [
        (
            "Training ensemble",
            setting(
                f"{seed_count} prespecified paired seeds;",
                f"initialization {init_seed_span};",
                f"training noise {noise_seed_span}",
            ),
        ),
        (
            "Problem",
            setting(
                f"seed {problem_seed};",
                f"`d/ds/da/k={d}/{ds}/{da}/{k}`;",
                f"`alpha={alpha:g}`",
            ),
        ),
        (
            "Rollout",
            setting(
                f"{iterations} iterations; `dt={dt:g}`;",
                f"`T={duration:g}`; {rollout_steps} fresh steps",
                "per branch and iteration",
            ),
        ),
        (
            "Discount/bootstrap",
            setting(
                f"`beta={beta:g}`; `gamma={gamma:.9f}`;",
                f"`L={horizon}` steps ({horizon * dt:g} time units);",
                "uniform start weighting",
            ),
        ),
        (
            "Noise/actions",
            setting(
                f"upstairs `epsilon={epsilon:g}`;",
                f"transition coefficient `{transition_noise:g}`;",
                f"constant behavior exploration `{exploration_std:g}`;",
                f"policy bound `{policy_bound:g}`;",
                f"executed-action clip `{action_clip:g}`",
            ),
        ),
        (
            "Critic optimization",
            setting(
                "value/advantage Cayley and Adam rates",
                f"`{critic_rate:g}`/`{advantage_rate:g}`;",
                f"updates 0–{critic_stop - 1}",
            ),
        ),
        (
            "Actor optimization",
            setting(
                f"effective Cayley and Adam rate `{actor_rate:g}`;",
                f"updates every {actor_cadence} iterations",
                f"from {actor_warmup} through {last_actor_update}",
            ),
        ),
        (
            "Freeze",
            setting(
                f"parameters and optimizer states frozen from {actor_stop}",
                f"through {iterations - 1}; rollout diagnostics continue",
            ),
        ),
        (
            "Evaluation",
            setting(
                f"{eval_episodes} fixed zero-exploration stochastic rollouts;",
                f"checkpoints 0,{eval_every},...,{last_regular_eval},",
                f"{iterations - 1}; evaluation seed {eval_seed}",
            ),
        ),
    ]
    setup_table = "\n".join(
        ["| Quantity | Setting |", "|---|---|"]
        + [f"| {name} | {value} |" for name, value in setup_rows]
    )

    return f"""## Extension of the original upstairs/downstairs experiment

This experiment retains the LQR problem and core CT-DDPG formulation from
`compare_upstairs_downstairs.py`, including the centered `[r-psi]` martingale
objective, multistep bootstrap, deterministic actor objective, and
Stiefel-constrained optimization. It changes the model class and tests a
stronger, unsynchronized protocol: does the correspondence survive when only
the representation is linear and the policy, value, and advantage heads are
nonlinear?

### Explicit representation followed by nonlinear heads

The observation geometry is retained. The noiseless upstairs observation is
the isometric lift

```text
o = M s,       M in R^(d x ds),       M^T M = I_ds.
```

Downstairs receives `x^down=s`, while upstairs receives

```text
x^up = M s + epsilon xi^o,       xi^o ~ Normal(0,I_d).
```

The original specially factored deep-linear actor and critics are replaced by
three explicit `k={k}` Stiefel representation layers, one for each
`q in (pi,V,A)`, followed by nonlinear heads:

```text
z_q^down = (X_q^down)^T x^down,   X_q^down in St(ds,k),
z_q^up   = (X_q^up)^T x^up,       X_q^up   in St(d,k),
X_(q,0)^up = M X_(q,0)^down.
```

In the clean case, whenever `X_q^up=M X_q^down` and the branch states match,

```text
z_q^up = (X_q^down)^T M^T M s
       = (X_q^down)^T s
       = z_q^down,
```

where the middle equality uses `M^T M=I_ds`. Corresponding head parameters and
Adam moments are also matched initially, but are stored separately thereafter.
Policy and value heads are `{k}-{hidden}-{hidden}-1` tanh MLPs, and the
advantage head is a `{k + da}-{hidden}-{hidden}-1` tanh MLP:

```text
pi(z_pi)   = B tanh(f_pi(z_pi) / B),
V(z_V)     = f_V(z_V),
Psi(z_A,a) = f_A([z_A,c_a a]) - c_R a^T R a,
psi_t      = Psi(z_(A,t),a_t) - Psi(z_(A,t),pi(z_(pi,t))).
```

This Stage-2 run uses `B={float(config["mlp_policy_action_limit"]):g}`,
`c_a={action_input_scale:g}`, and `c_R={curvature:g}`. The advantage critic is
therefore a pure MLP with no explicit `-a^T R a` curvature term.
The environment reward retains its usual action cost. Cayley updates apply
only to the three Stiefel representations, while the nonlinear heads use Adam.

### Retained losses and independent-action comparison

The value loss is the same `[r-psi]` martingale loss as in the original
experiment, and the policy loss is the negative critic-based deterministic
actor objective rather than a supervised policy error. Each
{rollout_steps}-step trajectory supplies all overlapping `L={horizon}`
bootstrap windows.

After matched initialization, the branches are never synchronized. Each branch
chooses its own actions and generates its own states, rewards, training batch,
gradients, Cayley updates, and Adam updates. They share only the process-noise
and behavior-exploration innovations as paired common random numbers, which
reduces comparison variance but does not copy behavior or experience between
branches.

For `epsilon=0`, matched outputs imply matched closed-loop trajectories in
exact arithmetic, so clean equivariance can propagate inductively without
synchronization. {experiment_regime}

### Exact run configuration

{setup_table}

### Evaluation and aggregation

Evaluation retains transition noise and upstairs observation noise but sets
action-exploration noise to zero. A single fixed bank of exogenous noise is
reused across branches, checkpoints, and training seeds. Episode return is

```text
J = dt sum_(t=0)^({rollout_steps - 1}) exp(-beta t dt) r_t.
```

Each return point first averages the {int(config["eval_episodes"])} fixed-bank
episodes within a trained seed. Lines then show the mean over the
{int(summary["n_seeds"])} training seeds; ribbons are the sample standard
deviation across those seed-level means divided by
`sqrt({int(summary["n_seeds"])})`. Policy and value losses are first smoothed
with a trailing-{int(summary["rolling_window"])} window within each seed and
then aggregated identically. Losses are measured on the current fresh training
rollout before that iteration update, while checkpoint returns are measured
after the update. The zero-action policy is evaluated on the same bank; the LQR
reference is intentionally omitted from this figure.
"""


def plot_reviewer_triptych(
    aggregate: Mapping[str, np.ndarray],
    summary: Mapping[str, Any],
    png_path: Path,
    pdf_path: Path,
) -> None:
    """Plot the three Stage-2 learning quantities as mean_s x_s +/- SE_s.

    Math:
        The return panel shows J_b(t) for branch b in {down, up}, where
        J=dt sum_n exp(-beta n dt) r_n on a fixed bank of exogenous-noise
        sequences.  The loss panels show a trailing-W average within each
        seed before computing the across-seed mean and standard error.

    Code map:
        Evaluation checkpoints use ``eval_iterations``; loss traces use
        ``smooth_iterations``.  ``draw_branch`` supplies the established
        blue-solid downstairs and orange-dashed upstairs styling and the
        corresponding mean-plus-or-minus-SE ribbons.
    """
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.15))
    return_axis, policy_axis, value_axis = axes

    eval_iterations = aggregate["eval_iterations"]
    draw_branch(
        return_axis,
        eval_iterations,
        aggregate["eval_return_down_mean"],
        aggregate["eval_return_down_se"],
        "downstairs",
    )
    draw_branch(
        return_axis,
        eval_iterations,
        aggregate["eval_return_up_mean"],
        aggregate["eval_return_up_se"],
        "upstairs",
    )
    return_axis.axhline(
        float(summary["zero_action_reference"]["mean"]),
        color=ZERO_COLOR,
        linestyle=":",
        linewidth=1.25,
        label="Zero action",
    )
    return_axis.set_title("Discounted return")
    return_axis.set_ylabel("Discounted return ↑")

    smooth_iterations = aggregate["smooth_iterations"]
    for axis, down_field, up_field, title in (
        (
            policy_axis,
            "down_policy_loss",
            "up_policy_loss",
            "Policy loss",
        ),
        (
            value_axis,
            "down_value_loss",
            "up_value_loss",
            "Value loss",
        ),
    ):
        draw_branch(
            axis,
            smooth_iterations,
            aggregate[f"{down_field}_smooth_mean"],
            aggregate[f"{down_field}_smooth_se"],
            "downstairs",
        )
        draw_branch(
            axis,
            smooth_iterations,
            aggregate[f"{up_field}_smooth_mean"],
            aggregate[f"{up_field}_smooth_se"],
            "upstairs",
        )
        axis.set_title(title)
        axis.set_ylabel(title)

    for axis in axes:
        axis.set_xlabel("Training iteration")
        axis.grid(True, alpha=0.22)
        axis.margins(x=0)

    handles, labels = return_axis.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def write_reviewer_summary(
    path: Path,
    runs: Sequence[Run],
    aggregate: Mapping[str, np.ndarray],
    summary: Mapping[str, Any],
    figure_name: str,
) -> None:
    """Write a concise interpretation tied to the plotted Stage-2 ensemble.

    Math:
        Delta J_s = J_pair,s(final) - J_zero,s.  Loss correspondence is
        summarized conservatively by max_(s,t) |L_up,s(t)-L_down,s(t)|.
    """
    policy_gap = float(
        np.max(
            np.abs(
                stack_field(runs, "up_policy_loss")
                - stack_field(runs, "down_policy_loss")
            )
        )
    )
    value_gap = float(
        np.max(
            np.abs(
                stack_field(runs, "up_value_loss")
                - stack_field(runs, "down_value_loss")
            )
        )
    )
    final_return = summary["final_paired_eval_return"]
    improvement = summary["eval_return_improvement"]
    return_gap = summary["final_paired_up_minus_down_return"]
    policy_tail = summary["last_window_policy_loss"]
    value_tail = summary["last_window_value_loss"]
    seed_count = int(summary["n_seeds"])
    epsilon = float(summary["epsilon"])
    rolling_window = int(summary["rolling_window"])
    freeze_iteration = int(summary["common_config"]["actor_stop_iters"])
    metrics_table = reviewer_metrics_table(aggregate)
    method_section = reviewer_method_section(summary)

    if np.isclose(epsilon, 0.0, rtol=0.0, atol=1e-12):
        result_text = f"""The final paired discounted return is
{format_mean_se(final_return)}, improving by
{format_mean_se(improvement)} over zero action, with
{summary['seeds_above_zero_action']}/{seed_count} seeds above the baseline.
The trailing policy loss is
{format_mean_se(policy_tail['downstairs'])} downstairs and
{format_mean_se(policy_tail['upstairs'])} upstairs; the corresponding value
loss is {format_mean_se(value_tail['downstairs'], 6)} and
{format_mean_se(value_tail['upstairs'], 6)}. Across every seed and iteration,
the largest upstairs/downstairs gaps are {policy_gap:.3e} for policy loss and
{value_gap:.3e} for value loss. Thus, the pure MLP head learns a policy that
improves over zero action while preserving the independent upstairs/downstairs
correspondence to numerical precision."""
        caption_ending = (
            "The two branches remain visually indistinguishable while "
            "discounted return improves over zero action."
        )
    else:
        result_text = f"""With observation noise, the final branch-specific
discounted returns are
{format_mean_se(summary['final_downstairs_eval_return'])} downstairs and
{format_mean_se(summary['final_upstairs_eval_return'])} upstairs, compared with
the zero-action reference {format_mean_se(summary['zero_action_reference'])}.
Their paired upstairs-minus-downstairs return difference is
{format_mean_se(return_gap)}. Their paired average is
{format_mean_se(final_return)}, improving by {format_mean_se(improvement)} over
the zero-action initialization. The trailing policy loss
is {format_mean_se(policy_tail['downstairs'])} downstairs and
{format_mean_se(policy_tail['upstairs'])} upstairs; the corresponding value
loss is {format_mean_se(value_tail['downstairs'], 6)} and
{format_mean_se(value_tail['upstairs'], 6)}. Across every seed and iteration,
the largest upstairs/downstairs gaps are {policy_gap:.3e} for policy loss and
{value_gap:.3e} for value loss. Exact correspondence is neither predicted nor
observed here: observation noise is present only upstairs, and the branch
separation measures degradation away from the clean orthogonal-lift case."""
        caption_ending = (
            f"At epsilon={epsilon:g}, upstairs-only observation noise breaks "
            "the exact clean-lift correspondence; visible branch separation "
            "measures degradation rather than numerical-precision "
            "matching."
        )

    text = f"""# Stage 2: pure-MLP reviewer summary

![Stage-2 learning curves]({figure_name})

{method_section}

## Results

The curves show the mean and standard error across all {seed_count}
prespecified paired training seeds. Fixed-bank return changes across
checkpoints reflect policy changes rather than newly sampled evaluation noise.
The zero-action reference is retained, while the LQR reference is intentionally
omitted. Policy and value losses use a trailing-{rolling_window} per-seed
average before aggregation. All parameters and optimizer states freeze at
iteration {freeze_iteration}; later losses can still fluctuate because they are
measured on fresh training rollouts. Policy loss is the negative critic-based
actor objective, not a supervised error, so its level need not track return
monotonically.

{result_text}

## Metrics at 100-iteration intervals

Each entry is the across-seed mean ± standard error. Returns are unsmoothed
fixed-bank evaluations; policy and value losses use the same
trailing-{rolling_window} per-seed averages as the figure. The final row is the
actual last checkpoint.

{metrics_table}

Suggested caption: *Independent-action upstairs/downstairs learning with paired
common random numbers, no synchronization after matched initialization, and a
pure MLP advantage head without an explicit
`-a^T R a` critic term. Lines show the
across-seed mean and shaded regions show ±1 standard error over {seed_count}
seeds. Evaluation uses a fixed bank of stochastic rollouts; losses use a
trailing-{rolling_window} average. Parameters freeze at iteration
{freeze_iteration}; subsequent loss variation comes from fresh rollouts.
{caption_ending}*
"""
    path.write_text(text)


def build_parser() -> argparse.ArgumentParser:
    """Map paths and ensemble-aggregation choices to the focused plot."""
    parser = argparse.ArgumentParser(
        description=("Plot the three reviewer-facing Stage-2 gentle metrics.")
    )
    parser.add_argument(
        "--stage2-root",
        type=Path,
        default=DEFAULT_STAGE2_ROOT,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--basename", default="stage2_gentle_reviewer")
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--expected-seeds", type=int, default=20)
    parser.add_argument("--rolling-window", type=int, default=25)
    parser.add_argument("--tail-eval-checkpoints", type=int, default=10)
    return parser


def main() -> None:
    """Validate Stage 2, aggregate all seeds, and write figure plus summary."""
    args = build_parser().parse_args()
    spec = StageSpec(
        key="stage2",
        label="Stage 2 (pure MLP)",
        curvature_scale=0.0,
        root=args.stage2_root,
        epsilon=args.epsilon,
    )
    runs = discover_runs(spec, args.expected_seeds)
    iterations = validate_stage(spec, runs)
    aggregate, summary = summarize_stage(
        spec,
        runs,
        iterations,
        args.rolling_window,
        args.tail_eval_checkpoints,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / f"{args.basename}.png"
    pdf_path = args.output_dir / f"{args.basename}.pdf"
    markdown_path = args.output_dir / f"{args.basename}.md"
    plot_reviewer_triptych(aggregate, summary, png_path, pdf_path)
    write_reviewer_summary(
        markdown_path,
        runs,
        aggregate,
        summary,
        png_path.name,
    )
    print(png_path)
    print(pdf_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
