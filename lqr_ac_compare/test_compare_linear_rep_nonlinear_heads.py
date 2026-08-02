"""Focused mathematical regression tests for the matched CT-DDPG experiment."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import compare_linear_rep_nonlinear_heads as experiment


jax.config.update("jax_enable_x64", True)


def _assert_tree_allclose(
    actual: object,
    expected: object,
    *,
    rtol: float = 1e-10,
    atol: float = 1e-11,
) -> None:
    """Assert leafwise equality of two parameter pytrees, theta ~= theta_hat."""
    assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(
        expected
    )
    for actual_leaf, expected_leaf in zip(
        jax.tree_util.tree_leaves(actual),
        jax.tree_util.tree_leaves(expected),
    ):
        np.testing.assert_allclose(
            np.asarray(actual_leaf),
            np.asarray(expected_leaf),
            rtol=rtol,
            atol=atol,
        )


def _assert_tree_equal(actual: object, expected: object) -> None:
    """Assert exact leafwise equality, theta_plus = theta."""
    assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(
        expected
    )
    for actual_leaf, expected_leaf in zip(
        jax.tree_util.tree_leaves(actual),
        jax.tree_util.tree_leaves(expected),
    ):
        np.testing.assert_array_equal(
            np.asarray(actual_leaf),
            np.asarray(expected_leaf),
        )


def _make_update_case() -> tuple[experiment.Params, dict[str, object]]:
    """Build one batch with r_t = -(s_t.T Q s_t + a_t.T R a_t)."""
    rng = np.random.default_rng(53)
    problem = experiment.construct_problem(d=7, ds=3, da=2, alpha=0.5, seed=59)
    params, _ = experiment.init_matched_models(
        problem,
        k=2,
        head_type="mlp",
        hidden_dim=6,
        seed=61,
        policy_output_scale=0.08,
        value_output_scale=0.11,
        advantage_output_scale=0.14,
    )
    ops = experiment.make_head_ops(
        "mlp",
        problem.R,
        action_curvature_scale=1.0,
        advantage_gradient="full_window",
    )
    n_steps = 6
    observations = rng.normal(scale=0.35, size=(n_steps + 1, problem.ds))
    actions = rng.normal(scale=0.25, size=(n_steps, problem.da))
    rewards = -(
        np.einsum(
            "bi,ij,bj->b",
            observations[:-1],
            problem.Q,
            observations[:-1],
        )
        + np.einsum("bi,ij,bj->b", actions, problem.R, actions)
    )
    update_kwargs = {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "ops": ops,
        "beta": 0.1,
        "dt": 0.02,
        "horizon": 2,
        "eta_actor": 2e-4,
        "eta_critic": 7e-4,
        "eta_advantage": 9e-4,
        "encoder_lr_scale": 0.3,
        "start_time_weighting": "discounted",
    }
    return params, update_kwargs


def test_mlp_curvature_toggle_adds_exact_quadratic_action_cost() -> None:
    """Math: Psi_(c_R=1)(z,a) - Psi_(c_R=0)(z,a) = -a.T R a."""
    k, da = 3, 2
    action_cost = np.array([[1.4, 0.2], [0.2, 0.7]])
    advantage_head = experiment.init_heads(
        "mlp",
        k=k,
        da=da,
        hidden_dim=6,
        seed=17,
        advantage_output_scale=0.13,
    )["a"]
    latents = jnp.asarray([[0.2, -0.1, 0.4], [-0.3, 0.7, 0.5], [0.9, -0.2, -0.6]])
    actions = jnp.asarray([[0.3, -0.8], [-0.5, 0.1], [0.6, 0.4]])

    pure_ops = experiment.make_head_ops("mlp", action_cost, action_curvature_scale=0.0)
    curved_ops = experiment.make_head_ops(
        "mlp", action_cost, action_curvature_scale=1.0
    )
    pure_psi = pure_ops["raw_advantage_batch"](advantage_head, latents, actions)
    curved_psi = curved_ops["raw_advantage_batch"](advantage_head, latents, actions)
    expected_difference = -np.einsum(
        "bi,ij,bj->b", np.asarray(actions), action_cost, np.asarray(actions)
    )

    np.testing.assert_allclose(
        np.asarray(curved_psi - pure_psi),
        expected_difference,
        rtol=2e-12,
        atol=2e-12,
    )


def test_horizon_one_full_gradient_is_dt_times_initial_semigradient() -> None:
    """Math: for L=1, grad L_full = dt g_initial, while L is identical."""
    rng = np.random.default_rng(23)
    ambient_dim, k, da, n_steps = 4, 3, 2, 5
    dt = 0.037
    action_cost = np.array([[1.2, -0.1], [-0.1, 0.8]])
    x_a = jnp.asarray(experiment.stiefel_init(ambient_dim, k, rng))
    advantage_head = experiment.init_heads(
        "mlp",
        k=k,
        da=da,
        hidden_dim=7,
        seed=29,
        advantage_output_scale=0.2,
    )["a"]
    observations = jnp.asarray(rng.normal(scale=0.4, size=(n_steps + 1, ambient_dim)))
    actions = jnp.asarray(rng.normal(scale=0.3, size=(n_steps, da)))
    policy_actions = jnp.asarray(rng.normal(scale=0.2, size=(n_steps, da)))
    base_residual = jnp.asarray(rng.normal(scale=0.15, size=(n_steps,)))
    start_weights = jnp.asarray(np.linspace(1.0, 0.6, n_steps))
    horizon_one_weights = jnp.ones((1,), dtype=jnp.float64)

    full_ops = experiment.make_head_ops(
        "mlp",
        action_cost,
        action_curvature_scale=1.0,
        advantage_gradient="full_window",
    )
    initial_ops = experiment.make_head_ops(
        "mlp",
        action_cost,
        action_curvature_scale=1.0,
        advantage_gradient="initial_semigradient",
    )
    args = (
        x_a,
        advantage_head,
        observations,
        actions,
        policy_actions,
        base_residual,
        start_weights,
        horizon_one_weights,
        jnp.asarray(dt),
    )
    full_loss, full_gradient = full_ops["advantage_grad"](*args)
    initial_loss, initial_gradient = initial_ops["advantage_grad"](*args)

    np.testing.assert_allclose(
        np.asarray(full_loss), np.asarray(initial_loss), rtol=2e-12, atol=2e-12
    )
    expected_full_gradient = jax.tree_util.tree_map(
        lambda gradient: dt * gradient, initial_gradient
    )
    _assert_tree_allclose(
        full_gradient,
        expected_full_gradient,
        rtol=3e-10,
        atol=3e-11,
    )


@pytest.mark.parametrize("action_curvature_scale", [0.0, 1.0])
def test_clean_shared_update_is_equivariant_with_cayley_and_adam(
    action_curvature_scale: float,
) -> None:
    """Math: one shared-data update preserves X_up^b=M X_down^b and h_up^b=h_down^b."""
    rng = np.random.default_rng(41)
    problem = experiment.construct_problem(d=7, ds=3, da=2, alpha=0.5, seed=43)
    down, up = experiment.init_matched_models(
        problem,
        k=2,
        head_type="mlp",
        hidden_dim=6,
        seed=47,
        policy_output_scale=0.08,
        value_output_scale=0.11,
        advantage_output_scale=0.14,
    )
    ops = experiment.make_head_ops(
        "mlp",
        problem.R,
        action_curvature_scale=action_curvature_scale,
        advantage_gradient="full_window",
    )

    n_steps = 6
    downstairs_observations = rng.normal(scale=0.35, size=(n_steps + 1, problem.ds))
    upstairs_observations = downstairs_observations @ problem.M.T
    actions = rng.normal(scale=0.25, size=(n_steps, problem.da))
    rewards = -(
        np.einsum(
            "bi,ij,bj->b",
            downstairs_observations[:-1],
            problem.Q,
            downstairs_observations[:-1],
        )
        + np.einsum("bi,ij,bj->b", actions, problem.R, actions)
    )
    update_kwargs = dict(
        actions=actions,
        rewards=rewards,
        ops=ops,
        beta=0.1,
        dt=0.02,
        horizon=2,
        eta_actor=2e-4,
        eta_critic=7e-4,
        eta_advantage=9e-4,
        update_actor=True,
        encoder_lr_scale=0.3,
        start_time_weighting="discounted",
    )
    down_next, down_stats = experiment.ctddpg_update(
        down,
        observations=downstairs_observations,
        **update_kwargs,
    )
    up_next, up_stats = experiment.ctddpg_update(
        up,
        observations=upstairs_observations,
        **update_kwargs,
    )

    # Cayley equivariance and Stiefel feasibility for all three representations.
    identity = np.eye(2)
    for encoder_key in experiment.ENCODER_KEYS:
        expected_upstairs = problem.M @ np.asarray(down_next[encoder_key])
        np.testing.assert_allclose(
            np.asarray(up_next[encoder_key]),
            expected_upstairs,
            rtol=2e-9,
            atol=2e-10,
        )
        np.testing.assert_allclose(
            np.asarray(down_next[encoder_key]).T @ np.asarray(down_next[encoder_key]),
            identity,
            rtol=2e-10,
            atol=2e-10,
        )
        np.testing.assert_allclose(
            np.asarray(up_next[encoder_key]).T @ np.asarray(up_next[encoder_key]),
            identity,
            rtol=2e-10,
            atol=2e-10,
        )

    # Adam receives equal gradients/states, so heads and optimizer states match.
    for branch in ("pi", "v", "a"):
        _assert_tree_allclose(
            up_next[f"h_{branch}"], down_next[f"h_{branch}"], rtol=2e-8, atol=2e-10
        )
        _assert_tree_allclose(
            up_next[f"adam_m_{branch}"],
            down_next[f"adam_m_{branch}"],
            rtol=2e-8,
            atol=2e-10,
        )
        _assert_tree_allclose(
            up_next[f"adam_v_{branch}"],
            down_next[f"adam_v_{branch}"],
            rtol=2e-8,
            atol=2e-10,
        )
        assert int(np.asarray(down_next[f"adam_step_{branch}"])) == 1
        assert int(np.asarray(up_next[f"adam_step_{branch}"])) == 1

    for metric in ("value_loss", "advantage_loss", "actor_objective"):
        np.testing.assert_allclose(
            up_stats[metric], down_stats[metric], rtol=2e-9, atol=2e-11
        )


def test_actor_and_critic_freeze_preserves_all_state_but_measures_losses() -> None:
    """Math: I_pi = I_C = 0 implies theta_plus = theta with finite L_V,L_A,J_pi."""
    params, update_kwargs = _make_update_case()

    frozen, stats = experiment.ctddpg_update(
        params,
        update_actor=False,
        update_critic=False,
        **update_kwargs,
    )

    _assert_tree_equal(frozen, params)
    assert stats["actor_updated"] == 0.0
    assert stats["critic_updated"] == 0.0
    for metric in (
        "value_loss",
        "advantage_loss",
        "actor_objective",
        "martingale_residual_rms",
        "grad_pi_norm",
        "grad_v_norm",
        "grad_a_norm",
    ):
        assert np.isfinite(stats[metric])


def test_unset_critic_stop_preserves_legacy_critic_update() -> None:
    """Math: t_C = infinity gives I_C(t) = 1, so critic Adam steps advance."""
    params, update_kwargs = _make_update_case()

    updated, stats = experiment.ctddpg_update(
        params,
        update_actor=False,
        **update_kwargs,
    )

    assert stats["critic_updated"] == 1.0
    assert stats["actor_updated"] == 0.0
    for branch in ("v", "a"):
        old_step = int(np.asarray(params[f"adam_step_{branch}"]))
        new_step = int(np.asarray(updated[f"adam_step_{branch}"]))
        assert new_step == old_step + 1
    for key in ("x_pi", "h_pi", "adam_m_pi", "adam_v_pi", "adam_step_pi"):
        _assert_tree_equal(updated[key], params[key])

    parser_args = experiment.build_parser().parse_args([])
    assert parser_args.critic_stop_iters is None


@pytest.mark.parametrize("critic_stop_iters", [-1, 11])
def test_validate_args_rejects_critic_stop_outside_closed_domain(
    critic_stop_iters: int,
) -> None:
    """Math: an exclusive critic stop must satisfy 0 <= t_C <= N."""
    args = experiment.build_parser().parse_args(
        [
            "--iters",
            "10",
            "--critic-stop-iters",
            str(critic_stop_iters),
        ]
    )

    with pytest.raises(ValueError, match="critic-stop-iters must be between"):
        experiment.validate_args(args)


def test_validate_args_requires_actor_stop_not_after_critic_stop() -> None:
    """Math: jointly defined stops must satisfy t_warm < t_pi <= t_C."""
    args = experiment.build_parser().parse_args(
        [
            "--iters",
            "10",
            "--actor-warmup-iters",
            "1",
            "--actor-stop-iters",
            "6",
            "--critic-stop-iters",
            "5",
        ]
    )

    with pytest.raises(ValueError, match="actor-stop-iters must not exceed"):
        experiment.validate_args(args)


def test_validate_args_accepts_equal_stops_and_zero_critic_stop() -> None:
    """Math: t_pi = t_C is valid, and t_C = 0 freezes the critic immediately."""
    parser = experiment.build_parser()
    equal_stops = parser.parse_args(
        [
            "--iters",
            "10",
            "--actor-warmup-iters",
            "1",
            "--actor-stop-iters",
            "5",
            "--critic-stop-iters",
            "5",
        ]
    )
    experiment.validate_args(equal_stops)

    zero_critic_stop = parser.parse_args(
        [
            "--iters",
            "10",
            "--critic-stop-iters",
            "0",
        ]
    )
    experiment.validate_args(zero_critic_stop)


def test_collectors_preserve_legacy_and_share_randomized_initial_state() -> None:
    """Math: a common s_0 preserves o_0=M s_0 and the clean paired path."""
    problem = experiment.construct_problem(d=7, ds=3, da=2, alpha=0.5, seed=71)
    down, up = experiment.init_matched_models(
        problem,
        k=2,
        head_type="mlp",
        hidden_dim=6,
        seed=73,
    )
    ops = experiment.make_head_ops("mlp", problem.R, action_curvature_scale=0.0)

    def collect_shared(initial_state: object = None) -> tuple[np.ndarray, ...]:
        return experiment.collect_shared_batch(
            down,
            problem,
            ops,
            epsilon=0.0,
            dt=0.02,
            n_steps=4,
            exploration_std=0.1,
            transition_noise=0.1,
            action_clip=20.0,
            process_rng=np.random.default_rng(79),
            exploration_rng=np.random.default_rng(83),
            observation_rng=np.random.default_rng(89),
            initial_state=initial_state,
        )

    legacy = collect_shared()
    explicit_legacy = collect_shared(np.full(problem.ds, 0.5))
    for actual, expected in zip(legacy, explicit_legacy):
        np.testing.assert_array_equal(actual, expected)

    initial_state = np.array([-0.4, 0.2, 0.8])
    batches = experiment.collect_independent_batches(
        down,
        up,
        problem,
        ops,
        epsilon=0.0,
        dt=0.02,
        n_steps=4,
        exploration_std=0.0,
        transition_noise=0.0,
        action_clip=20.0,
        process_rng=np.random.default_rng(97),
        exploration_rng=np.random.default_rng(101),
        observation_rng=np.random.default_rng(103),
        initial_state=initial_state,
    )
    np.testing.assert_array_equal(batches["down_states"][0], initial_state)
    np.testing.assert_array_equal(batches["up_states"][0], initial_state)
    np.testing.assert_allclose(
        batches["up_observations"],
        batches["up_states"] @ problem.M.T,
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        batches["probe_up_observations"],
        batches["down_states"] @ problem.M.T,
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        batches["up_states"],
        batches["down_states"],
        rtol=2e-11,
        atol=2e-12,
    )


def test_exploration_cosine_uses_exclusive_decay_end() -> None:
    """Math: sigma(0)=sigma_0 and sigma(t_end-1)=sigma_f."""
    values = np.array(
        [
            experiment.exploration_standard_deviation(
                0.3, 0.1, iteration, decay_end_iters=5, schedule="cosine"
            )
            for iteration in range(7)
        ]
    )
    np.testing.assert_allclose(values[0], 0.3, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(values[4:], 0.1, rtol=0.0, atol=1e-15)
    assert np.all(np.diff(values) <= 0.0)
    assert (
        experiment.exploration_standard_deviation(
            0.3, 0.1, iteration=4, decay_end_iters=5, schedule="constant"
        )
        == 0.3
    )


def test_delayed_actor_cosine_preserves_legacy_defaults_and_exclusive_end() -> None:
    """Math: delayed cosine is flat before t_start and zero at t_end-1."""
    base_rate = 2.0e-4
    legacy = experiment.actor_learning_rate(
        base_rate,
        "mlp",
        iteration=6,
        warmup_iters=2,
        total_iters=10,
        mlp_schedule="cosine",
    )
    legacy_progress = (6 - 2) / (10 - 1 - 2)
    expected_legacy = base_rate * 0.5 * (1.0 + np.cos(np.pi * legacy_progress))
    np.testing.assert_allclose(legacy, expected_legacy, rtol=0.0, atol=1e-18)

    rates = [
        experiment.actor_learning_rate(
            base_rate,
            "mlp",
            iteration=iteration,
            warmup_iters=2,
            total_iters=10,
            mlp_schedule="cosine",
            decay_start_iters=4,
            decay_end_iters=8,
        )
        for iteration in range(9)
    ]
    np.testing.assert_allclose(rates[:5], base_rate, rtol=0.0, atol=1e-18)
    np.testing.assert_allclose(rates[7:], 0.0, rtol=0.0, atol=1e-18)


def test_validate_args_resolves_new_schedule_defaults() -> None:
    """Math: omitted extensions recover fixed s_0, sigma, and legacy actor timing."""
    args = experiment.build_parser().parse_args(
        ["--iters", "10", "--critic-stop-iters", "6"]
    )
    experiment.validate_args(args)

    assert args.train_initial_state_std == 0.0
    assert args.exploration_schedule == "constant"
    assert args.exploration_final_std == 0.1
    assert args.exploration_decay_end_iters == 6
    assert args.mlp_actor_lr_decay_start_iters is None


@pytest.mark.parametrize(
    ("cli", "message"),
    [
        (
            ["--train-initial-state-std", "-0.1"],
            "train-initial-state-std must be finite and nonnegative",
        ),
        (
            ["--exploration-final-std", "-0.1"],
            "exploration standard deviations must be finite and nonnegative",
        ),
        (
            [
                "--iters",
                "10",
                "--exploration-schedule",
                "cosine",
                "--exploration-decay-end-iters",
                "1",
            ],
            "cosine exploration requires",
        ),
        (
            ["--mlp-actor-lr-decay-start-iters", "100"],
            "requires the cosine schedule",
        ),
        (
            [
                "--iters",
                "10",
                "--actor-warmup-iters",
                "2",
                "--actor-stop-iters",
                "8",
                "--mlp-actor-lr-schedule",
                "cosine",
                "--mlp-actor-lr-decay-start-iters",
                "7",
            ],
            "at least two iterations before",
        ),
    ],
)
def test_validate_args_rejects_invalid_new_schedule_domains(
    cli: list[str], message: str
) -> None:
    """Math: schedule scales and half-open endpoints stay in their domains."""
    args = experiment.build_parser().parse_args(cli)
    with pytest.raises(ValueError, match=message):
        experiment.validate_args(args)
