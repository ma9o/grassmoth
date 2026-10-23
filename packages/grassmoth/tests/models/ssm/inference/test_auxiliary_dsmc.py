"""Paper auxiliary laws, mixture weights and invariant conditional particle updates."""

from dataclasses import replace

import dynestyx as dsx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import norm

from grassmoth.artifacts.sampling import DsmcSpec
from grassmoth.models.ssm.inference.samplers.particle.auxiliary import build_auxiliary
from grassmoth.models.ssm.inference.samplers.particle.context import (
    build_smoother_context,
)
from grassmoth.models.ssm.inference.samplers.particle.contract import (
    PaidMixture,
)
from grassmoth.models.ssm.inference.samplers.particle.dsmc import step
from grassmoth.models.ssm.inference.samplers.particle.setup import build_smoother_setup
from grassmoth.models.ssm.inference.targets.gaussian_reference import GaussianReference
from grassmoth_test_support.interval_particle import interval_target

pytestmark = pytest.mark.inference(concern="sampling")


def _context(target, method, *, mixture=False, count=4):
    path = jnp.linspace(-0.2, 0.4, target.times.size + 1)[1:, None]
    static = build_smoother_setup(
        target,
        DsmcSpec(num_particles=count, drift=method),
        num_parameter_particles=2,
        free_mask=target.independent_mask,
        paid_mix=PaidMixture(jnp.zeros_like(path), jnp.ones_like(path), jnp.asarray([mixture])),
        diagnostic_metrics=frozenset(),
    )
    return build_smoother_context(
        static,
        jnp.array([[-0.4], [0.7]]),
        jnp.array([0.2, -0.3]),
        reference=path,
        delta=jnp.full(target.times.shape, 0.6),
        trajectory_dtype=path.dtype,
    )


def _local_gradients(target, ctx, path, method):
    local = target.tree_target()
    gradients = []
    for index in range(len(path)):

        def density(value, index=index):
            def per_label(bound):
                result = (
                    local.initial_log_prob(bound, value)
                    if index == 0
                    else local.transition_log_prob(bound, path[index - 1], value, index)
                )
                result += local.observation_increment(bound, value, index, target.observations)
                if method == "amala_plus" and index + 1 < len(path):
                    result += local.transition_log_prob(bound, value, path[index + 1], index + 1)
                return result

            return jax.scipy.special.logsumexp(
                ctx.initial_label_log_probs + jax.vmap(per_label)(ctx.contexts)
            )

        gradients.append(jax.grad(density)(path[index]))
    return jnp.stack(gradients)


def _auxiliary_density(auxiliary, path):
    return jnp.sum(
        jax.vmap(auxiliary.local_log_density)(
            jnp.concatenate((path[:1], path[:-1])),
            path,
            jnp.concatenate((path[1:], path[-1:])),
            jnp.arange(len(path)),
        )
    )


@pytest.mark.parametrize("interval", [False, True])
def test_local_auxiliary_keeps_exact_global_parameter_label_scores(interval):
    times = np.arange(5.0)
    observations = np.array([[0.1], [0.4], [-0.2], [0.6], [0.8]])
    starts = times[:, None].copy()
    if interval:
        observations[:2] = np.nan
        starts = np.array([[np.nan], [np.nan], [0.0], [1.0], [1.5]])
    target = interval_target(times, observations, starts, ("mean" if interval else "last",))
    ctx = _context(target, "amala")
    path = jnp.array([[-0.3], [0.1], [0.6], [-0.2], [0.5]])
    scores = ctx.score_labels(path)
    expected_path = jax.vmap(lambda bound: target.path_log_prob(bound, path, target.observations))(
        ctx.contexts
    )
    expected_joint = expected_path + jax.vmap(target.log_prior)(jnp.array([[-0.4], [0.7]]))
    np.testing.assert_allclose(scores.trajectory_log_probs, expected_path, atol=2e-5)
    np.testing.assert_allclose(scores.log_posteriors, expected_joint, atol=2e-5)
    np.testing.assert_allclose(
        scores.log_probs, jax.nn.log_softmax(expected_joint + jnp.array([0.2, -0.3])), atol=2e-5
    )


@pytest.mark.parametrize("method", ["amala", "amala_plus"])
@pytest.mark.parametrize("mixture", [False, True])
def test_auxiliary_draw_and_mixture_density_follow_paper_laws(method, mixture):
    times = np.arange(3.0)
    target = interval_target(times, np.array([[0.1], [0.4], [-0.2]]), times[:, None], ("last",))
    ctx = _context(target, method, mixture=mixture)
    reference = jnp.array([[-0.3], [0.2], [0.6]])
    mask = jnp.array([[True], [False], [True]])
    key = jax.random.key(83)
    auxiliary = build_auxiliary(ctx, key, reference, mask)
    variance = np.asarray(ctx.amala_delta)[:, None] / 2
    noise = np.sqrt(variance) * np.asarray(
        jax.random.normal(key, reference.shape, dtype=reference.dtype)
    )
    gradient = _local_gradients(target, ctx, reference, method)
    drift = variance * np.asarray(gradient)
    expected = np.where(mask, reference + drift + noise, reference)
    np.testing.assert_allclose(auxiliary.auxiliary, expected, atol=2e-5)
    candidate = reference + jnp.where(mask, 0.25, 0)
    gradient = _local_gradients(target, ctx, candidate, method)
    center = candidate + variance * gradient
    expected_density = np.where(
        mask, -0.5 * (np.log(2 * np.pi * variance) + (expected - center) ** 2 / variance), 0
    ).sum()
    np.testing.assert_allclose(
        _auxiliary_density(auxiliary, candidate), expected_density, atol=2e-5
    )
    particles, density = auxiliary.propose(jax.random.key(91))
    np.testing.assert_array_equal(particles[:, 0], reference)
    np.testing.assert_array_equal(particles[1, :, 0], np.full(4, reference[1, 0]))

    def normal(mean, var):
        return np.where(
            np.asarray(mask)[:, None],
            -0.5 * (np.log(2 * np.pi * var) + (np.asarray(particles) - mean) ** 2 / var),
            0,
        ).sum(axis=-1)

    local = normal(np.asarray(auxiliary.auxiliary)[:, None], variance[:, None])
    expected_proposal = local
    if mixture:
        expected_proposal = logsumexp(
            np.stack(
                (local + np.log(0.85), normal(0, 0.25) + np.log(0.1), normal(0, 4.0) + np.log(0.05))
            ),
            axis=0,
        )
    np.testing.assert_allclose(density, expected_proposal, atol=2e-5)


def _gaussian_smoother_case(interval):
    """Independent Gaussian conditioning supplies the exact joint posterior."""
    # Three times exercise an unbalanced tree while keeping compilation small.
    times = np.arange(3.0)
    if interval:
        # Overlapping windows [0, 1] and [0, 2]; the latter crosses the root seam.
        y = np.array([0.4, 0.8])
        observations = np.array([[np.nan], [y[0]], [y[1]]])
        starts = np.array([[np.nan], [0.0], [0.0]])
        rows = np.array([[0.5, 0.5, 0.0], [0.25, 0.5, 0.25]])
        summary = "mean"
    else:
        y = np.array([0.1, 0.4, -0.2])
        observations, starts, rows, summary = y[:, None], times[:, None], np.eye(3), "last"
    target = interval_target(times, observations, starts, (summary,))
    transition = np.tril(0.7 ** (np.arange(3)[:, None] - np.arange(3)[None]))
    prior = np.zeros((4, 4))
    prior[0, 0] = 1.0
    prior[1:, 1:] = transition @ np.diag([1.0, 0.3, 0.3]) @ transition.T
    design = np.column_stack((np.ones(len(y)), rows))
    gain = np.linalg.solve(design @ prior @ design.T + 0.2 * np.eye(len(y)), design @ prior).T
    return target, gain @ y, prior - gain @ design @ prior


def _stationarity_features(paths, mean, chol):
    """Check all first/second moments and bounded nonlinear functions at every time."""
    whitened = np.linalg.solve(chol, (paths - mean).T).T
    left, right = np.triu_indices(whitened.shape[1])
    return np.column_stack(
        (
            whitened,
            whitened[:, left] * whitened[:, right],
            np.sin(whitened),
            np.cos(whitened),
        )
    )


# Cover every pair of support, parameter count, drift and pilot-mixture choices.
@pytest.mark.parametrize(
    ("interval", "parameter_count", "drift", "mixture"),
    [
        (False, 1, "amala", False),
        (True, 2, "amala", True),
        (False, 2, "amala_plus", True),
        (True, 1, "amala_plus", False),
        (False, 1, "none", True),
        (True, 2, "none", False),
    ],
)
def test_conditional_tree_preserves_analytic_posterior(interval, parameter_count, drift, mixture):
    """One transition from exact independent draws isolates invariance without burn-in."""
    target, joint_mean, joint_covariance = _gaussian_smoother_case(interval)
    parameters = np.array([0.3]) if parameter_count == 1 else np.array([-0.4, 0.7])
    correction = np.array([0.2, -0.3])[:parameter_count]
    regression = joint_covariance[1:, 0] / joint_covariance[0, 0]
    means = joint_mean[None, 1:] + (parameters - joint_mean[0])[:, None] * regression
    covariance = joint_covariance[1:, 1:] - np.outer(regression, joint_covariance[0, 1:])
    conditional_chol = np.linalg.cholesky(covariance)
    # Integrating each component includes its evidence, not just its prior label weight.
    label_logits = correction - 0.5 * (parameters - joint_mean[0]) ** 2 / joint_covariance[0, 0]
    probabilities = np.exp(label_logits - logsumexp(label_logits))
    mean = probabilities @ means
    centered = means - mean
    total_covariance = covariance + (centered.T * probabilities) @ centered
    chol = np.linalg.cholesky(total_covariance)
    rng = np.random.default_rng(1801)
    size = 4096
    labels = rng.choice(parameter_count, size=size, p=probabilities)
    before = jnp.asarray(
        (means[labels] + rng.normal(size=(size, len(mean))) @ conditional_chol.T)[..., None]
    )
    spec = DsmcSpec(num_particles=4, drift=drift, pilot_mixture=mixture)
    setup = build_smoother_setup(
        target,
        spec,
        num_parameter_particles=parameter_count,
        free_mask=target.independent_mask,
        paid_mix=PaidMixture(
            jnp.zeros_like(before[0]), jnp.ones_like(before[0]), jnp.asarray([mixture])
        ),
        diagnostic_metrics=frozenset(),
    )

    def advance(path, key):
        ctx = build_smoother_context(
            setup,
            jnp.asarray(parameters[:, None]),
            jnp.asarray(correction),
            reference=path,
            delta=jnp.full(target.times.shape, 0.3),
            trajectory_dtype=path.dtype,
        )
        result = step(ctx, key, path, None)
        return result.latent_path, result.label_scores.log_probs

    keys = jax.random.split(jax.random.fold_in(jax.random.key(1801), 1), size)
    paths, log_probs = jax.jit(jax.vmap(advance))(before, keys)
    initial_paths = np.asarray(before)[..., 0]
    initial_features = _stationarity_features(initial_paths, mean, chol)
    # Bonferroni across six cases and every tested moment/function. Repeated sweeps
    # belong in the mixing benchmark; they are not needed to test invariance.
    critical = norm.isf(0.001 / (2 * 6 * initial_features.shape[1]))
    observed = np.asarray(paths)[..., 0]
    differences = _stationarity_features(observed, mean, chol) - initial_features
    uncertainty = differences.std(axis=0, ddof=1) / np.sqrt(size)
    assert np.all(np.abs(differences.mean(axis=0)) <= critical * uncertainty)

    # The identity kernel is invariant too; require movement at every time coordinate.
    displacement = np.mean((observed - initial_paths) ** 2, axis=0)
    assert np.all(displacement > 1e-4 * np.diag(total_covariance))

    residuals = observed[:, None, :] - means[None]
    standardized = np.linalg.solve(conditional_chol, residuals.reshape(-1, len(mean)).T).T
    expected_logits = label_logits[None] - 0.5 * np.sum(
        standardized.reshape(size, parameter_count, len(mean)) ** 2, axis=-1
    )
    expected = expected_logits - logsumexp(expected_logits, axis=-1, keepdims=True)
    np.testing.assert_allclose(log_probs, expected, atol=5e-5, rtol=1e-5)


@pytest.mark.parametrize(
    ("count", "steps", "drift"), [(4, 1, "none"), (4, 5, "amala"), (8, 33, "amala_plus")]
)
def test_root_replay_evaluates_only_linear_number_of_complete_paths(count, steps, drift):
    times = np.arange(float(steps))
    target = interval_target(times, np.full((steps, 1), 0.2), times[:, None], ("last",))
    ctx = _context(target, drift, count=count, mixture=True)
    calls = []

    def correction(path, local_prior):
        expected = jax.vmap(lambda bound: target.prior_path_log_prob(bound, path))(ctx.contexts)
        jax.debug.callback(
            lambda x, carried, direct: calls.append((x.shape, carried, direct)),
            path,
            local_prior,
            expected,
        )
        return jnp.zeros(2)

    result = jax.jit(lambda key: step(ctx, key, jnp.zeros((steps, 1)), correction, batch_size=2))(
        jax.random.key(43)
    )
    jax.block_until_ready(result.latent_path)
    jax.effects_barrier()
    assert [shape for shape, _, _ in calls] == [(steps, 1)] * count
    for _, carried, direct in calls:
        np.testing.assert_allclose(carried, direct, rtol=2e-5, atol=2e-5)


def test_zero_drift_uses_leaf_factors_without_path_or_transition_gradients():
    """The random-walk auxiliary is exactly local even for a parameter mixture."""
    from grassmoth.models.ssm.inference.samplers.particle.dsmc import trace_step

    times = np.arange(5.0)
    target = interval_target(times, np.full((5, 1), 0.2), times[:, None], ("last",))
    reference = jnp.linspace(-0.2, 0.3, 5)[:, None]

    def forbidden(*_args):
        raise AssertionError("Zero drift must not evaluate unused gradients")

    ctx = _context(target, "amala_plus", count=5)._replace(
        drift="none",
        transition_current_value_grad_by_param=forbidden,
        transition_next_value_grad_by_param=forbidden,
    )
    auxiliary = build_auxiliary(ctx, jax.random.key(910), reference, ctx.latent_free_mask)
    candidate = reference + 0.1
    expected = norm.logpdf(
        np.asarray(auxiliary.auxiliary), np.asarray(candidate), np.sqrt(0.3)
    ).sum()
    np.testing.assert_allclose(_auxiliary_density(auxiliary, candidate), expected, atol=1e-5)
    result = jax.jit(lambda: trace_step(ctx, jax.random.key(911), reference, None))()
    np.testing.assert_allclose(result.trace.root_selection.log_weights, -np.log(5), atol=1e-6)
    np.testing.assert_array_equal(auxiliary.gradient_norms, np.zeros(5))


@pytest.mark.parametrize("state_dependent", [False, True])
@pytest.mark.parametrize("beta", [None, 0.0, 0.4, 1.0])
def test_pairwise_transition_matrix_matches_individual_declared_densities(state_dependent, beta):
    times = np.arange(3.0)
    target = interval_target(times, np.full((3, 1), 0.2), times[:, None], ("last",), state_dim=3)
    if state_dependent:
        declared = target.dynamical_model

        def nonlinear(context):
            model = declared(context)
            evolution = dsx.GaussianStateEvolution(
                lambda x, _u, _t0, _t1: 0.7 * x + 0.2 * jnp.sin(x),
                lambda x, _u, _t0, _t1: jnp.diag(0.2 + jax.nn.softplus(x)) + 0.1 * jnp.outer(x, x),
            )
            return eqx.tree_at(lambda m: m.state_evolution, model, evolution)

        target = replace(target, dynamical_model=nonlinear)
    if beta is not None:
        bridge_reference = GaussianReference(
            parameter_mean=jnp.zeros(1),
            parameter_chol=jnp.eye(1),
            path_mean=jnp.zeros((3, 3)),
            autoregression=jnp.broadcast_to(0.4 * jnp.eye(3), (3, 3, 3)),
            innovation_chol=jnp.broadcast_to(0.7 * jnp.eye(3), (3, 3, 3)),
            free_mask=target.independent_mask,
        )
        target = target.at_temperature(bridge_reference, jnp.asarray(beta))
    context = target.context(jnp.zeros(1), target.times)
    before = jax.random.normal(jax.random.key(5), (7, 3))
    after = jax.random.normal(jax.random.key(8), (9, 3))

    def reference(a, b):
        return jax.vmap(
            lambda x: jax.vmap(lambda y: target.transition_log_prob(context, x, y, 1))(b)
        )(a)

    actual = target.pairwise_transition_log_prob(context, before, after, 1)
    np.testing.assert_allclose(actual, reference(before, after), rtol=2e-5, atol=2e-5)
    actual_grad = jax.grad(
        lambda a, b: target.pairwise_transition_log_prob(context, a, b, 1).sum(), argnums=(0, 1)
    )(before, after)
    expected_grad = jax.grad(lambda a, b: reference(a, b).sum(), argnums=(0, 1))(before, after)
    for actual, expected in zip(actual_grad, expected_grad, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)

    for current_gradient in (False, True):
        actual_scores = jax.jit(
            jax.vmap(
                lambda x, current_gradient=current_gradient: jax.vmap(
                    lambda y: target._transition_value_and_grad(
                        context, x, y, 1, current_gradient=current_gradient
                    )
                )(after)
            )
        )(before)
        expected_scores = jax.jit(
            jax.vmap(
                lambda x, current_gradient=current_gradient: jax.vmap(
                    lambda y: jax.value_and_grad(
                        target.transition_log_prob, argnums=2 if current_gradient else 1
                    )(context, x, y, 1)
                )(after)
            )
        )(before)
        for actual, expected in zip(actual_scores, expected_scores, strict=True):
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize(
    ("method", "steps", "mixture"),
    [
        ("amala", 1, False),
        ("amala_plus", 2, False),
        ("amala", 4, True),
        ("amala_plus", 5, True),
        ("amala_plus", 33, True),
    ],
)
def test_local_auxiliary_mixture_requires_no_root_auxiliary_correction(method, steps, mixture):
    from grassmoth.models.ssm.inference.samplers.particle.dsmc import trace_step

    times = np.arange(float(steps))
    target = interval_target(times, np.full((steps, 1), 0.2), times[:, None], ("last",))
    reference = jnp.linspace(-0.2, 0.3, steps)[:, None]
    options = DsmcSpec(num_particles=5, drift=method)
    setup = build_smoother_setup(
        target,
        options,
        num_parameter_particles=1,
        free_mask=target.independent_mask,
        paid_mix=PaidMixture(
            jnp.zeros_like(reference), jnp.ones_like(reference), jnp.asarray([mixture])
        ),
        diagnostic_metrics=frozenset(),
    )
    ctx = build_smoother_context(
        setup,
        jnp.array([[0.3]]),
        jnp.zeros(1),
        reference=reference,
        delta=jnp.full(steps, 0.3),
        trajectory_dtype=reference.dtype,
    )
    key = jax.random.key(610)
    full = jax.jit(lambda: trace_step(ctx, key, reference, None))()

    fast = jax.jit(lambda: trace_step(ctx, key, reference, None, batch_size=2))()
    if steps == 1:
        # With no merge/resampling level, the root still owns the leaf's exact
        # scientific and auxiliary importance weights.
        keys = jax.random.split(key, 6)
        auxiliary = build_auxiliary(ctx, keys[0], reference, ctx.latent_free_mask)
        candidates, proposal = auxiliary.propose(keys[2])
        context = target.context(jnp.array([0.3]), target.times)
        weights = (
            jax.vmap(
                lambda value: (
                    target.path_log_prob(context, value[None], target.observations)
                    + auxiliary.local_log_density(value, value, value, jnp.asarray(0))
                )
            )(candidates[0])
            - proposal[0]
        )
        expected = jax.nn.log_softmax(weights)
    else:
        expected = jnp.full(5, -np.log(5))
    np.testing.assert_allclose(fast.trace.root_selection.log_weights, expected, atol=1e-6)
    np.testing.assert_array_equal(fast.result.origin_path, full.result.origin_path)
    np.testing.assert_allclose(fast.result.latent_path, full.result.latent_path, atol=1e-6)
    np.testing.assert_allclose(
        fast.trace.root_selection.log_weights, full.trace.root_selection.log_weights, atol=1e-5
    )


@pytest.mark.parametrize(
    ("method", "kappa", "steps", "interval"),
    [
        ("amala", 1.0, 33, False),
        ("amala_plus", 1.0, 33, False),
        ("amala_plus", 0.0, 33, False),
        ("amala_plus", 0.0, 33, True),
    ],
)
def test_dsmc_batching_preserves_nonlocal_mixture_transition(
    method, kappa, steps, interval, monkeypatch
):
    from grassmoth.artifacts.probability.families import DistributionFamily
    from grassmoth.models.ssm.inference.samplers.particle import dsmc
    from grassmoth.models.ssm.inference.samplers.particle.dsmc import trace_step

    times = np.arange(float(steps))
    if interval:
        observations = np.full((steps, 2), np.nan)
        starts = observations.copy()
        observations[:, 0], starts[:, 0] = 0.2, times
        for row, start, value in ((7, 1, 2), (15, 9, 3), (31, 17, 4), (32, 28, 2)):
            observations[row, 1], starts[row, 1] = value, start
        target = interval_target(
            times,
            observations,
            starts,
            ("last", "sum"),
            families=(DistributionFamily.GAUSSIAN, DistributionFamily.NEGATIVE_BINOMIAL),
        )
    else:
        target = interval_target(times, np.full((steps, 1), 0.2), times[:, None], ("last",))
    ctx = _context(target, method, count=5, mixture=True)._replace(
        drift=method if kappa else "none", diagnostic_weights=True
    )
    path = jnp.linspace(-0.2, 0.3, steps)[:, None]
    key = jax.random.key(711)
    parallel = jax.jit(lambda: trace_step(ctx, key, path, None))()
    batched = jax.jit(lambda: trace_step(ctx, key, path, None, batch_size=2))()
    np.testing.assert_array_equal(batched.result.origin_path, parallel.result.origin_path)
    for metric, values in parallel.result.diagnostics.items():
        np.testing.assert_allclose(batched.result.diagnostics[metric], values, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(
        batched.trace.root_selection.log_weights,
        parallel.trace.root_selection.log_weights,
        atol=2e-5,
    )

    reductions = []
    original = dsmc.weight_statistics

    def observed_weight_work(logits):
        jax.debug.callback(lambda: reductions.append(True))
        return original(logits)

    monkeypatch.setattr(dsmc, "weight_statistics", observed_weight_work)
    disabled = jax.jit(
        lambda enabled: trace_step(ctx._replace(diagnostic_weights=enabled), key, path, None)
    )(jnp.asarray([False]))
    jax.effects_barrier()
    assert not reductions
    for name, values in disabled.result.diagnostics.items():
        if name.startswith("dsmc_weight_"):
            np.testing.assert_array_equal(values, jnp.zeros_like(values))
    np.testing.assert_array_equal(disabled.result.origin_path, parallel.result.origin_path)
    np.testing.assert_allclose(disabled.result.latent_path, parallel.result.latent_path)
