"""Independent density, Gaussian and nonlinear posterior oracles for carried stages."""

from dataclasses import replace
from typing import cast, override

import dynestyx as dsx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from dynestyx.models import DiscreteTimeStateEvolution
from numpy.polynomial.hermite import hermgauss
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, norm

from grassmoth.artifacts.sampling import GuidedPathSpec
from grassmoth.models.ssm.compile.state_space import (
    ParallelRecurrences,
    SequentialRecurrences,
    StateSpace,
)
from grassmoth.models.ssm.execution.contracts import MeasurementParams
from grassmoth.models.ssm.execution.dynamical_model import HeterogeneousObservation
from grassmoth.models.ssm.execution.supported_normal import SupportedNormal
from grassmoth.models.ssm.inference.samplers.particle.context import build_smoother_context
from grassmoth.models.ssm.inference.samplers.particle.guided import (
    guided_step,
    trace_guided_step,
)
from grassmoth.models.ssm.inference.samplers.particle.guided_geometry import _memory_guide
from grassmoth.models.ssm.inference.samplers.particle.setup import build_smoother_setup
from grassmoth.models.ssm.inference.targets.coordinates import TranslationChart
from grassmoth.models.ssm.inference.targets.sequential import sequential_observations
from grassmoth_test_support.interval_particle import interval_target

pytestmark = pytest.mark.inference(concern="sampling")

READINGS = np.array(
    [[0.2, np.nan, np.nan], [-0.1, np.nan, np.nan], [0.3, -0.1, 0.1], [0.1, 0.2, -0.1]]
)
WINDOWS = np.array(
    [[0, 0, 0, 0], [0, 0, 0, 0], [1 / 12, 7 / 12, 1 / 3, 0], [0, 1 / 12, 7 / 12, 1 / 3]]
)


class _Evolution(DiscreteTimeStateEvolution):
    label: jax.Array
    nonlinear: float

    @override
    def __call__(self, x, u, t_now, t_next):
        del u, t_now, t_next
        mean = jnp.array(
            [
                0.65 * x[0] + 0.25 * x[2] + self.nonlinear * 0.15 * jnp.tanh(x[1]),
                0.8 * x[1]
                + (0.4 + 0.2 * self.label) * x[0]
                + self.nonlinear * 0.15 * (jnp.tanh(x[1]) ** 3 + x[0] ** 2),
                0.5 * x[2] + 0.3 * x[1] + self.nonlinear * 0.1 * jnp.tanh(x[2]) ** 3,
            ]
        )
        return SupportedNormal(mean, jnp.diag(jnp.array([0.3, 0.0, 0.0])), (0,))


def _target(nonlinear, *, random_initial=False, steps=4):
    times = np.arange(float(steps))
    starts = np.column_stack((times, times, [np.nan, np.nan, 0.5, 1.5][:steps]))
    base = interval_target(
        times,
        READINGS[:steps],
        starts,
        ("last", "last", "mean"),
        state_dim=3,
        covariance=np.eye(3) * 0.45,
    )

    def context(position, grid):
        arrays, _, _, _ = base.context(position, grid)
        return arrays, grid, jnp.pad(position, (0, 2)), None

    def declared(bound):
        original = base.dynamical_model(bound)
        label = bound[2][0]
        return dsx.DynamicalModel(
            initial_condition=SupportedNormal(
                jnp.array([0.2 * label, 0.2 * label, -0.1 * label]),
                jnp.diag(jnp.array([0.65, 0.4 if random_initial else 0.0, 0.0])),
                (0, 1) if random_initial else (0,),
            ),
            state_evolution=_Evolution(label, nonlinear),
            observation_model=HeterogeneousObservation(
                MeasurementParams(
                    jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
                    jnp.zeros(3),
                    jnp.eye(3) * 0.45,
                ),
                cast("HeterogeneousObservation", original.observation_model).laws,
            ),
            t0=0.0,
        )

    recurrence = SequentialRecurrences((1, 2)) if nonlinear else ParallelRecurrences((1, 2))
    return replace(
        base,
        context=context,
        dynamical_model=declared,
        state_space=StateSpace(3, (), (0, 1) if random_initial else (0,), (0,), (), (recurrence,)),
        intervals=replace(base.intervals, dynamical_model=declared)
        if base.intervals is not None
        else None,
    )


def _prior(noise, label, nonlinear, *, random_initial=False):
    """Independent NumPy simulation of the declared conditional prior."""
    steps = noise.shape[-1] - int(random_initial)
    path = np.zeros((*noise.shape[:-1], steps, 3))
    path[..., 0, 0] = 0.2 * label + np.sqrt(0.65) * noise[..., 0]
    path[..., 0, 1] = 0.2 * label + (np.sqrt(0.4) * noise[..., -1] if random_initial else 0.0)
    path[..., 0, 2] = -0.1 * label
    for t in range(1, steps):
        x, z, w = np.moveaxis(path[..., t - 1, :], -1, 0)
        path[..., t, 0] = (
            0.65 * x + 0.25 * w + nonlinear * 0.15 * np.tanh(z) + np.sqrt(0.3) * noise[..., t]
        )
        path[..., t, 1] = (
            0.8 * z + (0.4 + 0.2 * label) * x + nonlinear * 0.15 * (np.tanh(z) ** 3 + x**2)
        )
        path[..., t, 2] = 0.5 * w + 0.3 * z + nonlinear * 0.1 * np.tanh(w) ** 3
    return path


def _responses(path):
    steps = path.shape[-2]
    return np.stack((path[..., 0], path[..., 2], path[..., 2] @ WINDOWS[:steps, :steps].T), axis=-1)


def _likelihood(path):
    residual = np.nan_to_num(READINGS[: path.shape[-2]] - _responses(path))
    return -0.5 * np.sum(residual**2, axis=(-2, -1)) / 0.45


@pytest.mark.parametrize(("steps", "random_initial"), [(4, False), (4, True), (1, True)])
def test_recurrence_guide_matches_dense_gaussian_conditioning(steps, random_initial):
    target = _target(0.0, random_initial=random_initial, steps=steps)
    guide = target.context(jnp.array([0.25]), target.times)
    memory = sequential_observations(target, jax.tree.map(lambda x: x[None], guide), guide)
    mask = np.asarray(target.independent_mask)
    count = int(mask.sum())
    origin = _prior(np.zeros(count), 0.25, 0.0, random_initial=random_initial)
    basis = _prior(np.eye(count), 0.25, 0.0, random_initial=random_initial)
    mapping = (basis[:, mask] - origin[mask]).T
    covariance = mapping @ mapping.T
    valid = np.isfinite(READINGS[:steps])
    response0 = _responses(origin)[valid]
    design = np.linalg.solve(mapping.T, (_responses(basis)[:, valid] - response0)).T
    offset = response0 - design @ origin[mask]
    auxiliary = jnp.full((steps, 3), 0.15)
    variance = 0.2
    # Only x has an emission depending on the current independent coordinate.
    current_x = np.zeros_like(mask)
    current_x[:, 0] = True
    diagonal = np.where(current_x[mask], 1 - variance / 0.45, 1)
    aux_offset = np.zeros_like(mask, dtype=float)
    aux_offset[:, 0] = variance * READINGS[:steps, 0] / 0.45
    precision = (
        np.linalg.inv(covariance) + design.T @ design / 0.45 + np.diag(diagonal**2 / variance)
    )
    posterior = np.linalg.inv(precision)
    mean = posterior @ (
        np.linalg.solve(covariance, origin[mask])
        + design.T @ (READINGS[:steps][valid] - offset) / 0.45
        + diagonal * (np.asarray(auxiliary)[mask] - aux_offset[mask]) / variance
    )

    def observe(value, index):
        return memory.log_probs(value, index), jax.grad(
            lambda state: memory.log_probs(memory.replace_current(value, state, index), index)[0]
        )(memory.state(value))

    _, geometry = jax.jit(
        lambda: _memory_guide(
            memory,
            observe,
            jnp.zeros(1),
            auxiliary,
            jnp.full(auxiliary.shape, variance),
            target.independent_mask,
            GuidedPathSpec(method="twisted_particle_agrad", kappa=1),
            "actual_law",
        )
    )()
    coordinates = np.full((steps, 3), 73.0)
    coordinates[mask] = np.linspace(-0.2, 0.3, count)
    carried = memory.path(jnp.asarray(coordinates))
    for t in range(steps):
        before = int(mask[:t].sum())
        after = before + int(mask[t].sum())
        local = jax.tree.map(lambda x, t=t: x[t], geometry)
        prior = (
            target.initial_moments(guide)[0]
            if t == 0
            else memory.guide_moments(carried[t - 1], jnp.asarray(t))[0]
        )
        center = local.center(prior, target.independent_mask[t]) + (
            local.parent_map @ carried[t - 1] if t else 0
        )
        expected_mean = mean[before:after]
        expected_covariance = posterior[before:after, before:after]
        if before:
            gain = np.linalg.solve(posterior[:before, :before], posterior[:before, before:after]).T
            expected_mean = expected_mean + gain @ (coordinates[mask][:before] - mean[:before])
            expected_covariance = expected_covariance - gain @ posterior[:before, before:after]
        np.testing.assert_allclose(np.asarray(center)[mask[t]], expected_mean, atol=3e-5)
        actual_covariance = np.asarray(local.chol @ local.chol.T)[np.ix_(mask[t], mask[t])]
        np.testing.assert_allclose(actual_covariance, expected_covariance, atol=3e-5)
        np.testing.assert_allclose(
            local.log_density(coordinates[t] - center, target.independent_mask[t]),
            multivariate_normal.logpdf(coordinates[t, mask[t]], expected_mean, expected_covariance),
            atol=3e-5,
        )


@pytest.mark.parametrize(
    ("method", "kappa"),
    [("twisted_particle_agrad", 1), ("particle_agrad", 0)],
)
def test_ancestral_recurrences_preserve_nonlinear_joint_posterior(method, kappa):
    target = _target(1.0)
    labels = np.array([-0.45, 0.55])
    label_prior = np.exp(norm.logpdf(labels) - logsumexp(norm.logpdf(labels)))
    nodes, weights = hermgauss(18)
    grid = np.stack(np.meshgrid(*(nodes * np.sqrt(2),) * 4, indexing="ij"), axis=-1).reshape(-1, 4)
    integration = np.prod(
        np.stack(np.meshgrid(*(weights / np.sqrt(np.pi),) * 4, indexing="ij"), axis=-1), axis=-1
    ).reshape(-1)
    oracle_paths = np.stack([_prior(grid, label, 1.0) for label in labels])
    masses = np.exp(_likelihood(oracle_paths)) * integration * label_prior[:, None]
    masses /= masses.sum()
    expected_mean = np.einsum("kn,knt->t", masses, oracle_paths[..., 0])
    expected_second = np.einsum(
        "kn,kni,knj->ij", masses, oracle_paths[..., 0], oracle_paths[..., 0]
    )
    # Exact posterior starts by prior rejection, independently of the quadrature and kernel.
    rng = np.random.default_rng(715)
    accepted = []
    while sum(len(batch) for batch in accepted) < 4096:
        selected = rng.choice(2, 8192, p=label_prior)
        paths = _prior(rng.normal(size=(8192, 4)), labels[selected], 1.0)
        accepted.append(paths[np.log(rng.uniform(size=8192)) < _likelihood(paths)])
    starts = jnp.asarray(np.concatenate(accepted)[:4096])
    spec = GuidedPathSpec(method=method, num_particles=8, kappa=kappa)
    setup = build_smoother_setup(
        target,
        spec,
        num_parameter_particles=2,
        free_mask=target.independent_mask,
        paid_mix=None,
        diagnostic_metrics=("particle_identity",),
    )
    ctx = build_smoother_context(
        setup,
        jnp.asarray(labels[:, None]),
        jnp.zeros(2),
        reference=starts[0],
        delta=jnp.full(4, 0.3),
        trajectory_dtype=starts.dtype,
    )
    guide = target.context(jnp.array([labels.mean()]), target.times)

    @jax.jit
    def run(paths):
        def sweep(current, key):
            updated = jax.vmap(
                lambda path, draw: guided_step(target, ctx, guide, spec, draw, path).latent_path
            )(current, jax.random.split(key, len(current)))
            return updated, None

        return jax.lax.scan(sweep, paths, jax.random.split(jax.random.key(371), 3))[0]

    paths = run(starts)
    draws = np.asarray(paths[..., 0])
    np.testing.assert_allclose(draws.mean(0), expected_mean, atol=0.035)
    np.testing.assert_allclose(draws.T @ draws / len(draws), expected_second, atol=0.035)
    actual_labels = jax.jit(jax.vmap(lambda path: jnp.exp(ctx.score_labels(path).log_probs)))(paths)
    np.testing.assert_allclose(np.asarray(actual_labels).mean(0), masses.sum(1), atol=0.025)
    assert np.min(np.mean(draws != np.asarray(starts[..., 0]), axis=0)) > 0.3
    traced = jax.jit(
        lambda path: trace_guided_step(target, ctx, guide, spec, jax.random.key(88), path)
    )(starts[0])
    selected = np.asarray(traced.trace.selected_particles)
    ancestors = np.asarray(traced.trace.ancestors)
    np.testing.assert_array_equal(selected[:-1], ancestors[np.arange(3), selected[1:]])
    for actual, expected in zip(
        traced.result.label_scores, ctx.score_labels(traced.result.latent_path), strict=True
    ):
        np.testing.assert_allclose(actual, expected, atol=3e-5)


@pytest.mark.parametrize("translated", [False, True])
def test_replayed_prefix_factors_match_whole_path_and_independent_recurrences(translated):
    physical = _target(1.0)
    chart = TranslationChart(
        jnp.broadcast_to(jnp.array([[-0.8], [0.0], [0.0]]), (4, 3, 1)), jnp.array([0.15])
    )
    target = physical.in_coordinates(chart) if translated else physical
    labels = jnp.array([[-0.45], [0.55]])
    contexts = jax.vmap(lambda label: target.context(label, target.times))(labels)
    guide = target.context(jnp.array([0.05]), target.times)
    memory = sequential_observations(target, contexts, guide)
    coordinates = jnp.array(
        [[0.2, 92.0, 93.0], [-0.3, 92.0, 93.0], [0.4, 92.0, 93.0], [-0.1, 92.0, 93.0]]
    )
    carried = jax.jit(memory.path)(coordinates)
    replayed = jax.vmap(memory.replayed_states)(carried)
    for k, label in enumerate([-0.45, 0.55, 0.05]):
        expected = np.array(
            chart.decode(jnp.array([label]), coordinates) if translated else coordinates
        )
        expected[0, 1:] = [0.2 * label, -0.1 * label]
        for t in range(1, 4):
            x, z, w = expected[t - 1]
            expected[t, 1] = 0.8 * z + (0.4 + 0.2 * label) * x + 0.15 * (np.tanh(z) ** 3 + x**2)
            expected[t, 2] = 0.5 * w + 0.3 * z + 0.1 * np.tanh(w) ** 3
        encoded = (
            chart.encode(jnp.array([label]), jnp.asarray(expected)) if translated else expected
        )
        np.testing.assert_allclose(replayed[:, k], encoded, atol=2e-6)

    def sequential(path):
        values = memory.path(path)
        return (
            memory.initial_log_probs(values[0])
            + jax.vmap(memory.transition_log_probs)(values[:-1], values[1:], jnp.arange(1, 4)).sum(
                0
            )
            + jax.vmap(memory.log_probs)(values, jnp.arange(4)).sum(0)
        )

    def whole(path):
        return jax.vmap(lambda bound: target.path_log_prob(bound, path, target.observations))(
            contexts
        )

    def physical_density(path):
        return jax.vmap(
            lambda label: physical.path_log_prob(
                physical.context(label, physical.times),
                chart.decode(label, path) if translated else path,
                physical.observations,
            )
        )(labels)

    np.testing.assert_allclose(
        jax.jit(sequential)(coordinates), jax.jit(whole)(coordinates), atol=3e-5
    )
    np.testing.assert_allclose(
        jax.jit(jax.jacrev(sequential))(coordinates),
        jax.jit(jax.jacrev(whole))(coordinates),
        atol=3e-5,
    )
    np.testing.assert_allclose(
        jax.jit(whole)(coordinates), jax.jit(physical_density)(coordinates), atol=3e-5
    )


def test_fixed_initial_state_without_memory_keeps_backward_selection():
    readings = np.array([0.15, -0.2, 0.3])
    base = interval_target(np.arange(3.0), readings[:, None], np.arange(3.0)[:, None], ("last",))

    def declared(bound):
        model = base.dynamical_model(bound)
        label = cast(
            "HeterogeneousObservation", model.observation_model
        ).measurement.manifest_means[0]
        return dsx.DynamicalModel(
            initial_condition=SupportedNormal(jnp.array([0.2 * label]), jnp.zeros((1, 1)), ()),
            state_evolution=model.state_evolution,
            observation_model=model.observation_model,
            t0=0.0,
        )

    target = replace(
        base, dynamical_model=declared, state_space=StateSpace(1, (), (), (0,), (), ())
    )
    labels = np.array([-0.45, 0.55])
    prior = 0.3 * np.array([[1.0, 0.7], [0.7, 1.49]])
    predictive = prior + np.eye(2) * 0.2
    gain = np.linalg.solve(predictive, prior).T
    prior_means = labels[:, None] * np.array([0.14, 0.098])
    means = prior_means + (readings[None, 1:] - labels[:, None] - prior_means) @ gain.T
    covariance = prior - gain @ prior
    evidence = (
        norm.logpdf(labels)
        + norm.logpdf(readings[0], 1.2 * labels, np.sqrt(0.2))
        + np.array(
            [
                multivariate_normal.logpdf(readings[1:], mean + label, predictive)
                for mean, label in zip(prior_means, labels, strict=True)
            ]
        )
    )
    probabilities = np.exp(evidence - logsumexp(evidence))
    rng = np.random.default_rng(921)
    chosen = rng.choice(2, 4096, p=probabilities)
    samples = means[chosen] + rng.normal(size=(4096, 2)) @ np.linalg.cholesky(covariance).T
    starts = jnp.asarray(np.column_stack((0.2 * labels[chosen], samples))[..., None])
    spec = GuidedPathSpec(method="twisted_particle_agrad", num_particles=8, kappa=1)
    setup = build_smoother_setup(
        target,
        spec,
        num_parameter_particles=2,
        free_mask=target.independent_mask,
        paid_mix=None,
        diagnostic_metrics=(),
    )
    ctx = build_smoother_context(
        setup,
        jnp.asarray(labels[:, None]),
        jnp.zeros(2),
        reference=starts[0],
        delta=jnp.full(3, 0.3),
        trajectory_dtype=starts.dtype,
    )
    guide = target.context(jnp.array([labels.mean()]), target.times)
    paths = jax.jit(
        jax.vmap(lambda path, key: guided_step(target, ctx, guide, spec, key, path).latent_path)
    )(starts, jax.random.split(jax.random.key(42), len(starts)))
    draws = np.asarray(paths[:, 1:, 0])
    np.testing.assert_allclose(draws.mean(0), probabilities @ means, atol=0.025)
    expected_second = covariance + np.einsum("k,ki,kj->ij", probabilities, means, means)
    np.testing.assert_allclose(draws.T @ draws / len(draws), expected_second, atol=0.025)
    scores = jax.jit(jax.vmap(lambda path: ctx.score_labels(path).log_probs))(paths)
    np.testing.assert_allclose(np.exp(scores).mean(0), probabilities, atol=0.025)
    traced = jax.jit(
        jax.vmap(lambda path, key: trace_guided_step(target, ctx, guide, spec, key, path))
    )(starts[:32], jax.random.split(jax.random.key(81), 32))
    picks = np.asarray(traced.trace.selected_particles)
    parents = np.asarray(traced.trace.ancestors)
    assert np.any(picks[:, 1] != parents[np.arange(32), 1, picks[:, 2]])
