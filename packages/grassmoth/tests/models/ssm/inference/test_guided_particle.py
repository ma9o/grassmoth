"""Independent posterior references for both guided marginalized path updates."""

from dataclasses import replace

import dynestyx as dsx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, norm

from grassmoth.artifacts.sampling import GuidedPathSpec
from grassmoth.models.ssm.inference.samplers.particle.context import build_smoother_context
from grassmoth.models.ssm.inference.samplers.particle.guided import guided_step, trace_guided_step
from grassmoth.models.ssm.inference.samplers.particle.setup import build_smoother_setup
from grassmoth.models.ssm.inference.samplers.particle.twist import GaussianTwist, future_information
from grassmoth.models.ssm.inference.targets.particle import ParticleTarget
from grassmoth.models.ssm.inference.tracing import SequentialSmootherTraceRecord, TracedResult
from grassmoth_test_support.interval_particle import interval_target

pytestmark = pytest.mark.inference(concern="sampling")


def _case(cubic, *, pinned=False, steps=3, dimension=1):
    if dimension == 2:
        return _correlated_gaussian_case()
    times = np.arange(float(steps))
    readings = np.array([0.15, -0.25, 0.45])[:steps]
    base = interval_target(times, readings[:, None], times[:, None], ("last",))

    def context(position, grid):
        arrays, _, _ = base.context(position, grid)
        return arrays, grid, position

    def declared(bound):
        model = base.dynamical_model(bound)
        evolution = dsx.GaussianStateEvolution(
            lambda x, _u, _t0, _t1: 0.65 * x - cubic * jnp.tanh(x) ** 3,
            lambda _x, _u, _t0, _t1: jnp.eye(1) * 0.2 * jnp.exp(2 * bound[2][0]),
        )
        return eqx.tree_at(lambda value: value.state_evolution, model, evolution)

    target = replace(base, context=context, dynamical_model=declared)
    labels = np.array([-0.45, 0.55])
    grid = np.linspace(-3.6, 3.6, 81)
    axes = tuple(np.array([0.2]) if pinned and index == 1 else grid for index in range(steps))
    paths = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, steps)
    log_densities = []
    for theta in labels:
        prior = norm.logpdf(paths[:, 0]) + norm.logpdf(theta)
        means = 0.65 * paths[:, :-1] - cubic * np.tanh(paths[:, :-1]) ** 3
        transitions = norm.logpdf(paths[:, 1:], means, np.sqrt(0.2) * np.exp(theta)).sum(axis=1)
        observed = norm.logpdf(readings, paths + theta, np.sqrt(0.2)).sum(axis=1)
        log_densities.append(prior + transitions + observed)
    log_densities = np.stack(log_densities, axis=-1)
    normalizer = logsumexp(log_densities)
    weights = np.exp(logsumexp(log_densities, axis=-1) - normalizer)
    means = weights @ paths
    seconds = np.einsum("n,ni,nj->ij", weights, paths, paths)
    label_weights = np.exp(logsumexp(log_densities, axis=0) - normalizer)
    starts = paths[np.random.default_rng(732).choice(len(paths), 4096, p=weights)]
    return (
        target,
        jnp.asarray(labels[:, None]),
        jnp.asarray(starts[..., None]),
        means,
        seconds,
        label_weights,
    )


def _correlated_gaussian_case():
    """Dense Gaussian conditioning, independent of the particle proposal algebra."""
    steps, dimension = 3, 2
    times = np.arange(float(steps))
    readings = np.array([[0.15, -0.1], [-0.25, 0.3], [0.45, 0.2]])
    base = interval_target(
        times,
        readings,
        np.broadcast_to(times[:, None], readings.shape),
        ("last", "last"),
        state_dim=dimension,
    )
    matrix = np.array([[0.7, 0.25], [-0.1, 0.6]])
    noise = np.array([[0.2, 0.06], [0.06, 0.28]])

    def context(position, grid):
        arrays, _, _ = base.context(position, grid)
        return arrays, grid, position

    def declared(bound):
        evolution = dsx.LinearGaussianStateEvolution(
            jnp.asarray(matrix), jnp.asarray(noise) * jnp.exp(2 * bound[2][0])
        )
        return eqx.tree_at(
            lambda value: value.state_evolution, base.dynamical_model(bound), evolution
        )

    target = replace(base, context=context, dynamical_model=declared)
    labels = np.array([-0.45, 0.55])
    transform = np.eye(steps * dimension)
    for index in range(1, steps):
        transform[2 * index : 2 * index + 2, : 2 * index] = (
            matrix @ transform[2 * index - 2 : 2 * index, : 2 * index]
        )
    design = np.kron(np.eye(steps), np.full((2, 2), 0.5))
    means, covariances, evidence = [], [], []
    for theta in labels:
        innovations = np.zeros((6, 6))
        innovations[:2, :2] = np.eye(2)
        for index in range(1, steps):
            innovations[2 * index : 2 * index + 2, 2 * index : 2 * index + 2] = noise * np.exp(
                2 * theta
            )
        prior = transform @ innovations @ transform.T
        observed = design @ prior @ design.T + np.eye(6) * 0.2
        gain = np.linalg.solve(observed, design @ prior).T
        means.append(gain @ (readings.ravel() - theta))
        covariances.append(prior - gain @ design @ prior)
        evidence.append(
            norm.logpdf(theta) + multivariate_normal.logpdf(readings.ravel() - theta, cov=observed)
        )
    label_weights = np.exp(evidence - logsumexp(evidence))
    means, covariances = np.asarray(means), np.asarray(covariances)
    rng = np.random.default_rng(732)
    selected = rng.choice(len(labels), 4096, p=label_weights)
    starts = means[selected] + np.einsum(
        "nij,nj->ni", np.linalg.cholesky(covariances[selected]), rng.normal(size=(4096, 6))
    )
    seconds = np.einsum(
        "k,kij->ij", label_weights, covariances + np.einsum("ki,kj->kij", means, means)
    )
    return (
        target,
        jnp.asarray(labels[:, None]),
        jnp.asarray(starts.reshape(-1, steps, dimension)),
        label_weights @ means,
        seconds,
        label_weights,
    )


@pytest.mark.parametrize(
    ("method", "cubic", "kappa", "pinned", "steps", "dimension", "prefix_mixture"),
    [
        (method, *case, False)
        for method in ("particle_agrad", "twisted_particle_agrad")
        for case in (
            (0.7, 0, False, 3, 1),
            (0.7, 1, True, 3, 1),
            (0.0, 1, False, 1, 1),
            (0.0, 1, False, 3, 2),
        )
    ]
    + [
        ("particle_mgrad", 0.0, 0, False, 3, 2, False),
        ("particle_mgrad", 0.7, 1, True, 3, 1, False),
        pytest.param(
            "twisted_particle_agrad", 0.0, 1, False, 3, 2, True, id="prefix_mixture_gaussian"
        ),
        pytest.param(
            "twisted_particle_agrad", 0.7, 1, True, 3, 1, True, id="prefix_mixture_nonlinear_pinned"
        ),
        pytest.param(
            "twisted_particle_agrad", 0.0, 1, False, 1, 1, True, id="prefix_mixture_single_time"
        ),
    ],
)
def test_guided_updates_preserve_path_and_label_posterior(
    method, cubic, kappa, pinned, steps, dimension, prefix_mixture
):
    target, labels, starts, expected_mean, expected_second, expected_labels = _case(
        cubic, pinned=pinned, steps=steps, dimension=dimension
    )
    spec = GuidedPathSpec(
        method=method, num_particles=8, kappa=kappa, prefix_mixture=prefix_mixture
    )
    mask = target.independent_mask
    if pinned:
        mask = mask.at[1, 0].set(False)
    static = build_smoother_setup(
        target,
        spec,
        num_parameter_particles=2,
        free_mask=mask,
        paid_mix=None,
        diagnostic_metrics=("particle_weights",),
    )
    ctx = build_smoother_context(
        static,
        labels,
        jnp.zeros(2),
        reference=starts[0],
        delta=(
            jnp.broadcast_to(jnp.array([0.12, 0.9]), (steps, dimension))
            if dimension == 2 and kappa == 1
            else jnp.full((steps,), 0.55)
        ),
        trajectory_dtype=starts.dtype,
    )
    guide = target.context(jnp.mean(labels, axis=0), target.times)

    def advance(path, key):
        result = guided_step(target, ctx, guide, spec, key, path)
        assert not isinstance(result, TracedResult)
        return result.latent_path

    @jax.jit
    def run(paths):
        def sweep(current, key):
            return jax.vmap(advance)(current, jax.random.split(key, len(current))), None

        return jax.lax.scan(sweep, paths, jax.random.split(jax.random.key(417), 8))[0]

    paths = run(starts)
    draws = np.asarray(paths).reshape(len(starts), -1)
    np.testing.assert_allclose(draws.mean(axis=0), expected_mean, atol=0.04)
    np.testing.assert_allclose(
        np.einsum("ni,nj->ij", draws, draws) / len(draws), expected_second, atol=0.04
    )
    probabilities = jax.jit(jax.vmap(lambda path: jnp.exp(ctx.score_labels(path).log_probs)))(paths)
    np.testing.assert_allclose(np.asarray(probabilities).mean(axis=0), expected_labels, atol=0.025)
    assert np.mean(np.any(draws != np.asarray(starts).reshape(len(starts), -1), axis=-1)) > 0.9
    if pinned:
        np.testing.assert_array_equal(draws[:, 1], np.asarray(starts)[:, 1, 0])
    traced = jax.jit(
        lambda path: trace_guided_step(target, ctx, guide, spec, jax.random.key(94), path)
    )(starts[0])
    assert isinstance(traced, TracedResult)
    assert isinstance(traced.trace, SequentialSmootherTraceRecord)
    assert traced.trace.filter_log_weights.shape == (steps, 8)
    assert np.isfinite(traced.result.diagnostics["particle_weight_ess"]).all()
    rescored = ctx.score_labels(traced.result.latent_path)
    for cached, exact in zip(traced.result.label_scores, rescored, strict=True):
        np.testing.assert_allclose(cached, exact, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(
        traced.result.latent_path, jax.jit(advance)(starts[0], jax.random.key(94)), atol=1e-7
    )


@pytest.mark.parametrize("method", ["particle_agrad", "twisted_particle_agrad"])
def test_guided_filter_keeps_zero_mass_candidates_out_of_selected_path(monkeypatch, method):
    def bounded_observation(self, context, state, index, observations):
        return dist.Uniform(state[0] - 0.35, state[0] + 0.35, validate_args=True).log_prob(
            observations[index, 0]
        )

    monkeypatch.setattr(ParticleTarget, "observation_increment", bounded_observation)
    monkeypatch.setattr(
        ParticleTarget,
        "observation_increment_with_noise",
        lambda self, context, _noise, state, index, _log_precision=None: bounded_observation(
            self, context, state, index, self.observations
        ),
    )
    times = np.arange(3.0)
    target = interval_target(times, np.zeros((3, 1)), times[:, None], ("last",))
    spec = GuidedPathSpec(method=method, num_particles=32, kappa=0)
    labels = jnp.array([[-0.45], [0.55]])
    reference = jnp.zeros((3, 1))
    setup = build_smoother_setup(
        target,
        spec,
        num_parameter_particles=2,
        free_mask=target.independent_mask,
        paid_mix=None,
        diagnostic_metrics=("particle_weights",),
    )
    ctx = build_smoother_context(
        setup,
        labels,
        jnp.zeros(2),
        reference=reference,
        delta=jnp.full((3,), 2.0),
        trajectory_dtype=reference.dtype,
    )
    guide = target.context(jnp.mean(labels, axis=0), target.times)
    traced = jax.jit(
        lambda: trace_guided_step(target, ctx, guide, spec, jax.random.key(28), reference)
    )()
    assert np.isneginf(traced.trace.filter_log_weights).any()
    assert not np.isnan(traced.trace.filter_log_weights).any()
    assert np.all(np.abs(traced.result.latent_path) <= 0.35)
    for cached, exact in zip(
        traced.result.label_scores, ctx.score_labels(traced.result.latent_path), strict=True
    ):
        np.testing.assert_allclose(cached, exact, rtol=2e-6, atol=2e-6)


def test_parallel_twist_matches_dense_correlated_gaussian_future_law():
    matrices = np.array(
        [[[0.7, 0.2], [-0.1, 0.8]], [[0.5, -0.3], [0.4, 0.6]], [[0.8, 0.1], [0.0, 0.9]]]
    )
    offsets = np.array([[0.1, -0.2], [-0.3, 0.1], [0.2, 0.4]])
    covariance = np.array([[0.3, 0.12], [0.12, 0.2]])
    covariances = np.repeat(covariance[None], 3, axis=0)
    auxiliary = np.array([[0.2, -0.1], [0.4, 0.5], [-0.2, 0.7], [0.6, -0.3]])
    variance = np.array([0.2, 0.4, 0.3, 0.5])
    mask = np.array([[True, True], [True, False], [True, True], [False, True]])
    twist = jax.jit(future_information)(
        jnp.asarray(matrices),
        jnp.asarray(offsets),
        jnp.asarray(covariances),
        GaussianTwist(
            jax.vmap(jnp.diag)(jnp.asarray(mask / variance[:, None])),
            jnp.asarray(np.where(mask, auxiliary / variance[:, None], 0.0)),
        ),
    )
    # Assemble the full future observation law independently of the information scan.
    for index in range(3):
        length = 3 - index
        mapping = np.zeros((2 * length, 2))
        noise = np.zeros((2 * length, 2 * length))
        shift = np.zeros(2 * length)
        previous_map, previous_noise, previous_shift = (
            np.eye(2),
            np.zeros((2, 2 * length)),
            np.zeros(2),
        )
        for row, time in enumerate(range(index, 3)):
            part = slice(2 * row, 2 * row + 2)
            mapping[part] = matrices[time] @ previous_map
            noise[part] = matrices[time] @ previous_noise
            noise[part, part] = np.linalg.cholesky(covariance)
            shift[part] = matrices[time] @ previous_shift + offsets[time]
            previous_map, previous_noise, previous_shift = mapping[part], noise[part], shift[part]
        full_covariance = noise @ noise.T + np.diag(np.repeat(variance[index + 1 :], 2))
        active = mask[index + 1 :].ravel()
        observed = auxiliary[index + 1 :].ravel()[active]
        joint_covariance = full_covariance[np.ix_(active, active)]
        base = multivariate_normal.logpdf(observed, shift[active], joint_covariance)
        for value in (np.array([0.2, -0.4]), np.array([-0.7, 0.5])):
            expected = (
                multivariate_normal.logpdf(
                    observed, (mapping @ value + shift)[active], joint_covariance
                )
                - base
            )
            np.testing.assert_allclose(
                twist.log_density(jnp.asarray(value), index), expected, atol=2e-6
            )
