"""Independent posterior references for both guided marginalized path updates."""

from dataclasses import replace
from typing import override

import dynestyx as dsx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from dynestyx.models import DiscreteTimeStateEvolution
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, norm

from grassmoth.artifacts.model.construct import replace_constructs
from grassmoth.artifacts.model.expressions import state
from grassmoth.artifacts.model.likelihood import DeltaLawSpec
from grassmoth.artifacts.model.parameter import SiteKind
from grassmoth.artifacts.sampling import (
    AuxiliaryKalmanMethodSpec,
    BlockedPathUpdateSpec,
    ConditionalParameterProposalSpec,
    FixedLatentStepSpec,
    GaussianInitializationSpec,
    GuidedMethodSpec,
    MarginalParticleGibbsSpec,
    ParameterMeanProposalSpec,
    ParameterMixtureProposalSpec,
    PathUpdateSpec,
    SamplerSpec,
    WholePathUpdateSpec,
)
from grassmoth.models.ssm.compile.state_space import ParallelRecurrences, StateSpace
from grassmoth.models.ssm.execution.contracts import MeasurementParams
from grassmoth.models.ssm.execution.dynamical_model import HeterogeneousObservation
from grassmoth.models.ssm.execution.supported_normal import SupportedNormal
from grassmoth.models.ssm.inference.conditioning import compile_exact_state_constraints
from grassmoth.models.ssm.inference.samplers.mpgibbs.kernel import (
    build_marginal_particle_gibbs_kernel,
)
from grassmoth.models.ssm.inference.samplers.mpgibbs.runner import run_marginal_particle_gibbs
from grassmoth.models.ssm.inference.samplers.particle.auxiliary_kalman import (
    auxiliary_kalman_move,
    auxiliary_kalman_step,
    trace_auxiliary_kalman_step,
)
from grassmoth.models.ssm.inference.samplers.particle.context import build_smoother_context
from grassmoth.models.ssm.inference.samplers.particle.guided import guided_step, trace_guided_step
from grassmoth.models.ssm.inference.samplers.particle.setup import build_smoother_setup
from grassmoth.models.ssm.inference.samplers.particle.twist import GaussianTwist, future_information
from grassmoth.models.ssm.inference.targets.particle import ParticleTarget
from grassmoth.models.ssm.inference.tracing import SequentialSmootherTraceRecord, TracedResult
from grassmoth.models.ssm.inference.warmup.initialization import (
    ParticleInitialization,
    prepare_particle_initialization,
)
from grassmoth.models.ssm.inference.warmup.outcomes import InitializationFailure
from grassmoth.models.ssm.structure.observation import DenseLoadings, ObservationStructure
from grassmoth_test_support.compute import EXECUTION_CONTEXT
from grassmoth_test_support.inference import compile_model_fixture
from grassmoth_test_support.interval_particle import initialize_interval_chain, interval_target
from grassmoth_test_support.model import one_state_gaussian_model, without_parameters
from grassmoth_test_support.values import revise_fixture
from packages.grassmoth.tests.models.ssm.inference.test_guided_recurrences import (
    _recurrence_posterior,
)
from packages.grassmoth.tests.models.ssm.inference.test_guided_windows import _window_posterior

pytestmark = pytest.mark.inference(concern="sampling")


@pytest.mark.inference(concern="warmup")
@pytest.mark.parametrize("fixed", [True, False])
@pytest.mark.parametrize("block_length", [None, 1])
def test_auxiliary_kalman_runner_retains_paths_and_freezes_step(fixed, block_length):
    target = interval_target(
        np.arange(2.0), np.array([[0.1], [-0.1]]), np.arange(2.0)[:, None], ("last",)
    )
    initialization = prepare_particle_initialization(
        target,
        spec=GaussianInitializationSpec(),
        key=jax.random.key(77),
        num_chains=1,
        execution_context=EXECUTION_CONTEXT,
    )
    assert isinstance(initialization, ParticleInitialization)
    spec = (
        PathUpdateSpec(
            method=AuxiliaryKalmanMethodSpec(),
            update=WholePathUpdateSpec()
            if (block_length) is None
            else BlockedPathUpdateSpec(block_length=block_length),
            proposal=ParameterMeanProposalSpec()
            if (block_length) is None
            else ConditionalParameterProposalSpec(),
            latent_step=FixedLatentStepSpec(delta=0.1),
        )
        if fixed
        else PathUpdateSpec(
            method=AuxiliaryKalmanMethodSpec(),
            update=WholePathUpdateSpec()
            if (block_length) is None
            else BlockedPathUpdateSpec(block_length=block_length),
            proposal=ParameterMeanProposalSpec()
            if (block_length) is None
            else ConditionalParameterProposalSpec(),
        )
    )
    kernel = build_marginal_particle_gibbs_kernel(
        target, path_update=spec, num_parameter_particles=2
    )
    result = run_marginal_particle_gibbs(
        target,
        kernel=kernel,
        sampler=SamplerSpec(
            num_warmup=0 if fixed else 110,
            num_samples_per_chain=3,
            num_chains=1,
            algorithm=MarginalParticleGibbsSpec(path_update=spec, diagnostic_metrics=()),
        ),
        key=jax.random.key(9),
        initialization=initialization,
        execution_context=replace(EXECUTION_CONTEXT, profile=True),
    )
    assert not isinstance(result, InitializationFailure)
    assert result.retained_draws == 3
    assert np.isfinite(result.latent_paths).all()
    assert np.isfinite(result.chain_extra_fields["path_block_acceptance"]).all()
    assert not any(name.startswith("particle_weight_") for name in result.chain_extra_fields)
    assert result.profile is not None
    assert result.profile.repetitions == 10
    assert result.profile.particle_dimensions is None
    if fixed:
        np.testing.assert_allclose(result.initial_latent_delta, 0.1)
        np.testing.assert_array_equal(result.initial_latent_delta, result.final_latent_delta)
    else:
        assert not np.array_equal(result.initial_latent_delta, result.final_latent_delta)


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


def _correlated_gaussian_case(sample_count=4096, *, independent=False):
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
    correlation = 0.0 if independent else 0.06
    noise = np.array([[0.2, correlation], [correlation, 0.28]])

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
    if independent:
        target = replace(
            target,
            state_space=replace(
                target.state_space, initial_covariance="diagonal", transition_covariance="diagonal"
            ),
        )
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
    selected = rng.choice(len(labels), sample_count, p=label_weights)
    starts = means[selected] + np.einsum(
        "nij,nj->ni", np.linalg.cholesky(covariances[selected]), rng.normal(size=(sample_count, 6))
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
        ("twisted_particle_agrad", *case, False)
        for case in (
            (0.7, 0, False, 3, 1),
            (0.7, 1, True, 3, 1),
            (0.0, 1, False, 1, 1),
            (0.0, 1, False, 3, 2),
        )
    ]
    + [
        pytest.param("particle_mgrad", 0.0, 1, False, 3, 2, True, id="mgrad_mixture_gaussian"),
        pytest.param("particle_mgrad", 0.0, 0, False, 3, 2, True, id="mgrad_mixture_unshifted"),
        pytest.param("particle_mgrad", 0.0, 1, False, 1, 1, True, id="mgrad_mixture_single_time"),
        pytest.param(
            "particle_mgrad", 0.7, 1, True, 3, 1, True, id="mgrad_mixture_nonlinear_pinned"
        ),
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
    if prefix_mixture and method == "particle_mgrad" and dimension == 2:
        target, labels, starts, expected_mean, expected_second, expected_labels = (
            _correlated_gaussian_case()
        )
    spec = PathUpdateSpec(
        method=GuidedMethodSpec(kind=method, num_particles=8, kappa=kappa),
        proposal=ParameterMixtureProposalSpec() if prefix_mixture else ParameterMeanProposalSpec(),
    )
    mask = target.independent_mask
    if pinned:
        mask = mask.at[1, 0].set(False)
    static = build_smoother_setup(
        target,
        num_parameter_particles=2,
        free_mask=mask,
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


@pytest.mark.parametrize(
    ("cubic", "dimension", "pinned"), [(0.7, 1, False), (0.0, 2, False), (0.7, 1, True)]
)
@pytest.mark.parametrize("block_length", [None, 1, 2])
def test_auxiliary_kalman_preserves_complete_parameter_mixture(
    cubic, dimension, pinned, block_length
):
    target, labels, starts, expected_mean, expected_second, expected_labels = _case(
        cubic, dimension=dimension, pinned=pinned
    )
    if block_length is not None and dimension == 2:
        target, labels, starts, expected_mean, expected_second, expected_labels = (
            _correlated_gaussian_case(16384)
        )
    mask = target.independent_mask
    if pinned:
        model = one_state_gaussian_model()
        construct = next(value for value in model.constructs if value.indicators)
        (indicator,) = construct.indicators
        assert indicator.likelihood is not None
        updated = (
            revise_fixture(
                construct,
                indicators=(
                    revise_fixture(
                        indicator,
                        likelihood=revise_fixture(
                            indicator.likelihood, law=DeltaLawSpec(v=state(construct.id))
                        ),
                    ),
                ),
            ),
        )
        parameters, distributions = without_parameters(
            model,
            next(
                parameter
                for parameter in model.parameters
                if model.parameter_context(parameter.id).quantity == SiteKind.MANIFEST_VAR_DIAG
            ),
        )
        model = model.with_entities(
            edges=replace_constructs(model.edges, updated),
            parameters=parameters,
            distributions=distributions,
        )
        constraints = compile_exact_state_constraints(
            compile_model_fixture(model), jnp.full_like(starts[0], jnp.nan).at[1, 0].set(0.2)
        )
        assert constraints is not None
        target = replace(target, exact_constraints=constraints)
        mask = mask & constraints.free_mask
    # This test exercises the shared context and independent smoother, not routing.
    static = build_smoother_setup(
        target,
        num_parameter_particles=2,
        free_mask=mask,
        diagnostic_metrics=(),
    )
    ctx = build_smoother_context(
        static,
        labels,
        jnp.zeros(2),
        reference=starts[0],
        delta=jnp.full((3,), 0.55),
        trajectory_dtype=starts.dtype,
    )
    guide = target.context(jnp.mean(labels, axis=0), target.times)

    def advance(path, key):
        return auxiliary_kalman_step(
            target, ctx, guide, key, path, block_length=block_length
        ).latent_path

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
    # Guard against the identity kernel passing a stationarity test. MH rejection
    # is expected; this is not the eight-particle filter's movement requirement.
    assert np.mean(np.any(draws != np.asarray(starts).reshape(len(starts), -1), axis=-1)) > 0.5
    if pinned:
        np.testing.assert_array_equal(draws[:, 1], np.asarray(starts)[:, 1, 0])
    result = jax.jit(
        lambda: auxiliary_kalman_step(
            target, ctx, guide, jax.random.key(94), starts[0], block_length=block_length
        )
    )()
    for cached, exact in zip(
        result.label_scores, ctx.score_labels(result.latent_path), strict=True
    ):
        np.testing.assert_allclose(cached, exact, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("block_length", [None, 1, 2])
@pytest.mark.parametrize("case", ["windows", "recurrences"])
def test_auxiliary_kalman_replays_memory_and_preserves_joint_posterior(case, block_length):
    """Window memory and nonlinear recurrences are replayed, never proposed.

    Length-1 blocks update in turn; length-2 blocks share colours for the windows,
    whose longest span is two steps, and update in turn for the recurrences.
    """
    target, labels, starts, expected_mean, expected_second, expected_labels = (
        _window_posterior() if case == "windows" else _recurrence_posterior()
    )
    static = build_smoother_setup(
        target,
        num_parameter_particles=2,
        free_mask=target.independent_mask,
        diagnostic_metrics=(),
    )
    ctx = build_smoother_context(
        static,
        jnp.asarray(labels[:, None]),
        jnp.zeros(2),
        reference=starts[0],
        delta=jnp.full((starts.shape[1],), 0.3),
        trajectory_dtype=starts.dtype,
    )
    guide = target.context(jnp.asarray([labels.mean()]), target.times)

    def advance(path, key):
        return auxiliary_kalman_step(
            target, ctx, guide, key, path, block_length=block_length
        ).latent_path

    @jax.jit
    def run(paths):
        def sweep(current, key):
            return jax.vmap(advance)(current, jax.random.split(key, len(current))), None

        return jax.lax.scan(sweep, paths, jax.random.split(jax.random.key(523), 6))[0]

    paths = run(starts)
    draws = np.asarray(paths[..., 0])
    np.testing.assert_allclose(draws.mean(axis=0), expected_mean, atol=0.035)
    np.testing.assert_allclose(draws.T @ draws / len(draws), expected_second, atol=0.035)
    probabilities = jax.jit(jax.vmap(lambda path: jnp.exp(ctx.score_labels(path).log_probs)))(paths)
    np.testing.assert_allclose(np.asarray(probabilities).mean(axis=0), expected_labels, atol=0.025)
    assert np.mean(np.any(draws != np.asarray(starts[..., 0]), axis=-1)) > 0.5
    if case == "recurrences" and block_length is None:
        # Both directions linearize at the guide's replay and score only free coordinates,
        # so the move cannot read whatever recurrence values a path happens to store.
        reference = starts[0]
        auxiliary = reference.at[:, 0].add(0.1)

        def move(path):
            return auxiliary_kalman_move(
                target, ctx, guide, path, auxiliary, jnp.full(path.shape, 0.15), jax.random.key(61)
            )

        exact, stale = (jax.jit(move)(path) for path in (reference, reference.at[:, 1:].add(0.7)))
        np.testing.assert_allclose(stale.proposed[:, 0], exact.proposed[:, 0], atol=1e-6)
        np.testing.assert_allclose(stale.log_acceptance, exact.log_acceptance, atol=1e-5)


class _Stock(DiscreteTimeStateEvolution):
    """x' = 0.7 x + noise and an exact stock E' = E + x."""

    @override
    def __call__(self, x, u, t_now, t_next):
        del u, t_now, t_next
        return SupportedNormal(
            jnp.array([0.7 * x[0], x[1] + x[0]]),
            jnp.diag(jnp.array([0.3, 0.0])),
            (0,),
            covariance_structure="dense",
        )


def _stock_posterior(count):
    """Dense Gaussian posterior when a precise reading of an exact stock couples the path."""
    steps = 6
    times = np.arange(float(steps))
    readings = np.full((steps, 2), np.nan)
    readings[1, 0], readings[3, 0], readings[5, 1] = 0.4, -0.2, 1.1
    noise = np.array([0.2, 0.01])
    base = interval_target(
        times,
        readings,
        np.column_stack((times, times)),
        ("last", "last"),
        state_dim=2,
        covariance=np.diag(noise),
    )

    def declared(bound):
        observation = base.dynamical_model(bound).observation_model
        return dsx.DynamicalModel(
            initial_condition=SupportedNormal(
                jnp.zeros(2), jnp.diag(jnp.array([1.0, 0.0])), (0,), covariance_structure="dense"
            ),
            state_evolution=_Stock(),
            observation_model=HeterogeneousObservation(
                MeasurementParams(
                    jnp.eye(2), observation.measurement.manifest_means, jnp.diag(jnp.asarray(noise))
                ),
                observation.laws,
                structure=ObservationStructure(DenseLoadings(), "dense"),
            ),
            t0=0.0,
        )

    target = replace(
        base,
        dynamical_model=declared,
        state_space=StateSpace(
            2,
            (),
            (0,),
            (0,),
            (),
            (ParallelRecurrences((1,)),),
            initial_covariance="dense",
            transition_covariance="dense",
        ),
    )
    innovations = np.eye(steps)
    for t in range(1, steps):
        innovations[t, :t] = 0.7 * innovations[t - 1, :t]
    prior = innovations @ np.diag([1.0, *([0.3] * (steps - 1))]) @ innovations.T
    # x at t = 1 and 3, and E at t = 5, the sum of x over t < 5.
    design = np.zeros((3, steps))
    design[0, 1] = design[1, 3] = 1.0
    design[2, :5] = 1.0
    values = np.array([0.4, -0.2, 1.1])
    predictive = design @ prior @ design.T + np.diag([0.2, 0.2, 0.01])
    gain = np.linalg.solve(predictive, design @ prior).T
    covariance = prior - gain @ design @ prior
    labels = np.array([-0.3, 0.5])
    means = np.stack([gain @ (values - label) for label in labels])
    evidence = np.array(
        [
            norm.logpdf(label) + multivariate_normal.logpdf(values - label, cov=predictive)
            for label in labels
        ]
    )
    probabilities = np.exp(evidence - logsumexp(evidence))
    expected_second = np.einsum(
        "k,kij->ij", probabilities, covariance[None] + np.einsum("ki,kj->kij", means, means)
    )
    rng = np.random.default_rng(11)
    x = (
        means[rng.choice(2, count, p=probabilities)]
        + rng.normal(size=(count, steps)) @ np.linalg.cholesky(covariance).T
    )
    stock = np.concatenate((np.zeros((count, 1)), np.cumsum(x, axis=1)[:, :-1]), axis=1)
    starts = jnp.asarray(np.stack((x, stock), axis=-1), dtype=jnp.float32)
    return target, labels, starts, probabilities @ means, expected_second, probabilities


def test_blocked_kalman_updates_blocks_coupled_by_a_stock_in_turn():
    """A stock reaches every later block, so blocks must update one at a time.

    Updating same-coloured blocks together here leaves second-moment errors near
    ten standard errors; the absolute errors stay below the usual 0.035 tolerance.
    """
    target, labels, starts, expected_mean, expected_second, expected_labels = _stock_posterior(
        16384
    )
    static = build_smoother_setup(
        target, num_parameter_particles=2, free_mask=target.independent_mask, diagnostic_metrics=()
    )
    ctx = build_smoother_context(
        static,
        jnp.asarray(labels[:, None]),
        jnp.zeros(2),
        reference=starts[0],
        delta=jnp.full((starts.shape[1],), 0.3),
        trajectory_dtype=starts.dtype,
    )
    guide = target.context(jnp.asarray([labels.mean()]), target.times)

    @jax.jit
    def run(paths):
        def sweep(current, key):
            moved = jax.vmap(
                lambda path, draw: (
                    auxiliary_kalman_step(
                        target, ctx, guide, draw, path, block_length=1
                    ).latent_path
                )
            )(current, jax.random.split(key, len(current)))
            return moved, None

        return jax.lax.scan(sweep, paths, jax.random.split(jax.random.key(7), 8))[0]

    paths = run(starts)
    draws = np.asarray(paths[..., 0], dtype=np.float64)
    seconds = np.einsum("ni,nj->nij", draws, draws)
    probabilities = np.asarray(
        jax.jit(jax.vmap(lambda path: jnp.exp(ctx.score_labels(path).log_probs)))(paths),
        dtype=np.float64,
    )
    count = len(draws)
    for samples, expected in (
        (draws, expected_mean),
        (seconds, expected_second),
        (probabilities, expected_labels),
    ):
        standard_errors = samples.std(axis=0) / np.sqrt(count)
        assert np.max(np.abs(samples.mean(axis=0) - expected) / standard_errors) < 4.5
    assert np.mean(np.any(draws != np.asarray(starts[..., 0]), axis=-1)) > 0.9


@pytest.mark.parametrize("block_length", [None, 1, 2])
def test_auxiliary_kalman_gaussian_limit_accepts_the_exact_conditional(block_length):
    target, labels, starts, *_ = _case(0.0, dimension=2)
    labels = jnp.full_like(labels, 0.05)
    static = build_smoother_setup(
        target,
        num_parameter_particles=2,
        free_mask=target.independent_mask,
        diagnostic_metrics=(),
    )
    ctx = build_smoother_context(
        static,
        labels,
        jnp.zeros(2),
        reference=starts[0],
        delta=jnp.full((3,), 0.55),
        trajectory_dtype=starts.dtype,
    )
    guide = target.context(jnp.mean(labels, axis=0), target.times)
    if block_length is not None:
        traced = jax.jit(
            lambda: trace_auxiliary_kalman_step(
                target, ctx, guide, jax.random.key(947), starts[0], block_length=block_length
            )
        )()
        np.testing.assert_allclose(traced.result.diagnostics["path_block_acceptance"], 1.0)
        np.testing.assert_allclose(
            np.asarray(traced.trace.log_acceptance)[np.asarray(traced.trace.accepted)],
            0.0,
            atol=2e-5,
        )
        return
    move = jax.jit(
        lambda: auxiliary_kalman_move(
            target,
            ctx,
            guide,
            starts[0],
            starts[0] + 0.2,
            jnp.full(starts.shape[1:], 0.275),
            jax.random.key(947),
        )
    )()
    np.testing.assert_allclose(move.log_acceptance, 0.0, atol=2e-5)


@pytest.mark.parametrize("method", ["particle_mgrad", "twisted_particle_agrad"])
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
    spec = PathUpdateSpec(method=GuidedMethodSpec(kind=method, num_particles=32, kappa=0))
    labels = jnp.array([[-0.45], [0.55]])
    reference = jnp.zeros((3, 1))
    setup = build_smoother_setup(
        target,
        num_parameter_particles=2,
        free_mask=target.independent_mask,
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


def test_auxiliary_kalman_keeps_mpgibbs_parameter_proposal_and_label_law():
    target, *_ = _case(0.7)
    reference = jnp.array([0.1, 0.2, -0.1, 0.3])
    key = jax.random.key(49)
    traces = []
    for spec in (
        PathUpdateSpec(method=GuidedMethodSpec(kind="particle_mgrad", num_particles=8)),
        PathUpdateSpec(
            method=AuxiliaryKalmanMethodSpec(),
            update=BlockedPathUpdateSpec(block_length=300),
            proposal=ConditionalParameterProposalSpec(),
        ),
    ):
        kernel = build_marginal_particle_gibbs_kernel(
            target, num_parameter_particles=2, path_update=spec
        )
        state = initialize_interval_chain(target, reference, kernel=kernel)
        traced = jax.jit(kernel.trace_step_fn)(state, key)
        plain = jax.jit(kernel.step_fn)(state, key)
        for actual, expected in zip(
            jax.tree.leaves(traced.result), jax.tree.leaves(plain), strict=True
        ):
            np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
        traces.append(traced)
    for left, right in zip(
        jax.tree.leaves(traces[0].trace.parameters),
        jax.tree.leaves(traces[1].trace.parameters),
        strict=True,
    ):
        np.testing.assert_array_equal(left, right)
    result, info = traces[1].result
    candidates = traces[1].trace.parameters.candidates
    expected = jax.nn.log_softmax(
        jax.vmap(
            lambda position: target.log_posterior(
                position, result.latent_trajectory, target.observations, target.times
            )
        )(candidates)
    )
    np.testing.assert_allclose(info["final_label_log_probs"], expected, atol=3e-6)
    assert np.isfinite(result.complete_log_posterior)


@pytest.mark.parametrize("mixture", [False, True])
@pytest.mark.parametrize("pinned", [False, True])
def test_diagonal_guided_geometry_preserves_dense_filter_and_backward_trace(mixture, pinned):
    target, labels, starts, *_ = _correlated_gaussian_case(2, independent=True)
    dense = replace(
        target,
        state_space=replace(
            target.state_space, initial_covariance="dense", transition_covariance="dense"
        ),
    )
    spec = PathUpdateSpec(
        method=GuidedMethodSpec(
            kind="particle_mgrad" if mixture else "twisted_particle_agrad", num_particles=8
        ),
        proposal=ParameterMixtureProposalSpec() if mixture else ParameterMeanProposalSpec(),
    )
    mask = target.independent_mask.at[1, 1].set(not pinned)

    def run(bound):
        setup = build_smoother_setup(
            bound, num_parameter_particles=2, free_mask=mask, diagnostic_metrics=()
        )
        ctx = build_smoother_context(
            setup,
            labels,
            jnp.zeros(2),
            reference=starts[0],
            delta=jnp.broadcast_to(jnp.array([0.12, 0.9]), (3, 2)),
            trajectory_dtype=starts.dtype,
        )
        guide = bound.context(jnp.mean(labels, axis=0), bound.times)
        return jax.jit(
            lambda: trace_guided_step(bound, ctx, guide, spec, jax.random.key(71), starts[0])
        )()

    actual, expected = run(target), run(dense)
    for measured, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        if jnp.issubdtype(measured.dtype, jnp.integer):
            np.testing.assert_array_equal(measured, reference)
        else:
            np.testing.assert_allclose(measured, reference, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("pinned", [False, True])
@pytest.mark.parametrize("kappa", [0, 1])
def test_scalar_tail_preserves_complete_guided_trace(pinned, kappa):
    dense, labels, starts, *_ = _case(0.7)
    target = replace(
        dense,
        state_space=replace(
            dense.state_space, initial_covariance="diagonal", transition_covariance="diagonal"
        ),
    )
    spec = PathUpdateSpec(
        method=GuidedMethodSpec(kind="particle_mgrad", num_particles=8, kappa=kappa),
        proposal=ParameterMixtureProposalSpec(),
    )
    mask = target.independent_mask.at[1, 0].set(not pinned)

    def run(bound):
        setup = build_smoother_setup(
            bound, num_parameter_particles=2, free_mask=mask, diagnostic_metrics=()
        )
        ctx = build_smoother_context(
            setup,
            labels,
            jnp.zeros(2),
            reference=starts[0],
            delta=jnp.full((3, 1), 0.2),
            trajectory_dtype=starts.dtype,
        )
        guide = bound.context(jnp.mean(labels, axis=0), bound.times)
        return jax.jit(
            lambda: trace_guided_step(bound, ctx, guide, spec, jax.random.key(71), starts[0])
        )()

    actual, expected = run(target), run(dense)
    for measured, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        if jnp.issubdtype(measured.dtype, jnp.integer):
            np.testing.assert_array_equal(measured, reference)
        else:
            np.testing.assert_allclose(measured, reference, atol=3e-5, rtol=3e-5)


def test_scalar_tail_zero_mass_with_disabled_gradient_preserves_other_particles():
    from grassmoth.models.ssm.inference.samplers.particle.guided import (
        _killing_noise,
        _killing_reference,
    )
    from grassmoth.models.ssm.inference.samplers.particle.guided_geometry import (
        _DiagonalProposalGeometry,
    )
    from grassmoth.models.ssm.inference.samplers.particle.guided_tail import diagonal_mixture_tail
    from grassmoth.models.ssm.inference.samplers.particle.marginal_mixture import (
        MarginalMixtureBank,
    )
    from grassmoth.models.ssm.inference.samplers.particle.math import label_logsumexp

    x = jnp.array([[0.4], [1.2], [-0.8], [0.9]])
    prior = x * 0.3
    chol = jnp.array([[0.7], [0.8], [1.2], [0.9]])
    geometry = _DiagonalProposalGeometry(
        jnp.zeros_like(x), jnp.zeros_like(x), chol, 1 / chol, jnp.zeros(4)
    )
    bank = MarginalMixtureBank(x, prior, prior, geometry, jnp.zeros(4))
    observed = jnp.array([[0.4, 0.7], [-1.0, 2.0], [-jnp.inf, -jnp.inf], [2.0, -0.3]])
    prior_labels = jnp.log(jnp.array([0.4, 0.6]))
    parent = jnp.broadcast_to(prior_labels, observed.shape)
    variance, mask = jnp.array([0.2]), jnp.array([True])
    noise = _killing_noise(jax.random.key(37), 4, x.dtype)
    actual = jax.jit(
        lambda: diagonal_mixture_tail(
            x,
            prior,
            chol,
            1 / chol,
            jnp.zeros(4),
            observed,
            jnp.zeros((4, 2, 1)),
            parent,
            jnp.zeros_like(observed),
            prior_labels,
            observed[:1],
            jnp.zeros((1, 1)),
            variance[None],
            mask[None],
            jax.tree.map(lambda value: value[None], noise),
            jnp.asarray(0),
            jnp.cumsum(jnp.exp(prior_labels)),
            jnp.full((1, x.shape[0]), 0.37),
            compute_gradient=False,
        )
    )()
    increment = label_logsumexp(parent + observed)
    weights = increment + bank.log_weights(jnp.zeros_like(x), variance, mask)
    scores = parent + observed - jnp.where(jnp.isneginf(increment), 0.0, increment)[:, None]
    np.testing.assert_allclose(actual[0], weights, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(actual[1], scores, rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(actual[2], observed)
    np.testing.assert_array_equal(actual[3], _killing_reference(weights, noise))
    assert np.isfinite(np.asarray(actual[0])[[0, 1, 3]]).all()
    assert np.isneginf(actual[0][2])


def test_shared_backward_points_preserve_fixed_start_diffused_coordinates():
    from grassmoth.models.ssm.inference.targets.sequential import sequential_observations

    times = np.arange(3.0)
    base = interval_target(times, np.array([[0.1], [-0.2], [0.3]]), times[:, None], ("last",))

    def context(position, grid):
        arrays, _, _ = base.context(position, grid)
        return arrays, grid, position

    def declared(bound):
        original = base.dynamical_model(bound)
        initial = SupportedNormal(bound[2][:1], jnp.zeros((1, 1)), (), covariance_structure="dense")
        return eqx.tree_at(lambda model: model.initial_condition, original, initial)

    target = replace(
        base,
        context=context,
        dynamical_model=declared,
        state_space=replace(base.state_space, initial=()),
    )
    labels = jnp.array([[-0.7], [0.4]])
    contexts = jax.vmap(lambda label: target.context(label, target.times))(labels)
    guide = target.context(jnp.array([0.1]), target.times)
    memory = sequential_observations(target, contexts, guide)
    initial = jax.vmap(memory.initial)(jnp.array([[0.0], [0.2], [-0.1]]))
    # Initial states differ by label even though the independently carried row is shared.
    np.testing.assert_allclose(memory.replayed_states(initial[0])[:-1, 0], labels[:, 0])
    current = jax.vmap(memory.advance, in_axes=(0, 0, None))(
        initial, jnp.array([[0.3], [-0.6], [0.8]]), jnp.array(1)
    )
    coordinates = jnp.asarray(target.state_space.transition)
    exact_points = jax.vmap(
        lambda value: jax.vmap(
            lambda context, state: target.project_state(context, state, 1)[coordinates]
        )(contexts, memory.replayed_states(value)[:-1])
    )(current)
    shared = memory.state(current)[..., coordinates]
    np.testing.assert_array_equal(
        exact_points, jnp.broadcast_to(shared[:, None], exact_points.shape)
    )
    normalized = memory.transition_log_probs_with_means(current[0], jnp.array(1), exact_points[0])
    np.testing.assert_allclose(normalized, memory.noise.transitions.log_normalizer[:, 1], atol=1e-7)


@pytest.mark.parametrize(
    ("dimension", "steps", "unroll"),
    [(3, 1, 8), (3, 2, 8), (3, 17, 8), (3, 19, 1), (3, 19, 8), (3, 19, 16), (9, 17, 8), (9, 19, 8)],
)
def test_vector_tail_carries_across_unrolled_scan(dimension, steps, unroll):
    times = np.arange(float(steps))
    readings = np.sin(times * 0.4)[:, None]
    base = interval_target(times, readings, times[:, None], ("last",), state_dim=dimension)

    def context(position, grid):
        arrays, _, _ = base.context(position, grid)
        return arrays, grid, position

    def model(bound):
        original = base.dynamical_model(bound)
        evolution = dsx.LinearGaussianStateEvolution(
            jnp.eye(dimension) * 0.7,
            jnp.eye(dimension) * 0.3 * jnp.exp(bound[2][0]),
            bias=jnp.full((dimension,), 0.2 * bound[2][0]),
        )
        return eqx.tree_at(lambda value: value.state_evolution, original, evolution)

    target = replace(
        base,
        context=context,
        dynamical_model=model,
        state_space=replace(
            base.state_space, initial_covariance="diagonal", transition_covariance="diagonal"
        ),
    )
    # Dense factors carry the same law through the unfused route.
    dense = replace(
        target,
        state_space=replace(
            target.state_space, initial_covariance="dense", transition_covariance="dense"
        ),
    )
    labels = jnp.array([[-0.3], [0.0], [0.4]])
    reference = jnp.broadcast_to(jnp.asarray(readings), (steps, dimension))

    def run(bound, scan_unroll):
        setup = build_smoother_setup(
            bound,
            num_parameter_particles=3,
            free_mask=bound.independent_mask,
            diagnostic_metrics=(),
        )
        ctx = build_smoother_context(
            setup,
            labels,
            jnp.zeros(3),
            reference=reference,
            delta=jnp.full((steps, dimension), 0.2),
            trajectory_dtype=reference.dtype,
        )
        guide = bound.context(jnp.mean(labels, axis=0), bound.times)
        spec = PathUpdateSpec(
            method=GuidedMethodSpec(
                kind="particle_mgrad", num_particles=16, kappa=1, scan_unroll=scan_unroll
            ),
            proposal=ParameterMixtureProposalSpec(),
        )
        return jax.jit(
            lambda: trace_guided_step(bound, ctx, guide, spec, jax.random.key(821), reference)
        )()

    fused = run(target, unroll)
    serial = run(target, 1)
    unfused = run(dense, unroll)
    for reference_result in (serial, unfused):
        for actual, expected in zip(
            jax.tree.leaves(fused), jax.tree.leaves(reference_result), strict=True
        ):
            if jnp.issubdtype(actual.dtype, jnp.integer):
                np.testing.assert_array_equal(actual, expected)
            else:
                np.testing.assert_allclose(actual, expected, atol=5e-5, rtol=5e-5)
