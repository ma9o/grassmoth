from dataclasses import replace

import dynestyx as dsx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from grassmoth.artifacts.joint_parameterization import MixedParameterizationRecord
from grassmoth.artifacts.probability.families import DistributionFamily
from grassmoth.models.ssm.execution.contracts import MeasurementParams
from grassmoth.models.ssm.execution.dynamical_model import HeterogeneousObservation
from grassmoth.models.ssm.execution.innovations import (
    GaussianPathMap,
    conditional_gaussian_step,
)
from grassmoth.models.ssm.inference.conditioning import compile_exact_state_constraints
from grassmoth.models.ssm.inference.joint_coordinates import build_joint_coordinates
from grassmoth.models.ssm.inference.parameter_transform import ParameterTransform
from grassmoth.models.ssm.inference.targets.gaussian_reference import GaussianReference
from grassmoth_test_support.compute import compile_program
from grassmoth_test_support.interval_particle import gaussian_interval_case, interval_target
from grassmoth_test_support.observation import observation_laws
from packages.grassmoth.tests.models.ssm.compile.test_exact_state_space import _model, _problem

"""Exact parameterization identities and decoded analytic posteriors, without fitting a study."""


pytestmark = pytest.mark.inference(concern="sampling")
PARAMETERIZATIONS = ("centred", "non_centred", "mixed")


@pytest.mark.parametrize("parameterization_name", PARAMETERIZATIONS)
def test_runtime_charts_preserve_each_fits_scientific_start_and_density(parameterization_name):
    target = _correlated_target()
    first = build_joint_coordinates(target, parameterization_name, compiler=compile_program)
    other_target = replace(
        target, parameters=replace(target.parameters, initial_position=jnp.array([0.7]))
    )
    second = build_joint_coordinates(other_target, parameterization_name, compiler=compile_program)
    dynamic = jax.jit(jax.value_and_grad(lambda point, chart: first.bind(chart).log_density(point)))
    for coordinates in (first, second):
        point = jnp.linspace(-0.2, 0.3, coordinates.dimension)
        expected = jax.jit(jax.value_and_grad(coordinates.log_density))(point)
        measured = dynamic(point, coordinates.chart)
        for actual, reference in zip(measured, expected, strict=True):
            np.testing.assert_allclose(actual, reference, atol=3e-6, rtol=3e-6)
        bound = first.bind(coordinates.chart)
        np.testing.assert_array_equal(bound.parameter_center, coordinates.parameter_center)
        assert bound.record == coordinates.record
        centred = build_joint_coordinates(coordinates.target, compiler=compile_program)
        start = centred.decode(jnp.zeros(centred.dimension))
        np.testing.assert_array_equal(bound.encode(*start), coordinates.encode(*start))


def _direct_constraints(value):
    from grassmoth.artifacts.identity import MechanismId
    from grassmoth.artifacts.model.construct import replace_constructs
    from grassmoth.artifacts.model.expressions import LiteralExpression, coefficient, state
    from grassmoth.artifacts.model.likelihood import (
        DeltaLawSpec,
        LikelihoodSpec,
        NormalLawSpec,
    )
    from grassmoth.artifacts.model.mechanism import DriftMechanismSpec
    from grassmoth_test_support.helpers import make_model
    from grassmoth_test_support.inference import compile_fit_fixture

    model = make_model(["x", "y"], [("x", "y")])
    constructs = tuple(
        c.revised(
            coefficients=(
                coefficient(0.0, role="initial_mean"),
                coefficient(1.0, role="initial_scale"),
                coefficient(0.4, role="diffusion_scale"),
            ),
            dynamics=(
                DriftMechanismSpec(
                    id=MechanismId("mechanism:self_" + c.name),
                    expression=LiteralExpression(value=0.0),
                ),
            ),
            indicators=tuple(
                i.revised(
                    observation=i.observation.revised(aggregation="last"),
                    likelihood=LikelihoodSpec(
                        law=DeltaLawSpec(v=state(c.id))
                        if c.name == "x"
                        else NormalLawSpec(
                            loc=state(c.id), scale=coefficient(0.2, role="observation_scale")
                        ),
                        standardized=c.name != "x",
                        reasoning="Analytic control instrument",
                    ),
                )
                for i in c.indicators
            ),
        )
        for c in model.constructs
    )
    edges = tuple(
        e.revised(
            mechanisms=(
                DriftMechanismSpec(id=MechanismId("mechanism:cross"), expression=state(e.cause.id)),
            )
        )
        for e in replace_constructs(model.edges, constructs)
    )
    compiled = compile_fit_fixture(model.with_entities(edges=edges)).compiled_dynamical_model
    assert compiled.states[0].name == "x"
    channel = next(
        i for i, o in enumerate(compiled.observations) if isinstance(o.law, DeltaLawSpec)
    )
    observations = jnp.full((4, len(compiled.observations)), jnp.nan).at[1, channel].set(value)
    return compile_exact_state_constraints(compiled, observations)


def _correlated_target(pinned=0.3):
    families = (DistributionFamily.DELTA, DistributionFamily.GAUSSIAN, DistributionFamily.GAUSSIAN)
    observed = np.array(
        [
            [np.nan, 0.1, np.nan],
            [pinned, np.nan, np.nan],
            [np.nan, -0.2, np.nan],
            [np.nan, np.nan, 0.2],
        ]
    )
    starts = np.full_like(observed, np.nan)
    starts[-1, -1] = 0.0
    target = interval_target(
        np.arange(4.0), observed, starts, ("last", "last", "mean"), families=families, state_dim=2
    )
    laws = observation_laws(families)

    def model(position):
        scale = jnp.exp(position[0])
        rho = 0.6 * jnp.tanh(position[0] + 0.5)
        initial_chol = jnp.array([[scale, 0.0], [rho * scale, 0.7]])
        step_chol = jnp.array([[0.4 * scale, 0.0], [0.3 * rho, 0.5 * scale]])
        return dsx.DynamicalModel(
            initial_condition=dist.MultivariateNormal(
                jnp.array([0.2, -0.3]), scale_tril=initial_chol
            ),
            state_evolution=dsx.LinearGaussianStateEvolution(
                jnp.array([[0.8, 0.1], [0.0, 0.7]]), step_chol @ step_chol.T
            ),
            observation_model=HeterogeneousObservation(
                MeasurementParams(
                    jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
                    jnp.zeros(3),
                    jnp.eye(3) * 0.2,
                ),
                laws,
            ),
            t0=0.0,
        )

    _, static = eqx.partition(model(jnp.zeros(1)), eqx.is_array)

    def context(position, times):
        return eqx.filter(model(position), eqx.is_array), times, jnp.zeros(2)

    def declared(context):
        return eqx.combine(context[0], static)

    assert target.intervals is not None
    return replace(
        target,
        context=context,
        dynamical_model=declared,
        parameters=ParameterTransform(
            jnp.array([-0.2]),
            lambda p: {"log_sigma": p[0]},
            lambda p: {"sigma": jnp.exp(p[0])},
            lambda p: dist.Normal(0.0, 1.0).log_prob(p[0]),
        ),
        intervals=replace(target.intervals, dynamical_model=declared),
        exact_constraints=_direct_constraints(pinned),
        point_reading_mask=jnp.array([[False, True], [True, False], [False, True], [False, False]]),
    )


@pytest.mark.parametrize("parameterization_name", PARAMETERIZATIONS)
def test_correlated_parameter_dependent_parameterization_keeps_exact_and_interval_factors(
    parameterization_name,
):
    target = _correlated_target()
    parameterization = build_joint_coordinates(
        target, parameterization_name, compiler=compile_program
    )
    point = jnp.linspace(-0.3, 0.4, parameterization.dimension)

    def scientific(q):
        parameters, path = parameterization.decode(q)
        return jnp.concatenate((parameters, path.ravel()[parameterization.free_indices]))

    parameters, path = parameterization.decode(point)
    assert path[1, 0] == 0.3
    assert parameterization.dimension == 8  # one parameter plus eight states minus the pinned state
    np.testing.assert_allclose(parameterization.encode(parameters, path), point, atol=2e-6)
    _, jacobian = jnp.linalg.slogdet(jax.jacfwd(scientific)(point))
    expected = target.log_posterior(parameters, path, target.observations, target.times) + jacobian
    value, gradient = jax.jit(jax.value_and_grad(parameterization.log_density))(point)
    np.testing.assert_allclose(value, expected, rtol=3e-6, atol=3e-6)
    direction = jnp.linspace(0.4, -0.2, point.size)
    h = 1e-3
    finite_difference = (
        parameterization.log_density(point + h * direction)
        - parameterization.log_density(point - h * direction)
    ) / (2 * h)
    np.testing.assert_allclose(gradient @ direction, finite_difference, rtol=3e-3, atol=2e-3)
    if parameterization_name == "mixed":
        record = parameterization.record
        assert isinstance(record, MixedParameterizationRecord)
        assert record.centred_mask == ((False, True), (False, False), (False, True), (False, False))
        assert not record.equals_non_centred
    # Conditional dynamics at a pinned reading remain in the transformed target.
    changed = _correlated_target(0.8)
    assert (
        abs(
            float(
                build_joint_coordinates(
                    changed, parameterization_name, compiler=compile_program
                ).log_density(point)
                - value
            )
        )
        > 0.01
    )


def test_conditional_gaussian_keeps_centred_marginal_and_noise_normalization():
    def scores(theta):
        chol = jnp.array([[jnp.exp(theta), 0.0], [0.4 * jnp.exp(theta), 0.7]])
        covariance = chol @ chol.T
        values = jnp.array([0.6, -0.2])  # first coordinate is pinned/centred; second is epsilon
        step = conditional_gaussian_step(
            jnp.array([0.1, -0.3]),
            covariance,
            density_mask=jnp.array([True, True]),
            innovation_mask=jnp.array([False, True]),
            values=values,
        )
        state = step.decode(values)
        transformed = (
            dist.MultivariateNormal(jnp.array([0.1, -0.3]), scale_tril=chol).log_prob(state)
            + step.log_abs_det
        )
        expected = dist.Normal(0.1, jnp.exp(theta)).log_prob(values[0]) + dist.Normal(
            0.0, 1.0
        ).log_prob(values[1])
        return transformed, expected

    actual, expected = scores(jnp.array(-0.4))
    np.testing.assert_allclose(actual, expected, rtol=2e-6)
    derivatives = jax.jacfwd(scores)(jnp.array(-0.4))
    np.testing.assert_allclose(*derivatives, rtol=2e-6)


@pytest.mark.parametrize("parameterization_name", PARAMETERIZATIONS)
def test_static_and_accumulator_replay_have_no_extra_noise_or_centred_drivers(
    parameterization_name,
):
    compiled, problem = _problem(_model())
    indices = {state.name: i for i, state in enumerate(compiled.states)}
    total_column = next(
        i for i, obs in enumerate(compiled.observations) if obs.state_index == indices["total"]
    )
    observed = problem.runtime.observations.at[2, total_column].set(0.2)
    _, problem = _problem(_model(), observed)
    target = problem.runtime
    parameterization = build_joint_coordinates(
        target, parameterization_name, compiler=compile_program
    )
    point = jnp.linspace(-0.2, 0.3, parameterization.dimension)
    parameters, path = parameterization.decode(point)
    np.testing.assert_allclose(parameterization.encode(parameters, path), point, atol=2e-6)
    np.testing.assert_allclose(
        path[1:, indices["total"]], jnp.cumsum(path[:-1, indices["driver"]]), atol=2e-6
    )
    np.testing.assert_array_equal(
        path[:, indices["baseline"]], jnp.repeat(path[0, indices["baseline"]], 4)
    )
    assert not np.asarray(parameterization.free_mask)[
        :, [indices["total"], indices["baseline"]]
    ].any()
    if parameterization_name == "mixed":
        record = parameterization.record
        assert isinstance(record, MixedParameterizationRecord)
        assert record.equals_non_centred
    context = target.context(parameters, target.times)
    np.testing.assert_allclose(path, target.replay_path(context, path), atol=2e-6)
    assert np.isfinite(jax.grad(parameterization.log_density)(point)).all()


def _point_gaussian_case():
    values = np.array([0.2, np.nan, -0.4, 0.7])
    target = interval_target(np.arange(4.0), values[:, None], np.full((4, 1), np.nan), ("last",))
    differences = np.eye(4) - np.diag(np.full(3, 0.7), -1)
    precision = np.zeros((5, 5))
    precision[0, 0] = 1.0
    precision[1:, 1:] = differences.T @ np.diag([1.0, 1 / 0.3, 1 / 0.3, 1 / 0.3]) @ differences
    design = np.column_stack((np.ones(3), np.eye(4)[[0, 2, 3]]))
    covariance = np.linalg.inv(precision + design.T @ design / 0.2)
    mean = covariance @ design.T @ values[[0, 2, 3]] / 0.2
    return target, mean, covariance


@pytest.mark.parametrize("fixture", [_point_gaussian_case, gaussian_interval_case])
@pytest.mark.parametrize("parameterization_name", PARAMETERIZATIONS)
def test_decoded_analytic_posterior_matches_in_all_parameterizations(
    fixture, parameterization_name
):
    target, expected_mean, expected_covariance = fixture()
    parameterization = build_joint_coordinates(
        target, parameterization_name, compiler=compile_program
    )
    zero = jnp.zeros(parameterization.dimension)
    # This control is Gaussian in each affine parameterization: integration is analytic,
    # without a sampler, adaptation, stochastic tolerance or production fit.
    precision = -jax.hessian(parameterization.log_density)(zero)
    covariance = jnp.linalg.inv(precision)
    mean = covariance @ jax.grad(parameterization.log_density)(zero)

    def scientific(q):
        parameters, path = parameterization.decode(q)
        return jnp.concatenate((parameters, path[:, 0]))

    derivative = jax.jacfwd(scientific)(mean)
    np.testing.assert_allclose(scientific(mean), expected_mean, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(
        derivative @ covariance @ derivative.T, expected_covariance, rtol=2e-5, atol=2e-5
    )
    if fixture == gaussian_interval_case and parameterization_name == "mixed":
        record = parameterization.record
        assert isinstance(record, MixedParameterizationRecord)
        assert record.equals_non_centred
    if parameterization_name == "non_centred":
        parameters, path = parameterization.decode(zero)
        context = target.context(parameters, target.times)
        expected = (
            target.log_prior(parameters)
            + dist.Normal(0.0, 1.0).log_prob(zero[1:]).sum()
            + target.observation_log_probs(context, path, target.observations).sum()
        )
        np.testing.assert_allclose(parameterization.log_density(zero), expected, atol=3e-6)


@pytest.mark.parametrize("parameterization_name", ["non_centred", "mixed"])
@pytest.mark.parametrize("beta", [0.0, 0.35, 1.0])
def test_scientific_tempering_reference_gets_the_same_parameterization_jacobian(
    parameterization_name, beta
):
    target = _correlated_target()
    affine = build_joint_coordinates(target, compiler=compile_program)
    _, anchor = affine.decode(jnp.zeros(affine.dimension))
    reference = GaussianReference(
        jnp.zeros(1),
        jnp.eye(1),
        anchor,
        jnp.zeros((4, 2, 2)),
        jnp.broadcast_to(jnp.eye(2), (4, 2, 2)),
        affine.free_mask,
    )
    target = target.at_temperature(reference, jnp.asarray(beta))
    parameterization = build_joint_coordinates(
        target, parameterization_name, compiler=compile_program
    )
    point = jnp.linspace(-0.2, 0.3, parameterization.dimension)

    def scientific(q):
        parameters, path = parameterization.decode(q)
        return jnp.concatenate((parameters, path.ravel()[parameterization.free_indices]))

    parameters, path = parameterization.decode(point)
    _, logdet = jnp.linalg.slogdet(jax.jacfwd(scientific)(point))
    np.testing.assert_allclose(
        parameterization.log_density(point),
        target.log_posterior(parameters, path, target.observations, target.times) + logdet,
        rtol=3e-6,
        atol=3e-6,
    )


# Every pair of coordinate system, temperature and support remains covered.
@pytest.mark.parametrize(
    ("parameterization_name", "beta", "support"),
    [
        ("non_centred", 0.0, "pinned_interval"),
        ("mixed", 0.0, "static_replay"),
        ("non_centred", 0.35, "static_replay"),
        ("mixed", 0.35, "pinned_interval"),
        ("non_centred", 1.0, "pinned_interval"),
        ("mixed", 1.0, "static_replay"),
    ],
)
def test_direct_scoring_identity_preserves_support_and_tempering(
    parameterization_name, beta, support
):
    target = _correlated_target() if support == "pinned_interval" else _problem(_model())[1].runtime
    affine = build_joint_coordinates(target, compiler=compile_program)
    _, anchor = affine.decode(jnp.zeros(affine.dimension))
    reference = GaussianReference(
        target.initial_position,
        jnp.eye(target.initial_position.size),
        anchor,
        jnp.zeros((*anchor.shape, anchor.shape[-1])),
        jnp.broadcast_to(jnp.eye(anchor.shape[-1]), (*anchor.shape, anchor.shape[-1])),
        affine.free_mask,
    )
    target = target.at_temperature(reference, jnp.asarray(beta))
    coordinates = build_joint_coordinates(target, parameterization_name, compiler=compile_program)
    point = jnp.linspace(-0.2, 0.3, coordinates.dimension)

    def scientific_density(native):
        parameters, values = coordinates._unpack(native)
        context = target.context(parameters, target.times)
        path, logdet = coordinates._path(context, values)
        return target.log_posterior(parameters, path, target.observations, target.times) + logdet

    measured = jax.jit(jax.value_and_grad(coordinates.log_density))(point)
    expected = jax.jit(jax.value_and_grad(scientific_density))(point)
    for actual, reference_value in zip(measured, expected, strict=True):
        np.testing.assert_allclose(actual, reference_value, rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize("centred_initial", [False, True])
def test_tiny_diffusion_scores_native_innovations_without_subtracting_rounded_states(
    centred_initial,
):
    sigma = jnp.asarray(1e-9, dtype=jnp.float32)
    innovation = jnp.array([[not centred_initial], [True], [True]])
    mapping = GaussianPathMap(
        jnp.array([10_000.0]),
        lambda previous, _index: previous,
        jnp.array([[[1.0]], [[sigma**2]], [[sigma**2]]]),
        jnp.ones((3, 1), dtype=bool),
        jnp.ones((3, 1), dtype=bool),
        innovation,
        jnp.full((3, 1), jnp.nan),
    )
    coordinates = jnp.array([[10_000.0 if centred_initial else 1.0], [1.0], [1.0]])

    def score(values):
        path, _ = mapping.decode(values)
        return mapping.transformed_log_prob(values, path)

    value, gradient = jax.jit(jax.value_and_grad(score))(coordinates)
    path, _ = mapping.decode(coordinates)
    assert float(path[-1, 0] - path[-2, 0]) == 0.0  # the old residual loses epsilon
    expected = -0.5 * (2.0 if centred_initial else 3.0) - 1.5 * jnp.log(2 * jnp.pi)
    np.testing.assert_allclose(value, expected, atol=1e-6)
    np.testing.assert_array_equal(gradient[1:], -jnp.ones((2, 1)))


def test_direct_scoring_keeps_correlated_static_initial_marginal_and_its_gradient():
    coordinates = jnp.array([[0.0, 0.7], [0.0, -0.4]])

    def scores(static):
        mapping = GaussianPathMap(
            jnp.array([0.1, -0.2]),
            lambda previous, _index: jnp.array([static, jnp.sin(previous[1]) + static]),
            jnp.array([[[4.0, 1.0], [1.0, 2.0]], [[0.0, 0.0], [0.0, 0.25]]]),
            jnp.array([[True, True], [False, True]]),
            jnp.array([[False, True], [False, True]]),
            jnp.array([[False, True], [False, True]]),
            jnp.array([[static, jnp.nan], [static, jnp.nan]]),
        )
        path, logdet = mapping.decode(coordinates)
        transformed = mapping.transformed_log_prob(coordinates, path)
        expected = dist.Normal(0.1, 2.0).log_prob(static) + jnp.sum(
            dist.Normal(0.0, 1.0).log_prob(coordinates[:, 1])
        )
        scientific = (
            dist.MultivariateNormal(mapping.initial_mean, mapping.covariances[0]).log_prob(path[0])
            + dist.Normal(jnp.sin(path[0, 1]) + static, 0.5).log_prob(path[1, 1])
            + logdet
        )
        return jnp.stack((transformed, expected, scientific))

    values = jax.jit(scores)(jnp.asarray(0.8))
    derivatives = jax.jit(jax.jacfwd(scores))(jnp.asarray(0.8))
    np.testing.assert_allclose(values, jnp.repeat(values[1], 3), rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(derivatives, jnp.repeat(derivatives[1], 3), rtol=3e-6, atol=3e-6)


def test_compiler_mask_tracks_available_direct_multi_state_readings_only():
    from grassmoth.artifacts.model.construct import replace_constructs
    from grassmoth.artifacts.model.expressions import coefficient, state
    from grassmoth.models.ssm.compile.point_readings import compile_point_reading_mask
    from grassmoth_test_support.inference import compile_fit_fixture

    model = _model()
    driver = next(c for c in model.constructs if c.name == "driver")
    outcome = next(c for c in model.constructs if c.name == "outcome")
    baseline = next(c for c in model.constructs if c.name == "baseline")
    changed = outcome.revised(
        indicators=tuple(
            i.revised(
                likelihood=i.likelihood.revised(
                    law=i.likelihood.law.revised(
                        loc=coefficient(0.0, role="observation_intercept")
                        + coefficient(1.0, role="loading") * state(driver.id)
                        + coefficient(1.0, role="loading") * state(outcome.id)
                        + coefficient(0.0, role="loading") * state(baseline.id)
                    ),
                    standardized=True,
                )
            )
            for i in outcome.indicators
        )
    )
    model = model.with_entities(edges=replace_constructs(model.edges, (changed,)))
    compiled = compile_fit_fixture(model).compiled_dynamical_model
    indices = {s.name: i for i, s in enumerate(compiled.states)}
    columns = {compiled.states[o.state_index].name: i for i, o in enumerate(compiled.observations)}
    observed = (
        jnp.full((4, len(compiled.observations)), jnp.nan)
        .at[0, columns["outcome"]]
        .set(0.2)
        .at[1, columns["total"]]
        .set(0.1)
        .at[2, columns["driver"]]
        .set(0.3)
    )
    mask = np.asarray(compile_point_reading_mask(compiled, observed))
    expected = np.zeros_like(mask)
    expected[0, [indices["driver"], indices["outcome"]]] = True
    expected[1, indices["total"]] = True
    expected[2, indices["driver"]] = True
    np.testing.assert_array_equal(mask, expected)


def test_zero_warmup_rejects_changed_parameterization_or_mask_before_kernel_execution():
    from grassmoth.artifacts.arrays import NumericalArray
    from grassmoth.artifacts.continuation import JointTuning, SamplerCheckpoint
    from grassmoth.artifacts.sampling import NutsDASpec, SamplerSpec
    from grassmoth.models.ssm.inference.samplers.nuts import run_nuts_da
    from grassmoth.models.ssm.inference.warmup.outcomes import InitializationFailure
    from grassmoth_test_support.compute import EXECUTION_CONTEXT

    target = _correlated_target()
    coordinates = build_joint_coordinates(target, "mixed", compiler=compile_program)
    parameters, path = coordinates.decode(jnp.zeros(coordinates.dimension))
    settings = SamplerSpec(
        algorithm=NutsDASpec(parameterization="mixed"),
        num_warmup=0,
        num_samples_per_chain=1,
        num_chains=1,
    )
    checkpoint = SamplerCheckpoint(
        settings=settings,
        parameters=NumericalArray.from_numpy(np.asarray(parameters[None])),
        paths=NumericalArray.from_numpy(np.asarray(path[None])),
        random_key=NumericalArray.from_numpy(np.array([0, 1], dtype=np.uint32)),
        native_state=(),
        tuning=JointTuning(parameterization=coordinates.record),
    )
    for changed_target, changed_settings in (
        (target, settings.revised(algorithm=NutsDASpec(parameterization="non_centred"))),
        (replace(target, point_reading_mask=jnp.zeros_like(target.point_reading_mask)), settings),
    ):
        result = run_nuts_da(
            changed_target,
            sampler=changed_settings,
            start=checkpoint,
            execution_context=EXECUTION_CONTEXT,
        )
        assert isinstance(result, InitializationFailure)
