"""Exact parameterization identities and decoded analytic posteriors, without fitting a study."""

from dataclasses import replace

import dynestyx as dsx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from nof1_causal_lab.artifacts.joint_parameterization import MixedParameterizationRecord
from nof1_causal_lab.artifacts.probability.families import DistributionFamily
from nof1_causal_lab.models.ssm.execution.contracts import MeasurementParams
from nof1_causal_lab.models.ssm.execution.dynamical_model import HeterogeneousObservation
from nof1_causal_lab.models.ssm.execution.innovations import conditional_gaussian_step
from nof1_causal_lab.models.ssm.inference.conditioning import compile_exact_state_constraints
from nof1_causal_lab.models.ssm.inference.joint_coordinates import build_joint_coordinates
from nof1_causal_lab.models.ssm.inference.parameter_transform import ParameterTransform
from nof1_causal_lab.models.ssm.inference.targets.gaussian_reference import GaussianReference
from tests.models.ssm.compile.test_exact_state_space import _model, _problem
from tests.support.interval_particle import gaussian_interval_case, interval_target
from tests.support.observation import observation_laws

pytestmark = pytest.mark.inference(concern="sampling")
PARAMETERIZATIONS = ("centred", "non_centred", "mixed")


def _direct_constraints(value):
    from nof1_causal_lab.artifacts.identity import MechanismId
    from nof1_causal_lab.artifacts.model.construct import replace_constructs
    from nof1_causal_lab.artifacts.model.expressions import LiteralExpression, coefficient, state
    from nof1_causal_lab.artifacts.model.likelihood import (
        DeltaLawSpec,
        LikelihoodSpec,
        NormalLawSpec,
    )
    from nof1_causal_lab.artifacts.model.mechanism import DriftMechanismSpec
    from tests.support.helpers import make_model
    from tests.support.inference import compile_fit_fixture

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
    parameterization = build_joint_coordinates(target, parameterization_name)
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
                build_joint_coordinates(changed, parameterization_name).log_density(point) - value
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
    parameterization = build_joint_coordinates(target, parameterization_name)
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
    parameterization = build_joint_coordinates(target, parameterization_name)
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
    affine = build_joint_coordinates(target)
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
    parameterization = build_joint_coordinates(target, parameterization_name)
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


def test_compiler_mask_tracks_available_direct_multi_state_readings_only():
    from nof1_causal_lab.artifacts.model.construct import replace_constructs
    from nof1_causal_lab.artifacts.model.expressions import coefficient, state
    from nof1_causal_lab.models.ssm.compile.point_readings import compile_point_reading_mask
    from tests.support.inference import compile_fit_fixture

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
    import time

    from nof1_causal_lab.artifacts.arrays import NumericalArray
    from nof1_causal_lab.artifacts.continuation import JointTuning, SamplerCheckpoint
    from nof1_causal_lab.artifacts.sampling import NutsDASpec, SamplerSpec
    from nof1_causal_lab.models.ssm.inference.samplers.nuts import run_nuts_da
    from nof1_causal_lab.models.ssm.inference.warmup.outcomes import InitializationFailure
    from tests.support.compute import COMPUTE_BUDGET

    target = _correlated_target()
    coordinates = build_joint_coordinates(target, "mixed")
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
            compute_budget=COMPUTE_BUDGET,
            key=jax.random.key(0),
            clock=time.monotonic,
            continuation=checkpoint,
        )
        assert isinstance(result, InitializationFailure)
