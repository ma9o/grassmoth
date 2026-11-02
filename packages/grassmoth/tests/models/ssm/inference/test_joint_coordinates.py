from dataclasses import replace

import dynestyx as dsx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from grassmoth.artifacts.probability.families import DistributionFamily
from grassmoth.models.ssm.execution.contracts import MeasurementParams
from grassmoth.models.ssm.execution.dynamical_model import HeterogeneousObservation
from grassmoth.models.ssm.inference.conditioning import compile_exact_state_constraints
from grassmoth.models.ssm.inference.joint_coordinates import build_joint_coordinates
from grassmoth.models.ssm.inference.parameter_transform import ParameterTransform
from grassmoth_test_support.compute import compile_program
from grassmoth_test_support.interval_particle import interval_target
from grassmoth_test_support.observation import observation_laws
from packages.grassmoth.tests.models.ssm.compile.test_exact_state_space import _model, _problem

"""Centred joint coordinates of particle starts, without fitting a study."""


pytestmark = pytest.mark.inference(concern="warmup")


def test_runtime_charts_preserve_each_fits_scientific_start():
    target = _correlated_target()
    first = build_joint_coordinates(target, compiler=compile_program)
    other_target = replace(
        target, parameters=replace(target.parameters, initial_position=jnp.array([0.7]))
    )
    second = build_joint_coordinates(other_target, compiler=compile_program)
    dynamic = jax.jit(lambda point, chart: first.bind(chart).decode(point))
    for coordinates in (first, second):
        point = jnp.linspace(-0.2, 0.3, coordinates.dimension)
        expected = jax.jit(coordinates.decode)(point)
        measured = dynamic(point, coordinates.chart)
        for actual, reference in zip(measured, expected, strict=True):
            np.testing.assert_allclose(actual, reference, atol=3e-6, rtol=3e-6)
        bound = first.bind(coordinates.chart)
        np.testing.assert_array_equal(bound.parameter_center, coordinates.parameter_center)


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
        return eqx.filter(model(position), eqx.is_array), times, jnp.zeros(2), None

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
    )


def test_static_and_accumulator_replay_have_no_extra_noise_or_free_coordinates():
    compiled, problem = _problem(_model())
    indices = {state.name: i for i, state in enumerate(compiled.states)}
    total_column = next(
        i for i, obs in enumerate(compiled.observations) if obs.state_index == indices["total"]
    )
    observed = problem.runtime.observations.at[2, total_column].set(0.2)
    _, problem = _problem(_model(), observed)
    target = problem.runtime
    parameterization = build_joint_coordinates(target, compiler=compile_program)
    point = jnp.linspace(-0.2, 0.3, parameterization.dimension)
    parameters, path = parameterization.decode(point)
    np.testing.assert_allclose(
        path[1:, indices["total"]], jnp.cumsum(path[:-1, indices["driver"]]), atol=2e-6
    )
    np.testing.assert_array_equal(
        path[:, indices["baseline"]], jnp.repeat(path[0, indices["baseline"]], 4)
    )
    assert not np.asarray(parameterization.free_mask)[
        :, [indices["total"], indices["baseline"]]
    ].any()
    context = target.context(parameters, target.times)
    np.testing.assert_allclose(path, target.replay_path(context, path), atol=2e-6)
