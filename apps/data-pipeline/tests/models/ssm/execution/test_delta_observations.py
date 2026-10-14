"""Exact measurement semantics across authoring, compilation, and observation execution."""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from pydantic import TypeAdapter, ValidationError

from nof1_causal_lab.artifacts.identity import ConstructId
from nof1_causal_lab.artifacts.model.construct import replace_constructs
from nof1_causal_lab.artifacts.model.dynamical_model_spec import DynamicalModelSpec
from nof1_causal_lab.artifacts.model.expressions import state
from nof1_causal_lab.artifacts.model.likelihood import (
    DeltaLawSpec,
    LikelihoodSpec,
    ObservationLawSpec,
)
from nof1_causal_lab.artifacts.model.parameter_context import iter_coefficient_uses
from nof1_causal_lab.artifacts.probability.families import DistributionFamily
from nof1_causal_lab.models.ssm import compiled_access as numeric
from nof1_causal_lab.models.ssm.execution.observation_model import compile_observation_model
from nof1_causal_lab.models.ssm.inference import problem as problem_module
from nof1_causal_lab.models.ssm.inference.conditioning import compile_exact_state_constraints
from nof1_causal_lab.models.ssm.inference.marginal_particle_gibbs._math import (
    _masked_mean,
    _masked_normal_log_prob,
)
from nof1_causal_lab.models.ssm.inference.marginal_particle_gibbs.kernel import (
    build_marginal_particle_gibbs_kernel,
)
from nof1_causal_lab.models.ssm.inference.marginal_particle_gibbs.runner import (
    _initialize_chain_state,
)
from nof1_causal_lab.models.structure import (
    StructuralSelection,
    selected_indicators,
    selected_state_ids,
)
from tests.support.compute import COMPUTE_BUDGET
from tests.support.helpers import make_model
from tests.support.inference import (
    bind_panel_fixture,
    compile_fit_fixture,
    compile_model_fixture,
    make_observation_support_runtime,
)
from tests.support.model import _exact_model_model, load_model_fixture
from tests.support.observation import observation_laws


def _authored_affine_delta_keeps_its_calibration_coefficients_complete_model() -> (
    DynamicalModelSpec
):
    return load_model_fixture(
        "delta_observations/authored_affine_delta_keeps_its_calibration_coefficients_complete_model.json"
    )


@pytest.fixture
def exact_model():
    return _exact_model_model()


@pytest.mark.contract
def test_exact_binding_roundtrips_without_authored_measurement_parameters(exact_model):
    dynamical_model_spec = DynamicalModelSpec.model_validate_json(
        exact_model.model_dump_json()
    ).materialized()
    owner = dynamical_model_spec.constructs[0]
    indicator = owner.indicators[0]
    likelihood = indicator.likelihood
    assert likelihood is not None
    assert isinstance(likelihood.law, DeltaLawSpec)
    assert likelihood.law.v == state(owner.id)
    assert likelihood.parsed.loadings[owner.id].value == 1
    assert likelihood.parsed.intercept.value == 0
    assert not likelihood.parsed.auxiliary
    assert all(
        isinstance(use.value, (int, float))
        for use in iter_coefficient_uses(dynamical_model_spec)
        if any(ref.id == indicator.observation.id for ref in use.owners)
    )
    np.testing.assert_array_equal(
        compile_model_fixture(dynamical_model_spec).loading_block.template, np.eye(2)
    )
    assert compile_model_fixture(dynamical_model_spec).observation_mean_block.template[0] == 0
    assert not compile_model_fixture(dynamical_model_spec).observation_noise_block.diag_support[0]
    np.testing.assert_array_equal(
        compile_model_fixture(dynamical_model_spec).observation_noise_block.template[0], [0, 0]
    )
    assert owner.id in selected_state_ids(StructuralSelection(dynamical_model_spec, None))
    assert "usage" not in type(owner).model_fields
    compile_model_fixture(dynamical_model_spec)


@pytest.mark.contract
def test_delta_constructor_requires_only_its_exact_value():
    for arguments in (
        {},
        {"loc": state(ConstructId("construct:x"))},
        {"v": state(ConstructId("construct:x")), "scale": 0},
    ):
        with pytest.raises(ValidationError):
            TypeAdapter(ObservationLawSpec).validate_python({"distribution": "Delta", **arguments})


@pytest.mark.contract
def test_authored_affine_delta_keeps_its_calibration_coefficients(exact_model):
    owner = exact_model.constructs[0]
    predictor = (
        TypeAdapter(ObservationLawSpec)
        .validate_json(
            (
                Path(__file__).resolve().parents[3]
                / "fixtures/models"
                / "delta_observations/gaussian_observation_law.json"
            ).read_text()
        )
        .loc
    )
    likelihood = LikelihoodSpec(
        law=DeltaLawSpec(v=predictor),
        reasoning="Exact measurement with an unknown calibration offset.",
    )
    indicator = owner.indicators[0].revised(likelihood=likelihood)
    dynamical_model_spec = (
        _authored_affine_delta_keeps_its_calibration_coefficients_complete_model()
    )
    completed = dynamical_model_spec.indicator(indicator.observation.id).likelihood
    assert completed is not None
    assert isinstance(completed.parsed.intercept.value, str)
    assert not completed.parsed.auxiliary
    compile_model_fixture(dynamical_model_spec)


@pytest.mark.contract
@pytest.mark.parametrize(
    ("dtype", "default_family", "observation_law_payload"),
    [
        pytest.param(
            "continuous",
            "gaussian",
            "delta_observations/exact_state_observation_law.json",
            id="continuous-gaussian",
        ),
        pytest.param(
            "binary",
            "bernoulli",
            "delta_observations/exact_state_observation_law.json",
            id="binary-bernoulli",
        ),
        pytest.param(
            "count",
            "poisson",
            "delta_observations/exact_state_observation_law.json",
            id="count-poisson",
        ),
        pytest.param(
            "ordinal",
            "ordered_logistic",
            "delta_observations/exact_state_observation_law.json",
            id="ordinal-ordered_logistic",
        ),
        pytest.param(
            "categorical",
            "categorical",
            "delta_observations/exact_state_observation_law.json",
            id="categorical-categorical",
        ),
    ],
)
def test_exact_measurement_is_available_but_never_selected_by_default(
    dtype, default_family, observation_law_payload
):
    dynamical_model_spec = make_model(["setting", "response"], [("setting", "response")])
    owner = dynamical_model_spec.constructs[0]
    updates = {"measurement_dtype": dtype, "aggregation": "last"}
    if dtype in {"ordinal", "categorical"}:
        updates[f"{dtype}_levels"] = ("low", "high")
    indicator = owner.indicators[0].revised(
        observation=owner.indicators[0].observation.revised(**updates)
    )
    dynamical_model_spec = dynamical_model_spec.with_entities(
        edges=replace_constructs(
            dynamical_model_spec.edges,
            (owner.revised(indicators=(indicator,)),),
        )
    )
    from nof1_causal_lab.artifacts.observations.dtypes import VALID_LIKELIHOODS_FOR_DTYPE

    assert VALID_LIKELIHOODS_FOR_DTYPE[indicator.observation.measurement_dtype][0] == default_family
    exact = LikelihoodSpec(
        law=TypeAdapter(ObservationLawSpec).validate_json(
            (
                Path(__file__).resolve().parents[3] / "fixtures/models" / observation_law_payload
            ).read_text()
        ),
        reasoning="Explicit exact measurement",
    )
    dynamical_model_spec.with_entities(
        edges=replace_constructs(
            dynamical_model_spec.edges,
            (owner.revised(indicators=(indicator.revised(likelihood=exact),)),),
        )
    )


@pytest.mark.inference(concern="sampling")
@pytest.mark.inference(concern="predictive")
def test_mixed_delta_density_draws_and_missingness_remain_exact():
    covariance = jnp.eye(2) * 100.0
    compiled = compile_observation_model(
        observation_laws([DistributionFamily.DELTA, DistributionFamily.POISSON], None, None),
        manifest_cov=covariance,
    )
    kernel = compiled.kernel
    predictor = jnp.array([-2.0, jnp.log(3.0)])
    observed = jnp.array([-2.0, 2.0])
    expected = dist.Poisson(3.0).log_prob(2.0)
    density = jax.jit(kernel.log_prob_fn)
    np.testing.assert_allclose(density(observed, predictor, covariance, jnp.ones(2)), expected)
    mismatch = observed.at[0].set(jnp.nextafter(observed[0], 0.0))
    assert jnp.isneginf(density(mismatch, predictor, covariance, jnp.ones(2)))
    np.testing.assert_allclose(
        density(observed.at[0].set(jnp.nan), predictor, covariance, jnp.array([0.0, 1.0])),
        expected,
    )
    assert density(jnp.full(2, jnp.nan), predictor, covariance, jnp.zeros(2)) == 0
    draw = compiled.point_sampler.sample_point(jax.random.key(2), predictor)
    assert draw[0] == predictor[0]


@pytest.mark.inference(concern="sampling")
@pytest.mark.inference(concern="predictive")
def test_exact_window_mean_constrains_the_summary_without_pinning_the_path():
    support = make_observation_support_runtime(
        anchor_times=np.array([0.0, 2.0]),
        manifest_names=("setting_mean",),
        support_kinds=("interval",),
        summary_operators=("mean",),
        anchor_policies=("support_end",),
        observation_windows=("2d",),
        support_start_times=np.array([[np.nan], [0.0]]),
        support_end_times=np.array([[np.nan], [2.0]]),
    )
    covariance = jnp.zeros((1, 1))
    compiled = compile_observation_model(
        observation_laws([DistributionFamily.DELTA], None, None),
        manifest_cov=covariance,
        observation_support=support,
    )
    assert compiled.observation_operator is not None
    assert compiled.mean_log_prob_fn is not None
    from nof1_causal_lab.models.ssm.execution.observation_dispatch import (
        build_interval_summary_sampler,
    )

    sampler = build_interval_summary_sampler(
        observation_laws([DistributionFamily.DELTA], None, None), covariance, (0,)
    )
    for path in ([[0.0], [4.0]], [[4.0], [0.0]], [[2.0], [2.0]]):
        means, mask = compiled.observation_operator.project_response_trajectory(jnp.array(path))
        np.testing.assert_array_equal(mask, [[0.0], [1.0]])
        assert means[-1, 0] == 2.0
        assert compiled.mean_log_prob_fn(jnp.array([2.0]), means[-1], covariance, mask[-1]) == 0
        assert jnp.isneginf(
            compiled.mean_log_prob_fn(jnp.array([3.0]), means[-1], covariance, mask[-1])
        )
        draws = sampler.sample_mean_trajectory(jax.random.key(0), means)
        np.testing.assert_array_equal(draws, means)


@pytest.mark.contract
@pytest.mark.parametrize("unsupported", ["affine", "interval"])
def test_unsupported_delta_constraints_fail_before_parameter_initialization(
    exact_model, monkeypatch, unsupported
):
    owner = exact_model.constructs[0]
    indicator = owner.indicators[0]
    if unsupported == "interval":
        indicator = indicator.revised(observation=indicator.observation.revised(aggregation="mean"))
    else:
        indicator = indicator.revised(
            likelihood=LikelihoodSpec(
                law=DeltaLawSpec(
                    v=TypeAdapter(ObservationLawSpec)
                    .validate_json(
                        (
                            Path(__file__).resolve().parents[3]
                            / "fixtures/models"
                            / "delta_observations/gaussian_observation_law.json"
                        ).read_text()
                    )
                    .loc
                ),
                reasoning="An affine equality needs a different constraint parameterization.",
            )
        )
    exact_model = exact_model.with_entities(
        edges=replace_constructs(
            exact_model.edges,
            (owner.revised(indicators=(indicator,)),),
        )
    )

    def unexpected_initialization(*_args):
        pytest.fail("An unsupported exact constraint must fail before numerical initialization")

    monkeypatch.setattr(problem_module, "prepare_model_parameters", unexpected_initialization)
    if unsupported == "affine":
        from nof1_causal_lab.models.ssm.compile.inputs import (
            IncompleteModel,
            compile_ssm_inputs_from_model,
        )

        result = compile_ssm_inputs_from_model(StructuralSelection(exact_model, None))
        assert isinstance(result, IncompleteModel)
        assert "measurement coefficients" in result.message
        return
    with pytest.raises(ValueError, match=r"setting_obs.*direct point binding"):
        problem_module.build_particle_problem(
            compile_fit_fixture(exact_model).prior_runtime_bundle,
            bind_panel_fixture(
                compile_model_fixture(exact_model),
                jnp.array([[jnp.nan, jnp.nan], [1.0, 0.5]]),
                jnp.array([0.0, 1.0]),
            ),
            scheme="euler_maruyama",
            trace_key=jax.random.key(0),
            reparam=None,
        )


@pytest.mark.contract
def test_sparse_exact_observations_leave_missing_coordinates_free(exact_model):
    constraints = compile_exact_state_constraints(
        compile_model_fixture(exact_model), jnp.array([[1.0, 0.0], [jnp.nan, 0.5], [2.0, 1.0]])
    )
    assert constraints is not None
    np.testing.assert_array_equal(
        constraints.free_mask, [[False, True], [True, True], [False, True]]
    )
    path = jnp.arange(6.0).reshape(3, 2)
    projected = constraints.project(path)
    np.testing.assert_array_equal(projected, [[1.0, 1.0], [2.0, 3.0], [2.0, 5.0]])
    np.testing.assert_array_equal(constraints.project(jnp.stack([path, path]))[1], projected)
    with pytest.raises(ValueError, match="finite or missing"):
        compile_exact_state_constraints(
            compile_model_fixture(exact_model), jnp.array([[jnp.inf, 0.0]])
        )


@pytest.mark.contract
def test_multiple_exact_indicators_must_agree_at_shared_times(exact_model):
    from nof1_causal_lab.artifacts.identity import scientific_id

    owner = exact_model.constructs[0]
    original = owner.indicators[0]
    duplicate = original.revised(
        observation=original.observation.revised(
            id=scientific_id("indicator", "second_recording"), name="second_recording"
        )
    )
    model = exact_model.with_entities(
        edges=replace_constructs(
            exact_model.edges,
            (owner.revised(indicators=(original, duplicate)),),
        )
    )
    columns = {
        identity: column
        for column, identity in enumerate(
            tuple(
                indicator.observation.id
                for indicator in selected_indicators(StructuralSelection(model, None))
            )
        )
    }
    observations = jnp.full((2, 3), jnp.nan)
    observations = observations.at[:, columns[original.observation.id]].set(
        jnp.array([1.0, jnp.nan])
    )
    observations = observations.at[:, columns[duplicate.observation.id]].set(jnp.array([1.0, 4.0]))
    constraints = compile_exact_state_constraints(compile_model_fixture(model), observations)
    assert constraints is not None
    np.testing.assert_array_equal(constraints.values[:, 0], [1.0, 4.0])
    with pytest.raises(ValueError, match="Conflicting exact observations"):
        compile_exact_state_constraints(
            compile_model_fixture(model),
            observations.at[0, columns[duplicate.observation.id]].set(2.0),
        )


@pytest.mark.inference(concern="sampling")
@pytest.mark.parametrize("mask", [[True, False], [False, True], [True, True], [False, False]])
def test_proposal_density_uses_the_free_coordinate_measure(mask):
    values, mean, variance = jnp.array([1.5, -2.0]), jnp.array([0.2, 0.4]), jnp.array([0.3, 2.0])
    free = np.asarray(mask)
    expected = dist.Normal(mean[free], jnp.sqrt(variance[free])).log_prob(values[free]).sum()
    np.testing.assert_allclose(
        _masked_normal_log_prob(values, mean, variance, free), expected, atol=1e-6
    )
    grad = jax.grad(lambda x: _masked_normal_log_prob(x, mean, variance, free))(values)
    np.testing.assert_array_equal(grad[~free], jnp.zeros((~free).sum()))


@pytest.mark.contract
def test_fixed_coordinates_are_excluded_from_sampler_freeze_diagnostics():
    free = jnp.array([[False, True], [False, False], [False, True]])
    frozen = jnp.array([[True, False], [True, True], [True, True]])
    assert _masked_mean(frozen, free) == 0.5
    np.testing.assert_array_equal(_masked_mean(frozen, free, axis=0), [0.0, 0.5])
    np.testing.assert_array_equal(_masked_mean(frozen, free, axis=-1), [0.0, 0.0, 1.0])


@pytest.fixture
def point_problem(exact_model):
    inputs = compile_fit_fixture(exact_model)
    return problem_module.build_particle_problem(
        inputs.prior_runtime_bundle,
        bind_panel_fixture(
            inputs.compiled_dynamical_model,
            jnp.array([[1.0, jnp.nan], [jnp.nan, jnp.nan], [2.0, jnp.nan]]),
            jnp.array([0.0, 0.4, 1.0]),
        ),
        scheme="euler_maruyama",
        trace_key=jax.random.key(0),
        reparam=None,
    )


def _conditioned_initial_state(problem):
    runtime = problem.runtime
    return _initialize_chain_state(
        runtime.initial_position,
        observations=runtime.observations,
        times=runtime.times,
        target=runtime,
        initial_latent_delta=jnp.full(runtime.times.shape, 0.2),
        param_step_size=0.01,
        param_min_scale=1e-6,
        param_max_scale=1e3,
        param_target_accept=0.35,
        initial_latent_trajectory=jnp.full((3, 2), 0.25),
        exact_constraints=problem.exact_constraints,
    )


@pytest.mark.inference(concern="sampling")
def test_conditioned_target_preserves_initial_and_transition_evidence(point_problem):
    runtime = point_problem.runtime
    state = _conditioned_initial_state(point_problem)
    path = state.latent_trajectory
    np.testing.assert_array_equal(path[:, 0], [1.0, 0.25, 2.0])
    dynamics = runtime.dynamical_model(state.latent_context)
    expected = dynamics.initial_condition.log_prob(path[0])
    for index in (1, 2):
        expected += dynamics.state_evolution(
            path[index - 1], None, runtime.times[index - 1], runtime.times[index]
        ).log_prob(path[index])
    np.testing.assert_allclose(state.trajectory_log_prob, expected, rtol=1e-6)
    assert jnp.isfinite(state.complete_log_posterior)
    coordinates = path.at[0, 0].set(0.0)
    np.testing.assert_array_equal(runtime.replay_path(state.latent_context, coordinates), path)
    np.testing.assert_allclose(
        runtime.path_log_prob(state.latent_context, coordinates, runtime.observations),
        expected,
        rtol=1e-6,
    )
    # Emission factors score the supplied state, including exact-reading support.
    assert jnp.isneginf(
        runtime.observation_log_probs(state.latent_context, coordinates, runtime.observations)[0]
    )
    coordinate_gradient = jax.grad(
        lambda coords: runtime.path_log_prob(state.latent_context, coords, runtime.observations)
    )(coordinates)
    assert jnp.isfinite(coordinate_gradient).all()
    np.testing.assert_array_equal(
        coordinate_gradient[~point_problem.exact_constraints.free_mask], 0.0
    )
    derivative = jax.grad(
        lambda z: runtime.log_posterior(z, path, runtime.observations, runtime.times)
    )(runtime.initial_position)
    assert jnp.isfinite(derivative).all()


@pytest.mark.inference(concern="sampling")
@pytest.mark.parametrize("proposal", ["amala_exact", "paid_mix"])
def test_conditioned_smoother_traces_with_missing_and_observed_coordinates(point_problem, proposal):
    runtime = point_problem.runtime
    state = _conditioned_initial_state(point_problem)
    path = state.latent_trajectory
    kernel = build_marginal_particle_gibbs_kernel(
        runtime,
        num_particles=4,
        num_parameter_particles=2,
        param_step_size=0.01,
        exact_constraints=point_problem.exact_constraints,
        dsmc_leaf_proposal=proposal,
        latent_block_coords=1,
        pilot_means=path if proposal == "paid_mix" else None,
        pilot_vars=jnp.ones_like(path) if proposal == "paid_mix" else None,
        pilot_wide_vars=jnp.ones_like(path) * 4 if proposal == "paid_mix" else None,
    )
    shape, _ = jax.eval_shape(kernel.step_fn, state, jax.random.key(1))
    assert shape.latent_trajectory.shape == (3, 2)


@pytest.mark.inference(concern="sampling")
@pytest.mark.parametrize("proposal", ["amala_exact", "paid_mix"])
def test_particle_updates_keep_exact_readings_and_move_missing_states(point_problem, proposal):
    runtime = point_problem.runtime
    state = _conditioned_initial_state(point_problem)
    path = state.latent_trajectory
    kernel = build_marginal_particle_gibbs_kernel(
        runtime,
        num_particles=16,
        num_parameter_particles=2,
        param_step_size=0.01,
        exact_constraints=point_problem.exact_constraints,
        dsmc_leaf_proposal=proposal,
        latent_block_coords=1,
        pilot_means=path,
        pilot_vars=jnp.ones_like(path),
        pilot_wide_vars=jnp.ones_like(path) * 4,
    )
    step = jax.jit(kernel.step_fn)
    missing_values = []
    for key in jax.random.split(jax.random.key(7), 8):
        state, _ = step(state, key)
        np.testing.assert_array_equal(state.latent_trajectory[[0, 2], 0], [1.0, 2.0])
        assert jnp.isfinite(state.complete_log_posterior)
        missing_values.append(state.latent_trajectory[1, 0])
    assert float(jnp.var(jnp.stack(missing_values))) > 0


@pytest.mark.inference(concern="predictive")
def test_exact_observations_pass_through_full_construct_diagnostics(exact_model):
    from nof1_causal_lab.models.ssm.diagnostics.simulation import (
        ConstructSimulationTarget,
        DesignInfo,
        measure_construct_simulation,
    )
    from nof1_causal_lab.models.ssm.parameterization import assemble_deterministics_from_registry
    from nof1_causal_lab.models.ssm.predictive.types import PredictiveDraws, PredictiveTrajectory
    from tests.support.inference import compile_model_fixture, parameter_draws

    draws, ticks = 2, 6
    compiled_dynamical_model = compile_model_fixture(exact_model)
    parameters = parameter_draws(exact_model, draws)
    parameters.update(
        assemble_deterministics_from_registry(parameters, compiled_dynamical_model, n_draws=draws)
    )
    paths = jnp.broadcast_to(jnp.linspace(-1.0, 1.0, ticks)[None, :, None], (draws, ticks, 2))
    prediction = PredictiveDraws(
        parameters=parameters,
        trajectory=PredictiveTrajectory(
            paths, paths, paths, jnp.ones_like(paths, dtype=bool), paths
        ),
    )
    target = exact_model.constructs[0]
    indicator_id = target.indicators[0].observation.id
    results, _ = measure_construct_simulation(
        compiled_dynamical_model,
        prediction,
        DesignInfo(
            t_grid=jnp.arange(ticks, dtype=float),
            manifest_ids=tuple(numeric.observation_ids(compiled_dynamical_model)),
            obs_index_by_indicator={indicator_id: np.arange(ticks)},
            values_by_indicator={indicator_id: np.asarray(paths[0, :, 0])},
        ),
        ConstructSimulationTarget(
            next(state for state in compiled_dynamical_model.states if state.id == target.id)
        ),
        clock=time.monotonic,
        edge_contrasts=COMPUTE_BUDGET,
    )
    transmission = next(result for result in results if result.check == "C5c transmission")
    assert transmission.passed
    assert transmission.evidence is not None
    np.testing.assert_array_equal(transmission.evidence["conditional_variance"], 0.0)
    np.testing.assert_array_equal(transmission.evidence["signal_fraction"], 1.0)
