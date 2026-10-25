"""Expression ownership, parameter binding, and exact local drift; no fitting or rollouts."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from numpyro import handlers
from pydantic import TypeAdapter

from grassmoth.artifacts.identity import ConstructId, scientific_id
from grassmoth.artifacts.model.construct import (
    CausalEdgeSpec,
    ConstructSpec,
    replace_constructs,
)
from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec
from grassmoth.artifacts.model.expressions import (
    CallExpression,
    CoefficientExpression,
    coefficient,
    expression_coefficients,
    expression_states,
    hill,
    restoring_coefficients,
    restoring_force,
    restoring_potential,
    state,
)
from grassmoth.artifacts.model.mechanism import DriftMechanismSpec, DynamicsMechanismSpec
from grassmoth.models.ssm.dynamics.expression import ExpressionComponentSpec
from grassmoth.models.ssm.dynamics.intervention import (
    ConstantValueFn,
    EdgeInputOverride,
    Intervention,
    VariableOverride,
)
from grassmoth.models.ssm.dynamics.spec import (
    DynamicsSpec,
    compile_dynamics,
    pack_component_params_from_samples,
)
from grassmoth.models.ssm.dynamics.vector_field import VectorFieldArgs


def _component(value, *, source=None):
    return ExpressionComponentSpec(
        expression=value,
        target=1,
        state_ids=tuple(ConstructId(f"construct:{name}") for name in ("x", "y", "z")),
        source=source,
    )


@pytest.mark.contract
def test_direct_values_preserve_fixed_zero_shared_identity_and_partial_authoring():
    identity = scientific_id("parameter", "shared")
    operands = [CoefficientExpression(role="weight", value=value) for value in (0, identity, None)]
    assert [operand.value for operand in operands] == [0, identity, None]
    for operand in operands:
        assert CoefficientExpression.model_validate_json(operand.model_dump_json()) == operand
    for invalid in (float("inf"), float("nan"), "not-a-parameter-id"):
        with pytest.raises(ValueError, match=r"finite_number|string_pattern_mismatch"):
            CoefficientExpression(role="weight", value=invalid)


@pytest.mark.inference(concern="simulation")
def test_existing_functions_and_composition_preserve_drift_and_intervention_inputs():
    restoring = restoring_force(ConstructId("construct:y"), center=0.3, stiffness=0.7, quartic=0.2)
    saturation = hill(state(ConstructId("construct:x")), emax=2, ec50=1.5, n=2)
    components = (
        _component(restoring),
        _component(coefficient(0.4, "intercept")),
        _component(coefficient(0.6, "weight") * state(ConstructId("construct:x")), source=0),
        _component(
            coefficient(0.5, "weight")
            * state(ConstructId("construct:x"))
            * state(ConstructId("construct:z")),
            source=0,
        ),
        _component(saturation, source=0),
        _component(saturation * state(ConstructId("construct:z")), source=0),
    )
    native = compile_dynamics(DynamicsSpec(n_latent=3, components=components))
    params = tuple({} for _ in components)

    def derivative(x, intervention):
        return native.vector_field(
            jnp.asarray(0.0), x, VectorFieldArgs(params=params, intervention=intervention)
        )

    values = jnp.asarray([1.2, 0.8, 0.9])
    expected_hill = 2 * 1.2**2 / (1.5**2 + 1.2**2 + 1e-12)
    expected = -0.7 * 0.5 - 0.2 * 0.5**3 + 0.4 + 0.6 * 1.2 + 0.5 * 1.2 * 0.9 + expected_hill * 1.9
    np.testing.assert_allclose(derivative(values, Intervention.none()), [0, expected, 0], rtol=1e-6)
    edges_off = Intervention(
        overrides=(
            EdgeInputOverride(source=0, target=1, value_fn=ConstantValueFn(jnp.asarray(0.0))),
        )
    )
    np.testing.assert_allclose(
        derivative(values, edges_off), [0, -0.7 * 0.5 - 0.2 * 0.5**3 + 0.4, 0], atol=1e-7
    )
    clamped = Intervention(
        overrides=(VariableOverride(index=1, value_fn=ConstantValueFn(jnp.asarray(2.0))),)
    )
    assert derivative(values, clamped)[1] == 0
    assert (
        native.vector_field.initial_condition(
            values, VectorFieldArgs(params=params, intervention=clamped)
        )[1]
        == 2
    )
    gradient = jax.grad(lambda x: derivative(x, Intervention.none())[1])(values)
    assert np.all(np.isfinite(gradient))
    np.testing.assert_allclose(gradient[1], -0.7 - 3 * 0.2 * 0.5**2, rtol=1e-6)


@pytest.mark.inference(concern="simulation")
def test_symbolic_compilation_preserves_native_values_gradients_and_domains():
    x = state(ConstructId("construct:x"))
    identity = scientific_id("parameter", "gain")
    gain = coefficient(identity, "weight")
    formula = (
        hill(x, emax=2, ec50=1.5, n=2)
        + CallExpression(function="exp", arguments=(gain * x,))
        - CallExpression(function="sigmoid", arguments=(x,))
    ) / (CallExpression(function="normal_cdf", arguments=(x,)) + 1)
    component = _component(formula, source=0).build()

    def compiled(values):
        return component.evaluate(jnp.array([values[0], 0.0, 0.0]), {identity: values[1]})

    def native(values):
        dose, amplitude = values
        positive = jnp.maximum(dose, 0.0)
        saturation = 2 * positive**2 / (1.5**2 + positive**2 + 1e-12)
        return (saturation + jnp.exp(amplitude * dose) - jax.nn.sigmoid(dose)) / (
            1 + jax.scipy.special.ndtr(dose)
        )

    values = jnp.array([1.2, 0.4])
    actual = jax.jit(jax.value_and_grad(compiled))(values)
    expected = jax.value_and_grad(native)(values)
    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(result, reference, rtol=1e-6, atol=1e-7)
    quotient = _component(x / x, source=0).build()
    assert np.isnan(quotient.evaluate(jnp.zeros(3), {}))
    assert expression_states(x / x) == {"construct:x"}


@pytest.mark.inference(concern="simulation")
@pytest.mark.parametrize("component_count", [0, 3])
def test_shared_drift_preserves_component_parameters_and_edge_ownership(component_count):
    identity = scientific_id("parameter", "gain")
    formula = coefficient(identity, "weight") * CallExpression(
        function="exp", arguments=(state(ConstructId("construct:x")),)
    )
    components = (
        _component(formula),
        _component(formula, source=0),
        replace(_component(formula, source=0), target=2),
    )[:component_count]
    native = compile_dynamics(DynamicsSpec(n_latent=3, components=components))
    intervention = Intervention(
        overrides=(
            EdgeInputOverride(source=0, target=1, value_fn=ConstantValueFn(jnp.asarray(-0.5))),
        )
    )

    def compiled(values: jax.Array, weights: jax.Array):
        params: tuple[dict[str, jax.Array], ...] = tuple(
            {identity: weights[index]} for index in range(component_count)
        )
        return native.vector_field(
            jnp.asarray(0.0), values, VectorFieldArgs(params=params, intervention=intervention)
        )

    def reference(values, weights):
        if component_count == 0:
            return jnp.zeros_like(values)
        return jnp.array(
            [
                0.0,
                weights[0] * jnp.exp(values[0]) + weights[1] * jnp.exp(-0.5),
                weights[2] * jnp.exp(values[0]),
            ]
        )

    values, weights = jnp.array([1.2, 0.8, 0.9]), jnp.array([0.4, 0.7, 1.2])
    np.testing.assert_allclose(compiled(values, weights), reference(values, weights), rtol=1e-6)
    actual = jax.jit(jax.jacrev(compiled, argnums=(0, 1)))(values, weights)
    expected = jax.jacrev(reference, argnums=(0, 1))(values, weights)
    for result, target in zip(actual, expected, strict=True):
        np.testing.assert_allclose(result, target, rtol=1e-6, atol=1e-7)


@pytest.mark.inference(concern="sampling")
def test_multiple_same_role_operands_bind_directly_and_repeated_references_sample_once():
    first = scientific_id("parameter", "first")
    second = scientific_id("parameter", "second")
    value = coefficient(first, "weight") * state(ConstructId("construct:x")) + coefficient(
        second, "weight"
    ) * state(ConstructId("construct:x")) * coefficient(first, "weight")
    component = _component(value, source=0)
    native = compile_dynamics(DynamicsSpec(n_latent=3, components=(component,)))
    assert expression_states(value) == {"construct:x"}
    assert len(expression_coefficients(value)) == 2
    sites = dict(component.parameter_sites("vf_0"))
    assert sites.keys() == {first, second}
    assert len({site.name for site in sites.values()}) == 2

    def draw():
        return native.sample_params(lambda _name: dist.Normal(0, 1))

    with handlers.seed(rng_seed=3):
        trace = handlers.trace(draw).get_trace()
    sampled = {name: item["value"] for name, item in trace.items() if item["type"] == "sample"}
    assert sampled.keys() == {site.name for site in sites.values()}
    packed = pack_component_params_from_samples(native.spec, sampled)
    assert packed[0].keys() == sites.keys()
    assert native.spec.components[0] is component
    mechanism = DriftMechanismSpec(id="mechanism:sum", expression=value)
    assert (
        TypeAdapter(DynamicsMechanismSpec).validate_json(mechanism.model_dump_json()) == mechanism
    )


def _model(expression):
    nodes = {
        name: ConstructSpec(
            id=f"construct:{name}",
            name=name,
            description=name,
            role="endogenous",
            temporal_status="time_varying",
        )
        for name in ("x", "y", "z")
    }
    return DynamicalModelSpec.from_entities(
        edges=(
            CausalEdgeSpec(
                id="edge:xy",
                cause=nodes["x"],
                effect=nodes["y"],
                description="x affects y",
                mechanisms=(DriftMechanismSpec(id="mechanism:effect", expression=expression),),
            ),
            CausalEdgeSpec(
                id="edge:xz", cause=nodes["x"], effect=nodes["z"], description="x affects z"
            ),
        )
    )


@pytest.mark.contract
def test_composition_requires_all_causal_dependencies_and_valid_parameter_references():
    base = _model(coefficient(1, "weight") * state(ConstructId("construct:x")))
    compound = hill(state(ConstructId("construct:x")), emax=1, ec50=1, n=2) * state(
        ConstructId("construct:z")
    )
    edge = base.edges[0].revised(
        mechanisms=(DriftMechanismSpec(id="mechanism:effect", expression=compound),)
    )
    with pytest.raises(ValueError, match="explicit causal edges"):
        base.with_entities(edges=(edge, *base.edges[1:]))
    extended = base.with_entities(
        edges=(
            edge,
            CausalEdgeSpec(
                id="edge:zy",
                cause=base.get_construct("construct:z"),
                effect=base.get_construct("construct:y"),
                description="z moderates x",
            ),
        )
    )
    assert expression_states(extended.edges[0].mechanisms[0].expression) == {
        "construct:x",
        "construct:z",
    }
    with pytest.raises(ValueError, match="unknown constructs"):
        _model(coefficient(1, "weight") * state(ConstructId("construct:missing")))
    with pytest.raises(ValueError, match="undeclared parameter"):
        _model(
            coefficient(scientific_id("parameter", "missing"), "weight")
            * state(ConstructId("construct:x"))
        )
    with pytest.raises(ValueError, match="owning construct"):
        base.with_entities(
            edges=replace_constructs(
                base.edges,
                (
                    base.constructs[0].revised(
                        dynamics=(
                            DriftMechanismSpec(
                                id="mechanism:intrinsic",
                                expression=state(ConstructId("construct:y")),
                            ),
                        )
                    ),
                    *base.constructs[1:],
                ),
            )
        )


@pytest.mark.contract
@pytest.mark.parametrize(
    ("kind", "constructor"), [("drift", restoring_force), ("potential", restoring_potential)]
)
def test_restoring_anchor_requires_the_complete_function_and_supported_coefficients(
    kind, constructor
):
    value = constructor("construct:y", center=0, stiffness=1, quartic=0)
    assert {
        operand.role
        for operand in restoring_coefficients(value, ConstructId("construct:y"), kind=kind)
    } == {
        "center",
        "decay",
        "quartic",
    }
    assert not restoring_coefficients(
        coefficient(0, "center") + state(ConstructId("construct:y")),
        ConstructId("construct:y"),
        kind=kind,
    )
    assert not restoring_coefficients(
        value, ConstructId("construct:y"), kind="potential" if kind == "drift" else "drift"
    )
    with pytest.raises(ValueError, match="decay must be positive"):
        restoring_force(ConstructId("construct:y"), center=0, stiffness=0, quartic=0)
    with pytest.raises(ValueError, match="exponent must be positive"):
        CoefficientExpression(role="exponent", value=-1)
    with pytest.raises(ValueError, match="literal_error"):
        TypeAdapter(DynamicsMechanismSpec).validate_python(
            {
                "id": "mechanism:unknown",
                "kind": kind,
                "expression": {
                    "kind": "binary",
                    "operator": "eval",
                    "left": {"kind": "literal", "value": 1},
                    "right": {"kind": "literal", "value": 2},
                },
            }
        )


@pytest.mark.contract
def test_solver_steps_follow_coefficient_meanings_including_fixed_rates():
    from grassmoth.models.ssm.predictive.registry_runtime import (
        _predictive_draw_order,
        _predictive_max_rates,
        _predictive_sde_config,
    )

    rate = scientific_id("parameter", "relaxation")
    components = (
        _component(
            restoring_force(ConstructId("construct:y"), center=0, stiffness=rate, quartic=0)
        ),
        _component(restoring_force(ConstructId("construct:y"), center=0, stiffness=4, quartic=0)),
    )
    compiled = compile_dynamics(DynamicsSpec(3, components))
    site = next(iter(components[0].parameter_sites("vf_0")))[1]
    rates = _predictive_max_rates(compiled, {site.name: jnp.asarray([8, 2, 16])})
    np.testing.assert_array_equal(rates, [8, 4, 16])
    steps = _predictive_sde_config(rates, 60).brownian.step_size
    assert steps is not None
    np.testing.assert_allclose(np.asarray(steps), 0.25 / np.asarray([8, 4, 16]))
    # The work budget cannot coarsen a requested stability refinement.
    fast = _predictive_sde_config(jnp.asarray(1e8), 60).brownian.step_size
    np.testing.assert_allclose(np.asarray(fast), 2.5e-9)
    order, inverse = _predictive_draw_order(rates, 60)
    np.testing.assert_array_equal(order, [1, 0, 2])
    np.testing.assert_array_equal(rates[order][inverse], rates)
