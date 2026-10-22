"""Stable scientific subjects through authoring, compilation, and retained results."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest
from pydantic import TypeAdapter

from grassmoth.artifacts.identity import DistributionId, ParameterId
from grassmoth.artifacts.model.construct import replace_constructs
from grassmoth.artifacts.model.expressions import Expression, coefficient, state
from grassmoth.artifacts.model.likelihood import (
    LikelihoodSpec,
    ObservationLawSpec,
    StudentTLawSpec,
)
from grassmoth.artifacts.model.parameter import SiteKind
from grassmoth.artifacts.model.parameter_spec import ParameterSpec
from grassmoth.artifacts.probability.families import DistributionFamily
from grassmoth.models.ssm.compile.bindings import parameter_bindings
from grassmoth.models.structure import StructuralSelection, selected_state_ids
from tests.support.inference import compile_model_fixture
from tests.support.model import (
    additive_a_b_model,
    construct_named,
    indicator_named,
    likelihood_named,
    load_model_fixture,
    parameter_named,
    replace_parameters,
)


def _shared_likelihood_parameter_owns_only_active_channels_complete_model() -> DynamicalModelSpec:
    _OBS_DF_PARAMETER_ID = ParameterId(
        "parameter:e9c414236abf8cbc520db01a7912d59c8e29aa890ad71833dceb4f3255f67e2e"
    )
    _OBS_DF_DISTRIBUTION_ID = DistributionId(
        "distribution:97e1aff2ec6681a794c808b35e7b30e9609ba25913dbfc5351baa0398a40b016"
    )
    dynamical_model_spec = additive_a_b_model()
    a = construct_named(dynamical_model_spec, "A")
    a_obs = indicator_named(dynamical_model_spec, "A_obs")
    a_obs_likelihood = likelihood_named(dynamical_model_spec, "A_obs")
    a_obs_revised = a_obs.revised(
        likelihood=a_obs_likelihood.revised(
            law=StudentTLawSpec[Expression](
                df=coefficient(_OBS_DF_PARAMETER_ID, "degrees_of_freedom"),
                loc=(
                    coefficient(0.0, "observation_intercept")
                    + (coefficient(1.0, "loading") * state(a.id))
                ),
                scale=coefficient(0.0, "observation_scale"),
            ),
            reasoning="Test tails",
        )
    )
    a_revised = a.revised(indicators=(a_obs_revised,))
    return dynamical_model_spec.with_entities(
        edges=replace_constructs(dynamical_model_spec.edges, (a_revised,)),
        parameters=(
            *dynamical_model_spec.parameters,
            ParameterSpec(
                id=_OBS_DF_PARAMETER_ID,
                name="obs_df",
                description="degrees of freedom for obs_df",
                distribution=_OBS_DF_DISTRIBUTION_ID,
            ),
        ),
        distributions={
            **dynamical_model_spec.distributions,
            _OBS_DF_DISTRIBUTION_ID: dist.Gamma(concentration=5.0, rate=1.0, validate_args=False),
        },
    )


def _shared_likelihood_parameter_owns_only_active_channels_complete_model_2() -> DynamicalModelSpec:
    dynamical_model_spec = _shared_likelihood_parameter_owns_only_active_channels_complete_model()
    b = construct_named(dynamical_model_spec, "B")
    b_obs = indicator_named(dynamical_model_spec, "B_obs")
    b_obs_likelihood = likelihood_named(dynamical_model_spec, "B_obs")
    obs_df = parameter_named(dynamical_model_spec, "obs_df")
    b_obs_revised = b_obs.revised(
        likelihood=b_obs_likelihood.revised(
            law=StudentTLawSpec[Expression](
                df=coefficient(obs_df.id, "degrees_of_freedom"),
                loc=(
                    coefficient(0.0, "observation_intercept")
                    + (coefficient(1.0, "loading") * state(b.id))
                ),
                scale=coefficient(0.0, "observation_scale"),
            ),
            reasoning="Test tails",
        )
    )
    b_revised = b.revised(indicators=(b_obs_revised,))
    return dynamical_model_spec.with_entities(
        edges=replace_constructs(dynamical_model_spec.edges, (b_revised,))
    )


def _student_innovation_model() -> DynamicalModelSpec:
    _PROC_DF_PARAMETER_ID = ParameterId(
        "parameter:cda322d07f43753b6303431fe88a8d86238b54e2a7f366cecd07b25c754971b6"
    )
    _PROC_DF_DISTRIBUTION_ID = DistributionId(
        "distribution:8b5923f7a2b2d83dbc2d3961d34d000cfa280f81457d2e3b222014b086954b66"
    )
    dynamical_model_spec = additive_a_b_model()
    a = construct_named(dynamical_model_spec, "A")
    sigma_a = parameter_named(dynamical_model_spec, "sigma_A")
    b = construct_named(dynamical_model_spec, "B")
    sigma_b = parameter_named(dynamical_model_spec, "sigma_B")
    a_revised = a.revised(
        coefficients=(
            coefficient(sigma_a.id, "diffusion_scale"),
            coefficient(0.0, "initial_mean"),
            coefficient(1.0, "initial_scale"),
            coefficient(_PROC_DF_PARAMETER_ID, "process_degrees_of_freedom"),
        ),
        innovation_family=DistributionFamily.STUDENT_T,
    )
    b_revised = b.revised(
        coefficients=(
            coefficient(sigma_b.id, "diffusion_scale"),
            coefficient(0.0, "initial_mean"),
            coefficient(1.0, "initial_scale"),
            coefficient(_PROC_DF_PARAMETER_ID, "process_degrees_of_freedom"),
        ),
        innovation_family=DistributionFamily.STUDENT_T,
    )
    return dynamical_model_spec.with_entities(
        edges=replace_constructs(
            dynamical_model_spec.edges,
            (
                a_revised,
                b_revised,
            ),
        ),
        parameters=(
            *dynamical_model_spec.parameters,
            ParameterSpec(
                id=_PROC_DF_PARAMETER_ID,
                name="proc_df",
                description="degrees of freedom for proc_df",
                distribution=_PROC_DF_DISTRIBUTION_ID,
            ),
        ),
        distributions={
            **dynamical_model_spec.distributions,
            _PROC_DF_DISTRIBUTION_ID: dist.Gamma(concentration=5.0, rate=1.0, validate_args=False),
        },
    )


def _rename_preserves_parameter_and_element_identity__compile_2() -> DynamicalModelSpec:
    dynamical_model_spec = additive_a_b_model()
    a = construct_named(dynamical_model_spec, "A")
    a_obs = indicator_named(dynamical_model_spec, "A_obs")
    b = construct_named(dynamical_model_spec, "B")
    b_obs = indicator_named(dynamical_model_spec, "B_obs")
    rho_a = parameter_named(dynamical_model_spec, "rho_A")
    rho_b = parameter_named(dynamical_model_spec, "rho_B")
    beta_a_b = parameter_named(dynamical_model_spec, "beta_A_B")
    sigma_a = parameter_named(dynamical_model_spec, "sigma_A")
    sigma_b = parameter_named(dynamical_model_spec, "sigma_B")
    a_obs_revised = a_obs.revised(
        observation=a_obs.observation.revised(name="renamed measurement 0")
    )
    a_revised = a.revised(name="renamed construct 0", indicators=(a_obs_revised,))
    b_obs_revised = b_obs.revised(
        observation=b_obs.observation.revised(name="renamed measurement 1")
    )
    b_revised = b.revised(name="renamed construct 1", indicators=(b_obs_revised,))
    return dynamical_model_spec.with_entities(
        edges=replace_constructs(
            dynamical_model_spec.edges,
            (
                a_revised,
                b_revised,
            ),
        ),
        parameters=replace_parameters(
            dynamical_model_spec.parameters,
            rho_a.revised(
                name="rho_renamed construct 0", description="stiffness of rho_renamed construct 0"
            ),
            rho_b.revised(
                name="rho_renamed construct 1", description="stiffness of rho_renamed construct 1"
            ),
            beta_a_b.revised(
                name="beta_renamed construct 0_renamed construct 1",
                description="weight of beta_renamed construct 0_renamed construct 1",
            ),
            sigma_a.revised(
                name="sigma_renamed construct 0",
                description="innovation.scale for sigma_renamed construct 0",
            ),
            sigma_b.revised(
                name="sigma_renamed construct 1",
                description="innovation.scale for sigma_renamed construct 1",
            ),
        ),
    )


def _additive_hill_and_linear_terms_survive_parameter_renaming_with_parameter_distributions() -> (
    DynamicalModelSpec
):
    return load_model_fixture(
        "scientific_parameter_identity/additive_hill_and_linear_terms_survive_parameter_renaming_with_parameter_distributions.json"
    )


def _initial_state_defaults_are_authored_before_compilation_complete_model_initial_mean() -> (
    DynamicalModelSpec
):
    _T0_SD_A_PARAMETER_ID = ParameterId(
        "parameter:78a820c678a86f96a375ac708343df8e24b0d62b74d344a2ab12637121eadd5e"
    )
    _T0_SD_B_PARAMETER_ID = ParameterId(
        "parameter:2314214a83c2bf4026ec9a51ea9d67f25eca66e6e8f9a21aca204f1009a31302"
    )
    _T0_SD_A_DISTRIBUTION_ID = DistributionId(
        "distribution:ce98d58c7dc08717d8b053705c0a6ab7c00b7b2f0d4fa6142ac250c346e7778c"
    )
    _T0_SD_B_DISTRIBUTION_ID = DistributionId(
        "distribution:4405cbecb2b23c024b8f1ca97701f3be9df5104733d4425b50e1955801b005d1"
    )
    dynamical_model_spec = additive_a_b_model()
    a = construct_named(dynamical_model_spec, "A")
    sigma_a = parameter_named(dynamical_model_spec, "sigma_A")
    b = construct_named(dynamical_model_spec, "B")
    sigma_b = parameter_named(dynamical_model_spec, "sigma_B")
    a_revised = a.revised(
        coefficients=(
            coefficient(sigma_a.id, "diffusion_scale"),
            coefficient(0.0, "initial_mean"),
            coefficient(_T0_SD_A_PARAMETER_ID, "initial_scale"),
        )
    )
    b_revised = b.revised(
        coefficients=(
            coefficient(sigma_b.id, "diffusion_scale"),
            coefficient(0.0, "initial_mean"),
            coefficient(_T0_SD_B_PARAMETER_ID, "initial_scale"),
        )
    )
    return dynamical_model_spec.with_entities(
        edges=replace_constructs(
            dynamical_model_spec.edges,
            (
                a_revised,
                b_revised,
            ),
        ),
        parameters=(
            *dynamical_model_spec.parameters,
            ParameterSpec(
                id=_T0_SD_A_PARAMETER_ID,
                name="t0_sd_A",
                description="initial state.scale for t0_sd_A",
                distribution=_T0_SD_A_DISTRIBUTION_ID,
            ),
            ParameterSpec(
                id=_T0_SD_B_PARAMETER_ID,
                name="t0_sd_B",
                description="initial state.scale for t0_sd_B",
                distribution=_T0_SD_B_DISTRIBUTION_ID,
            ),
        ),
        distributions={
            **dynamical_model_spec.distributions,
            _T0_SD_A_DISTRIBUTION_ID: dist.HalfNormal(scale=2.0, validate_args=False),
            _T0_SD_B_DISTRIBUTION_ID: dist.HalfNormal(scale=2.0, validate_args=False),
        },
    )


def _initial_state_defaults_are_authored_before_compilation_complete_model_initial_scale() -> (
    DynamicalModelSpec
):
    _T0_MEAN_A_PARAMETER_ID = ParameterId(
        "parameter:1776151c61f063b055fe8347ef429b23283aba9338c3b772714bc055b2f26cf4"
    )
    _T0_MEAN_B_PARAMETER_ID = ParameterId(
        "parameter:40b11a0aae69fc9e9dad5839abc5f20dea0e5b49ebcc2c6e2da03a5dd18cb5c3"
    )
    _T0_MEAN_A_DISTRIBUTION_ID = DistributionId(
        "distribution:646704dcb25d1ed8816be8aef4ee24c8f54efad46bfedf9391c03de528a9c394"
    )
    _T0_MEAN_B_DISTRIBUTION_ID = DistributionId(
        "distribution:b600d6c6f149d7851b1186b692be93f2838f38762e670f3e4d130a37f5a128b0"
    )
    dynamical_model_spec = additive_a_b_model()
    a = construct_named(dynamical_model_spec, "A")
    sigma_a = parameter_named(dynamical_model_spec, "sigma_A")
    b = construct_named(dynamical_model_spec, "B")
    sigma_b = parameter_named(dynamical_model_spec, "sigma_B")
    a_revised = a.revised(
        coefficients=(
            coefficient(sigma_a.id, "diffusion_scale"),
            coefficient(1.0, "initial_scale"),
            coefficient(_T0_MEAN_A_PARAMETER_ID, "initial_mean"),
        )
    )
    b_revised = b.revised(
        coefficients=(
            coefficient(sigma_b.id, "diffusion_scale"),
            coefficient(1.0, "initial_scale"),
            coefficient(_T0_MEAN_B_PARAMETER_ID, "initial_mean"),
        )
    )
    return dynamical_model_spec.with_entities(
        edges=replace_constructs(
            dynamical_model_spec.edges,
            (
                a_revised,
                b_revised,
            ),
        ),
        parameters=(
            *dynamical_model_spec.parameters,
            ParameterSpec(
                id=_T0_MEAN_A_PARAMETER_ID,
                name="t0_mean_A",
                description="initial state.mean for t0_mean_A",
                distribution=_T0_MEAN_A_DISTRIBUTION_ID,
            ),
            ParameterSpec(
                id=_T0_MEAN_B_PARAMETER_ID,
                name="t0_mean_B",
                description="initial state.mean for t0_mean_B",
                distribution=_T0_MEAN_B_DISTRIBUTION_ID,
            ),
        ),
        distributions={
            **dynamical_model_spec.distributions,
            _T0_MEAN_A_DISTRIBUTION_ID: dist.Normal(loc=0.0, scale=2.0, validate_args=False),
            _T0_MEAN_B_DISTRIBUTION_ID: dist.Normal(loc=0.0, scale=2.0, validate_args=False),
        },
    )


def _initial_state_defaults_are_authored_before_compilation_complete_model_none() -> (
    DynamicalModelSpec
):
    return load_model_fixture(
        "scientific_parameter_identity/initial_state_defaults_are_authored_before_compilation_complete_model_none.json"
    )


def _ordinal_components_have_label_identity_and_padding_is_explicit__compile() -> (
    DynamicalModelSpec
):
    return load_model_fixture(
        "scientific_parameter_identity/ordinal_components_have_label_identity_and_padding_is_explicit__compile.json"
    )


def _scalar_identity_survives_reordered_execution_axes__compile_2() -> DynamicalModelSpec:
    return load_model_fixture(
        "scientific_parameter_identity/scalar_identity_survives_reordered_execution_axes__compile_2.json"
    )


def _scalar_identity_survives_reordered_execution_axes__compile() -> DynamicalModelSpec:
    dynamical_model_spec = _scalar_identity_survives_reordered_execution_axes__compile_2()
    a = construct_named(dynamical_model_spec, "A")
    b = construct_named(dynamical_model_spec, "B")
    a_to_b = next(
        edge
        for edge in dynamical_model_spec.edges
        if edge.cause.id == a.id and edge.effect.id == b.id
    )
    b_to_a = next(
        edge
        for edge in dynamical_model_spec.edges
        if edge.cause.id == b.id and edge.effect.id == a.id
    )
    rho_a = parameter_named(dynamical_model_spec, "rho_A")
    rho_b = parameter_named(dynamical_model_spec, "rho_B")
    beta_a_b = parameter_named(dynamical_model_spec, "beta_A_B")
    beta_b_a = parameter_named(dynamical_model_spec, "beta_B_A")
    sigma_a = parameter_named(dynamical_model_spec, "sigma_A")
    sigma_b = parameter_named(dynamical_model_spec, "sigma_B")
    return dynamical_model_spec.with_entities(
        edges=(
            a_to_b,
            b_to_a,
        ),
        parameters=(
            rho_a,
            rho_b,
            beta_a_b,
            beta_b_a,
            sigma_a,
            sigma_b,
        ),
    )


if TYPE_CHECKING:
    from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec


@pytest.mark.contract
def test_rename_preserves_parameter_and_element_identity():
    old_model = additive_a_b_model()
    new_model = _rename_preserves_parameter_and_element_identity__compile_2()
    assert {
        b.parameter_id: set(b.elements)
        for b in parameter_bindings(compile_model_fixture(old_model))[0]
    } == {
        b.parameter_id: set(b.elements)
        for b in parameter_bindings(compile_model_fixture(new_model))[0]
    }
    assert {p.name for p in old_model.parameters} != {p.name for p in new_model.parameters}


@pytest.mark.contract
def test_scalar_identity_survives_reordered_execution_axes():
    dynamical_model_spec = _scalar_identity_survives_reordered_execution_axes__compile()
    reordered = _scalar_identity_survives_reordered_execution_axes__compile_2()
    assert selected_state_ids(StructuralSelection(dynamical_model_spec, None)) == tuple(
        reversed(selected_state_ids(StructuralSelection(reordered, None)))
    )
    decay = next(
        p
        for p in dynamical_model_spec.parameters
        if dynamical_model_spec.parameter_context(p.id).quantity == SiteKind.DYNAMICS_DECAY
    )
    old_binding = next(
        b
        for b in parameter_bindings(compile_model_fixture(dynamical_model_spec))[0]
        if b.parameter_id == decay.id
    )
    new_binding = next(
        b
        for b in parameter_bindings(compile_model_fixture(reordered))[0]
        if b.parameter_id == decay.id
    )
    assert old_binding.elements == new_binding.elements
    assert old_binding.coordinates != new_binding.coordinates


@pytest.mark.contract
def test_model_rejects_forged_owner_before_compilation():
    dynamical_model_spec = additive_a_b_model()
    payload = dynamical_model_spec.model_dump(mode="json")
    next(iter(payload["parameters"].values()))["owners"] = [
        {"kind": "construct", "id": "construct:forged"}
    ]
    with pytest.raises(ValueError, match=r"Extra inputs|owner"):
        _ = type(dynamical_model_spec).model_validate(payload).parameters


@pytest.mark.contract
def test_posterior_writer_uses_declared_subject_and_rejects_unknown_coordinate():
    dynamical_model_spec = additive_a_b_model()
    binding = parameter_bindings(compile_model_fixture(dynamical_model_spec))[0][0]
    element, coordinate = next(iter(binding.coordinates.items()))
    from grassmoth.artifacts.identity import ParameterRef
    from grassmoth.artifacts.model.parameter import ParameterCoordinate
    from grassmoth.models.ssm.inference.diagnostics.plots import compute_posterior_marginals

    label = binding.elements[element]
    subject = ParameterRef(parameter_id=binding.parameter_id, element_id=element)
    references = {coordinate: (label, subject)}
    # The numerical producer attaches the reference directly; no serialized row is rebound.
    shape = tuple(i + 1 for i in coordinate.indices)
    samples = jnp.zeros((4, *shape))
    marginals = compute_posterior_marginals({coordinate.site_name: samples}, references, 5)
    assert next(row for row in marginals if row.subject == subject).parameter == label
    with pytest.raises(KeyError):
        compute_posterior_marginals({"unknown": jnp.zeros(4)}, references, 5)
    assert ParameterCoordinate(site_name="unknown", indices=()) not in references


@pytest.mark.contract
def test_ordinal_components_have_label_identity_and_padding_is_explicit():
    dynamical_model_spec = (
        _ordinal_components_have_label_identity_and_padding_is_explicit__compile()
    )
    gaps = {
        p.id
        for p in dynamical_model_spec.parameters
        if dynamical_model_spec.parameter_context(p.id).quantity == SiteKind.OBS_ORDERED_GAPS
    }
    gap_bindings = [
        b
        for b in parameter_bindings(compile_model_fixture(dynamical_model_spec))[0]
        if b.parameter_id in gaps
    ]
    assert len(gap_bindings) == 1
    assert list(gap_bindings[0].elements.values()) == ["A_obs: gap low / mid / high"]
    assert parameter_bindings(compile_model_fixture(dynamical_model_spec))[1]


@pytest.mark.contract
def test_shared_likelihood_parameter_owns_only_active_channels():
    dynamical_model_spec = additive_a_b_model()
    first, second = dynamical_model_spec.constructs
    student = LikelihoodSpec(
        law=TypeAdapter(ObservationLawSpec).validate_json(
            (
                Path(__file__).resolve().parents[3]
                / "fixtures/models"
                / "scientific_parameter_identity/shared_likelihood_parameter_owns_only_active_channels_observation_law.json"
            ).read_text()
        ),
        reasoning="Test tails",
        standardized=True,
    )
    first = first.revised(indicators=(first.indicators[0].revised(likelihood=student),))
    dynamical_model_spec = _shared_likelihood_parameter_owns_only_active_channels_complete_model()
    shared = next(
        p
        for p in dynamical_model_spec.parameters
        if dynamical_model_spec.parameter_context(p.id).quantity == SiteKind.OBS_DF
    )
    assert {o.id for o in dynamical_model_spec.parameter_context(shared.id).owners} == {
        first.id,
        first.indicators[0].observation.id,
    }
    second = second.revised(
        indicators=(
            second.indicators[0].revised(
                likelihood=LikelihoodSpec(
                    law=TypeAdapter(ObservationLawSpec).validate_json(
                        (
                            Path(__file__).resolve().parents[3]
                            / "fixtures/models"
                            / "scientific_parameter_identity/shared_likelihood_parameter_owns_only_active_channels_observation_law_2.json"
                        ).read_text()
                    ),
                    reasoning="Test tails",
                    standardized=True,
                )
            ),
        )
    )
    expanded = _shared_likelihood_parameter_owns_only_active_channels_complete_model_2()
    newer = next(
        p
        for p in expanded.parameters
        if expanded.parameter_context(p.id).quantity == SiteKind.OBS_DF
    )
    assert newer.id == shared.id
    assert newer.distribution == shared.distribution
    assert {o.id for o in expanded.parameter_context(newer.id).owners} == {
        first.id,
        second.id,
        first.indicators[0].observation.id,
        second.indicators[0].observation.id,
    }
    compile_model_fixture(expanded)


@pytest.mark.contract
def test_student_innovation_tail_is_explicit_and_shared_through_completion():

    dynamical_model_spec = _student_innovation_model()
    parameter = next(p for p in dynamical_model_spec.parameters if p.name == "proc_df")
    for construct in dynamical_model_spec.constructs:
        assert parameter in dynamical_model_spec.parameters_for(construct.id)
    compile_model_fixture(dynamical_model_spec)
    first, second = dynamical_model_spec.constructs
    candidate = dynamical_model_spec.with_entities(
        edges=replace_constructs(
            dynamical_model_spec.edges,
            (
                first.revised(
                    coefficients=tuple(
                        operand
                        for operand in first.coefficients
                        if operand.role != "process_degrees_of_freedom"
                    )
                ),
                second,
            ),
        )
    )
    with pytest.raises(ValueError, match="degrees_of_freedom requires a prior parameter"):
        compile_model_fixture(candidate)
    completed = _student_innovation_model()
    assert completed.parameter(parameter.id) == parameter
    assert completed.get_construct(first.id).coefficients == first.coefficients


@pytest.mark.contract
@pytest.mark.parametrize(
    ("retained_role", "_compile_payload", "complete_model_payload"),
    [
        pytest.param(
            None,
            additive_a_b_model,
            _initial_state_defaults_are_authored_before_compilation_complete_model_none,
            id="None",
        ),
        pytest.param(
            "initial_mean",
            additive_a_b_model,
            _initial_state_defaults_are_authored_before_compilation_complete_model_initial_mean,
            id="initial_mean",
        ),
        pytest.param(
            "initial_scale",
            additive_a_b_model,
            _initial_state_defaults_are_authored_before_compilation_complete_model_initial_scale,
            id="initial_scale",
        ),
    ],
)
def test_initial_state_defaults_are_authored_before_compilation(
    retained_role, _compile_payload, complete_model_payload
):
    model = _compile_payload()

    free = model.with_entities(
        edges=replace_constructs(
            model.edges,
            tuple(
                c.revised(
                    coefficients=tuple(
                        operand
                        for operand in c.coefficients
                        if not operand.role.startswith("initial_") or operand.role == retained_role
                    )
                )
                for c in model.constructs
            ),
        )
    )
    with pytest.raises(ValueError, match="initial-state coefficients"):
        compile_model_fixture(free)
    completed = complete_model_payload()
    before = completed.model_dump(mode="json")
    compile_model_fixture(completed)
    initial = [
        p
        for p in completed.parameters
        if completed.parameter_context(p.id).quantity in {SiteKind.T0_MEANS, SiteKind.T0_VAR_DIAG}
    ]
    assert initial
    assert all(p.distribution is not None for p in initial)
    assert {p.id for p in initial} <= {
        b.parameter_id for b in parameter_bindings(compile_model_fixture(completed))[0]
    }
    assert completed.model_dump(mode="json") == before
    assert all(
        "elements" not in p and "role" not in p and "constraint" not in p
        for p in before["parameters"].values()
    )


@pytest.mark.inference(concern="sampling")
def test_parameter_labels_do_not_change_mechanisms_bindings_or_prior_laws():
    dynamical_model_spec = additive_a_b_model()
    renamed = dynamical_model_spec.with_entities(
        parameters=tuple(
            p.revised(name=f"display {n}") for n, p in enumerate(dynamical_model_spec.parameters)
        )
    )
    compile_model_fixture(renamed)
    _assert_same_prior_laws(dynamical_model_spec, renamed)
    assert {
        b.parameter_id: b.coordinates
        for b in parameter_bindings(compile_model_fixture(dynamical_model_spec))[0]
    } == {
        b.parameter_id: b.coordinates for b in parameter_bindings(compile_model_fixture(renamed))[0]
    }


@pytest.mark.inference(concern="sampling")
def test_additive_hill_and_linear_terms_survive_parameter_renaming():
    dynamical_model_spec = additive_a_b_model()

    additive = (
        _additive_hill_and_linear_terms_survive_parameter_renaming_with_parameter_distributions()
    )
    compile_model_fixture(additive)
    renamed = additive.with_entities(
        parameters=tuple(p.revised(name=f"opaque {n}") for n, p in enumerate(additive.parameters))
    )
    compile_model_fixture(renamed)
    _assert_same_prior_laws(additive, renamed)
    components = compile_model_fixture(additive).dynamics.spec.components
    original_components = compile_model_fixture(dynamical_model_spec).dynamics.spec.components
    assert isinstance(components, tuple)
    assert isinstance(original_components, tuple)
    assert len(components) == len(original_components) + 1


def _assert_same_prior_laws(first, second):
    import numpy as np

    from grassmoth.models.ssm.compile.prior_compilation import compile_priors

    first_laws = compile_priors(compile_model_fixture(first), StructuralSelection(first, None))[0]
    second_laws = compile_priors(compile_model_fixture(second), StructuralSelection(second, None))[
        0
    ]
    assert first_laws.keys() == second_laws.keys()
    for name, law in first_laws.items():
        for value in (0.15, 0.5, 1.25):
            np.testing.assert_allclose(law.log_prob(value), second_laws[name].log_prob(value))
