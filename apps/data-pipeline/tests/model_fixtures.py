"""Test-owned DynamicalModelSpec construction helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.dynamical_model_spec import DynamicalModelSpec
from nof1_causal_lab.artifacts.expressions import Expression, coefficient, restoring_force, state
from nof1_causal_lab.artifacts.likelihood import DeltaLawSpec
from nof1_causal_lab.artifacts.mechanism import DriftMechanismSpec
from nof1_causal_lab.artifacts.parameter import SiteKind

if TYPE_CHECKING:
    from collections.abc import Mapping

    from nof1_causal_lab.artifacts.construct import ConstructSpec
    from nof1_causal_lab.artifacts.identity import DistributionId, ParameterId
    from nof1_causal_lab.artifacts.indicator import IndicatorSpec
    from nof1_causal_lab.artifacts.likelihood import LikelihoodSpec
    from nof1_causal_lab.artifacts.parameter_spec import ParameterSpec
    from nof1_causal_lab.numpyro_json import NumPyroDistribution


def load_model_fixture(name: str) -> DynamicalModelSpec:
    """Parse one current-format canonical model without study converters."""
    return DynamicalModelSpec.model_validate_json(
        (Path(__file__).parent / "fixtures/models" / name).read_text()
    ).materialized()


def construct_named(dynamical_model_spec: DynamicalModelSpec, name: str) -> ConstructSpec:
    return next(node for node in dynamical_model_spec.constructs if node.name == name)


def indicator_named(dynamical_model_spec: DynamicalModelSpec, name: str) -> IndicatorSpec:
    return next(
        indicator
        for _, indicator in dynamical_model_spec.iter_indicators()
        if indicator.observation.name == name
    )


def likelihood_named(dynamical_model_spec: DynamicalModelSpec, name: str) -> LikelihoodSpec:
    return next(
        likelihood
        for indicator, likelihood in dynamical_model_spec.iter_likelihoods()
        if indicator.observation.name == name
    )


def parameter_named(dynamical_model_spec: DynamicalModelSpec, name: str) -> ParameterSpec:
    return next(
        parameter for parameter in dynamical_model_spec.parameters if parameter.name == name
    )


def parameter_for(
    dynamical_model_spec: DynamicalModelSpec, quantity: SiteKind, *owners: str
) -> ParameterSpec:
    names: dict[str, str] = {node.id: node.name for node in dynamical_model_spec.constructs} | {
        indicator.observation.id: indicator.observation.name
        for _, indicator in dynamical_model_spec.iter_indicators()
    }
    (parameter,) = tuple(
        p
        for p in dynamical_model_spec.parameters
        if dynamical_model_spec.parameter_context(p.id).quantity == quantity
        and {
            names[ref.id]
            for ref in dynamical_model_spec.parameter_context(p.id).owners
            if ref.id in names
        }
        == set(owners)
    )
    return parameter


def replace_parameters(
    parameters: tuple[ParameterSpec, ...], *revisions: ParameterSpec
) -> tuple[ParameterSpec, ...]:
    updated = {parameter.id: parameter for parameter in revisions}
    return tuple(updated.get(parameter.id, parameter) for parameter in parameters)


def without_parameters(
    dynamical_model_spec: DynamicalModelSpec, *removed: ParameterSpec
) -> tuple[tuple[ParameterSpec, ...], Mapping[DistributionId, NumPyroDistribution]]:
    ids = {parameter.id for parameter in removed}
    retained = tuple(
        parameter for parameter in dynamical_model_spec.parameters if parameter.id not in ids
    )
    used = {parameter.distribution for parameter in retained} | {
        node.distribution for node in dynamical_model_spec.constructs
    }
    return retained, {
        identity: law
        for identity, law in dynamical_model_spec.distributions.items()
        if identity in used
    }


def parameter_laws(
    dynamical_model_spec: DynamicalModelSpec, updates: Mapping[ParameterId, NumPyroDistribution]
) -> Mapping[DistributionId, NumPyroDistribution]:
    replacements = {}
    for identity, law in updates.items():
        distribution = dynamical_model_spec.parameter(identity).distribution
        assert distribution is not None, "Fixture parameter must own a law"
        replacements[distribution] = law
    return {**dynamical_model_spec.distributions, **replacements}


def x_model() -> DynamicalModelSpec:
    return load_model_fixture("common/x_model.json")


def x_y_model() -> DynamicalModelSpec:
    return load_model_fixture("common/x_y_model.json")


def stress_sleep_model() -> DynamicalModelSpec:
    return load_model_fixture("common/stress_sleep_model.json")


def stress_sleep_causal_model() -> DynamicalModelSpec:
    """The two measured states with Sleep as the question's downstream outcome."""
    from nof1_causal_lab.artifacts.identity import MechanismId

    dynamical_model_spec = stress_sleep_model()
    stress = construct_named(dynamical_model_spec, "Stress")
    return dynamical_model_spec.with_entities(
        edges=(
            dynamical_model_spec.edges[0].revised(
                effect=construct_named(dynamical_model_spec, "Sleep"),
                mechanisms=(
                    DriftMechanismSpec(
                        id=MechanismId("mechanism:stress_to_sleep"),
                        expression=coefficient(0.1, role="weight") * state(stress.id),
                    ),
                ),
            ),
        )
    )


def one_state_gaussian_model() -> DynamicalModelSpec:
    return load_model_fixture("common/one_state_gaussian_model.json")


def two_state_gaussian_model() -> DynamicalModelSpec:
    return load_model_fixture("common/two_state_gaussian_model.json")


def additive_a_b_model() -> DynamicalModelSpec:
    return load_model_fixture("common/additive_a_b_model.json")


def three_state_gaussian_model() -> DynamicalModelSpec:
    return load_model_fixture(
        "dag_to_ssm/testdynamicsmask_test_dynamics_support_zeros_non_edges__make_3latent_spec.json"
    )


def fixed_hill_model() -> DynamicalModelSpec:
    return load_model_fixture("simulation_checks/fixed_hill_model.json")


def mixed_family_model() -> DynamicalModelSpec:
    return load_model_fixture("observation_support/mixed_family_model.json")


def _exact_model_model() -> DynamicalModelSpec:
    dynamical_model_spec = load_model_fixture(
        "delta_observations/authored_affine_delta_keeps_its_calibration_coefficients_complete_model.json"
    )
    setting = construct_named(dynamical_model_spec, "setting")
    setting_obs = indicator_named(dynamical_model_spec, "setting_obs")
    setting_obs_likelihood = likelihood_named(dynamical_model_spec, "setting_obs")
    manifest_mean_setting_obs = parameter_named(dynamical_model_spec, "manifest_mean_setting_obs")
    setting_obs_revised = setting_obs.revised(
        likelihood=setting_obs_likelihood.revised(
            law=DeltaLawSpec[Expression](v=state(setting.id)),
            reasoning="The recorded setting is exact at its observation anchor.",
        )
    )
    setting_revised = setting.revised(indicators=(setting_obs_revised,))
    parameters, distributions = without_parameters(dynamical_model_spec, manifest_mean_setting_obs)
    return dynamical_model_spec.with_entities(
        edges=replace_constructs(dynamical_model_spec.edges, (setting_revised,)),
        parameters=parameters,
        distributions=distributions,
    )


def _two_state_fixed_drift_model() -> DynamicalModelSpec:
    dynamical_model_spec = load_model_fixture(
        "dynamics_config/scientific_model_roundtrip_preserves_derived_dynamics_model_fixture.json"
    )
    latent_0 = construct_named(dynamical_model_spec, "latent_0")
    (latent_0_potential,) = latent_0.dynamics
    latent_0_dynamics_decay = parameter_for(
        dynamical_model_spec, SiteKind.DYNAMICS_DECAY, "latent_0"
    )
    latent_0_latent_1_hill_emax = parameter_for(
        dynamical_model_spec, SiteKind.HILL_EMAX, "latent_0", "latent_1"
    )
    latent_0_latent_1_hill_n = parameter_for(
        dynamical_model_spec, SiteKind.HILL_N, "latent_0", "latent_1"
    )
    latent_0_latent_1_hill_ec50 = parameter_for(
        dynamical_model_spec, SiteKind.HILL_EC50, "latent_0", "latent_1"
    )
    latent_0_revised = latent_0.revised(
        dynamics=(
            DriftMechanismSpec(
                id=latent_0_potential.id,
                expression=restoring_force(
                    latent_0.id, center=0.0, stiffness=latent_0_dynamics_decay.id, quartic=0.0
                ),
            ),
        )
    )
    parameters, distributions = without_parameters(
        dynamical_model_spec,
        latent_0_latent_1_hill_emax,
        latent_0_latent_1_hill_n,
        latent_0_latent_1_hill_ec50,
    )
    return dynamical_model_spec.with_entities(
        edges=replace_constructs(
            tuple(
                edge
                for edge in dynamical_model_spec.edges
                if (edge.cause.name, edge.effect.name) not in (("latent_0", "latent_1"),)
            ),
            (latent_0_revised,),
        ),
        parameters=parameters,
        distributions=distributions,
    )
