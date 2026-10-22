"""Fixed model parameters do not declare sample sites."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from grassmoth.artifacts.model.construct import replace_constructs
from grassmoth.artifacts.model.expressions import (
    Expression,
    coefficient,
    restoring_force,
    state,
)
from grassmoth.artifacts.model.likelihood import NormalLawSpec
from grassmoth_test_support.inference import compile_model_fixture
from grassmoth_test_support.model import (
    construct_named,
    indicator_named,
    likelihood_named,
    load_model_fixture,
)


def _all_fixed_spec_yields_no_sites__all_fixed_spec() -> DynamicalModelSpec:
    dynamical_model_spec = load_model_fixture(
        "runtime_ssm/testssmmodeldynamicsdispatch_test_nonlinear_dynamics_uses_vector_field_backend_method_model_fixture.json"
    )
    latent_0 = construct_named(dynamical_model_spec, "latent_0")
    manifest_0 = indicator_named(dynamical_model_spec, "manifest_0")
    manifest_0_likelihood = likelihood_named(dynamical_model_spec, "manifest_0")
    (latent_0_drift,) = latent_0.dynamics
    latent_1 = construct_named(dynamical_model_spec, "latent_1")
    manifest_1 = indicator_named(dynamical_model_spec, "manifest_1")
    manifest_1_likelihood = likelihood_named(dynamical_model_spec, "manifest_1")
    (latent_1_drift,) = latent_1.dynamics
    manifest_0_revised = manifest_0.revised(
        likelihood=manifest_0_likelihood.revised(
            law=NormalLawSpec[Expression](
                loc=(
                    coefficient(0.0, "observation_intercept")
                    + (coefficient(1.0, "loading") * state(latent_0.id))
                ),
                scale=coefficient(1.0, "observation_scale"),
            )
        )
    )
    latent_0_revised = latent_0.revised(
        indicators=(manifest_0_revised,),
        dynamics=(
            latent_0_drift.revised(
                expression=restoring_force(latent_0.id, center=0.0, stiffness=1.0, quartic=0.0)
            ),
        ),
        coefficients=(
            coefficient(1.0, "diffusion_scale"),
            coefficient(0.0, "initial_mean"),
            coefficient(1.0, "initial_scale"),
        ),
    )
    manifest_1_revised = manifest_1.revised(
        likelihood=manifest_1_likelihood.revised(
            law=NormalLawSpec[Expression](
                loc=(
                    coefficient(0.0, "observation_intercept")
                    + (coefficient(1.0, "loading") * state(latent_1.id))
                ),
                scale=coefficient(1.0, "observation_scale"),
            )
        )
    )
    latent_1_revised = latent_1.revised(
        indicators=(manifest_1_revised,),
        dynamics=(
            latent_1_drift.revised(
                expression=restoring_force(latent_1.id, center=0.0, stiffness=1.0, quartic=0.0)
            ),
        ),
        coefficients=(
            coefficient(1.0, "diffusion_scale"),
            coefficient(0.0, "initial_mean"),
            coefficient(1.0, "initial_scale"),
        ),
    )
    return dynamical_model_spec.with_entities(
        edges=replace_constructs(
            dynamical_model_spec.edges,
            (
                latent_0_revised,
                latent_1_revised,
            ),
        )
    )


if TYPE_CHECKING:
    from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec


pytestmark = pytest.mark.contract


def test_all_fixed_spec_yields_no_sites():
    dynamical_model_spec = _all_fixed_spec_yields_no_sites__all_fixed_spec()
    assert list(compile_model_fixture(dynamical_model_spec).site_registry) == []
