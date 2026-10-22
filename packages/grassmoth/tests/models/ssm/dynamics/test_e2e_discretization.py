"""Model structure and discrete-time to continuous-time prior compilation.

Checks reference intervals, construct-specific priors, structural support,
and parameter identity.
"""

import math
from typing import Any

import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import polars as pl
import pytest

from grassmoth.artifacts.model.construct import replace_constructs
from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec
from grassmoth.artifacts.model.expressions import expression_coefficients
from grassmoth.artifacts.model.parameter import SiteKind
from grassmoth.artifacts.probability.native import prior_reference_value
from grassmoth.models.ssm import compiled_access as numeric
from grassmoth.models.ssm.compile.bindings import parameter_bindings
from grassmoth.models.ssm.compile.inputs import compile_priors as compile_ssm_priors
from grassmoth.models.structure import StructuralSelection
from tests.support.helpers import graph_constructs
from tests.support.inference import compile_fit_fixture, compile_model_fixture
from tests.support.model import (
    load_model_fixture,
    parameter_laws,
    parameter_named,
    replace_parameters,
)


def _teste2espectodiscretization_test_time_invariant_states_drop_static_target_dynamics_and_diffusion_su() -> (
    DynamicalModelSpec
):
    return load_model_fixture(
        "e2e_discretization/teste2espectodiscretization_test_time_invariant_states_drop_static_target_dynamics_and_diffusion_support_complete_test_model.json"
    )


def _weekly_effect_rate() -> DynamicalModelSpec:
    return load_model_fixture("e2e_discretization/weekly_effect_rate.json")


def _daily_effect_rate() -> DynamicalModelSpec:
    dynamical_model_spec = _weekly_effect_rate()
    beta_stress_mood = parameter_named(dynamical_model_spec, "beta_stress_mood")
    return dynamical_model_spec.with_entities(
        parameters=replace_parameters(
            dynamical_model_spec.parameters,
            beta_stress_mood.revised(
                transform=beta_stress_mood.transform.revised(interval_days=1.0)
            ),
        )
    )


def _equal_intervals_elementwise_priors() -> DynamicalModelSpec:
    dynamical_model_spec = _weekly_effect_rate()
    rho_stress = parameter_named(dynamical_model_spec, "rho_stress")
    rho_mood = parameter_named(dynamical_model_spec, "rho_mood")
    return dynamical_model_spec.with_entities(
        parameters=replace_parameters(
            dynamical_model_spec.parameters,
            rho_stress.revised(transform=rho_stress.transform.revised(interval_days=7.0)),
            rho_mood.revised(transform=rho_mood.transform.revised(interval_days=7.0)),
        ),
        distributions=parameter_laws(
            dynamical_model_spec,
            {
                rho_mood.id: dist.Beta(
                    concentration1=jnp.array(3.0, dtype=jnp.float32),
                    concentration0=jnp.array(2.0, dtype=jnp.float32),
                    validate_args=True,
                )
            },
        ),
    )


def _stress_mood_model() -> DynamicalModelSpec:
    dynamical_model_spec = _daily_effect_rate()
    beta_stress_mood = parameter_named(dynamical_model_spec, "beta_stress_mood")
    rho_stress = parameter_named(dynamical_model_spec, "rho_stress")
    rho_mood = parameter_named(dynamical_model_spec, "rho_mood")
    return dynamical_model_spec.with_entities(
        parameters=replace_parameters(
            dynamical_model_spec.parameters,
            beta_stress_mood.revised(
                transform=beta_stress_mood.transform.revised(interval_days="model_clock")
            ),
        ),
        distributions=parameter_laws(
            dynamical_model_spec,
            {
                rho_stress.id: dist.Beta(
                    concentration1=2.0, concentration0=2.0, validate_args=False
                ),
                rho_mood.id: dist.Beta(concentration1=2.0, concentration0=2.0, validate_args=False),
                beta_stress_mood.id: dist.Normal(loc=0.0, scale=0.5, validate_args=False),
            },
        ),
    )


def _construct_specific_residual_scales() -> DynamicalModelSpec:
    dynamical_model_spec = _daily_effect_rate()
    beta_stress_mood = parameter_named(dynamical_model_spec, "beta_stress_mood")
    rho_mood = parameter_named(dynamical_model_spec, "rho_mood")
    sigma_stress = parameter_named(dynamical_model_spec, "sigma_stress")
    lambda_stress_cortisol_stress = parameter_named(
        dynamical_model_spec, "lambda_stress_cortisol_stress"
    )
    sigma_mood = parameter_named(dynamical_model_spec, "sigma_mood")
    return dynamical_model_spec.with_entities(
        parameters=replace_parameters(
            dynamical_model_spec.parameters,
            beta_stress_mood.revised(
                transform=beta_stress_mood.transform.revised(interval_days="model_clock")
            ),
        ),
        distributions=parameter_laws(
            dynamical_model_spec,
            {
                rho_mood.id: dist.Beta(
                    concentration1=jnp.array(3.0, dtype=jnp.float32),
                    concentration0=jnp.array(2.0, dtype=jnp.float32),
                    validate_args=True,
                ),
                sigma_stress.id: dist.HalfNormal(
                    scale=jnp.array(0.8999999761581421, dtype=jnp.float32), validate_args=True
                ),
                lambda_stress_cortisol_stress.id: dist.Normal(
                    loc=jnp.array(0.800000011920929, dtype=jnp.float32),
                    scale=jnp.array(0.20000000298023224, dtype=jnp.float32),
                    validate_args=True,
                ),
                sigma_mood.id: dist.HalfNormal(
                    scale=jnp.array(0.10000000149011612, dtype=jnp.float32), validate_args=True
                ),
            },
        ),
    )


def _weekly_reference_intervals() -> DynamicalModelSpec:
    dynamical_model_spec = _equal_intervals_elementwise_priors()
    rho_stress = parameter_named(dynamical_model_spec, "rho_stress")
    sigma_stress = parameter_named(dynamical_model_spec, "sigma_stress")
    obs_sd_stress_self_report = parameter_named(dynamical_model_spec, "obs_sd_stress_self_report")
    lambda_stress_cortisol_stress = parameter_named(
        dynamical_model_spec, "lambda_stress_cortisol_stress"
    )
    obs_sd_stress_cortisol = parameter_named(dynamical_model_spec, "obs_sd_stress_cortisol")
    sigma_mood = parameter_named(dynamical_model_spec, "sigma_mood")
    return dynamical_model_spec.with_entities(
        parameters=replace_parameters(
            dynamical_model_spec.parameters,
            rho_stress.revised(transform=rho_stress.transform.revised(interval_days="model_clock")),
        ),
        distributions=parameter_laws(
            dynamical_model_spec,
            {
                sigma_stress.id: dist.HalfNormal(
                    scale=jnp.array(1.0, dtype=jnp.float32), validate_args=True
                ),
                obs_sd_stress_self_report.id: dist.HalfNormal(
                    scale=jnp.array(0.5, dtype=jnp.float32), validate_args=True
                ),
                lambda_stress_cortisol_stress.id: dist.HalfNormal(
                    scale=jnp.array(0.800000011920929, dtype=jnp.float32), validate_args=True
                ),
                obs_sd_stress_cortisol.id: dist.HalfNormal(
                    scale=jnp.array(0.5, dtype=jnp.float32), validate_args=True
                ),
                sigma_mood.id: dist.HalfNormal(
                    scale=jnp.array(1.0, dtype=jnp.float32), validate_args=True
                ),
            },
        ),
    )


pytestmark = pytest.mark.contract


def _compile_structure(payload: dict[str, Any]) -> DynamicalModelSpec:

    return DynamicalModelSpec.model_validate(payload).materialized()


def _compile_priors_for_test(scientific_model: DynamicalModelSpec):
    prior_registry, index_maps, _diagnostics = compile_ssm_priors(
        compile_model_fixture(scientific_model), StructuralSelection(scientific_model, None)
    )
    return prior_registry, index_maps


def _prior_reference_value(prior, flat_index: int = 0) -> float:
    return float(np.asarray(prior_reference_value(prior)).reshape(-1)[flat_index])


def _decay_reference_values(dynamical_model_spec: DynamicalModelSpec, prior_registry) -> np.ndarray:
    values = np.zeros(numeric.n_states(compile_model_fixture(dynamical_model_spec)), dtype=float)
    for index, component in enumerate(
        compile_model_fixture(dynamical_model_spec).dynamics.spec.components
    ):
        for _, site in component.parameter_sites(f"vf_{index}"):
            if site.site_kind == SiteKind.DYNAMICS_DECAY:
                values[component.target] += _prior_reference_value(prior_registry[site.name])
    return values


def _linear_edge_weight(
    dynamical_model_spec: DynamicalModelSpec, prior_registry, *, source: int, target: int
) -> float:
    for index, component in enumerate(
        compile_model_fixture(dynamical_model_spec).dynamics.spec.components
    ):
        if component.source == source and component.target == target:
            for _, site in component.parameter_sites(f"vf_{index}"):
                if site.site_kind == SiteKind.DYNAMICS_WEIGHT:
                    return _prior_reference_value(prior_registry[site.name])
    raise AssertionError(f"No linear coefficient for source={source}, target={target}")


def _decay_support(dynamical_model_spec: DynamicalModelSpec) -> np.ndarray:
    mask = np.zeros(numeric.n_states(compile_model_fixture(dynamical_model_spec)), dtype=bool)
    for component in compile_model_fixture(dynamical_model_spec).dynamics.spec.components:
        if any(
            operand.role == "decay" for operand in expression_coefficients(component.expression)
        ):
            mask[component.target] = True
    return mask


def _linear_edge_support(dynamical_model_spec: DynamicalModelSpec) -> np.ndarray:
    mask = np.zeros(
        (
            numeric.n_states(compile_model_fixture(dynamical_model_spec)),
            numeric.n_states(compile_model_fixture(dynamical_model_spec)),
        ),
        dtype=bool,
    )
    for component in compile_model_fixture(dynamical_model_spec).dynamics.spec.components:
        if component.source is not None and any(
            operand.role == "weight" for operand in expression_coefficients(component.expression)
        ):
            mask[component.target, component.source] = True
    return mask


def _state_intercept_mask(dynamical_model_spec: DynamicalModelSpec) -> np.ndarray:
    mask = np.zeros(numeric.n_states(compile_model_fixture(dynamical_model_spec)), dtype=bool)
    for component in compile_model_fixture(dynamical_model_spec).dynamics.spec.components:
        if not component.edge_owned and any(
            operand.role in {"center", "intercept"} for _, operand in component.parameters
        ):
            mask[component.target] = True
    return mask


# ═══════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def two_construct_structure() -> DynamicalModelSpec:
    """Realistic 2-construct causal design: stress → mood.

    - Both constructs are daily time-varying
    - 3 indicators: mood_rating, stress_self_report, stress_cortisol
    - stress_cortisol is a second indicator for stress (free loading)
    """
    return _compile_structure(
        {
            "constructs": {
                "construct:6b04dc42c531e7091eb8": {
                    "name": "stress",
                    "description": "Daily stress level",
                    "role": "endogenous",
                    "temporal_status": "time_varying",
                    "indicators": {
                        "indicator:4ff8be7491bd87d28af4": {
                            "observation": {
                                "name": "stress_self_report",
                                "measurement_dtype": "continuous",
                                "aggregation": "mean",
                            },
                            "construct_polarity": "positive",
                        },
                        "indicator:522342c2385e38d5e750": {
                            "observation": {
                                "name": "stress_cortisol",
                                "measurement_dtype": "continuous",
                                "aggregation": "mean",
                            },
                            "construct_polarity": "positive",
                        },
                    },
                },
                "construct:bbc87212909e45b9e6c3": {
                    "name": "mood",
                    "description": "Daily mood state",
                    "role": "endogenous",
                    "temporal_status": "time_varying",
                    "indicators": {
                        "indicator:e05e217de7f4442abdc5": {
                            "observation": {
                                "name": "mood_rating",
                                "measurement_dtype": "continuous",
                                "aggregation": "mean",
                            },
                            "construct_polarity": "positive",
                        }
                    },
                },
            },
            "edges": {
                "edge:923689028b6b177617c2": {
                    "cause": "construct:6b04dc42c531e7091eb8",
                    "effect": "construct:bbc87212909e45b9e6c3",
                    "description": "Stress impairs mood",
                }
            },
            "measurement_clock": "1d",
        }
    )


@pytest.fixture
def two_construct_model(two_construct_structure) -> DynamicalModelSpec:
    return _stress_mood_model()


# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: First-order DT→CT with interval_days
# ═══════════════════════════════════════════════════════════════════════


class TestE2ESpecToDiscretization:
    """End-to-end: DynamicalModelSpec → DynamicalModelSpec → dict[str, dist.Distribution] → discretize → roundtrip."""

    def test_source_model_structure_from_dag(self, two_construct_structure, two_construct_model):
        """Compilation produces correct DynamicalModelSpec from DAG structure."""

        # Dimensions
        assert numeric.n_states(compile_model_fixture(two_construct_model)) == 2  # mood, stress
        assert (
            numeric.n_observations(compile_model_fixture(two_construct_model)) == 3
        )  # mood_rating, stress_self_report, stress_cortisol
        assert numeric.state_names(compile_model_fixture(two_construct_model)) == ("stress", "mood")

        # Dynamics support: diagonal decay (AR) + stress→mood linear edge.
        np.testing.assert_array_equal(_decay_support(two_construct_model), [True, True])
        edge_support = _linear_edge_support(two_construct_model)
        assert edge_support[1, 0]  # stress→mood coupling (effect=mood row, cause=stress col)
        assert not edge_support[0, 1]  # no mood→stress edge

        # Lambda mask: stress_cortisol has free loading for stress
        assert compile_model_fixture(two_construct_model).loading_block.free_support is not None
        # mood_rating loads on mood (fixed=1.0), stress_self_report loads on stress (fixed=1.0)
        # stress_cortisol loads on stress (free)
        manifest_names = numeric.observation_names(compile_model_fixture(two_construct_model))
        assert manifest_names is not None
        stress_cortisol_idx = manifest_names.index("stress_cortisol")
        assert numeric.state_names(compile_model_fixture(two_construct_model)) is not None
        stress_latent_idx = numeric.state_names(compile_model_fixture(two_construct_model)).index(
            "stress"
        )
        assert compile_model_fixture(two_construct_model).loading_block.free_support[
            stress_cortisol_idx, stress_latent_idx
        ]

    def test_model_owns_latent_identity(self, two_construct_model):

        model = two_construct_model
        renamed = model.with_entities(
            edges=replace_constructs(
                model.edges,
                tuple(
                    c.revised(name="renamed") if c.name == "mood" else c for c in model.constructs
                ),
            )
        )
        spec = renamed
        assert numeric.state_names(compile_model_fixture(spec)) == ("stress", "renamed")
        assert numeric.state_ids(compile_model_fixture(spec)) == tuple(
            c.id for c in model.constructs
        )
        assert [p.id for p in renamed.parameters] == [p.id for p in model.parameters]

    def test_time_invariant_states_drop_static_target_dynamics_and_diffusion_support(self):

        dynamical_model_spec = _teste2espectodiscretization_test_time_invariant_states_drop_static_target_dynamics_and_diffusion_su()
        dynamical_model_spec = dynamical_model_spec
        assert not dynamical_model_spec.constructs[0].dynamics
        static_index = numeric.state_names(compile_model_fixture(dynamical_model_spec)).index(
            "baseline"
        )
        dynamic_index = numeric.state_names(compile_model_fixture(dynamical_model_spec)).index(
            "mood"
        )
        assert not _decay_support(dynamical_model_spec)[static_index]
        assert _decay_support(dynamical_model_spec)[dynamic_index]
        assert not compile_model_fixture(
            dynamical_model_spec
        ).diffusion_block.diffusion_chol_support[static_index, static_index]
        assert compile_model_fixture(dynamical_model_spec).diffusion_block.diffusion_chol_support[
            dynamic_index, dynamic_index
        ]
        assert not _linear_edge_support(dynamical_model_spec)[static_index].any()
        assert not _state_intercept_mask(dynamical_model_spec)[static_index]

    def test_model_rejects_mechanism_reference_outside_its_owners(self, two_construct_model):
        payload = two_construct_model.model_dump(mode="json")
        target = next(iter(graph_constructs(payload)[0]["dynamics"].values()))
        target["expression"] = {
            "kind": "coefficient",
            "role": "decay",
            "value": "parameter:foreign",
        }
        with pytest.raises(ValueError, match="parameter"):
            DynamicalModelSpec.model_validate(payload).materialized()

    def test_compiled_artifact_roundtrips_grounded_structure(
        self,
        two_construct_structure,
        two_construct_model,
    ):
        """Compiled artifacts preserve the grounded latent and measurement layout."""

        typed_scientific_model = DynamicalModelSpec.model_validate(
            two_construct_model
        ).materialized()
        compile_model_fixture(_weekly_reference_intervals())

        assert numeric.state_names(compile_model_fixture(_weekly_reference_intervals())) == (
            "stress",
            "mood",
        )
        assert numeric.observation_names(compile_model_fixture(typed_scientific_model)) == (
            "stress_self_report",
            "stress_cortisol",
            "mood_rating",
        )
        binding_rows = [
            {
                "parameter": next(
                    p.name
                    for p in typed_scientific_model.parameters
                    if p.id == binding.parameter_id
                ),
                "site_name": binding.site.name,
                "flat_index": binding.flat_index,
            }
            for binding in parameter_bindings(compile_model_fixture(_weekly_reference_intervals()))[
                0
            ]
        ]
        bindings = {item["parameter"]: item for item in binding_rows}
        assert bindings["beta_stress_mood"]["site_name"] == "vf_2_p0"
        assert bindings["rho_mood"]["site_name"] == "vf_1_p0"
        assert bindings["rho_stress"]["site_name"] == "vf_0_p0"
        assert bindings["lambda_stress_cortisol_stress"]["flat_index"] == 0
        assert set(bindings) == {p.name for p in typed_scientific_model.parameters}

        data_for_model = pl.DataFrame(
            {
                "indicator": [
                    "mood_rating",
                    "stress_self_report",
                    "stress_cortisol",
                    "mood_rating",
                    "stress_self_report",
                    "stress_cortisol",
                ],
                "value": [6.0, 4.0, 10.0, 7.0, 5.0, 11.0],
                "anchor_time": [
                    "2024-01-01T00:00:00",
                    "2024-01-01T00:00:00",
                    "2024-01-01T00:00:00",
                    "2024-01-02T00:00:00",
                    "2024-01-02T00:00:00",
                    "2024-01-02T00:00:00",
                ],
            }
        )

        indicator_ids = {
            item.observation.name: item.observation.id
            for item in two_construct_structure.indicators
        }
        data_for_model = data_for_model.with_columns(
            pl.col("indicator").replace_strict(indicator_ids).alias("indicator_id")
        ).drop("indicator")
        source = _weekly_reference_intervals()
        model = compile_fit_fixture(source)
        compiled_dynamical_model = model.compiled_dynamical_model
        assert numeric.state_names(compiled_dynamical_model) == ("stress", "mood")
        edge_support = _linear_edge_support(source)
        assert edge_support[1, 0]
        assert not edge_support[0, 1]
        assert compiled_dynamical_model.loading_block.free_support is not None
        assert compiled_dynamical_model.loading_block.free_support[1, 0]
        runtime = model.prior_runtime_bundle
        assert runtime.priors["vf_0_p0"].batch_shape == ()
        assert runtime.priors["vf_1_p0"].batch_shape == ()
        assert (
            compiled_dynamical_model.bindings
            == parameter_bindings(compile_model_fixture(source))[0]
        )

    def test_residual_sd_priors_are_construct_specific(
        self, two_construct_structure, two_construct_model
    ):
        """Construct-specific sigma priors compile to per-latent diffusion scales."""

        ssm_priors, _idx = _compile_priors_for_test(_construct_specific_residual_scales())

        np.testing.assert_allclose(ssm_priors["diffusion_diag_free"].scale, [0.9, 0.1])

    def test_dt_to_ct_uses_interval_days(
        self,
        two_construct_structure,
        two_construct_model,
    ):
        """Priors with interval_days use that dt.

        rho_mood has interval_days=7 → dt=7
        rho_stress has interval_days="model_clock" → daily dt=1
        beta_stress_mood has interval_days=7 → dt=7
        """
        ssm_priors, _idx = _compile_priors_for_test(_weekly_reference_intervals())

        # --- rho_mood: Beta(3,2) → E=0.6, interval_days=7 ---
        # dynamics decay for mood = -ln(0.6) / 7 ≈ 0.073
        mu_ar_mood = 3.0 / 5.0  # E[Beta(3,2)] = 0.6
        expected_dynamics_mood = -math.log(mu_ar_mood) / 7.0
        mu_dynamics = _decay_reference_values(two_construct_model, ssm_priors)
        mu_mood = mu_dynamics[1]
        assert abs(mu_mood - expected_dynamics_mood) < 0.01, (
            f"mood dynamics: got {mu_mood}, expected {expected_dynamics_mood} "
            f"(using interval_days=7)"
        )

        # --- rho_stress: Beta(2,2) → E=0.5, interval_days="model_clock" → daily dt=1 ---
        # dynamics decay for stress = -ln(0.5) / 1.0 ≈ 0.693
        mu_ar_stress = 0.5
        expected_dynamics_stress = -math.log(mu_ar_stress) / 1.0
        mu_stress = mu_dynamics[0]
        assert abs(mu_stress - expected_dynamics_stress) < 0.01, (
            f"stress dynamics: got {mu_stress}, expected {expected_dynamics_stress} "
            f"(fallback to daily dt=1)"
        )

        # --- beta_stress_mood: Normal(0.3, 0.15), interval_days=7 ---
        # linear-edge weight = 0.3 / 7 ≈ 0.043
        expected_offdiag = 0.3 / 7.0
        mu_offdiag_val = _linear_edge_weight(two_construct_model, ssm_priors, source=0, target=1)
        assert abs(mu_offdiag_val - expected_offdiag) < 0.01, (
            f"stress→mood dynamics: got {mu_offdiag_val}, expected {expected_offdiag} "
            f"(using interval_days=7)"
        )

    def test_different_intervals_produce_different_rates(self, two_construct_model):
        """Same DT beta at different study intervals → different CT rates.

        beta=0.3 from weekly (dt=7) → CT rate ≈ 0.043
        beta=0.3 from daily  (dt=1) → CT rate ≈ 0.300
        This is the Kuiper & Ryan (2018) sign-reversal effect in action.
        """

        scientific_model = two_construct_model

        # Weekly study priors
        # Daily study priors (same beta value, different interval)

        source_model = scientific_model

        ssm_priors_w, _idx = _compile_priors_for_test(_weekly_effect_rate())

        ssm_priors_d, _idx = _compile_priors_for_test(_daily_effect_rate())

        # Weekly: mixed intervals (beta=7d, rho=1d) → first-order: 0.3 / 7 ≈ 0.043
        mu_w_val = _linear_edge_weight(source_model, ssm_priors_w, source=0, target=1)
        assert abs(mu_w_val - 0.3 / 7.0) < 0.01

        # Daily: beta_CT = beta_DT / dt = 0.3 / 1 = 0.3
        mu_d_val = _linear_edge_weight(source_model, ssm_priors_d, source=0, target=1)
        expected_daily = 0.3
        assert abs(mu_d_val - expected_daily) < 0.05, (
            f"Daily case uses beta/dt scaling: got {mu_d_val}, expected {expected_daily}"
        )

        # Rates should differ significantly because beta/dt depends on the interval.
        assert mu_d_val > mu_w_val, "Daily rate should be larger than weekly rate"


# ═══════════════════════════════════════════════════════════════════════
# Prior factorization and reference intervals
# ═══════════════════════════════════════════════════════════════════════


class TestPriorCompilationMetadata:
    """Compiler-owned prior factorization and reference intervals."""

    def test_compile_keeps_elementwise_priors_when_intervals_match(self, two_construct_structure):
        """Compilation keeps factorized DT→CT priors even when dt values match."""
        scientific_model = _stress_mood_model()

        # All parameters at dt=7 (weekly)

        source_model = scientific_model

        ssm_priors, _idx = _compile_priors_for_test(_equal_intervals_elementwise_priors())

        dynamics_decay = _decay_reference_values(source_model, ssm_priors)
        linear_edge_weight = _linear_edge_weight(source_model, ssm_priors, source=0, target=1)

        assert abs(dynamics_decay[1] - (-math.log(0.6) / 7.0)) < 0.01
        assert abs(dynamics_decay[0] - (-math.log(0.5) / 7.0)) < 0.01
        assert abs(linear_edge_weight - (0.3 / 7.0)) < 0.01
