"""Causal structure, measurement masks, and authored priors in executable models.

Parameter traces use the prior-only backend; likelihood numerics are exercised
by the inference tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro.distributions as dist
import numpyro.handlers as handlers
import polars as pl
import pytest

from nof1_causal_lab.artifacts.identity import ConstructId, DistributionId, MechanismId, ParameterId
from nof1_causal_lab.artifacts.model.construct import replace_constructs
from nof1_causal_lab.artifacts.model.expressions import coefficient, restoring_potential, state
from nof1_causal_lab.artifacts.model.mechanism import DriftMechanismSpec
from nof1_causal_lab.artifacts.model.parameter import SiteKind, SupportClass
from nof1_causal_lab.artifacts.model.parameter_spec import (
    InitialCorrelationTransformSpec,
    IntervalEffectTransformSpec,
    ParameterSpec,
)
from nof1_causal_lab.artifacts.probability.families import PriorDistributionFamily
from nof1_causal_lab.artifacts.probability.native import distribution_from_params
from nof1_causal_lab.models.ssm import compiled_access as numeric
from nof1_causal_lab.models.ssm.inference.prior_program import numpyro_model
from nof1_causal_lab.models.ssm.parameterization import build_site_registry
from nof1_causal_lab.models.ssm.priors import resolve_site_priors
from nof1_causal_lab.models.ssm.structure.sites import SiteDescriptor
from nof1_causal_lab.models.structure import StructuralSelection, validate_execution_structure
from tests.support.inference import bind_panel_fixture, compile_fit_fixture, compile_model_fixture
from tests.support.model import (
    construct_named,
    load_model_fixture,
    one_state_gaussian_model,
    parameter_named,
    three_state_gaussian_model,
    without_parameters,
)


def _static_baseline_stress_sleep() -> DynamicalModelSpec:
    return load_model_fixture("dag_to_ssm/static_baseline_stress_sleep.json")


def _freed_quartic_and_hill_sites_sample_finite_complete_test_model() -> DynamicalModelSpec:
    return load_model_fixture(
        "dag_to_ssm/testgradualbuildcomponents_test_freed_quartic_and_hill_sites_sample_finite_complete_test_model.json"
    )


def _site_registry_with_dynamics_support_model_fixture() -> DynamicalModelSpec:
    return load_model_fixture(
        "dag_to_ssm/testsiteregistrymasks_test_site_registry_with_dynamics_support_model_fixture.json"
    )


def _hill_edge_emitted_for_saturating_edge_complete_test_model() -> DynamicalModelSpec:
    dynamical_model_spec = _freed_quartic_and_hill_sites_sample_finite_complete_test_model()
    y = construct_named(dynamical_model_spec, "Y")
    (y_potential,) = y.dynamics
    rho_y = parameter_named(dynamical_model_spec, "rho_Y")
    self_limit_y = parameter_named(dynamical_model_spec, "self_limit_Y")
    y_revised = y.revised(
        dynamics=(
            y_potential.revised(
                expression=restoring_potential(y.id, center=0.0, stiffness=rho_y.id, quartic=0.0)
            ),
        )
    )
    parameters, distributions = without_parameters(dynamical_model_spec, self_limit_y)
    return dynamical_model_spec.with_entities(
        edges=replace_constructs(dynamical_model_spec.edges, (y_revised,)),
        parameters=parameters,
        distributions=distributions,
    )


def _x_z_measurements() -> DynamicalModelSpec:
    _X_TO_Y_DRIFT_MECHANISM_ID = MechanismId(
        "mechanism:2ed9052dbe3b4b4ab136228e0619ab1d3ae146cdec8d269f39bf5f6854f0579d"
    )
    _BETA_X_Y_PARAMETER_ID = ParameterId(
        "parameter:5ea4a485fc5b2e52a5c20036687f0abf45b3c8ed88ec29a0f2565d81b168e8e6"
    )
    _BETA_X_Y_DISTRIBUTION_ID = DistributionId(
        "distribution:5af2717690bf131509e0e3929a3b66d08d6661bc9462a1b0e63ccad62bbfcdfe"
    )
    dynamical_model_spec = _hill_edge_emitted_for_saturating_edge_complete_test_model()
    x = construct_named(dynamical_model_spec, "X")
    y = construct_named(dynamical_model_spec, "Y")
    x_to_y = next(
        edge
        for edge in dynamical_model_spec.edges
        if edge.cause.id == x.id and edge.effect.id == y.id
    )
    z = construct_named(dynamical_model_spec, "Z")
    y_to_z = next(
        edge
        for edge in dynamical_model_spec.edges
        if edge.cause.id == y.id and edge.effect.id == z.id
    )
    hill_emax_x_y = parameter_named(dynamical_model_spec, "hill_emax_X_Y")
    hill_ec50_x_y = parameter_named(dynamical_model_spec, "hill_ec50_X_Y")
    hill_n_x_y = parameter_named(dynamical_model_spec, "hill_n_X_Y")
    rho_x = parameter_named(dynamical_model_spec, "rho_X")
    rho_y = parameter_named(dynamical_model_spec, "rho_Y")
    rho_z = parameter_named(dynamical_model_spec, "rho_Z")
    beta_y_z = parameter_named(dynamical_model_spec, "beta_Y_Z")
    sigma_x = parameter_named(dynamical_model_spec, "sigma_X")
    obs_sd_x1 = parameter_named(dynamical_model_spec, "obs_sd_x1")
    lambda_x2_x = parameter_named(dynamical_model_spec, "lambda_x2_X")
    obs_sd_x2 = parameter_named(dynamical_model_spec, "obs_sd_x2")
    sigma_y = parameter_named(dynamical_model_spec, "sigma_Y")
    sigma_z = parameter_named(dynamical_model_spec, "sigma_Z")
    _parameters, distributions = without_parameters(
        dynamical_model_spec, hill_emax_x_y, hill_ec50_x_y, hill_n_x_y
    )
    return dynamical_model_spec.with_entities(
        edges=(
            x_to_y.revised(
                mechanisms=(
                    DriftMechanismSpec(
                        id=_X_TO_Y_DRIFT_MECHANISM_ID,
                        expression=(coefficient(_BETA_X_Y_PARAMETER_ID, "weight") * state(x.id)),
                    ),
                )
            ),
            y_to_z,
        ),
        parameters=(
            rho_x,
            rho_y,
            rho_z,
            ParameterSpec(
                id=_BETA_X_Y_PARAMETER_ID,
                name="beta_X_Y",
                description="weight of beta_X_Y",
                transform=IntervalEffectTransformSpec(interval_days="model_clock"),
                distribution=_BETA_X_Y_DISTRIBUTION_ID,
            ),
            beta_y_z,
            sigma_x,
            obs_sd_x1,
            lambda_x2_x,
            obs_sd_x2,
            sigma_y,
            sigma_z,
        ),
        distributions={
            **distributions,
            _BETA_X_Y_DISTRIBUTION_ID: dist.Normal(loc=0.0, scale=0.5, validate_args=False),
        },
    )


def _translate_spec_rejects_initial_state_correlation_parameters_with_scientific_model_attach_test_coeff() -> (
    DynamicalModelSpec
):
    _COR0_PARAMETER_ID = ParameterId(
        "parameter:418d0717cfc9857102f86ab03eaa94a90f0757b6ac7826fc239d3167f7adaf50"
    )
    _COR0_DISTRIBUTION_ID = DistributionId(
        "distribution:37add4c0f63e0477b02d5f8319fc8527c467007b2ece06c089b72e3de2c48b09"
    )
    dynamical_model_spec = _x_z_measurements()
    z = construct_named(dynamical_model_spec, "Z")
    sigma_z = parameter_named(dynamical_model_spec, "sigma_Z")
    x = construct_named(dynamical_model_spec, "X")
    z_revised = z.revised(
        coefficients=(
            coefficient(sigma_z.id, "diffusion_scale"),
            coefficient(0.0, "initial_mean"),
            coefficient(1.0, "initial_scale"),
            coefficient(_COR0_PARAMETER_ID, "initial_correlation", construct_ids=(x.id,)),
        )
    )
    return dynamical_model_spec.with_entities(
        edges=replace_constructs(dynamical_model_spec.edges, (z_revised,)),
        parameters=(
            *dynamical_model_spec.parameters,
            ParameterSpec(
                id=_COR0_PARAMETER_ID,
                name="cor0",
                description="Unsupported pairwise initial correlation",
                transform=InitialCorrelationTransformSpec(),
                distribution=_COR0_DISTRIBUTION_ID,
            ),
        ),
        distributions={
            **dynamical_model_spec.distributions,
            _COR0_DISTRIBUTION_ID: dist.Normal(loc=0.2, scale=0.8, validate_args=False),
        },
    )


def _quartic_freed_only_for_self_limiting_construct_complete_test_model() -> DynamicalModelSpec:
    _SELF_LIMIT_Y_PARAMETER_ID = ParameterId(
        "parameter:ab64427aa9da4f5f72753f200c603d31725edc0deb5ccbb6958c1186f5fd041b"
    )
    _SELF_LIMIT_Y_DISTRIBUTION_ID = DistributionId(
        "distribution:c23e111a0f275ce7c22f2d507b6e4d06ed3faba49cd67be9841ff42a322a0d16"
    )
    dynamical_model_spec = _x_z_measurements()
    y = construct_named(dynamical_model_spec, "Y")
    (y_potential,) = y.dynamics
    rho_y = parameter_named(dynamical_model_spec, "rho_Y")
    rho_x = parameter_named(dynamical_model_spec, "rho_X")
    rho_z = parameter_named(dynamical_model_spec, "rho_Z")
    beta_x_y = parameter_named(dynamical_model_spec, "beta_X_Y")
    beta_y_z = parameter_named(dynamical_model_spec, "beta_Y_Z")
    sigma_x = parameter_named(dynamical_model_spec, "sigma_X")
    obs_sd_x1 = parameter_named(dynamical_model_spec, "obs_sd_x1")
    lambda_x2_x = parameter_named(dynamical_model_spec, "lambda_x2_X")
    obs_sd_x2 = parameter_named(dynamical_model_spec, "obs_sd_x2")
    sigma_y = parameter_named(dynamical_model_spec, "sigma_Y")
    sigma_z = parameter_named(dynamical_model_spec, "sigma_Z")
    y_revised = y.revised(
        dynamics=(
            y_potential.revised(
                expression=restoring_potential(
                    y.id, center=0.0, stiffness=rho_y.id, quartic=_SELF_LIMIT_Y_PARAMETER_ID
                )
            ),
        )
    )
    return dynamical_model_spec.with_entities(
        edges=replace_constructs(dynamical_model_spec.edges, (y_revised,)),
        parameters=(
            rho_x,
            rho_y,
            ParameterSpec(
                id=_SELF_LIMIT_Y_PARAMETER_ID,
                name="self_limit_Y",
                description="quartic of self_limit_Y",
                distribution=_SELF_LIMIT_Y_DISTRIBUTION_ID,
            ),
            rho_z,
            beta_x_y,
            beta_y_z,
            sigma_x,
            obs_sd_x1,
            lambda_x2_x,
            obs_sd_x2,
            sigma_y,
            sigma_z,
        ),
        distributions={
            **dynamical_model_spec.distributions,
            _SELF_LIMIT_Y_DISTRIBUTION_ID: dist.HalfNormal(scale=1.0, validate_args=False),
        },
    )


def _site_registry_with_lambda_support_model_fixture() -> DynamicalModelSpec:
    return load_model_fixture(
        "dag_to_ssm/testsiteregistrymasks_test_site_registry_with_lambda_support_model_fixture.json"
    )


def _per_element_prior_in_model_with_parameter_distributions() -> DynamicalModelSpec:
    return load_model_fixture(
        "dag_to_ssm/testperelementpriors_test_per_element_prior_in_model_with_parameter_distributions.json"
    )


def _three_latent_unmasked() -> DynamicalModelSpec:
    return load_model_fixture("dag_to_ssm/three_latent_unmasked.json")


def _lambda_no_mask_returns_fixed__make_3latent_spec() -> DynamicalModelSpec:
    return load_model_fixture(
        "dag_to_ssm/testlambdamask_test_lambda_no_mask_returns_fixed__make_3latent_spec.json"
    )


def _lambda_template_plus_mask__make_3latent_spec() -> DynamicalModelSpec:
    return load_model_fixture(
        "dag_to_ssm/testlambdamask_test_lambda_template_plus_mask__make_3latent_spec.json"
    )


if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.model.dynamical_model_spec import DynamicalModelSpec


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════
# Fix 1: DAG-constrained dynamics
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.inference(concern="sampling")
class TestDynamicsMask:
    """Test that component translation constrains linear edge sampling."""

    def test_dynamics_support_zeros_non_edges(self):
        """Dynamics entries where mask is False should be zero."""
        offdiag_support = np.zeros((3, 3), dtype=bool)
        offdiag_support[1, 0] = True  # X→Y
        offdiag_support[2, 1] = True  # Y→Z

        dynamical_model_spec = three_state_gaussian_model()
        model = compile_fit_fixture(dynamical_model_spec)

        rng = random.PRNGKey(42)
        trace = handlers.trace(handlers.seed(numpyro_model, rng)).get_trace(
            bind_panel_fixture(
                model.compiled_dynamical_model, jnp.zeros((2, 4)), jnp.arange(2, dtype=jnp.float32)
            ),
            priors=model.prior_runtime_bundle,
            likelihood_backend=None,
        )

        weight_sites = [
            site
            for site in build_site_registry(compile_model_fixture(dynamical_model_spec))
            if site.site_kind == SiteKind.DYNAMICS_WEIGHT
        ]
        assert len(weight_sites) == 2
        assert all(site.name in trace for site in weight_sites)
        assert {site.positions[0] for site in weight_sites} == {
            (1, 0),
            (2, 1),
        }

    def test_no_mask_fully_free(self):
        """Default dynamics mask expands to a fully free dynamics structure."""
        dynamical_model_spec = _three_latent_unmasked()
        model = compile_fit_fixture(dynamical_model_spec)

        rng = random.PRNGKey(0)
        trace = handlers.trace(handlers.seed(numpyro_model, rng)).get_trace(
            bind_panel_fixture(
                model.compiled_dynamical_model, jnp.zeros((2, 4)), jnp.arange(2, dtype=jnp.float32)
            ),
            priors=model.prior_runtime_bundle,
            likelihood_backend=None,
        )

        weight_sites = [
            site
            for site in build_site_registry(compile_model_fixture(dynamical_model_spec))
            if site.site_kind == SiteKind.DYNAMICS_WEIGHT
        ]
        assert len(weight_sites) == 6
        assert all(site.name in trace for site in weight_sites)

    def test_dynamics_support_single_latent(self):
        """Single latent: no off-diagonal, mask should be identity."""
        dynamical_model_spec = one_state_gaussian_model()
        model = compile_fit_fixture(dynamical_model_spec)

        rng = random.PRNGKey(0)
        trace = handlers.trace(handlers.seed(numpyro_model, rng)).get_trace(
            bind_panel_fixture(
                model.compiled_dynamical_model, jnp.zeros((2, 1)), jnp.arange(2, dtype=jnp.float32)
            ),
            priors=model.prior_runtime_bundle,
            likelihood_backend=None,
        )

        dynamics_sites = [
            site
            for site in build_site_registry(compile_model_fixture(dynamical_model_spec))
            if site.assembly_group == "dynamics"
        ]
        assert [site.site_kind for site in dynamics_sites] == [SiteKind.DYNAMICS_DECAY]
        assert dynamics_sites[0].name in trace


# ═══════════════════════════════════════════════════════════════════════
# Fix 2: Structured lambda
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.inference(concern="sampling")
class TestLambdaMask:
    """Test that lambda_support constrains factor loadings."""

    def test_lambda_template_plus_mask(self):
        """Template+mask mode: fixed reference + free additional loadings."""
        # X has 2 indicators (x1 ref, x2 free), Y has 1, Z has 1

        lambda_support = np.zeros((4, 3), dtype=bool)
        lambda_support[1, 0] = True  # x2→X (free)

        dynamical_model_spec = _lambda_template_plus_mask__make_3latent_spec()
        model = compile_fit_fixture(dynamical_model_spec)

        rng = random.PRNGKey(0)
        trace = handlers.trace(handlers.seed(numpyro_model, rng)).get_trace(
            bind_panel_fixture(
                model.compiled_dynamical_model, jnp.zeros((2, 4)), jnp.arange(2, dtype=jnp.float32)
            ),
            priors=model.prior_runtime_bundle,
            likelihood_backend=None,
        )

        # Only 1 free loading sampled
        assert trace["lambda_free"]["value"].shape == (1,)

        # Check the assembled lambda
        lam = trace["lambda"]["value"]
        assert float(lam[0, 0]) == 1.0  # Fixed reference
        assert float(lam[2, 1]) == 1.0  # Fixed reference
        assert float(lam[3, 2]) == 1.0  # Fixed reference
        assert float(lam[1, 0]) != 0.0  # Free loading was sampled

    def test_lambda_no_mask_returns_fixed(self):
        """Array lambda_mat with default zero free-mask is returned as-is."""
        # Both x indicators belong to X; fixed loadings must follow that ownership.
        lambda_mat = jnp.array(
            [[1.0, 0.0, 0.0], [0.75, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        dynamical_model_spec = _lambda_no_mask_returns_fixed__make_3latent_spec()
        model = compile_fit_fixture(dynamical_model_spec)

        rng = random.PRNGKey(0)
        trace = handlers.trace(handlers.seed(numpyro_model, rng)).get_trace(
            bind_panel_fixture(
                model.compiled_dynamical_model, jnp.zeros((2, 4)), jnp.arange(2, dtype=jnp.float32)
            ),
            priors=model.prior_runtime_bundle,
            likelihood_backend=None,
        )

        # No lambda_free sampled
        assert "lambda_free" not in trace
        # Lambda deterministic IS emitted (the block always emits its
        # assembled output, fixed or sampled); the value equals the template.
        assert "lambda" in trace
        np.testing.assert_allclose(np.asarray(trace["lambda"]["value"]), np.asarray(lambda_mat))


# ═══════════════════════════════════════════════════════════════════════
# Fix 3: Per-element priors
# ═══════════════════════════════════════════════════════════════════════


class TestPerElementPriors:
    """Test array-valued priors through canonical site runtime state."""

    @staticmethod
    def _real_site(shape: tuple[int, ...]) -> SiteDescriptor:
        return SiteDescriptor(
            name="test_site",
            shape=shape,
            support=SupportClass.REAL,
            assembly_group="test",
            site_kind=SiteKind.DYNAMICS_WEIGHT,
        )

    @pytest.mark.contract
    def test_make_prior_dist_scalar(self):
        """Scalar mu/sigma produces scalar Normal."""
        site = self._real_site(())
        priors = {
            "test_site": distribution_from_params(
                PriorDistributionFamily.NORMAL, {"mu": 0.0, "sigma": 1.0}
            )
        }
        state = resolve_site_priors([site], priors)
        d = state[site.name]
        assert d.batch_shape == ()

    @pytest.mark.contract
    def test_make_prior_dist_array(self):
        """Array mu/sigma produces batched Normal."""
        site = self._real_site((3,))
        priors = {
            "test_site": distribution_from_params(
                PriorDistributionFamily.NORMAL,
                {"mu": [0.1, 0.2, 0.3], "sigma": [1.0, 0.5, 0.3]},
            )
        }
        state = resolve_site_priors([site], priors)
        d = state[site.name]
        assert d.batch_shape == (3,)

    @pytest.mark.contract
    def test_make_prior_batch_scalar_expand(self):
        """Scalar prior expanded to batch shape."""
        site = self._real_site((5,))
        priors = {
            "test_site": distribution_from_params(
                PriorDistributionFamily.NORMAL, {"mu": 0.0, "sigma": 1.0}
            )
        }
        state = resolve_site_priors([site], priors)
        d = state[site.name]
        assert d.batch_shape == (5,)

    @pytest.mark.contract
    def test_make_prior_batch_array_passthrough(self):
        """Array prior with correct shape passes through."""
        site = self._real_site((2,))
        priors = {
            "test_site": distribution_from_params(
                PriorDistributionFamily.NORMAL,
                {"mu": [0.1, 0.2], "sigma": [1.0, 0.5]},
            )
        }
        state = resolve_site_priors([site], priors)
        d = state[site.name]
        assert d.batch_shape == (2,)

    @pytest.mark.contract
    def test_make_prior_batch_mismatch_raises(self):
        """Array prior with wrong shape raises."""
        site = self._real_site((3,))
        priors = {
            "test_site": distribution_from_params(
                PriorDistributionFamily.NORMAL,
                {"mu": [0.1, 0.2], "sigma": [1.0, 0.5]},
            )
        }
        with pytest.raises(ValueError, match="broadcast"):
            resolve_site_priors([site], priors)

    @pytest.mark.inference(concern="sampling")
    def test_per_element_prior_in_model(self):
        """Per-element dynamics priors are used in sampling."""
        offdiag_support = np.zeros((2, 2), dtype=bool)
        offdiag_support[1, 0] = True  # X→Y

        # Per-element prior: single off-diagonal has mu=2.0

        dynamical_model_spec = _per_element_prior_in_model_with_parameter_distributions()
        model = compile_fit_fixture(dynamical_model_spec)

        rng = random.PRNGKey(0)
        trace = handlers.trace(handlers.seed(numpyro_model, rng)).get_trace(
            bind_panel_fixture(
                model.compiled_dynamical_model, jnp.zeros((2, 2)), jnp.arange(2, dtype=jnp.float32)
            ),
            priors=model.prior_runtime_bundle,
            likelihood_backend=None,
        )

        # Verify the authored law, independently of one lucky prior draw.
        prior = trace["vf_2_p0"]["fn"]
        assert float(prior.mean) == pytest.approx(2.0)
        assert float(prior.variance) == pytest.approx(0.01)


# ═══════════════════════════════════════════════════════════════════════
# Runtime structural-support construction
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.contract
class TestRuntimeStructuralSupport:
    """Test that compilation constructs correct block support from DynamicalModelSpec."""

    def test_compiled_mechanisms_and_loadings_preserve_structural_coordinates(self):
        dynamical_model_spec = _three_latent_unmasked()
        compiled_dynamical_model = compile_model_fixture(dynamical_model_spec)
        components = compiled_dynamical_model.dynamics.spec.components
        assert {
            (component.source, component.target) for component in components if component.edge_owned
        } == {(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)}
        assert {component.target for component in components if not component.edge_owned} == {
            0,
            1,
            2,
        }
        assert compiled_dynamical_model.loading_block.free_support.shape == (4, 3)
        np.testing.assert_array_equal(
            compiled_dynamical_model.loading_block.template[[0, 2, 3], [0, 1, 2]], 1.0
        )

    def test_model_build_accepts_only_already_compiled_ssm_spec(self):
        """Runtime construction consumes an DynamicalModelSpec without structural authoring inputs."""

        pl.DataFrame(
            {
                "time": list(range(5)),
                "x1": [1.0] * 5,
                "x2": [2.0] * 5,
                "y1": [3.0] * 5,
                "z1": [4.0] * 5,
            }
        )

        model = compile_fit_fixture(_three_latent_unmasked())
        assert numeric.n_states(model.compiled_dynamical_model) == 3

    @pytest.mark.parametrize(
        ("source_count", "complete_test_model_payload"),
        [
            pytest.param(
                1,
                _static_baseline_stress_sleep,
                id="1",
            ),
            pytest.param(
                2,
                _static_baseline_stress_sleep,
                id="2",
            ),
        ],
    )
    def test_translate_spec_compiles_static_baseline_factor_from_induced_dependency(
        self, source_count, complete_test_model_payload
    ):

        model = complete_test_model_payload()
        if source_count == 2:
            source = model.get_construct(ConstructId("construct:766f6091724c163a3404"))
            second = source.revised(id="construct:second-common-cause", name="second_common_cause")
            model = model.with_entities(
                edges=replace_constructs(
                    (
                        *model.edges,
                        *(
                            edge.revised(id=f"{edge.id}-second", cause=second)
                            for edge in model.edges
                        ),
                    ),
                    (*model.constructs, second),
                )
            )
            # Equivalent marginalized roots reference one aggregate scale, not two draws.
            assert second.coefficient("initial_scale") == source.coefficient("initial_scale")
        compile_model_fixture(model)
        validate_execution_structure(StructuralSelection(model, None))
        spec = model
        np.testing.assert_array_equal(
            compile_model_fixture(spec).static_scale_block.free_support, [True]
        )
        np.testing.assert_allclose(
            compile_model_fixture(spec).static_scale_block.template, np.zeros(1)
        )
        np.testing.assert_allclose(
            compile_model_fixture(spec).static_factor_loadings, [[1.0], [1.0]]
        )
        assert numeric.static_factor_names(compile_model_fixture(spec)) == ("tau_u_shared",)
        np.testing.assert_array_equal(
            compile_model_fixture(spec).initial_covariance_block.correlation_support,
            np.zeros((2, 2), dtype=bool),
        )
        np.testing.assert_array_equal(
            compile_model_fixture(spec).initial_mean_block.free_support, [False, False]
        )
        np.testing.assert_array_equal(
            compile_model_fixture(spec).initial_covariance_block.diag_support,
            [False, False],
        )

    def test_translate_spec_marks_standardizable_gaussian_mean_indicators(self):

        dynamical_model_spec = _x_z_measurements()
        dynamical_model_spec = dynamical_model_spec
        assert numeric.observation_standardized(compile_model_fixture(dynamical_model_spec)) == (
            True,
            True,
            True,
            True,
        )

    def test_translate_spec_fixes_manifest_noise_for_single_indicator_constructs(self):

        dynamical_model_spec = _x_z_measurements()
        dynamical_model_spec = dynamical_model_spec
        assert isinstance(
            compile_model_fixture(dynamical_model_spec).observation_noise_block.template,
            jnp.ndarray,
        )
        np.testing.assert_array_equal(
            compile_model_fixture(dynamical_model_spec).observation_noise_block.diag_support,
            [True, True, False, False],
        )
        np.testing.assert_allclose(
            compile_model_fixture(dynamical_model_spec).observation_noise_block.template,
            np.zeros((4, 4)),
        )

    def test_translate_spec_rejects_initial_state_correlation_parameters_with_scientific_model(
        self,
    ):

        dynamical_model_spec = _translate_spec_rejects_initial_state_correlation_parameters_with_scientific_model_attach_test_coeff()
        with pytest.raises(
            ValueError, match=r"explicit latent confounder|two distinct state owners"
        ):
            compile_model_fixture(dynamical_model_spec)

    def test_translate_spec_rejects_self_initial_state_correlation_with_scientific_model(self):
        from nof1_causal_lab.artifacts.model.parameter_spec import (
            InitialCorrelationTransformSpec,
            ParameterSpec,
        )

        dynamical_model_spec = _x_z_measurements()
        parameter = ParameterSpec(
            id="parameter:77f3ac548e5c828bd95649d677ae53ce71c8dd33c2b1526961592ea56d79f013",
            name="cor0",
            description="Unsupported pairwise initial correlation",
            transform=InitialCorrelationTransformSpec(),
        )

        with pytest.raises(ValueError, match="Joint coefficients require one other construct"):
            dynamical_model_spec.constructs[0].revised(
                coefficients=(
                    *dynamical_model_spec.constructs[0].coefficients,
                    coefficient(
                        parameter.id,
                        "initial_correlation",
                        construct_ids=(dynamical_model_spec.constructs[0].id,),
                    ),
                )
            )

    def test_model_build_end_to_end(self):

        science = _x_z_measurements()
        pl.DataFrame(
            {
                "time": list(range(10)),
                "x1": [1.0] * 10,
                "x2": [2.0] * 10,
                "y1": [3.0] * 10,
                "z1": [4.0] * 10,
            }
        )
        compile_model_fixture(science)
        dynamical_model_spec = science
        assert (
            sum(
                site.site_kind == SiteKind.DYNAMICS_DECAY
                for site in compile_model_fixture(dynamical_model_spec).site_registry
            )
            == 3
        )
        assert compile_model_fixture(dynamical_model_spec).loading_block.free_support is not None
        assert numeric.n_states(compile_model_fixture(dynamical_model_spec)) == 3
        assert numeric.n_observations(compile_model_fixture(dynamical_model_spec)) == 4


# ═══════════════════════════════════════════════════════════════════════
# Site-registry mask awareness
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.contract
class TestSiteRegistryMasks:
    """Test that the canonical site registry respects SSM masks."""

    def test_site_registry_with_dynamics_support(self):
        """Site registry should size masked dynamics entries correctly."""
        from nof1_causal_lab.models.ssm.parameterization import build_site_registry

        # 3 latent, X→Y and Y→Z = 2 off-diagonal entries
        offdiag_support = np.zeros((3, 3), dtype=bool)
        offdiag_support[1, 0] = True
        offdiag_support[2, 1] = True

        dynamical_model_spec = _site_registry_with_dynamics_support_model_fixture()

        registry = {
            site.name: site
            for site in build_site_registry(compile_model_fixture(dynamical_model_spec))
        }

        weight_sites = sorted(
            site.name for site in registry.values() if site.site_kind == SiteKind.DYNAMICS_WEIGHT
        )
        assert len(weight_sites) == 2
        assert {
            site.positions[0]
            for site in build_site_registry(compile_model_fixture(dynamical_model_spec))
            if site.name in weight_sites
        } == {
            (1, 0),
            (2, 1),
        }
        assert [
            site.shape for site in registry.values() if site.site_kind == SiteKind.DYNAMICS_DECAY
        ] == [
            (),
            (),
            (),
        ]

    def test_site_registry_with_lambda_support(self):
        """Site registry should size masked loading entries correctly."""
        from nof1_causal_lab.models.ssm.parameterization import build_site_registry

        dynamical_model_spec = _site_registry_with_lambda_support_model_fixture()

        registry = {
            site.name: site
            for site in build_site_registry(compile_model_fixture(dynamical_model_spec))
        }
        assert registry["lambda_free"].shape == (1,)


# ═══════════════════════════════════════════════════════════════════════
# Integration: trace verification
# ═══════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════
# Gradual-build surface: self-limiting quartic + Hill (saturating) edges
# ═══════════════════════════════════════════════════════════════════════


class TestGradualBuildComponents:
    """Quartic self-limitation and Hill edges materialize from the DynamicalModelSpec."""

    @pytest.mark.contract
    def test_quartic_freed_only_for_self_limiting_construct(self):

        from nof1_causal_lab.artifacts.model.expressions import expression_coefficients

        dynamical_model_spec = _quartic_freed_only_for_self_limiting_construct_complete_test_model()
        quartics = {
            component.target: next(
                operand.value
                for operand in expression_coefficients(component.expression)
                if operand.role == "quartic"
            )
            for component in compile_model_fixture(dynamical_model_spec).dynamics.spec.components
            if not component.edge_owned
        }
        assert isinstance(quartics[1], str)
        assert quartics[0] == 0
        assert quartics[2] == 0

    @pytest.mark.contract
    def test_hill_edge_emitted_for_saturating_edge(self):
        from nof1_causal_lab.artifacts.model.expressions import hill_applications

        dynamical_model_spec = _hill_edge_emitted_for_saturating_edge_complete_test_model()
        edge_components = [
            item
            for item in compile_model_fixture(dynamical_model_spec).dynamics.spec.components
            if item.edge_owned
        ]
        hill = [item for item in edge_components if any(hill_applications(item.expression))]
        linear = [item for item in edge_components if not any(hill_applications(item.expression))]
        assert [(item.source, item.target) for item in hill] == [(0, 1)]
        assert (1, 2) in [(item.source, item.target) for item in linear]
        assert (0, 1) not in [(item.source, item.target) for item in linear]

    @pytest.mark.inference(concern="sampling")
    def test_freed_quartic_and_hill_sites_sample_finite(self):

        science = _freed_quartic_and_hill_sites_sample_finite_complete_test_model()
        dynamical_model_spec = science
        model = compile_fit_fixture(dynamical_model_spec)
        trace = handlers.trace(handlers.seed(numpyro_model, random.PRNGKey(0))).get_trace(
            bind_panel_fixture(
                model.compiled_dynamical_model,
                # This prior trace has no readings, including incomplete interval windows.
                jnp.full((2, 4), jnp.nan),
                jnp.arange(2, dtype=jnp.float32),
            ),
            priors=model.prior_runtime_bundle,
            likelihood_backend=None,
        )
        registry = build_site_registry(compile_model_fixture(dynamical_model_spec))
        quartic_sites = [
            site.name for site in registry if site.site_kind == SiteKind.DYNAMICS_POTENTIAL_QUARTIC
        ]
        emax_sites = [site.name for site in registry if site.site_kind == SiteKind.HILL_EMAX]
        assert len(quartic_sites) == 1
        assert len(emax_sites) == 1
        for name in quartic_sites + emax_sites:
            assert bool(jnp.all(jnp.isfinite(trace[name]["value"])))
