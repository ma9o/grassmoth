"""Tests for the canonical site registry and compile-stable prior evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro.distributions as dist
import pytest
from numpyro import handlers

from nof1_causal_lab.artifacts.identity import DistributionId, ParameterId
from nof1_causal_lab.artifacts.model.construct import replace_constructs
from nof1_causal_lab.artifacts.model.expressions import Expression, coefficient, state
from nof1_causal_lab.artifacts.model.likelihood import NormalLawSpec
from nof1_causal_lab.artifacts.model.parameter import SiteKind, SupportClass
from nof1_causal_lab.artifacts.model.parameter_spec import ParameterSpec
from nof1_causal_lab.artifacts.probability.families import DistributionFamily
from nof1_causal_lab.models.ssm import compiled_access as numeric
from nof1_causal_lab.models.ssm.execution.parameters import sample_sites
from nof1_causal_lab.models.ssm.inference.parameter_tracing import (
    _discover_sites,
)
from nof1_causal_lab.models.ssm.inference.prior_program import sample_parameters
from nof1_causal_lab.models.ssm.parameterization import (
    assemble_deterministics_from_registry,
    build_site_registry,
    process_sites,
)
from nof1_causal_lab.models.ssm.priors import resolve_site_priors
from tests.support.inference import bind_panel_fixture, compile_fit_fixture, compile_model_fixture
from tests.support.model import (
    _two_state_fixed_drift_model,
    construct_named,
    indicator_named,
    likelihood_named,
    load_model_fixture,
    parameter_for,
    two_state_gaussian_model,
    without_parameters,
)


def _mood_model() -> DynamicalModelSpec:
    return load_model_fixture("parameterization/mood_model.json")


def _static_state_model() -> DynamicalModelSpec:
    return load_model_fixture("parameterization/static_state_model.json")


def _ordered_threshold_priors_bind_per_manifest_component_and_row_model_with_prior_payloads() -> (
    DynamicalModelSpec
):
    return load_model_fixture(
        "parameterization/testcompiledartifactintegration_test_ordered_threshold_priors_bind_per_manifest_component_and_row_model_with_prior_payloads.json"
    )


def _assemble_deterministics_from_registry_fixed_blocks__make_spec() -> DynamicalModelSpec:
    return load_model_fixture(
        "parameterization/testdeterministicassembly_test_assemble_deterministics_from_registry_fixed_blocks__make_spec.json"
    )


def _diag_diffusion_excludes_lower__make_spec() -> DynamicalModelSpec:
    return load_model_fixture(
        "parameterization/testsiteregistry_test_diag_diffusion_excludes_lower__make_spec.json"
    )


def _sparse_initial_state_correlations_only_include_authored_pairs__make_spec() -> (
    DynamicalModelSpec
):
    return load_model_fixture(
        "parameterization/testsiteregistry_test_sparse_initial_state_correlations_only_include_authored_pairs__make_spec.json"
    )


def _partial_manifest_variance_model() -> DynamicalModelSpec:
    dynamical_model_spec = two_state_gaussian_model()
    latent_0 = construct_named(dynamical_model_spec, "latent_0")
    manifest_0 = indicator_named(dynamical_model_spec, "manifest_0")
    manifest_0_likelihood = likelihood_named(dynamical_model_spec, "manifest_0")
    latent_0_manifest_0_manifest_var_diag = parameter_for(
        dynamical_model_spec, SiteKind.MANIFEST_VAR_DIAG, "latent_0", "manifest_0"
    )
    manifest_0_revised = manifest_0.revised(
        likelihood=manifest_0_likelihood.revised(
            law=NormalLawSpec[Expression](
                loc=(
                    coefficient(0.0, "observation_intercept")
                    + (coefficient(1.0, "loading") * state(latent_0.id))
                ),
                scale=coefficient(0.4000000059604645, "observation_scale"),
            )
        )
    )
    latent_0_revised = latent_0.revised(indicators=(manifest_0_revised,))
    parameters, distributions = without_parameters(
        dynamical_model_spec, latent_0_manifest_0_manifest_var_diag
    )
    return dynamical_model_spec.with_entities(
        edges=replace_constructs(dynamical_model_spec.edges, (latent_0_revised,)),
        parameters=parameters,
        distributions=distributions,
    )


def _student_innovation_model() -> DynamicalModelSpec:
    _PROC_DF_PARAMETER_ID = ParameterId(
        "parameter:cda322d07f43753b6303431fe88a8d86238b54e2a7f366cecd07b25c754971b6"
    )
    _PROC_DF_DISTRIBUTION_ID = DistributionId(
        "distribution:8b5923f7a2b2d83dbc2d3961d34d000cfa280f81457d2e3b222014b086954b66"
    )
    dynamical_model_spec = two_state_gaussian_model()
    latent_1 = construct_named(dynamical_model_spec, "latent_1")
    latent_1_diffusion_diag = parameter_for(
        dynamical_model_spec, SiteKind.DIFFUSION_DIAG, "latent_1"
    )
    latent_1_t0_means = parameter_for(dynamical_model_spec, SiteKind.T0_MEANS, "latent_1")
    latent_1_t0_var_diag = parameter_for(dynamical_model_spec, SiteKind.T0_VAR_DIAG, "latent_1")
    latent_0_latent_1_diffusion_lower = parameter_for(
        dynamical_model_spec, SiteKind.DIFFUSION_LOWER, "latent_0", "latent_1"
    )
    latent_0 = construct_named(dynamical_model_spec, "latent_0")
    latent_0_latent_1_t0_var_lower = parameter_for(
        dynamical_model_spec, SiteKind.T0_VAR_LOWER, "latent_0", "latent_1"
    )
    latent_1_revised = latent_1.revised(
        coefficients=(
            coefficient(latent_1_diffusion_diag.id, "diffusion_scale"),
            coefficient(latent_1_t0_means.id, "initial_mean"),
            coefficient(latent_1_t0_var_diag.id, "initial_scale"),
            coefficient(
                latent_0_latent_1_diffusion_lower.id,
                "diffusion_loading",
                construct_ids=(latent_0.id,),
            ),
            coefficient(
                latent_0_latent_1_t0_var_lower.id,
                "initial_correlation",
                construct_ids=(latent_0.id,),
            ),
            coefficient(_PROC_DF_PARAMETER_ID, "process_degrees_of_freedom"),
        ),
        innovation_family=DistributionFamily.STUDENT_T,
    )
    return dynamical_model_spec.with_entities(
        edges=replace_constructs(dynamical_model_spec.edges, (latent_1_revised,)),
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


def _global_ordered_threshold_priors_are_not_authorable__make_spec() -> DynamicalModelSpec:
    return load_model_fixture(
        "parameterization/testcompiledartifactintegration_test_global_ordered_threshold_priors_are_not_authorable__make_spec.json"
    )


def _dag_spec_model() -> DynamicalModelSpec:
    return load_model_fixture("parameterization/dag_spec_model.json")


def _assemble_deterministics_repairs_invalid_initial_correlation_matrix__make_spec() -> (
    DynamicalModelSpec
):
    return load_model_fixture(
        "parameterization/testdeterministicassembly_test_assemble_deterministics_repairs_invalid_initial_correlation_matrix__make_spec.json"
    )


if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.model.dynamical_model_spec import DynamicalModelSpec


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_spec():
    return two_state_gaussian_model()


@pytest.fixture
def simple_model(simple_spec):
    return compile_fit_fixture(simple_spec)


@pytest.fixture
def dag_spec():
    return _dag_spec_model()


@pytest.fixture
def dag_model(dag_spec):
    return compile_fit_fixture(dag_spec)


# ---------------------------------------------------------------------------
# Site registry tests
# ---------------------------------------------------------------------------


def _assert_registry_matches_trace(registry, site_info):
    """Keep the registry/trace integration assertion owned by this test module."""
    assert {site.name for site in registry} == set(site_info)
    for site in registry:
        assert site.shape == site_info[site.name].shape


class TestSiteRegistry:
    @pytest.mark.inference(concern="sampling")
    def test_registry_names_match_trace(self, simple_model):
        """Registry produces the same site names as model tracing."""
        compiled = simple_model.compiled_dynamical_model
        registry = build_site_registry(compiled)
        T = 2
        obs = jnp.zeros((T, numeric.n_observations(compiled)))
        times = jnp.linspace(0, 1, T)
        site_info = _discover_sites(
            simple_model.prior_runtime_bundle,
            bind_panel_fixture(simple_model.compiled_dynamical_model, obs, times),
            random.PRNGKey(0),
        )
        _assert_registry_matches_trace(registry, site_info)

    @pytest.mark.inference(concern="sampling")
    def test_registry_names_match_trace_dag(self, dag_model):
        """Registry matches trace for DAG-constrained model with cint."""
        compiled = dag_model.compiled_dynamical_model
        registry = build_site_registry(compiled)
        T = 2
        obs = jnp.zeros((T, numeric.n_observations(compiled)))
        times = jnp.linspace(0, 1, T)
        site_info = _discover_sites(
            dag_model.prior_runtime_bundle,
            bind_panel_fixture(dag_model.compiled_dynamical_model, obs, times),
            random.PRNGKey(0),
        )
        _assert_registry_matches_trace(registry, site_info)

    @pytest.mark.inference(concern="sampling")
    def test_registry_shapes_match_trace_partial_manifest_variance_mask(self):
        """Masked manifest variance exposes only free diagonal entries as a site."""
        dynamical_model_spec = _partial_manifest_variance_model()
        model = compile_fit_fixture(dynamical_model_spec)
        registry = build_site_registry(compile_model_fixture(dynamical_model_spec))
        T = 2
        obs = jnp.zeros((T, numeric.n_observations(compile_model_fixture(dynamical_model_spec))))
        times = jnp.linspace(0, 1, T)
        site_info = _discover_sites(
            model.prior_runtime_bundle,
            bind_panel_fixture(model.compiled_dynamical_model, obs, times),
            random.PRNGKey(0),
        )

        _assert_registry_matches_trace(registry, site_info)
        manifest_site = next(site for site in registry if site.name == "manifest_var_diag_free")
        assert manifest_site.shape == (1,)

    @pytest.mark.contract
    def test_fixed_dynamics_excludes_dynamics_sites(self):
        """When dynamics is a fixed array, no dynamics sites appear."""
        dynamical_model_spec = _two_state_fixed_drift_model()
        registry = build_site_registry(compile_model_fixture(dynamical_model_spec))
        assert len([site for site in registry if site.site_kind == SiteKind.DYNAMICS_DECAY]) == 2

    @pytest.mark.contract
    def test_diag_diffusion_excludes_lower(self):
        """Diagonal diffusion has no lower-triangle sites."""
        dynamical_model_spec = _diag_diffusion_excludes_lower__make_spec()
        registry = build_site_registry(compile_model_fixture(dynamical_model_spec))
        names = {s.name for s in registry}
        assert "diffusion_diag_free" in names
        assert "diffusion_lower_free" not in names

    @pytest.mark.contract
    def test_free_diffusion_includes_lower(self):
        """Free diffusion includes lower-triangle sites."""
        dynamical_model_spec = two_state_gaussian_model()
        registry = build_site_registry(compile_model_fixture(dynamical_model_spec))
        names = {s.name for s in registry}
        assert "diffusion_diag_free" in names
        assert "diffusion_lower_free" in names

    @pytest.mark.contract
    def test_sparse_initial_state_correlations_only_include_authored_pairs(self):
        """Initial-state correlation sites should only exist for authored pairs."""
        mask = np.zeros((3, 3), dtype=bool)
        mask[2, 0] = True
        dynamical_model_spec = (
            _sparse_initial_state_correlations_only_include_authored_pairs__make_spec()
        )
        registry = build_site_registry(compile_model_fixture(dynamical_model_spec))
        site_map = {site.name: site for site in registry}
        assert site_map["t0_var_lower_free"].shape == (1,)

    @pytest.mark.contract
    def test_support_classes(self, simple_spec):
        """Check that support classes are correctly assigned."""
        registry = build_site_registry(compile_model_fixture(simple_spec))
        support_map = {s.name: s.support for s in registry}
        # POSITIVE support sites
        assert all(
            site.support == SupportClass.POSITIVE
            for site in registry
            if site.site_kind == SiteKind.DYNAMICS_DECAY
        )
        # POSITIVE support sites (HalfNormal priors)
        assert support_map["diffusion_diag_free"] == SupportClass.POSITIVE
        assert support_map["manifest_var_diag_free"] == SupportClass.POSITIVE
        assert support_map["t0_var_diag_free"] == SupportClass.POSITIVE

    @pytest.mark.contract
    def test_mixed_diffusion_includes_proc_df_site(self):
        """Any student-t latent in diffusion_dists should expose proc_df."""
        dynamical_model_spec = _student_innovation_model()
        registry = build_site_registry(compile_model_fixture(dynamical_model_spec))
        assert "proc_df" in {site.name for site in registry}

    @pytest.mark.inference(concern="sampling")
    def test_mixed_diffusion_sampling_emits_proc_df(self):
        """The traced model should sample proc_df when diffusion_dists include student_t."""
        dynamical_model_spec = _student_innovation_model()
        model = compile_fit_fixture(dynamical_model_spec)

        with handlers.seed(rng_seed=0):
            trace = handlers.trace(
                lambda: sample_sites(
                    process_sites(model.compiled_dynamical_model),
                    model.prior_runtime_bundle.priors.__getitem__,
                )
            ).get_trace()

        assert "proc_df" in trace

    @pytest.mark.inference(concern="sampling")
    def test_static_state_sd_site_is_registered_and_traced(self):
        """Compiled baseline factors should expose a positive static-state SD site."""
        dynamical_model_spec = _static_state_model()
        model = compile_fit_fixture(dynamical_model_spec)

        registry = build_site_registry(compile_model_fixture(dynamical_model_spec))
        site_map = {site.name: site for site in registry}
        assert site_map["static_state_sd_free"].shape == (1,)
        assert site_map["static_state_sd_free"].support == SupportClass.POSITIVE

        obs = jnp.zeros((5, numeric.n_observations(compile_model_fixture(dynamical_model_spec))))
        times = jnp.arange(5, dtype=jnp.float32)
        site_info = _discover_sites(
            model.prior_runtime_bundle,
            bind_panel_fixture(model.compiled_dynamical_model, obs, times),
            random.PRNGKey(0),
        )
        _assert_registry_matches_trace(registry, site_info)
        assert site_info["static_state_sd_free"].shape == (1,)


@pytest.mark.inference(concern="sampling")
class TestSpecBlockAssembly:
    def test_assemble_t0_cov_adds_low_rank_baseline_factor_covariance(self):
        """Static baseline factors should add `B diag(tau^2) B^T` to the t0 covariance."""
        dynamical_model_spec = _static_state_model()
        model = compile_fit_fixture(dynamical_model_spec)
        values = {
            site.name: jnp.ones(site.shape)
            for block in numeric.parameter_blocks(compile_model_fixture(dynamical_model_spec))
            for site in block.iter_sites()
        }
        values["static_state_sd_free"] = jnp.array([2.0])
        with handlers.substitute(data=values), handlers.trace() as trace:
            cov = sample_parameters(model.compiled_dynamical_model, model.prior_runtime_bundle)[
                "t0_cov"
            ]
        assert "t0_correlation_positive_definite" in trace

        np.testing.assert_allclose(
            np.asarray(cov),
            np.array([[5.0, 4.0], [4.0, 5.0]]),
            atol=1e-6,
        )


# ---------------------------------------------------------------------------
# Deterministic assembly
# ---------------------------------------------------------------------------


class TestDeterministicAssembly:
    @pytest.mark.contract
    def test_assemble_deterministics_from_registry_free_spec(self, simple_spec):
        """Registry-driven assembly builds the expected matrices."""
        samples = {
            "diffusion_diag_free": jnp.array([[0.4, 0.6]], dtype=jnp.float32),
            "diffusion_lower_free": jnp.array([[0.25]], dtype=jnp.float32),
            "lambda_free": jnp.array([], dtype=jnp.float32).reshape(1, 0),
            "manifest_var_diag_free": jnp.array([[0.7, 0.8]], dtype=jnp.float32),
            "t0_means_free": jnp.array([[1.0, -1.0]], dtype=jnp.float32),
            "t0_var_diag_free": jnp.array([[0.9, 1.1]], dtype=jnp.float32),
            "t0_var_lower_free": jnp.zeros((1, 1), dtype=jnp.float32),
        }

        det = assemble_deterministics_from_registry(samples, compile_model_fixture(simple_spec))
        assert jnp.allclose(det["diffusion"][0], jnp.array([[0.4, 0.0], [0.25, 0.6]]))
        assert det["lambda"].shape == (1, 2, 2)
        assert jnp.allclose(det["manifest_cov"][0], jnp.diag(jnp.array([0.49, 0.64])))
        assert jnp.allclose(det["t0_means"][0], jnp.array([1.0, -1.0]))
        assert jnp.allclose(det["t0_cov"][0], jnp.diag(jnp.array([0.81, 1.21])))

    @pytest.mark.contract
    def test_missing_declared_free_value_is_rejected(self, simple_spec):
        with pytest.raises(KeyError, match="diffusion_diag_free"):
            assemble_deterministics_from_registry({}, compile_model_fixture(simple_spec), n_draws=2)

    @pytest.mark.contract
    def test_assemble_deterministics_from_registry_fixed_blocks(self):
        """Fixed spec matrices are broadcast without any sampled sites."""
        dynamical_model_spec = _assemble_deterministics_from_registry_fixed_blocks__make_spec()
        det = assemble_deterministics_from_registry(
            {}, compile_model_fixture(dynamical_model_spec), n_draws=3
        )
        assert jnp.allclose(
            det["diffusion"],
            jnp.broadcast_to(
                compile_model_fixture(dynamical_model_spec).diffusion_block.assemble(), (3, 2, 2)
            ),
        )
        assert jnp.allclose(
            det["lambda"],
            jnp.broadcast_to(
                compile_model_fixture(dynamical_model_spec).loading_block.assemble(), (3, 2, 2)
            ),
        )
        manifest_chol = compile_model_fixture(
            dynamical_model_spec
        ).observation_noise_block.assemble()
        expected_manifest_cov = manifest_chol @ manifest_chol.T
        assert jnp.allclose(det["manifest_cov"], jnp.broadcast_to(expected_manifest_cov, (3, 2, 2)))
        assert isinstance(
            compile_model_fixture(dynamical_model_spec).initial_mean_block.assemble(), jnp.ndarray
        )
        assert jnp.allclose(
            det["t0_means"],
            jnp.broadcast_to(
                compile_model_fixture(dynamical_model_spec).initial_mean_block.assemble(), (3, 2)
            ),
        )
        expected_t0_cov = compile_model_fixture(
            dynamical_model_spec
        ).initial_covariance_block.assemble_cov()
        assert jnp.allclose(det["t0_cov"], jnp.broadcast_to(expected_t0_cov, (3, 2, 2)))

    @pytest.mark.contract
    def test_assemble_deterministics_from_registry_partial_manifest_variance_mask(self):
        """Registry assembly respects mixed fixed/free manifest-noise diagonals."""
        dynamical_model_spec = _partial_manifest_variance_model()
        samples = {
            "diffusion_diag_free": jnp.array([[0.4, 0.6]], dtype=jnp.float32),
            "diffusion_lower_free": jnp.array([[0.25]], dtype=jnp.float32),
            "lambda_free": jnp.array([], dtype=jnp.float32).reshape(1, 0),
            "manifest_var_diag_free": jnp.array([[0.9]], dtype=jnp.float32),
            "t0_means_free": jnp.array([[1.0, -1.0]], dtype=jnp.float32),
            "t0_var_diag_free": jnp.array([[0.9, 1.1]], dtype=jnp.float32),
            "t0_var_lower_free": jnp.zeros((1, 1), dtype=jnp.float32),
        }

        det = assemble_deterministics_from_registry(
            samples, compile_model_fixture(dynamical_model_spec)
        )
        assert jnp.allclose(det["manifest_cov"][0], jnp.diag(jnp.array([0.16, 0.81])))

    @pytest.mark.contract
    def test_assemble_deterministics_from_registry_initial_state_correlations(self):
        """Initial-state off-diagonal samples are interpreted as correlations."""
        mask = np.zeros((2, 2), dtype=bool)
        mask[1, 0] = True
        dynamical_model_spec = two_state_gaussian_model()
        samples = {
            "diffusion_diag_free": jnp.array([[0.4, 0.6]], dtype=jnp.float32),
            "diffusion_lower_free": jnp.array([[0.25]], dtype=jnp.float32),
            "lambda_free": jnp.array([], dtype=jnp.float32).reshape(1, 0),
            "manifest_var_diag_free": jnp.array([[0.7, 0.8]], dtype=jnp.float32),
            "t0_means_free": jnp.array([[1.0, -1.0]], dtype=jnp.float32),
            "t0_var_diag_free": jnp.array([[2.0, 3.0]], dtype=jnp.float32),
            "t0_var_lower_free": jnp.array([[0.25]], dtype=jnp.float32),
        }

        det = assemble_deterministics_from_registry(
            samples, compile_model_fixture(dynamical_model_spec)
        )

        assert jnp.allclose(
            det["t0_cov"][0],
            jnp.array([[4.0, 1.5], [1.5, 9.0]], dtype=jnp.float32),
        )

    @pytest.mark.inference(concern="sampling")
    def test_assembly_preserves_initial_correlations_for_density_rejection(self):
        """Assembly preserves the authored covariance; the model density rejects invalid support."""
        mask = np.zeros((3, 3), dtype=bool)
        mask[1, 0] = True
        mask[2, 0] = True
        mask[2, 1] = True
        dynamical_model_spec = (
            _assemble_deterministics_repairs_invalid_initial_correlation_matrix__make_spec()
        )
        samples = {
            "diffusion_diag_free": jnp.array([[0.4, 0.6, 0.5]], dtype=jnp.float32),
            "diffusion_lower_free": jnp.array([[0.25, 0.1, -0.15]], dtype=jnp.float32),
            "lambda_free": jnp.array([], dtype=jnp.float32).reshape(1, 0),
            "manifest_var_diag_free": jnp.array([[0.7, 0.8, 0.9]], dtype=jnp.float32),
            "t0_means_free": jnp.array([[1.0, -1.0, 0.5]], dtype=jnp.float32),
            "t0_var_diag_free": jnp.array([[1.0, 1.0, 1.0]], dtype=jnp.float32),
            "t0_var_lower_free": jnp.array([[0.9, 0.9, -0.9]], dtype=jnp.float32),
        }

        det = assemble_deterministics_from_registry(
            samples, compile_model_fixture(dynamical_model_spec)
        )
        min_eig = jnp.min(jnp.linalg.eigvalsh(det["t0_cov"][0]))

        assert bool(jnp.isfinite(det["t0_cov"]).all())
        np.testing.assert_allclose(
            det["t0_cov"][0], [[1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0]]
        )
        assert float(min_eig) < 0

        model = compile_fit_fixture(dynamical_model_spec)
        with handlers.substitute(data={name: value[0] for name, value in samples.items()}):
            trace = handlers.trace(sample_parameters).get_trace(
                model.compiled_dynamical_model, model.prior_runtime_bundle
            )
        np.testing.assert_allclose(trace["t0_cov"]["value"], det["t0_cov"][0], atol=1e-6)
        factor = trace["t0_correlation_positive_definite"]
        assert float(factor["fn"].log_prob(factor["value"])) == pytest.approx(-800000.01, rel=1e-5)


# ---------------------------------------------------------------------------
# Prior runtime state
# ---------------------------------------------------------------------------


@pytest.mark.contract
class TestNativeRuntimePriors:
    def test_sites_have_native_distributions_with_the_declared_shapes(self, simple_spec):
        registry = build_site_registry(compile_model_fixture(simple_spec))
        priors = resolve_site_priors(registry)
        assert set(priors) == {site.name for site in registry}
        for site in registry:
            assert isinstance(priors[site.name], dist.Distribution)
            assert priors[site.name].batch_shape == site.shape

    def test_partial_native_overrides_preserve_other_defaults(self, simple_spec):
        registry = build_site_registry(compile_model_fixture(simple_spec))
        decay_site = next(
            site.name for site in registry if site.site_kind == SiteKind.DYNAMICS_DECAY
        )
        priors = resolve_site_priors(registry, {decay_site: dist.Gamma(4.0, 2.0)})
        np.testing.assert_allclose(priors[decay_site].mean, 2.0)
        assert priors["diffusion_diag_free"].batch_shape == (2,)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Serialization roundtrip
# ---------------------------------------------------------------------------


class TestCompiledArtifactIntegration:
    """Test that compiled_prior_semantics is emitted and correctly consumed."""

    @pytest.mark.contract
    def test_global_ordered_threshold_priors_are_not_authorable(self):

        dynamical_model_spec = _global_ordered_threshold_priors_are_not_authorable__make_spec()
        parameter = ParameterSpec(
            id="parameter:bd5e7c989f1fe5b6752e958831d287e988f8a43f9633886ba7281220ae168533",
            name="obs_ordered_base",
            description="Unbound threshold",
        )
        with pytest.raises(ValueError, match="not referenced by component slots"):
            dynamical_model_spec.with_entities(
                parameters=(*dynamical_model_spec.parameters, parameter)
            )

    @pytest.mark.contract
    def test_ordered_threshold_priors_bind_per_manifest_component_and_row(self):
        from nof1_causal_lab.models.ssm.compile.prior_compilation import compile_priors
        from nof1_causal_lab.models.structure import StructuralSelection

        priors, bindings, _diagnostics = compile_priors(
            compile_model_fixture(
                _ordered_threshold_priors_bind_per_manifest_component_and_row_model_with_prior_payloads()
            ),
            StructuralSelection(
                _ordered_threshold_priors_bind_per_manifest_component_and_row_model_with_prior_payloads(),
                None,
            ),
        )

        binding_by_parameter = {
            binding.parameter_name: binding
            for binding in {binding.parameter_id: binding for binding in bindings}.values()
        }
        assert binding_by_parameter["obs_ordered_base_short_scale"].flat_index == 0
        assert binding_by_parameter["obs_ordered_base_long_scale"].flat_index == 1
        for row, name in enumerate(("obs_ordered_gaps_short_scale", "obs_ordered_gaps_long_scale")):
            assert {
                native.coordinate.indices[0]
                for native in binding_by_parameter[name].native_coordinates
            } == {row}

        base_prior = priors["obs_ordered_base"]
        np.testing.assert_allclose(base_prior.loc, [-1.0, -3.0])
        np.testing.assert_allclose(base_prior.scale, [0.5, 1.0])

        gap_prior = priors["obs_ordered_gaps"]
        gap_scales = np.asarray(gap_prior.scale).reshape(2, 8)
        np.testing.assert_allclose(gap_scales[0], 2.0)
        np.testing.assert_allclose(gap_scales[1], 0.5)

    @pytest.mark.contract
    def test_execution_checks_accept_complete_model(self):
        """A model with authored priors satisfies execution requirements."""

        compile_model_fixture(_mood_model())

    @pytest.mark.inference(concern="sampling")
    def test_runtime_derives_the_authored_priors(self):

        definition = _mood_model()
        model = compile_fit_fixture(definition)
        assert set(model.prior_runtime_bundle.priors) == {
            site.name for site in build_site_registry(compile_model_fixture(definition))
        }
