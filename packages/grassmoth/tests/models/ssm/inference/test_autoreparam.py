"""Comprehensive tests for AutoReparam: automatic reparameterization strategies.

Check project strategy selection, trace structure, and exact location-scale
reconstruction with deterministic standardized variates.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import dynestyx as dsx
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from numpy.testing import assert_allclose
from numpyro import handlers
from numpyro.infer.reparam import LocScaleReparam, ProjectedNormalReparam

from grassmoth.artifacts.model.construct import replace_constructs
from grassmoth.artifacts.model.expressions import restoring_force
from grassmoth.artifacts.model.mechanism import DriftMechanismSpec
from grassmoth.artifacts.model.parameter import SiteKind
from grassmoth.models.ssm.dynamics.vector_field import StructuralDrift
from grassmoth.models.ssm.inference.prior_program import numpyro_model
from grassmoth.models.ssm.inference.problem import build_particle_problem
from grassmoth.models.ssm.inference.reparameterization import (
    AutoReparam,
    _is_unconstrained,
    _loc_scale_reparam,
    _minimal_reparam,
)
from grassmoth_test_support.compute import compile_program
from grassmoth_test_support.inference import (
    MinimalReparam,
    bind_panel_fixture,
    compile_fit_fixture,
    compile_model_fixture,
)
from grassmoth_test_support.model import (
    construct_named,
    load_model_fixture,
    parameter_for,
    parameter_laws,
    two_state_gaussian_model,
)


def _particle_runtime_reconstructs_log_normal_hill_sites_with_parameter_distributions() -> (
    DynamicalModelSpec
):
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
    return dynamical_model_spec.with_entities(
        edges=replace_constructs(dynamical_model_spec.edges, (latent_0_revised,)),
        distributions=parameter_laws(
            dynamical_model_spec,
            {
                latent_0_latent_1_hill_emax.id: dist.LogNormal(
                    loc=jnp.array(-0.20000000298023224, dtype=jnp.float32),
                    scale=jnp.array(0.30000001192092896, dtype=jnp.float32),
                    validate_args=True,
                )
            },
        ),
    )


if TYPE_CHECKING:
    from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def trace_name_type(model_fn, *args, **kwargs):
    """Trace the model and return [(name, type)] for all sample/deterministic sites.

    Mirrors Pyro's trace_name_is_observed but adapted for NumPyro's trace
    structure where reparameterized sites become 'deterministic' type.
    """
    with handlers.seed(rng_seed=0):
        trace = handlers.trace(model_fn).get_trace(*args, **kwargs)
    return [
        (name, site["type"])
        for name, site in trace.items()
        if site["type"] in ("sample", "deterministic")
    ]


# ---------------------------------------------------------------------------
# Test models
# ---------------------------------------------------------------------------


def comprehensive_model():
    """Model exercising the full AutoReparam cascade.

    Covers: Normal (loc-scale, real), LogNormal (TransformedDistribution),
    HalfNormal (no loc, positive), Gamma (positive), StudentT (loc-scale + shape),
    batched Normal, Independent-wrapped Normal, observed sites.
    """
    a = numpyro.sample("a", dist.Normal(0, 1))
    b = numpyro.sample("b", dist.LogNormal(0, 1))
    c = numpyro.sample("c", dist.Normal(a, b))
    d = numpyro.sample("d", dist.HalfNormal(1.0))
    e = numpyro.sample("e", dist.Gamma(2.0, 1.0))
    f = numpyro.sample("f", dist.StudentT(5.0, a, b))
    g = numpyro.sample("g", dist.Normal(jnp.zeros(2), 1.0).to_event(1))
    h = numpyro.sample("h", dist.Normal(0, 1), obs=a)
    return a, b, c, d, e, f, g, h


def projected_normal_model():
    x = numpyro.sample("x", dist.ProjectedNormal(jnp.zeros(3)))
    numpyro.sample("obs", dist.Normal(jnp.sum(x), 0.1), obs=jnp.array(0.5))


def observed_projected_normal_model():
    numpyro.sample(
        "x",
        dist.ProjectedNormal(jnp.zeros(3)),
        obs=jnp.array([1.0, 0.0, 0.0]),
    )


def truncated_normal_model():
    """TruncatedNormal: has loc/scale but constrained support. Should NOT be reparameterized."""
    x = numpyro.sample("x", dist.TruncatedNormal(0.0, 1.0, low=-2.0, high=2.0))
    numpyro.sample("obs", dist.Normal(x, 0.1), obs=jnp.array(0.5))


def plated_model():
    """Model with numpyro.plate."""
    mu = numpyro.sample("mu", dist.Normal(0, 1))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    with numpyro.plate("data", 5):
        numpyro.sample("x", dist.Normal(mu, sigma))


# ---------------------------------------------------------------------------
# I. Unit tests: helper functions
# ---------------------------------------------------------------------------


@pytest.mark.contract
class TestIsUnconstrained:
    def test_real(self):
        assert _is_unconstrained(dist.constraints.real) is True

    def test_positive(self):
        assert _is_unconstrained(dist.constraints.positive) is False

    def test_unit_interval(self):
        assert _is_unconstrained(dist.constraints.unit_interval) is False

    def test_independent_real(self):
        assert _is_unconstrained(dist.constraints.independent(dist.constraints.real, 1)) is True

    def test_independent_positive(self):
        assert (
            _is_unconstrained(dist.constraints.independent(dist.constraints.positive, 1)) is False
        )


@pytest.mark.contract
class TestLocScaleReparamHelper:
    def test_normal(self):
        assert isinstance(_loc_scale_reparam("x", dist.Normal(0.0, 1.0), 0.0), LocScaleReparam)

    def test_half_normal_skipped(self):
        assert _loc_scale_reparam("sigma", dist.HalfNormal(1.0), 0.0) is None

    def test_gamma_skipped(self):
        assert _loc_scale_reparam("df", dist.Gamma(5.0, 1.0), 0.0) is None

    def test_laplace(self):
        assert isinstance(_loc_scale_reparam("x", dist.Laplace(0.0, 1.0), 0.5), LocScaleReparam)

    def test_decentered_name_skipped(self):
        assert _loc_scale_reparam("x_decentered", dist.Normal(0.0, 1.0), 0.0) is None

    def test_student_t_shape_params(self):
        result = _loc_scale_reparam("x", dist.StudentT(5.0, 0.0, 1.0), None)
        assert isinstance(result, LocScaleReparam)
        assert "df" in result.shape_params

    def test_truncated_normal_skipped(self):
        """TruncatedNormal has interval support, not real."""
        result = _loc_scale_reparam("x", dist.TruncatedNormal(0.0, 1.0, low=-2.0, high=2.0), 0.0)
        assert result is None

    def test_log_normal_skipped(self):
        """LogNormal has positive support."""
        result = _loc_scale_reparam("x", dist.LogNormal(0.0, 1.0), 0.0)
        assert result is None


@pytest.mark.contract
class TestMinimalReparamHelper:
    def test_normal_returns_none(self):
        assert _minimal_reparam(dist.Normal(0.0, 1.0), is_observed=False) is None

    def test_projected_normal(self):
        assert isinstance(
            _minimal_reparam(dist.ProjectedNormal(jnp.zeros(3)), is_observed=False),
            ProjectedNormalReparam,
        )

    def test_transformed_with_normal_base(self):
        td = dist.TransformedDistribution(dist.Normal(0.0, 1.0), dist.transforms.ExpTransform())
        assert _minimal_reparam(td, is_observed=False) is None

    def test_observed_projected_normal_returns_none(self):
        assert _minimal_reparam(dist.ProjectedNormal(jnp.zeros(3)), is_observed=True) is None


@pytest.mark.contract
class TestAutoReparamValidation:
    def test_centered_above_one(self):
        with pytest.raises(ValueError, match="centered must be in"):
            AutoReparam(centered=1.5)

    def test_centered_negative(self):
        with pytest.raises(ValueError, match="centered must be in"):
            AutoReparam(centered=-0.1)


# ---------------------------------------------------------------------------
# II. Trace structure (Pyro-style trace_name_is_observed)
# ---------------------------------------------------------------------------


@pytest.mark.inference(concern="sampling")
class TestTraceStructure:
    """Verify exact trace structure after reparameterization.

    Ported from Pyro's test_strategies.py::test_normal_auto pattern.
    """

    def test_comprehensive_minimal(self):
        """MinimalReparam should not touch normal/loc-scale sites."""
        model = MinimalReparam()(comprehensive_model)
        actual = trace_name_type(model)
        expected = [
            ("a", "sample"),
            ("b", "sample"),
            ("c", "sample"),
            ("d", "sample"),
            ("e", "sample"),
            ("f", "sample"),
            ("g", "sample"),
            ("h", "sample"),
        ]
        assert actual == expected

    def test_comprehensive_auto_decentered(self):
        """AutoReparam(centered=0.0): Normal/StudentT → decentered, LogNormal → TransformReparam."""
        strategy = AutoReparam(centered=0.0)
        model = strategy(comprehensive_model)
        actual = trace_name_type(model)
        expected = [
            # a: Normal → LocScaleReparam → decentered aux + deterministic original
            ("a_decentered", "sample"),
            ("a", "deterministic"),
            # b: LogNormal = TransformedDistribution → TransformReparam → base (Normal)
            #    Then b_base (Normal) is also reparameterized by LocScaleReparam
            ("b_base_decentered", "sample"),
            ("b_base", "deterministic"),
            ("b", "deterministic"),
            # c: Normal(a, b) → LocScaleReparam
            ("c_decentered", "sample"),
            ("c", "deterministic"),
            # d: HalfNormal → not reparameterized (positive support)
            ("d", "sample"),
            # e: Gamma → not reparameterized (positive support)
            ("e", "sample"),
            # f: StudentT → LocScaleReparam (has loc, scale, real support)
            ("f_decentered", "sample"),
            ("f", "deterministic"),
            # g: Normal(..).to_event(1) → LocScaleReparam (unwraps Independent)
            ("g_decentered", "sample"),
            ("g", "deterministic"),
            # h: observed Normal → not reparameterized
            ("h", "sample"),
        ]
        assert actual == expected

    def test_comprehensive_auto_centered(self):
        """AutoReparam(centered=1.0): fully centered = no-op for loc-scale, but TransformReparam still fires."""
        strategy = AutoReparam(centered=1.0)
        model = strategy(comprehensive_model)
        actual = trace_name_type(model)
        expected = [
            ("a", "sample"),  # centered=1.0 is identity
            # b: LogNormal → TransformReparam still applies (not loc-scale cascade)
            ("b_base", "sample"),
            ("b", "deterministic"),
            ("c", "sample"),
            ("d", "sample"),
            ("e", "sample"),
            ("f", "sample"),
            ("g", "sample"),
            ("h", "sample"),
        ]
        assert actual == expected

    def test_projected_normal_auto(self):
        """ProjectedNormalReparam creates x_normal (Normal), which then
        gets LocScaleReparam-ed to x_normal_decentered."""
        strategy = AutoReparam(centered=0.0)
        model = strategy(projected_normal_model)
        actual = trace_name_type(model)
        expected = [
            ("x_normal_decentered", "sample"),
            ("x_normal", "deterministic"),
            ("x", "deterministic"),
            ("obs", "sample"),
        ]
        assert actual == expected

    @pytest.mark.parametrize(
        ("strategy_factory"),
        [MinimalReparam, lambda: AutoReparam(centered=0.0)],
    )
    def test_observed_projected_normal_not_reparameterized(self, strategy_factory):
        model = strategy_factory()(observed_projected_normal_model)
        actual = trace_name_type(model)
        assert actual == [("x", "sample")]

    def test_truncated_normal_not_reparameterized(self):
        """TruncatedNormal has constrained support — should be left alone."""
        strategy = AutoReparam(centered=0.0)
        model = strategy(truncated_normal_model)
        actual = trace_name_type(model)
        expected = [
            ("x", "sample"),
            ("obs", "sample"),
        ]
        assert actual == expected

    def test_plated_sites(self):
        """Sites inside numpyro.plate should be reparameterized correctly."""
        strategy = AutoReparam(centered=0.0)
        model = strategy(plated_model)
        actual = trace_name_type(model)
        expected = [
            ("mu_decentered", "sample"),
            ("mu", "deterministic"),
            ("sigma", "sample"),  # HalfNormal, not reparameterized
            ("x_decentered", "sample"),
            ("x", "deterministic"),
        ]
        assert actual == expected

    def test_config_dict_reuse(self):
        """After first run, strategy.config can be used as standalone dict config.

        Ported from Pyro's test_strategies.py::test_normal_auto.
        """
        strategy = AutoReparam(centered=0.0)
        model = strategy(comprehensive_model)
        first_result = trace_name_type(model)

        # Extract config dict and use it directly
        config_dict = strategy.config
        assert isinstance(config_dict, dict)
        model_from_dict = handlers.reparam(comprehensive_model, config=config_dict)
        second_result = trace_name_type(model_from_dict)

        assert first_result == second_result


# ---------------------------------------------------------------------------
# III. Exact reconstruction and gradients
# ---------------------------------------------------------------------------


@pytest.mark.inference(concern="sampling")
class TestLocScalePreservation:
    """Exercise AutoReparam without Monte Carlo error or large random draws."""

    @pytest.mark.parametrize("shape", [(), (4,), (3, 2)], ids=str)
    @pytest.mark.parametrize("centered", [0.0, 0.6, 1.0, None])
    @pytest.mark.parametrize("dist_type", ["Normal", "StudentT"])
    def test_reconstruction_and_gradients(self, dist_type, centered, shape):
        rng = np.random.default_rng(0)
        loc = jnp.asarray(rng.uniform(-1.0, 1.0, shape), dtype=jnp.float32)
        scale = jnp.asarray(rng.uniform(0.5, 1.5, shape), dtype=jnp.float32)
        noise = jnp.asarray(rng.normal(size=shape), dtype=jnp.float32)

        def model(loc, scale):
            with numpyro.plate_stack("plates", shape):
                if dist_type == "Normal":
                    numpyro.sample("x", dist.Normal(loc, scale))
                else:
                    numpyro.sample("x", dist.StudentT(10.0, loc, scale))

        def standardized_value(msg):
            if msg["type"] == "sample":
                law = msg["fn"]
                assert isinstance(law, dist.Normal if dist_type == "Normal" else dist.StudentT)
                # Fix the standardized variate while retaining the selected law.
                # This couples both parameterizations without relying on RNG keys.
                if dist_type == "StudentT":
                    assert_allclose(law.df, 10.0)
                return law.loc + law.scale * noise
            return None

        def reconstruct(loc, scale):
            strategy = AutoReparam(centered=centered)
            with handlers.substitute(substitute_fn=standardized_value):
                trace = handlers.trace(strategy(model)).get_trace(loc, scale)
            assert isinstance(strategy.config["x"], LocScaleReparam)
            assert trace["x"]["type"] == ("sample" if centered == 1.0 else "deterministic")
            return trace["x"]["value"]

        assert_allclose(reconstruct(loc, scale), loc + scale * noise, atol=1e-6)
        loc_grad, scale_grad = jax.jacfwd(reconstruct, argnums=(0, 1))(loc, scale)
        identity = np.eye(loc.size).reshape(shape + shape)
        assert_allclose(loc_grad, identity, atol=1e-6)
        assert_allclose(scale_grad, identity * np.asarray(noise), atol=1e-6)


# ---------------------------------------------------------------------------
# IV. SSM-specific integration
# ---------------------------------------------------------------------------


@pytest.mark.inference(concern="sampling")
class TestAutoReparamSSM:
    """Test AutoReparam with the actual SSM model."""

    def test_ssm_site_classification(self):
        """Verify which SSM sites get reparameterized and which don't."""
        model = compile_fit_fixture(two_state_gaussian_model())
        strategy = AutoReparam(centered=0.0)

        model_fn = functools.partial(numpyro_model, likelihood_backend=None)
        reparam_model = handlers.reparam(model_fn, config=strategy)

        T = 2
        observations = jnp.zeros((T, 2))
        times = jnp.linspace(0, 1, T)

        with handlers.seed(rng_seed=42):
            trace = handlers.trace(reparam_model).get_trace(
                bind_panel_fixture(model.compiled_dynamical_model, observations, times),
                priors=model.prior_runtime_bundle,
            )

        # Normal sites (loc-scale, real support) → LocScaleReparam
        for site in ["vf_2_p0", "t0_means_free"]:
            assert isinstance(strategy.config[site], LocScaleReparam), (
                f"{site} should be LocScaleReparam"
            )

        # Positive-support sites → None
        for site in [
            "vf_0_p0",
            "diffusion_diag_free",
            "manifest_var_diag_free",
            "t0_var_diag_free",
        ]:
            assert site in trace
            assert strategy.config[site] is None, f"{site} should NOT be reparameterized"

        # All values finite
        for name, site in trace.items():
            if site["type"] in ("sample", "deterministic"):
                assert jnp.all(jnp.isfinite(site["value"])), f"Non-finite at {name}"

    def test_extract_constrained_samples_filters_auxiliary_sites(self):
        """Report original parameters, excluding reparam auxiliaries and assembled matrices."""
        from grassmoth.models.ssm.inference.parameter_tracing import (
            extract_constrained_samples,
            prepare_model_parameters,
        )
        from grassmoth.models.ssm.parameterization import build_site_registry

        model = compile_fit_fixture(two_state_gaussian_model())
        observations = jnp.zeros((5, 2))
        times = jnp.linspace(0, 1, 5)
        parameters, _, public_sites = prepare_model_parameters(
            model.prior_runtime_bundle,
            bind_panel_fixture(model.compiled_dynamical_model, observations, times),
            jax.random.PRNGKey(0),
            AutoReparam(centered=0.0),
            compiler=compile_program,
        )
        particles = jnp.stack([parameters.initial_position, parameters.initial_position + 0.05])
        samples = extract_constrained_samples(
            particles, parameters, public_sites, compiler=compile_program
        )

        assert "vf_0_p0" in samples
        assert "diffusion_diag_free" in samples
        assert all("_decentered" not in name for name in samples)
        assert samples["vf_0_p0"].shape[0] == 2
        assert samples["diffusion_diag_free"].shape[0] == 2
        assert set(samples) == {
            site.name for site in build_site_registry(model.compiled_dynamical_model)
        }

    def test_particle_runtime_reconstructs_log_normal_hill_sites(self):
        """Nested TransformReparam + LocScaleReparam restores the public Hill site."""

        dynamical_model_spec = (
            _particle_runtime_reconstructs_log_normal_hill_sites_with_parameter_distributions()
        )
        model = compile_fit_fixture(dynamical_model_spec)
        observations = jnp.zeros((3, 2))
        times = jnp.arange(3, dtype=jnp.float32)

        bundle = build_particle_problem(
            model.prior_runtime_bundle,
            bind_panel_fixture(model.compiled_dynamical_model, observations, times),
            trace_key=jax.random.PRNGKey(0),
            reparam=AutoReparam(centered=0.0),
            compiler=compile_program,
        )
        context = bundle.runtime.context(bundle.runtime.initial_position, times)

        assert "vf_2_p0_base_decentered" in bundle.site_info

        hill = compile_model_fixture(dynamical_model_spec).dynamics.spec.components[2]
        evolution = context[0].state_evolution
        assert isinstance(evolution, dsx.StochasticContinuousTimeStateEvolution)
        assert isinstance(evolution.drift, StructuralDrift)
        params = evolution.drift.args.params[2]
        assert set(params) == dict(hill.parameter_sites("vf_2")).keys()
        assert bool(jnp.all(jnp.isfinite(jnp.stack(tuple(params.values())))))
