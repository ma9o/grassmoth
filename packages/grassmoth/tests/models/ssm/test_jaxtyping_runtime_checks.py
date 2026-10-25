"""Runtime shape/dtype checking via the jaxtyping + beartype import hook.

The ``--jaxtyping-packages=...,beartype.beartype`` pytest flag (see
``pyproject.toml``) instruments the listed modules so their jaxtyping
annotations are enforced at call time. These tests confirm the hook rejects
named-axis and dtype violations at the function boundary (before the body runs).
Valid inputs are covered by the covariance, guided-particle and parameter
inference tests. Configuration lint also requires instrumented modules to omit
``from __future__ import annotations`` so beartype can resolve its annotations.
"""

import jax
import jax.numpy as jnp
import jaxtyping
import pytest

from grassmoth.artifacts.reports.posterior_diagnostics import GaussianInitializationDiagnostics
from grassmoth.models.ssm.covariance import symmetrize
from grassmoth.models.ssm.inference.samplers.mpgibbs.kernel import _bounded_parameter_drift
from grassmoth.models.ssm.inference.samplers.particle.guided_geometry import _ProposalGeometry
from grassmoth.models.ssm.inference.samplers.particle.guided_information import point_information
from grassmoth.models.ssm.inference.samplers.particle.twist import GaussianTwist, future_information
from grassmoth.models.ssm.inference.warmup.initialization import ParticleInitialization
from grassmoth.models.ssm.inference.warmup.parameter_adaptation import (
    initialize_parameter_covariance,
)

pytestmark = pytest.mark.contract


def test_warmup_covariance_matches_parameter_dimension():
    with pytest.raises(jaxtyping.TypeCheckError, match="chol"):
        initialize_parameter_covariance(jnp.zeros((3, 2)), jnp.eye(1))


def test_initialization_pairs_parameter_and_path_chains():
    with pytest.raises(jaxtyping.TypeCheckError, match="paths"):
        ParticleInitialization(  # noqa: CORE001 -- Tests constructor rejection; no compiled evidence is created.
            positions=jnp.zeros((2, 1)),
            paths=jnp.zeros((3, 4, 2)),
            parameter_chol=jnp.eye(1),
            pilot_mean=jnp.zeros((4, 2)),
            pilot_variance=jnp.ones((4, 2)),
            diagnostics=GaussianInitializationDiagnostics(joint_dimension=9, scale=1.0),
        )


def test_covariance_named_axis_mismatch_raises():
    # A covariance must use the same named axis for both matrix dimensions.
    with pytest.raises(jaxtyping.TypeCheckError):
        symmetrize(jnp.ones((3, 2)))


def test_covariance_dtype_mismatch_raises():
    # Float[Array, "*batch N N"] rejects an integer-dtyped covariance.
    with pytest.raises(jaxtyping.TypeCheckError):
        symmetrize(jnp.eye(3, dtype=jnp.int32))


@pytest.mark.parametrize("compiled", [False, True])
def test_parameter_gradient_must_match_preconditioner(compiled):
    drift = jax.jit(_bounded_parameter_drift) if compiled else _bounded_parameter_drift
    with pytest.raises(jaxtyping.TypeCheckError, match="chol"):
        drift(jnp.ones((2, 3)), jnp.eye(4), jnp.array(0.2))


@pytest.mark.parametrize("mismatch", ["time", "state", "mask_dtype"])
def test_guide_information_rejects_inconsistent_path_arrays(mismatch):
    variance = jnp.ones((3, 2))
    mask = jnp.ones((3, 2), dtype=bool)
    if mismatch == "time":
        variance = variance[:2]
    elif mismatch == "state":
        variance = variance[:, :1]
    else:
        mask = mask.astype(jnp.int32)

    def observe(value, index):
        pytest.fail("The shape contract must reject the input before evaluating the target")

    with pytest.raises(
        jaxtyping.TypeCheckError, match="mask" if mismatch == "mask_dtype" else "variance"
    ):
        point_information("actual_law", observe, jnp.zeros((3, 2)), variance, mask)


def test_forward_proposal_noise_excludes_the_conditioned_particle():
    geometry = _ProposalGeometry(
        jnp.eye(2), jnp.zeros(2), jnp.eye(2), jnp.eye(2), jnp.array(0.0), jnp.zeros((2, 2))
    )
    with pytest.raises(jaxtyping.TypeCheckError, match="noise"):
        geometry.next_particles(
            jnp.zeros((4, 2)),
            jnp.zeros((4, 2)),  # Only the three unconditioned particles receive noise.
            jnp.zeros((3, 2)),
            jnp.ones((3, 2), dtype=bool),
            jnp.array(1),
        )


def test_twist_requires_one_transition_per_pair_of_time_points():
    information = GaussianTwist(jnp.zeros((4, 2, 2)), jnp.zeros((4, 2)))
    with pytest.raises(jaxtyping.TypeCheckError, match="matrices"):
        future_information(
            jnp.zeros((4, 2, 2)), jnp.zeros((3, 2)), jnp.zeros((3, 2, 2)), information
        )


def test_twist_value_matches_its_information_dimension():
    twist = GaussianTwist(jnp.zeros((3, 2, 2)), jnp.zeros((3, 2)))
    with pytest.raises(jaxtyping.TypeCheckError, match="value"):
        twist.log_density(jnp.zeros((4, 3)), 0)
