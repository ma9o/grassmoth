"""Joint marginal-model quadrature checks, including the ν–precision collapse order."""

from dataclasses import replace

import dynestyx as dsx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from grassmoth.artifacts.probability.families import DistributionFamily
from grassmoth.artifacts.sampling import (
    AuxiliaryKalmanMethodSpec,
    BlockedPathUpdateSpec,
    ConditionalParameterProposalSpec,
    FixedLatentStepSpec,
    GuidedMethodSpec,
    ParameterMeanProposalSpec,
    PathUpdateSpec,
    WholePathUpdateSpec,
)
from grassmoth.models.ssm.execution.contracts import MeasurementParams
from grassmoth.models.ssm.execution.dynamical_model import HeterogeneousObservation
from grassmoth.models.ssm.inference.parameter_transform import ParameterTransform
from grassmoth.models.ssm.inference.samplers.mpgibbs.kernel import (
    build_marginal_particle_gibbs_kernel,
)
from grassmoth.models.ssm.inference.samplers.mpgibbs.runner import _initialize_chain_state
from grassmoth.models.ssm.structure.observation import DenseLoadings, ObservationStructure
from grassmoth_test_support.interval_particle import interval_target
from grassmoth_test_support.observation import observation_laws
from grassmoth_test_support.student_reference import student_features, student_quadrature

pytestmark = pytest.mark.inference(concern="sampling")


def _target(dimension=1):
    times = np.arange(2.0)
    readings = np.array([[1.4, -0.4], [-0.8, 0.6]])
    base = interval_target(
        times,
        readings,
        np.broadcast_to(times[:, None], readings.shape),
        ("last", "last"),
        state_dim=dimension,
    )

    template_laws = observation_laws(
        (DistributionFamily.STUDENT_T,) * 2, parameters={"obs_df": 3.0}
    )

    def model(z):
        df = 3 * jnp.exp(0.7 * z[0])
        scale = jnp.exp(-0.3 + 0.3 * z[1])
        return dsx.DynamicalModel(
            initial_condition=dist.MultivariateNormal(jnp.zeros(dimension), jnp.eye(dimension)),
            state_evolution=dsx.LinearGaussianStateEvolution(
                jnp.eye(dimension) * 0.7, jnp.eye(dimension) * 0.3
            ),
            observation_model=HeterogeneousObservation(
                MeasurementParams(
                    jnp.broadcast_to(jnp.array([[1.0], [-0.7]]), (2, dimension))
                    / dimension
                    * (1 + 0.3 * z[2]),
                    jnp.array([0.2, -0.1]),
                    jnp.eye(2) * scale**2,
                ),
                tuple(
                    eqx.tree_at(lambda value: value.df.values, law, (df,)) for law in template_laws
                ),
                structure=ObservationStructure(DenseLoadings(), "dense"),
            ),
            t0=0.0,
        )

    _, static = eqx.partition(model(jnp.zeros(3)), eqx.is_array)
    return replace(
        base,
        parameters=ParameterTransform(
            jnp.zeros(3),
            lambda z: {"z": z},
            lambda z: {"z": z},
            lambda z: dist.Normal(0.0, 1.0).log_prob(z).sum(),
        ),
        context=lambda z, grid: (eqx.filter(model(z), eqx.is_array), grid, jnp.zeros(dimension)),
        dynamical_model=lambda context: eqx.combine(context[0], static),
    )


@pytest.fixture(scope="module")
def marginal_reference():
    """Independent marginal-model tensor quadrature, refined along the latent path."""
    return student_quadrature()


def _initialize(target, kernel, joint):
    return _initialize_chain_state(
        joint[:3],
        observations=target.observations,
        times=target.times,
        target=target,
        parameter_gradient_fn=kernel.parameter_gradient_fn,
        parameter_chol=kernel.initial_parameter_chol,
        initial_latent_delta=jnp.full((2,), 0.55),
        param_step_size=0.18,
        param_min_scale=1e-6,
        param_max_scale=1e3,
        initial_latent_trajectory=joint[3:, None],
    )


@pytest.mark.parametrize(
    ("method", "kappa", "proposal"),
    [
        ("particle_mgrad", 0, "random_walk"),
        ("twisted_particle_agrad", 0, "pseudo_langevin"),
        ("particle_mgrad", 1, "random_walk"),
        ("twisted_particle_agrad", 1, "pseudo_langevin"),
        ("kalman_full", 1, "random_walk"),
        ("kalman_blocked", 1, "random_walk"),
    ],
)
def test_augmented_sweep_preserves_joint_marginal_posterior(
    marginal_reference, method, kappa, proposal
):
    starts, expected, variance = marginal_reference
    # Round the starts to float32 first, so unmoved coordinates compare equal.
    starts = jnp.asarray(starts)
    target = _target()
    kernel = build_marginal_particle_gibbs_kernel(
        target,
        num_parameter_particles=3,
        parameter_proposal=proposal,
        parameter_reference_path=jnp.zeros((2, 1)),
        path_update=(
            PathUpdateSpec(
                method=AuxiliaryKalmanMethodSpec(),
                update=BlockedPathUpdateSpec(block_length=1)
                if method == "kalman_blocked"
                else WholePathUpdateSpec(),
                proposal=ConditionalParameterProposalSpec()
                if method == "kalman_blocked"
                else ParameterMeanProposalSpec(),
            )
            if method.startswith("kalman")
            else PathUpdateSpec(method=GuidedMethodSpec(kind=method, num_particles=8, kappa=kappa))
        ),
    )

    def one(joint, key):
        state = _initialize(target, kernel, joint)
        step_key = jax.random.split(key, 3)[1]
        updated, _ = kernel.step_fn(state, step_key)
        complete, trajectory = target.log_posterior_from_context(
            updated.position, updated.latent_context, updated.latent_trajectory, target.observations
        )
        return jnp.concatenate((updated.position, updated.latent_trajectory[:, 0])), jnp.stack(
            (updated.complete_log_posterior - complete, updated.trajectory_log_prob - trajectory)
        )

    # Native one-step invariance from independent posterior starts avoids mixing-based tolerances.
    moved, cache_error = jax.jit(jax.vmap(one))(
        starts, jax.random.split(jax.random.key(926), len(starts))
    )
    np.testing.assert_allclose(cache_error, 0, atol=1e-5)
    observed = student_features(np.asarray(moved)).mean(axis=0)
    assert np.max(np.abs(observed - expected) / np.sqrt(variance / len(starts))) < 5.5
    assert np.mean(np.any(np.asarray(moved) != np.asarray(starts), axis=1)) > 0.8


@pytest.mark.parametrize("fixed", [False, True])
def test_trace_and_split_sweeps_preserve_marginal_caches_and_fixed_steps(fixed):
    target = _target()
    options = {"latent_step": FixedLatentStepSpec(delta=0.55)} if fixed else {}
    kernel = build_marginal_particle_gibbs_kernel(
        target,
        num_parameter_particles=3,
        parameter_proposal="pseudo_langevin",
        parameter_reference_path=jnp.zeros((2, 1)),
        path_update=PathUpdateSpec(
            method=GuidedMethodSpec(kind="twisted_particle_agrad", num_particles=8, kappa=1),
            **options,
        ),
    )
    state = _initialize(target, kernel, jnp.zeros(5))
    key = jax.random.key(321)
    traced = jax.jit(kernel.trace_step_fn)(state, key)
    plain = jax.jit(kernel.step_fn)(state, key)
    for left, right in zip(jax.tree.leaves(plain), jax.tree.leaves(traced.result), strict=True):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(plain[0].latent_delta, state.latent_delta)

    def scan(initial, keys):
        return jax.lax.scan(
            lambda current, draw_key: (kernel.step_fn(current, draw_key)[0], None), initial, keys
        )[0]

    run = jax.jit(scan)
    keys = jax.random.split(jax.random.key(491), 4)
    whole = run(state, keys)
    continued = run(run(state, keys[:2]), keys[2:])
    for left, right in zip(jax.tree.leaves(whole), jax.tree.leaves(continued), strict=True):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("kappa", [0, 1])
@pytest.mark.parametrize("dimension", [1, 3])
def test_fused_tail_preserves_student_precision_conditioning(kappa, dimension):
    from grassmoth.artifacts.sampling import ParameterMixtureProposalSpec
    from grassmoth.models.ssm.inference.samplers.particle.context import build_smoother_context
    from grassmoth.models.ssm.inference.samplers.particle.guided import trace_guided_step
    from grassmoth.models.ssm.inference.samplers.particle.setup import build_smoother_setup

    dense = _target(dimension)
    diagonal = replace(
        dense,
        state_space=replace(
            dense.state_space, initial_covariance="diagonal", transition_covariance="diagonal"
        ),
    )
    labels = jnp.array([[0.1, -0.2, 0.3], [-0.3, 0.1, -0.4]])
    reference = jnp.broadcast_to(jnp.array([[0.1], [-0.3]]), (2, dimension))
    log_precisions = jnp.array([[0.2, -0.7], [1.1, 0.5]])
    spec = PathUpdateSpec(
        method=GuidedMethodSpec(kind="particle_mgrad", num_particles=8, kappa=kappa),
        proposal=ParameterMixtureProposalSpec(),
    )

    def run(target):
        setup = build_smoother_setup(
            target,
            num_parameter_particles=2,
            free_mask=target.independent_mask,
            diagnostic_metrics=(),
        )
        context = build_smoother_context(
            setup,
            labels,
            jnp.array([0.1, -0.3]),
            reference=reference,
            delta=jnp.full((2, dimension), 0.2),
            trajectory_dtype=reference.dtype,
        )
        guide = target.context(jnp.mean(labels, axis=0), target.times)
        return jax.jit(
            lambda: trace_guided_step(
                target,
                context,
                guide,
                spec,
                jax.random.key(71),
                reference,
                log_precisions=log_precisions,
            )
        )()

    for actual, expected in zip(
        jax.tree.leaves(run(diagonal)), jax.tree.leaves(run(dense)), strict=True
    ):
        np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-5)
