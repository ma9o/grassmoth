"""Tracing observes production moves without changing their random transitions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from grassmoth.artifacts.reports.posterior_diagnostics import ParticleWeightMetric
from grassmoth.artifacts.sampling import DsmcSpec
from grassmoth.models.ssm.inference.samplers.mpgibbs.kernel import (
    build_marginal_particle_gibbs_kernel,
)
from grassmoth.models.ssm.inference.targets.particle import ParticleTarget
from grassmoth_test_support.interval_particle import initialize_interval_chain, interval_target


@pytest.mark.inference(concern="sampling")
@pytest.mark.parametrize(
    ("steps", "proposal", "labels"),
    [(1, "pseudo_langevin", 2), (3, "pseudo_langevin", 3), (3, "random_walk", 2)],
)
def test_particle_trace_records_the_same_transition_and_selected_path(
    steps, proposal, labels, monkeypatch
):
    times = np.arange(float(steps))
    target = interval_target(times, np.full((steps, 1), 0.4), times[:, None], ("last",))
    joint = jnp.linspace(-0.2, 0.3, steps + 1)
    kernel = build_marginal_particle_gibbs_kernel(
        target,
        dsmc=DsmcSpec(num_particles=8),
        num_parameter_particles=labels,
        param_step_size=0.3,
        parameter_proposal=proposal,
        parameter_preconditioner_chol=jnp.array([[0.7]]),
        parameter_reference_path=joint[1:, None],
        diagnostic_metrics=("particle_weights",),
    )
    # Count executed pilot evaluations, including vmapped candidates, rather
    # than Python traces. Retained candidates must not be reevaluated next sweep.
    evaluations = []
    log_posterior = ParticleTarget.log_posterior

    def counted(self, position, path, observations, times):
        jax.debug.callback(lambda point: evaluations.append(np.asarray(point)), position)
        return log_posterior(self, position, path, observations, times)

    monkeypatch.setattr(ParticleTarget, "log_posterior", counted)
    initial = initialize_interval_chain(target, joint, kernel=kernel)
    key = jax.random.key(219)
    step = jax.jit(kernel.step_fn)
    ordinary = step(initial, key)
    traced = jax.jit(kernel.trace_step_fn)(initial, key)
    for actual, expected in zip(
        jax.tree.leaves(traced.result), jax.tree.leaves(ordinary), strict=True
    ):
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)

    updated, info = traced.result
    trace = traced.trace
    np.testing.assert_array_equal(trace.parameters.candidates[0], initial.position)
    np.testing.assert_array_equal(trace.smoother.particles[:, 0], initial.latent_trajectory)
    np.testing.assert_allclose(
        updated.position, trace.parameters.candidates[trace.label_selection.selected]
    )
    np.testing.assert_allclose(
        updated.latent_trajectory,
        trace.smoother.particles[jnp.arange(steps), trace.smoother.selected_particles],
    )
    for selection in (trace.smoother.root_selection, trace.label_selection):
        np.testing.assert_allclose(jnp.exp(selection.log_weights).sum(), 1, atol=2e-6)
        assert 0 <= int(selection.selected) < selection.log_weights.size
    assert trace.smoother.particles.shape == (steps, 8, 1)
    # Native histories contain every real leaf/merge and the final corrected law.
    stages = (
        sum(len(range(0, steps, 1 << level)) for level in range((steps - 1).bit_length() + 1)) + 1
    )
    for metric in ParticleWeightMetric:
        assert info[metric.history_key].shape == (stages,)
    probabilities = np.exp(np.asarray(trace.smoother.root_selection.log_weights))
    np.testing.assert_allclose(
        info[ParticleWeightMetric.ESS.history_key][-1],
        1 / np.square(probabilities).sum(),
        rtol=2e-6,
    )
    np.testing.assert_allclose(
        info[ParticleWeightMetric.REFERENCE_PROBABILITY.history_key][-1],
        probabilities[0],
        rtol=2e-6,
    )

    # Rescore independently after several transitions: the oracle stays on the
    # fixed pilot, while exact densities follow the newly selected path.
    states = [initial, updated]
    for seed in (220, 221):
        updated, _ = step(updated, jax.random.key(seed))
        states.append(updated)
    jax.effects_barrier()
    assert len(evaluations) == (1 + 4 * (labels - 1) if proposal == "pseudo_langevin" else 0)
    for state in states:
        expected_gradient = (
            jax.grad(
                lambda position: log_posterior(
                    target, position, joint[1:, None], target.observations, target.times
                )
            )(state.position)
            if proposal == "pseudo_langevin"
            else jnp.zeros_like(state.position)
        )
        np.testing.assert_allclose(
            state.parameter_gradient, expected_gradient, rtol=2e-6, atol=2e-6
        )
        complete, trajectory = target.log_posterior_from_context(
            state.position, state.latent_context, state.latent_trajectory, target.observations
        )
        np.testing.assert_allclose(state.trajectory_log_prob, trajectory, rtol=2e-6, atol=2e-6)
        np.testing.assert_allclose(state.complete_log_posterior, complete, rtol=2e-6, atol=2e-6)
