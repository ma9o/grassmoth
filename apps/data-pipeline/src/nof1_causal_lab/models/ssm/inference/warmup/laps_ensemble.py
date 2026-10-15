"""Single-device ensemble execution of BlackJAX's two LAPS initialization phases.

The BlackJAX kernels and adaptation rules are unchanged. Compiled batches run
the ensemble on the selected device. All members contribute to one adaptation
update, regardless of the batch width selected from compiler memory estimates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import numpy as np
from blackjax.adaptation import laps, laps_burn_in
from blackjax.mcmc.adjusted_mclmc import build_kernel
from blackjax.mcmc.hmc import HMCState
from blackjax.mcmc.integrators import (
    IntegratorState,
    generate_isokinetic_integrator,
    mclachlan_coefficients,
    omelyan_coefficients,
)

from nof1_causal_lab.artifacts.reports.posterior_diagnostics import LapsInitializationDiagnostics
from nof1_causal_lab.compute.batching import compiled_memory
from nof1_causal_lab.models.ssm.inference.tracing import EnsemblePhaseTraceRecord
from nof1_causal_lab.models.ssm.inference.warmup.outcomes import InitializationFailure

if TYPE_CHECKING:
    from collections.abc import Callable

    from nof1_causal_lab.artifacts.sampling import LapsInitializationSpec
    from nof1_causal_lab.compute.budget import ComputeBudget


@dataclass(frozen=True)
class NoLapsTrace:
    """Initialization ran without retaining its ensemble history."""


@dataclass(frozen=True)
class LapsTraceRecord:
    """Actual warmup transitions in the initializer's frozen affine coordinates.

    Positions are recorded after each move; step sizes are those used for that
    move. These adaptive and unadjusted positions are never posterior draws.
    """

    unadjusted: EnsemblePhaseTraceRecord  # noqa: F841, RUF100 - Notebook consumers are outside Vulture's source-only pass.
    adjusted: EnsemblePhaseTraceRecord  # noqa: F841, RUF100 - Notebook consumers are outside Vulture's source-only pass.
    adjusted_acceptance: jax.Array


@dataclass(frozen=True)
class LapsEnsemble:
    """Final joint positions and measurements of both initialization phases."""

    positions: jax.Array
    diagnostics: LapsInitializationDiagnostics
    trace: LapsTraceRecord | NoLapsTrace = field(default_factory=NoLapsTrace)


def _execute[State, Result](
    function: Callable[[State, int], Result],
    initial: State,
    budget: ComputeBudget,
    *,
    max_batch_size: int,
) -> Result:
    """Select a fitting batch width while preserving the full ensemble state.

    Compile the largest candidate first: full-state outputs and adaptation
    history are fixed costs, so scaling a one-member estimate would count them
    repeatedly. Halving only changes the compiled map width; no kernel executes
    until the complete phase fits, and no numerical run is retried after OOM.
    """
    batch_size = max_batch_size
    while True:
        compiled = jax.jit(function, static_argnums=1).lower(initial, batch_size).compile()
        if budget.capacity(compiled_memory(compiled), limit=1):
            return cast("Result", compiled(initial))
        if batch_size == 1:
            raise MemoryError("The LAPS ensemble exceeds the numerical execution memory budget")
        batch_size = max(1, batch_size // 2)


def run_laps_ensemble(
    log_density: Callable[[jax.Array], jax.Array],
    *,
    dimension: int,
    scientific_dimension: int,
    key: jax.Array,
    spec: LapsInitializationSpec,
    compute_budget: ComputeBudget,
    record_trace: bool = False,
) -> LapsEnsemble | InitializationFailure:
    """Run unadjusted LAPS burn-in followed by its Metropolis-adjusted refinement."""
    initial_key, burn_key, adjusted_key = jax.random.split(key, 3)
    positions = spec.initial_scale * jax.random.normal(initial_key, (spec.ensemble_size, dimension))

    def initialize(points: jax.Array, batch_size: int) -> IntegratorState:
        values, gradients = jax.lax.map(
            jax.value_and_grad(log_density), points, batch_size=batch_size
        )
        signs = jnp.where(jnp.mean(-points * gradients, axis=0) < 1, -1.0, 1.0)
        # Rescale before taking the norm, preserving direction without squaring
        # very large scores in the original physical units.
        scaled = gradients / jnp.max(jnp.abs(gradients), axis=-1, keepdims=True)
        momentum = signs * scaled / jnp.linalg.norm(scaled, axis=-1, keepdims=True)
        return IntegratorState(points, momentum, values, gradients)

    state = _execute(initialize, positions, compute_budget, max_batch_size=spec.ensemble_size)
    if not all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(state)):
        return InitializationFailure.unavailable(
            "LAPS starts have a non-finite joint density or gradient"
        )

    adaptation = laps_burn_in.Adaptation(
        dimension,
        microcanonical=True,
        alpha=1.9,
        C=0.1,
        r_end=0.01,
        bias_type=3,
        save_num=max(2, round(0.2 * spec.num_unadjusted_steps)),
    )
    burn_kernel = laps_burn_in.build_kernel(log_density, dimension, microcanonical=True)

    def burn(initial: IntegratorState, batch_size: int):
        def condition(carry):
            _, tuning, index, keep_going, _, _ = carry
            return (
                (index < spec.num_unadjusted_steps)
                & keep_going
                & jnp.isfinite(tuning.step_size)
                & (tuning.step_size > 0)
            )

        def step(carry):
            current, tuning, index, _, history, step_sizes = carry
            sampling_key, summary_key = jax.random.split(jax.random.fold_in(burn_key, index))
            current, info = jax.lax.map(
                lambda member: burn_kernel(member[0], member[1], tuning),
                (jax.random.split(sampling_key, spec.ensemble_size), current),
                batch_size=batch_size,
            )
            if record_trace:
                history = history.at[index].set(current.position)
                step_sizes = step_sizes.at[index].set(tuning.step_size)
            summaries = jax.vmap(adaptation.summary_statistics_fn, in_axes=(0, 0, None))(
                current, info, summary_key
            )
            tuning, summary = adaptation.update(
                tuning, jax.tree.map(lambda values: jnp.mean(values, axis=0), summaries)
            )
            return (
                current,
                tuning,
                index + 1,
                adaptation.while_cond(summary, index),
                history,
                step_sizes,
            )

        history_size = spec.num_unadjusted_steps if record_trace else 0
        return jax.lax.while_loop(
            condition,
            step,
            (
                initial,
                adaptation.initial_state,
                jnp.asarray(0),
                jnp.asarray(True),
                jnp.zeros((history_size, spec.ensemble_size, dimension), dtype=positions.dtype),
                jnp.zeros(history_size, dtype=positions.dtype),
            ),
        )

    state, burn_tuning, steps, _, burn_positions, burn_step_sizes = _execute(
        burn, state, compute_budget, max_batch_size=spec.ensemble_size
    )
    burn_step = float(burn_tuning.step_size)
    inverse_mass = np.asarray(burn_tuning.inverse_mass_matrix)
    if not np.isfinite(burn_step) or burn_step <= 0 or not np.isfinite(inverse_mass).all():
        return InitializationFailure.unavailable(
            "LAPS unadjusted step size or ensemble scale collapsed"
        )
    if np.any(inverse_mass <= 0):
        return InitializationFailure.unavailable(
            "LAPS ensemble has a constant coordinate after burn-in"
        )

    coefficients = omelyan_coefficients if dimension > 200 else mclachlan_coefficients
    adjusted_kernel = build_kernel(integrator=generate_isokinetic_integrator(coefficients))
    adjusted_adaptation = laps.Adaptation(
        burn_tuning._replace(
            step_size=burn_tuning.step_size / jnp.sqrt(jnp.mean(burn_tuning.inverse_mass_matrix))
        ),
        num_adaptation_samples=spec.num_adjusted_steps,
        steps_per_sample=spec.steps_per_sample,
        acc_prob_target=0.9 if dimension > 200 else 0.7,
    )

    def refine(initial: HMCState, batch_size: int):
        def step(carry, index):
            current, tuning = carry
            sampling_key, summary_key = jax.random.split(jax.random.fold_in(adjusted_key, index))

            def move(chain_key, chain_state):
                return adjusted_kernel(
                    rng_key=chain_key,
                    state=chain_state,
                    logdensity_fn=log_density,
                    step_size=tuning.step_size,
                    integration_steps_params=(spec.steps_per_sample,),
                    inverse_mass_matrix=burn_tuning.inverse_mass_matrix,
                    L_proposal_factor=1.25,
                )

            current, info = jax.lax.map(
                lambda member: move(member[0], member[1]),
                (jax.random.split(sampling_key, spec.ensemble_size), current),
                batch_size=batch_size,
            )
            used_step_size = tuning.step_size
            summaries = jax.vmap(adjusted_adaptation.summary_statistics_fn, in_axes=(0, 0, None))(
                current, info, summary_key
            )
            tuning, summary = adjusted_adaptation.update(
                tuning, jax.tree.map(lambda values: jnp.mean(values, axis=0), summaries)
            )
            return (current, tuning), (
                summary["acc_prob"],
                current.position if record_trace else jnp.empty((0,), dtype=positions.dtype),
                used_step_size if record_trace else jnp.empty((0,), dtype=positions.dtype),
            )

        return jax.lax.scan(
            step,
            (initial, adjusted_adaptation.initial_state),
            jnp.arange(spec.num_adjusted_steps),
        )

    (final, tuning), (acceptance, adjusted_positions, adjusted_step_sizes) = _execute(
        refine,
        HMCState(state.position, state.logdensity, state.logdensity_grad),
        compute_budget,
        max_batch_size=spec.ensemble_size,
    )
    mean_acceptance = float(jnp.mean(acceptance))
    final_step = float(tuning.step_size)
    if (
        not np.isfinite(np.asarray(final.position)).all()
        or not np.isfinite(np.asarray(final.logdensity)).all()
        or not bool(jnp.any(jnp.asarray(final.position) != jnp.asarray(state.position)))
        or not np.isfinite(final_step)
        or final_step <= 0
        or not np.isfinite(mean_acceptance)
        or mean_acceptance <= 0
    ):
        return InitializationFailure.unavailable(
            "LAPS adjusted refinement did not produce a finite moving ensemble"
        )
    return LapsEnsemble(
        positions=jnp.asarray(final.position),
        diagnostics=LapsInitializationDiagnostics(
            ensemble_size=spec.ensemble_size,
            joint_dimension=scientific_dimension,
            unadjusted_steps=int(steps),
            adjusted_steps=spec.num_adjusted_steps,
            unadjusted_step_size=burn_step,
            adjusted_step_size=final_step,
            adjusted_acceptance=mean_acceptance,
            early_stopped=int(steps) < spec.num_unadjusted_steps,
        ),
        trace=LapsTraceRecord(
            unadjusted=EnsemblePhaseTraceRecord(
                positions=burn_positions[: int(steps)],
                step_sizes=burn_step_sizes[: int(steps)],
            ),
            adjusted=EnsemblePhaseTraceRecord(
                positions=adjusted_positions,
                step_sizes=adjusted_step_sizes,
            ),
            adjusted_acceptance=acceptance,
        )
        if record_trace
        else NoLapsTrace(),
    )
