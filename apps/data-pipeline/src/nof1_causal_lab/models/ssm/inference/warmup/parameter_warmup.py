"""Shared joint initialization and preconditioner setup for blocked SSM MCMC.

LAPS supplies paired parameter/path starts and an ensemble covariance from the
exact joint density. Its ensemble is initialization evidence, never a posterior.
Other explicit init positions (Pathfinder draws) and preconditioners (Pathfinder or
Laplace/MAP covariance) are Gaussian approximations used only to warm-start the
sampler. Seeding from them is valid no matter how poor the linearization is: the
invariant distribution is the exact posterior for any seed, so the approximation
sets burn-in, not the target (Andrieu, Doucet & Holenstein 2010). init_positions
consumes only the location, never the covariance, so a bad linearization cannot
bias the result; the preconditioner uses the covariance but only as a proposal
scale that the Metropolis accept/reject corrects and adaptation re-tunes. These
Gaussians are seeds, never the reported posterior.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from jax.typing import DTypeLike

    from nof1_causal_lab.compute.budget import ComputeBudget
    from nof1_causal_lab.models.ssm.autoreparam import Strategy
    from nof1_causal_lab.models.ssm.parameterization import PriorRuntimeBundle
    from nof1_causal_lab.models.ssm.runtime import BoundPanel
    from nof1_causal_lab.sampler_config import (
        LapsInitializationSpec,
        ParameterInitialization,
        ParameterPreconditioner,
    )

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from nof1_causal_lab.artifacts.posterior_diagnostics import (
    ParameterWarmupDiagnostics,
    PathfinderDiagnostics,
)
from nof1_causal_lab.models.ssm.inference.warmup.laps import (
    LapsInitializationResult,
    prepare_laps_initialization,
)
from nof1_causal_lab.models.ssm.inference.warmup.outcomes import InitializationFailure
from nof1_causal_lab.models.ssm.inference.warmup.scipy_pathfinder import (
    ScipyPathfinderResult,
    run_scipy_pathfinder_approximation,
    sample_scipy_pathfinder_init_positions,
    scipy_pathfinder_preconditioner_chol,
)

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.inference.targets.particle import ParticleTarget
    from nof1_causal_lab.models.ssm.inference.types import WarmupProposal

logger = logging.getLogger(__name__)

DEFAULT_PRIOR_RELEASED_SITE_NAMES: tuple[str, ...] = ("obs_df",)


@dataclass(frozen=True)
class ParameterWarmupResult:
    """Resolved parameter initialisation and preconditioning artifacts."""

    init_positions: jnp.ndarray | None
    preconditioner_chol: jnp.ndarray | None
    warmup_diagnostics: ParameterWarmupDiagnostics
    pathfinder_state: ScipyPathfinderResult | None
    laps_initialization: LapsInitializationResult | None


def _laplace_preconditioner_chol_from_map_result(
    map_result: WarmupProposal, jitter: float = 1e-6
) -> jnp.ndarray:
    """Build a parameter-kernel preconditioner Cholesky from ``fit_map`` covariance."""
    covariance = np.asarray(map_result.diagnostics.parameter_covariance)
    covariance = 0.5 * (covariance + covariance.T)
    covariance = covariance + jitter * np.eye(covariance.shape[0], dtype=covariance.dtype)
    return jnp.asarray(np.linalg.cholesky(covariance), dtype=jnp.float32)


def _validate_initial_positions_override(
    initial_positions_override: jnp.ndarray,
    *,
    num_chains: int,
    dim: int,
    dtype: DTypeLike,
) -> jnp.ndarray:
    init_positions = jnp.asarray(initial_positions_override, dtype=dtype)
    if init_positions.shape != (num_chains, dim):
        raise ValueError(
            "initial_positions_override must have shape (num_chains, dim); got "
            f"{init_positions.shape}"
        )
    return init_positions


def _log_pathfinder_completion(
    *,
    phase_label: str,
    started_at: float,
    pathfinder_diagnostics: PathfinderDiagnostics,
    clock: Callable[[], float],
) -> None:
    best_elbo = pathfinder_diagnostics.best_pathfinder_elbo
    elbo_spread = pathfinder_diagnostics.pathfinder_elbo_spread
    logger.info(
        "%s: scipy_pathfinder complete in %.1fs "
        "(setup=%.1fs, jax_compile=%.1fs, runtime=%.1fs, best_elbo=%s, "
        "elbo_spread=%s, n_starts_finite=%s)",
        phase_label,
        (clock() - started_at),
        pathfinder_diagnostics.pathfinder_setup_seconds,
        pathfinder_diagnostics.pathfinder_jax_compile_seconds,
        pathfinder_diagnostics.pathfinder_runtime_seconds,
        f"{best_elbo:.2f}",
        f"{elbo_spread:.2f}",
        pathfinder_diagnostics.n_pathfinder_starts_finite,
    )


def prepare_parameter_warmup(
    priors: PriorRuntimeBundle,
    panel: BoundPanel,
    *,
    compute_budget: ComputeBudget,
    bundle: ParticleTarget,
    method_label: str,
    phase_label: str,
    trace_key: jnp.ndarray,
    pathfinder_key: jnp.ndarray,
    sample_key: jnp.ndarray,
    reparam: Strategy | None,
    seed: int,
    n_ieks_iters: int,
    num_chains: int,
    init_method: ParameterInitialization,
    laps_spec: LapsInitializationSpec,
    initial_positions_override: jnp.ndarray | None,
    parameter_preconditioner_chol: jnp.ndarray | None,
    auto_preconditioner_method: ParameterPreconditioner,
    auto_preconditioner_maxiter: int,
    pathfinder_num_elbo_samples: int,
    pathfinder_maxiter: int,
    n_pathfinder_starts: int,
    pathfinder_parallel_workers: int | None,
    pathfinder_init_scale: float | None,
    prior_released_sites: tuple[str, ...] = DEFAULT_PRIOR_RELEASED_SITE_NAMES,
    prior_release_scale: float = 0.05,
    release_jitter_key: jnp.ndarray | None = None,
    clock: Callable[[], float],
) -> ParameterWarmupResult | InitializationFailure:
    """Resolve parameter init positions and the parameter-kernel preconditioner.

    The policy is intentionally centralized: if Pathfinder is needed for both
    initialization and preconditioning, this function runs the fit once and hands
    the same best-ELBO Gaussian to both consumers.
    """
    total_t0 = clock()
    dim = int(bundle.initial_position.shape[0])
    dtype = bundle.initial_position.dtype
    pathfinder_state: ScipyPathfinderResult | None = None
    pathfinder_diagnostics: PathfinderDiagnostics | None = None
    laps_result: LapsInitializationResult | None = None
    init_positions: jnp.ndarray | None = None
    preconditioner_chol = parameter_preconditioner_chol

    if (initial_positions_override is None and init_method == "laps") or (
        parameter_preconditioner_chol is None and auto_preconditioner_method == "laps"
    ):
        logger.info(
            "%s: running joint LAPS initialization (ensemble=%d)",
            phase_label,
            laps_spec.ensemble_size,
        )
        laps_outcome = prepare_laps_initialization(
            bundle,
            spec=laps_spec,
            key=pathfinder_key,
            num_chains=num_chains,
            compute_budget=compute_budget,
        )
        if isinstance(laps_outcome, InitializationFailure):
            return laps_outcome
        laps_result = laps_outcome
        measured = laps_result.diagnostics
        logger.info(
            "%s: LAPS ready (dim=%d, ensemble=%d, unadjusted=%d, adjusted=%d, "
            "early_stop=%s, steps=%.4g/%.4g, adjusted_accept=%.3f)",
            phase_label,
            measured.joint_dimension,
            measured.ensemble_size,
            measured.unadjusted_steps,
            measured.adjusted_steps,
            measured.early_stopped,
            measured.unadjusted_step_size,
            measured.adjusted_step_size,
            measured.adjusted_acceptance,
        )

    sampling_mode = None
    released_names: tuple[str, ...] = ()
    released_indices: tuple[int, ...] = ()
    released_scale = 0.0
    preconditioner_device = None
    pathfinder_consumers: list[str] = []
    if initial_positions_override is None and init_method == "pathfinder":
        pathfinder_consumers.append("init")
    if parameter_preconditioner_chol is None and auto_preconditioner_method == "pathfinder":
        pathfinder_consumers.append("preconditioner")

    if pathfinder_consumers:
        pathfinder_t0 = clock()
        logger.info(
            "%s: running scipy_pathfinder for %s (n_starts=%d, maxiter=%d, "
            "n_ieks_iters=%d, elbo_samples=%d, parallel_workers=%s)...",
            phase_label,
            "+".join(pathfinder_consumers),
            n_pathfinder_starts,
            pathfinder_maxiter,
            n_ieks_iters,
            pathfinder_num_elbo_samples,
            pathfinder_parallel_workers if pathfinder_parallel_workers is not None else "starts",
        )
        pathfinder_state, pathfinder_diagnostics = run_scipy_pathfinder_approximation(
            priors,
            panel,
            compute_budget=compute_budget,
            trace_key=trace_key,
            pathfinder_key=pathfinder_key,
            reparam=reparam,
            n_ieks_iters=n_ieks_iters,
            num_elbo_samples=pathfinder_num_elbo_samples,
            maxiter=pathfinder_maxiter,
            n_pathfinder_starts=n_pathfinder_starts,
            pathfinder_parallel_workers=pathfinder_parallel_workers,
            clock=clock,
        )
        _log_pathfinder_completion(
            phase_label=phase_label,
            started_at=pathfinder_t0,
            pathfinder_diagnostics=pathfinder_diagnostics,
            clock=clock,
        )
    else:
        logger.info("%s: scipy_pathfinder skipped", phase_label)

    init_t0 = clock()
    if initial_positions_override is not None:
        init_positions = _validate_initial_positions_override(
            initial_positions_override,
            num_chains=num_chains,
            dim=dim,
            dtype=dtype,
        )
        init_source = "user_provided"
    elif init_method == "laps":
        assert laps_result is not None  # The LAPS consumer above owns this package.
        init_positions = laps_result.positions
        init_source = "laps"
    elif init_method == "pathfinder":
        if pathfinder_state is None or pathfinder_diagnostics is None:
            raise RuntimeError("Pathfinder state missing for Pathfinder initialization.")
        (init_positions, sampling_mode, released_names, released_indices, released_scale) = (
            sample_scipy_pathfinder_init_positions(
                pathfinder_state,
                sample_key=sample_key,
                num_chains=num_chains,
                dtype=dtype,
                pathfinder_init_scale=pathfinder_init_scale,
                init_bundle=bundle,
                prior_released_sites=prior_released_sites,
                prior_release_scale=prior_release_scale,
                release_jitter_key=release_jitter_key,
                method_label=method_label,
            )
        )
        init_source = "pathfinder"
    else:
        init_source = "random"
    logger.info(
        "%s: parameter init source=%s ready in %.1fs",
        phase_label,
        init_source,
        (clock() - init_t0),
    )

    preconditioner_t0 = clock()
    if parameter_preconditioner_chol is None:
        if auto_preconditioner_method == "laps":
            assert laps_result is not None  # The LAPS consumer above owns this package.
            preconditioner_chol = laps_result.parameter_chol
            preconditioner_source = "laps"
            preconditioner_device = jax.default_backend()
        elif auto_preconditioner_method == "pathfinder":
            if pathfinder_state is None or pathfinder_diagnostics is None:
                raise RuntimeError("Pathfinder state missing for Pathfinder preconditioner.")
            preconditioner_chol = jax.device_put(
                scipy_pathfinder_preconditioner_chol(pathfinder_state)
            )
            preconditioner_device = jax.default_backend()
            preconditioner_source = "pathfinder"
        elif auto_preconditioner_method == "map":
            from nof1_causal_lab.models.ssm.inference.warmup.map import fit_map

            map_result = fit_map(
                priors,
                panel,
                num_samples=1,
                seed=seed,
                n_ieks_iters=n_ieks_iters,
                maxiter=auto_preconditioner_maxiter,
                parameter_covariance_method="optimizer_hess_inv",
                reparam=reparam,
                clock=clock,
            )
            preconditioner_chol = _laplace_preconditioner_chol_from_map_result(map_result)
            preconditioner_source = "map"
        else:
            preconditioner_source = "none"
    else:
        preconditioner_source = "caller_provided"
    logger.info(
        "%s: parameter preconditioner source=%s ready in %.1fs",
        phase_label,
        preconditioner_source,
        (clock() - preconditioner_t0),
    )

    warmup_diagnostics = ParameterWarmupDiagnostics(
        pathfinder=pathfinder_diagnostics,
        laps=None if laps_result is None else laps_result.diagnostics,
        pathfinder_run_count=1 if pathfinder_consumers else 0,
        pathfinder_consumers=tuple(pathfinder_consumers),
        init_source=init_source,
        preconditioner_source=preconditioner_source,
        preconditioner_device=preconditioner_device,
        dim=dim,
        duration_seconds=(clock() - total_t0),
        pathfinder_sampling_mode=sampling_mode,
        pathfinder_init_scale=pathfinder_init_scale if init_source == "pathfinder" else None,
        prior_released_site_names=released_names,
        prior_released_site_indices=released_indices,
        prior_release_scale=released_scale,
    )
    logger.info(
        "%s: parameter warmup complete in %.1fs (pathfinder_runs=%d, init=%s, preconditioner=%s)",
        phase_label,
        warmup_diagnostics.duration_seconds,
        warmup_diagnostics.pathfinder_run_count,
        init_source,
        preconditioner_source,
    )

    return ParameterWarmupResult(
        init_positions=init_positions,
        preconditioner_chol=preconditioner_chol,
        warmup_diagnostics=warmup_diagnostics,
        pathfinder_state=pathfinder_state,
        laps_initialization=laps_result if init_source == "laps" else None,
    )
