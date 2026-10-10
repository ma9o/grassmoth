"""Laplace-approximated likelihood backend for non-Gaussian SSMs.

Computes log p(y|theta) by combining an Iterated Extended Kalman Smoother
(IEKS) mode-finding inner loop with a Laplace approximation to the marginal
likelihood.  Three solver strategies are dispatched automatically:

- **Point IEKS**: block-tridiagonal O(T D^3) — used when every observation is
  a point measurement.
- **Support-aware IEKS**: tridiagonal precision plus low-rank corrections for
  observed interval summaries (e.g. means/sums over windows).
- **Dense support**: full joint Hessian — fallback for very short series with
  interval summaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp

from nof1_causal_lab.models.ssm.dynamics.linearisation import infer_linearisation
from nof1_causal_lab.models.ssm.execution.contracts import (
    EMPTY_LAPLACE_STATE,
    LaplaceEvaluationResult,
    LaplaceSolverState,
)
from nof1_causal_lab.models.ssm.execution.observation_model import compile_observation_model
from nof1_causal_lab.models.ssm.execution.observation_operator import (
    get_summary_operator_codes,
)
from nof1_causal_lab.models.ssm.inference.backend_factory import initialization_observation_laws
from nof1_causal_lab.models.ssm.inference.targets.transitions import build_discrete_transitions

from .point import (
    _dense_dynamic_support_laplace_log_lik,
    _dense_support_laplace_log_lik,
    _ieks_smooth,
    _point_dynamic_transition_ieks_laplace,
)
from .shared import (
    _block_banded_logdet,
    _build_ieks_system_from_prior,
    _build_prior_tridiagonal_system,
    _compute_profile_lower_bandwidths,
    _factor_block_profile_cholesky,
    _infer_support_groups,
    _predictive_latent_init,
    _should_use_dense_support_laplace,
    _solve_block_profile_from_cholesky,
    block_profile_logdet_packed_cotangent,
)
from .support import (
    _assemble_support_aware_observation_system,
    _make_support_window_derivatives,
    _support_aware_ieks_laplace,
    _support_dynamic_transition_ieks_laplace,
)

if TYPE_CHECKING:
    import numpy as np
    from dynestyx import StochasticContinuousTimeStateEvolution
    from numpyro.distributions import MultivariateNormal

    from nof1_causal_lab.models.ssm.dynamics.vector_field import StructuralDrift
    from nof1_causal_lab.models.ssm.execution.contracts import (
        MeasurementParams,
        ObservationLaws,
    )
    from nof1_causal_lab.models.ssm.execution.observation_model import CompiledObservationModel
    from nof1_causal_lab.models.ssm.observation_support import ObservationSupportRuntime

    from .shared import SupportObservationWindowBatch
    from .support import SupportWindowDerivatives


@dataclass(frozen=True, init=False, eq=False)
class LaplaceLikelihood:
    """Laplace-approximated likelihood backend.

    Computes log p(y|theta) via IEKS + Laplace approximation.
    Implements the default marginal likelihood path.

    Receives bound native laws per evaluation, including dynamic observation
    parameters in the custom gradient path.
    """

    # The support-aware Laplace path constructs runtime callables and custom-VJP
    # closures that are not remat-safe under large traced outer evaluations.
    checkpoint_loglik = False
    n_latent: int
    n_manifest: int
    n_ieks_iters: int
    observation_support: ObservationSupportRuntime | None
    _summary_operator_codes: jax.Array
    _support_window_batches: tuple[SupportObservationWindowBatch, ...]

    def __init__(
        self,
        n_latent: int,
        n_manifest: int,
        n_ieks_iters: int = 5,
        observation_support: ObservationSupportRuntime | None = None,
        *,
        observed_mask: np.ndarray,
    ) -> None:
        """Bind state dimensions and support-window structure for initialization-only Laplace solves."""
        object.__setattr__(self, "n_latent", n_latent)
        object.__setattr__(self, "n_manifest", n_manifest)
        object.__setattr__(self, "n_ieks_iters", n_ieks_iters)
        object.__setattr__(self, "observation_support", observation_support)
        if observation_support is not None:
            object.__setattr__(
                self, "_summary_operator_codes", get_summary_operator_codes(observation_support)
            )
        else:
            object.__setattr__(
                self, "_summary_operator_codes", jnp.zeros((n_manifest,), dtype=jnp.int32)
            )
        if (
            observation_support is not None
            and observation_support.requires_interval_summary_handling
        ):
            object.__setattr__(
                self,
                "_support_window_batches",
                _infer_support_groups(observation_support, observed_mask),
            )
        else:
            object.__setattr__(self, "_support_window_batches", ())

    def _build_support_window_derivatives(
        self, observation_model: CompiledObservationModel
    ) -> tuple[SupportWindowDerivatives, ...]:
        return tuple(
            _make_support_window_derivatives(
                max_state_len=batch.max_state_len,
                n_latent=self.n_latent,
                n_manifest=self.n_manifest,
                summary_operator_codes=self._summary_operator_codes,
                obs_kernel=observation_model.kernel,
                mean_log_prob_fn=observation_model.mean_log_prob_fn,
            )
            for batch in self._support_window_batches
        )

    def _compute_log_likelihood_impl(
        self,
        dynamics: StochasticContinuousTimeStateEvolution,
        measurement_params: MeasurementParams,
        initial_state: MultivariateNormal,
        observations: jnp.ndarray,
        time_intervals: jnp.ndarray,
        *,
        obs_mask: jnp.ndarray | None = None,
        observation_laws: ObservationLaws,
        latent_mode_init: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Shared Laplace likelihood implementation with caller-owned solver initialization."""
        if obs_mask is None:
            obs_mask = ~jnp.isnan(observations)
        initial_covariance: jax.Array = initial_state.covariance_matrix  # pyright: ignore[reportAssignmentType] - NumPyro lazy_property returns the covariance array on instances.
        clean_obs = jnp.nan_to_num(observations, nan=0.0)

        with jax.named_scope("map/compile_observation_model"):
            observation_model = compile_observation_model(
                initialization_observation_laws(observation_laws),
                manifest_cov=measurement_params.manifest_cov,
                observation_support=self.observation_support,
            )
        obs_kernel = observation_model.kernel

        def _discretize_base_system() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            with jax.named_scope("map/discretize_system"):
                transitions = build_discrete_transitions(
                    dynamics,
                    time_intervals,
                )
            return transitions.A, transitions.cov, jnp.asarray(transitions.bias)

        if (
            self.observation_support is not None
            and self.observation_support.requires_interval_summary_handling
        ):
            uses_dynamic_transitions = (
                infer_linearisation(cast("StructuralDrift", dynamics.drift).vector_field)
                == "trajectory"
            )
            if uses_dynamic_transitions:
                support_mode_init = latent_mode_init
                if _should_use_dense_support_laplace(
                    n_time=clean_obs.shape[0],
                    n_latent=self.n_latent,
                ):
                    with jax.named_scope("map/dense_dynamic_support_backend"):
                        log_lik, inner_eval_aux = _dense_dynamic_support_laplace_log_lik(
                            clean_obs,
                            obs_mask,
                            dynamics,
                            time_intervals,
                            measurement_params.lambda_mat,
                            measurement_params.manifest_means,
                            measurement_params.manifest_cov,
                            initial_state.mean,
                            jnp.asarray(initial_state.scale_tril)
                            @ jnp.asarray(initial_state.scale_tril).T,
                            obs_kernel,
                            observation_model.mean_log_prob_fn,
                            self.observation_support,
                            self.n_ieks_iters,
                            z_init=support_mode_init,
                        )
                    return log_lik, inner_eval_aux

                with jax.named_scope("map/support_dynamic_backend"):
                    window_derivatives = self._build_support_window_derivatives(observation_model)
                    log_lik, _z_mode, inner_eval_aux = _support_dynamic_transition_ieks_laplace(
                        clean_obs,
                        obs_mask,
                        dynamics,
                        time_intervals,
                        measurement_params.lambda_mat,
                        measurement_params.manifest_means,
                        measurement_params.manifest_cov,
                        initial_state.mean,
                        jnp.asarray(initial_state.scale_tril)
                        @ jnp.asarray(initial_state.scale_tril).T,
                        obs_kernel,
                        observation_model.mean_log_prob_fn,
                        self.observation_support,
                        self._support_window_batches,
                        window_derivatives,
                        self.n_ieks_iters,
                        z_init=support_mode_init,
                    )
                return log_lik, inner_eval_aux

            def _build_support_measurement_objects(
                manifest_cov: jnp.ndarray,
                runtime_observation_laws: ObservationLaws,
            ) -> tuple[CompiledObservationModel, tuple[SupportWindowDerivatives, ...]]:
                runtime_observation_model = compile_observation_model(
                    initialization_observation_laws(runtime_observation_laws),
                    manifest_cov=manifest_cov,
                    observation_support=self.observation_support,
                )
                return runtime_observation_model, self._build_support_window_derivatives(
                    runtime_observation_model
                )

            Ad, Qd, cd = _discretize_base_system()
            support_mode_init = latent_mode_init
            if _should_use_dense_support_laplace(
                n_time=clean_obs.shape[0],
                n_latent=self.n_latent,
            ):
                with jax.named_scope("map/dense_support_backend"):
                    log_lik, inner_eval_aux = _dense_support_laplace_log_lik(
                        clean_obs,
                        obs_mask,
                        Ad,
                        Qd,
                        cd,
                        measurement_params.lambda_mat,
                        measurement_params.manifest_means,
                        measurement_params.manifest_cov,
                        initial_state.mean,
                        jnp.asarray(initial_state.scale_tril)
                        @ jnp.asarray(initial_state.scale_tril).T,
                        obs_kernel,
                        observation_model.mean_log_prob_fn,
                        self.observation_support,
                        self.n_ieks_iters,
                    )
                return log_lik, inner_eval_aux
            with jax.named_scope("map/support_aware_backend"):
                window_derivatives = self._build_support_window_derivatives(observation_model)
                log_lik, _z_mode, inner_eval_aux = _support_aware_ieks_laplace(
                    clean_obs,
                    obs_mask,
                    Ad,
                    Qd,
                    cd,
                    measurement_params.lambda_mat,
                    measurement_params.manifest_means,
                    measurement_params.manifest_cov,
                    initial_state.mean,
                    initial_covariance,
                    obs_kernel,
                    observation_model.mean_log_prob_fn,
                    self.observation_support,
                    self._support_window_batches,
                    window_derivatives=window_derivatives,
                    build_measurement_objects=_build_support_measurement_objects,
                    observation_laws=observation_laws,
                    n_ieks_iters=self.n_ieks_iters,
                    z_init=support_mode_init,
                )
                return log_lik, inner_eval_aux

        point_mode_init = latent_mode_init

        uses_dynamic_transitions = (
            infer_linearisation(cast("StructuralDrift", dynamics.drift).vector_field)
            == "trajectory"
        )
        T_obs = clean_obs.shape[0]
        H_rows = jnp.broadcast_to(
            measurement_params.lambda_mat[None, :, :],
            (T_obs, *measurement_params.lambda_mat.shape),
        )
        d_rows = jnp.broadcast_to(
            measurement_params.manifest_means[None, :],
            (T_obs, *measurement_params.manifest_means.shape),
        )

        def _build_point_measurement_objects(
            manifest_cov: jnp.ndarray,
            runtime_observation_laws: ObservationLaws,
        ) -> CompiledObservationModel:
            return compile_observation_model(
                initialization_observation_laws(runtime_observation_laws),
                manifest_cov=manifest_cov,
                observation_support=self.observation_support,
            )

        with jax.named_scope("map/ieks_backend"):
            if uses_dynamic_transitions:
                _z_mode, log_lik, inner_eval_aux = _point_dynamic_transition_ieks_laplace(
                    clean_obs,
                    obs_mask,
                    dynamics,
                    time_intervals,
                    H_rows,
                    d_rows,
                    measurement_params.manifest_cov,
                    initial_state.mean,
                    initial_covariance,
                    obs_kernel,
                    n_ieks_iters=self.n_ieks_iters,
                    z_init=point_mode_init,
                )
            else:
                Ad, Qd, cd = _discretize_base_system()
                _z_mode, log_lik, inner_eval_aux = _ieks_smooth(
                    clean_obs,
                    obs_mask,
                    Ad,
                    Qd,
                    cd,
                    H_rows,
                    d_rows,
                    measurement_params.manifest_cov,
                    initial_state.mean,
                    initial_covariance,
                    obs_kernel,
                    n_ieks_iters=self.n_ieks_iters,
                    z_init=point_mode_init,
                    build_measurement_objects=_build_point_measurement_objects,
                    observation_laws=observation_laws,
                )

        return log_lik, inner_eval_aux

    def compute_log_likelihood(
        self,
        dynamics: StochasticContinuousTimeStateEvolution,
        measurement_params: MeasurementParams,
        initial_state: MultivariateNormal,
        observations: jnp.ndarray,
        time_intervals: jnp.ndarray,
        *,
        obs_mask: jnp.ndarray | None = None,
        observation_laws: ObservationLaws,
        solver_state: LaplaceSolverState = EMPTY_LAPLACE_STATE,
    ) -> jnp.ndarray:
        """Compute the Laplace log-likelihood used for sampler initialization.

        Args:
            dynamics: Continuous-time state evolution to linearize around the mode path.
            measurement_params: Loadings, observation intercepts, and measurement covariance.
            initial_state: Gaussian law at the initial latent time.
            observations: Recorded values with shape ``(time, indicator)``.
            time_intervals: Model-time increments whose cumulative sum gives the latent grid.
            obs_mask: Optional observation mask aligned with ``observations``.
            observation_laws: Bound measurement laws in observation-column order.
            solver_state: Optional retained latent mode used to initialize the solve.

        Returns:
            Scalar approximate log marginal likelihood for the complete observation
            sequence. This approximation is restricted to particle-sampler initialization.
        """
        log_lik, _aux = self._compute_log_likelihood_impl(
            dynamics,
            measurement_params,
            initial_state,
            observations,
            time_intervals,
            obs_mask=obs_mask,
            observation_laws=observation_laws,
            latent_mode_init=solver_state.latent_mode,
        )
        return log_lik

    def compute_log_likelihood_with_aux(
        self,
        dynamics: StochasticContinuousTimeStateEvolution,
        measurement_params: MeasurementParams,
        initial_state: MultivariateNormal,
        observations: jnp.ndarray,
        time_intervals: jnp.ndarray,
        *,
        obs_mask: jnp.ndarray | None = None,
        observation_laws: ObservationLaws,
        solver_state: LaplaceSolverState = EMPTY_LAPLACE_STATE,
    ) -> LaplaceEvaluationResult:
        """Compute Laplace-approximated log-likelihood plus host-log aux."""
        log_lik, inner_eval_aux = self._compute_log_likelihood_impl(
            dynamics,
            measurement_params,
            initial_state,
            observations,
            time_intervals,
            obs_mask=obs_mask,
            observation_laws=observation_laws,
            latent_mode_init=solver_state.latent_mode,
        )
        return LaplaceEvaluationResult(
            log_lik,
            LaplaceSolverState(inner_eval_aux["latent_mode"]),
            {key: value for key, value in inner_eval_aux.items() if key != "latent_mode"},
        )


__all__ = [
    "LaplaceLikelihood",
    # point.py re-exports
    "_dense_support_laplace_log_lik",
    "_ieks_smooth",
    # shared.py re-exports
    "_block_banded_logdet",
    "_build_ieks_system_from_prior",
    "_build_prior_tridiagonal_system",
    "_compute_profile_lower_bandwidths",
    "_factor_block_profile_cholesky",
    "_infer_support_groups",
    "_predictive_latent_init",
    "_should_use_dense_support_laplace",
    "_solve_block_profile_from_cholesky",
    "block_profile_logdet_packed_cotangent",
    # support.py re-exports
    "_assemble_support_aware_observation_system",
    "_make_support_window_derivatives",
    "_support_aware_ieks_laplace",
]
