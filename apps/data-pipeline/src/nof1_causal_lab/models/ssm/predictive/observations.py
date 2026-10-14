"""Shared predictive observation simulation for prior and posterior workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np

from nof1_causal_lab.artifacts.model.likelihood import LinkFunction
from nof1_causal_lab.models.ssm.execution.dynamical_model import HeterogeneousObservation
from nof1_causal_lab.models.ssm.execution.observation_dispatch import build_interval_summary_sampler
from nof1_causal_lab.models.ssm.execution.observation_operator import compile_observation_operator

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import dynestyx as dsx

    from nof1_causal_lab.models.ssm.execution.contracts import ObservationLaws
    from nof1_causal_lab.models.ssm.execution.observation_operator import ObservationOperator
    from nof1_causal_lab.models.ssm.preparation.observations import ObservationSupportRuntime


class PredictiveObservationMeanOverflow(RuntimeError):
    """Raised when a log-link predictive mean exceeds finite numeric range."""

    def __init__(
        self,
        *,
        bad_manifest_names: tuple[str, ...],
        manifest_indices: tuple[int, ...],
        failing_draw_indices: tuple[int, ...],
        n_draws: int,
        first_bad_time_index: int,
        max_linear_predictor: float,
        overflow_threshold: float,
        n_nonfinite: int = 0,
    ) -> None:
        """Retain affected indicators, draws, and first failing time in a predictive overflow error."""
        self.bad_manifest_names = bad_manifest_names
        self.manifest_indices = manifest_indices
        self.failing_draw_indices = failing_draw_indices
        self.n_draws = n_draws
        self.first_bad_time_index = first_bad_time_index
        self.max_linear_predictor = max_linear_predictor
        self.overflow_threshold = overflow_threshold
        self.n_nonfinite = n_nonfinite
        manifest_summary = ", ".join(bad_manifest_names) if bad_manifest_names else "unknown"
        if n_nonfinite:
            cause = (
                f"linear predictor contains {n_nonfinite} non-finite (NaN/Inf) values for "
                f"{manifest_summary} — the latent simulation diverged under these model laws "
                "(rein in feedback edge gains, diffusion, or persistence), "
                f"max finite eta={max_linear_predictor:.2f}"
            )
        else:
            cause = (
                f"linear predictor exceeded the finite exp range for {manifest_summary} "
                f"(max eta={max_linear_predictor:.2f}, threshold={overflow_threshold:.2f}"
            )
        super().__init__(
            "Predictive log-link mean overflow before observation sampling: "
            f"{cause}, first bad time index={first_bad_time_index}; "
            f"{len(failing_draw_indices)} of {n_draws} predictive draws affected — "
            "a small fraction means heavy tails (tighten sigma/upper bounds); "
            "most draws means the central mass is wrong (lower the loading/edge-gain "
            "or intercept location feeding this log link)."
        )


def _resolve_effective_observation_mask(
    target_shape: tuple[int, ...],
    semantic_mask: jnp.ndarray | None,
    observation_mask: jnp.ndarray | None,
) -> jnp.ndarray:
    """Return the explicit emitted-observation mask for one simulated draw."""
    if len(target_shape) == 2:
        mask_shape = target_shape
    else:
        mask_shape = target_shape[1:]
    effective_mask = jnp.ones(mask_shape, dtype=bool)
    if semantic_mask is not None:
        effective_mask = effective_mask & (semantic_mask > 0.5)
    if observation_mask is not None:
        effective_mask = effective_mask & observation_mask.astype(bool)
    return effective_mask


def _apply_observation_mask(
    y_sim: jnp.ndarray,
    semantic_mask: jnp.ndarray | None,
    observation_mask: jnp.ndarray | None,
) -> jnp.ndarray:
    """Set structurally absent observations to NaN."""
    effective_mask = _resolve_effective_observation_mask(
        y_sim.shape,
        semantic_mask,
        observation_mask,
    )
    if y_sim.ndim == 2:
        return jnp.where(effective_mask, y_sim, jnp.nan)
    return jnp.where(effective_mask[None, :, :], y_sim, jnp.nan)


def _raise_if_log_link_mean_overflow(
    linear_predictors: jnp.ndarray,
    *,
    laws: ObservationLaws,
    manifest_names: Sequence[str] | None,
) -> None:
    """Fail fast when a log-link predictive mean would overflow before sampling."""
    log_link_mask = np.asarray(
        [next(iter(law.operands()))[1].link == LinkFunction.LOG for law in laws], dtype=bool
    )
    if not bool(log_link_mask.any()):
        return

    linear_np = np.asarray(linear_predictors)
    overflow_threshold = float(np.log(np.finfo(linear_np.dtype).max))
    bad_mask = np.zeros_like(linear_np, dtype=bool)
    bad_mask[..., log_link_mask] = (~np.isfinite(linear_np[..., log_link_mask])) | (
        linear_np[..., log_link_mask] > overflow_threshold
    )
    if not bool(bad_mask.any()):
        return

    manifest_mask = bad_mask.any(axis=(0, 1))
    manifest_indices = tuple(int(idx) for idx in np.flatnonzero(manifest_mask))
    names = manifest_names or [f"var_{idx}" for idx in range(linear_np.shape[2])]
    bad_manifest_names = tuple(names[idx] for idx in manifest_indices if idx < len(names))
    draw_mask = bad_mask.reshape(bad_mask.shape[0], -1).any(axis=1)
    failing_draw_indices = tuple(int(idx) for idx in np.flatnonzero(draw_mask))
    time_mask = bad_mask.any(axis=(0, 2))
    first_bad_time_index = int(np.flatnonzero(time_mask)[0])
    max_linear_predictor = float(np.nanmax(linear_np[..., log_link_mask]))
    raise PredictiveObservationMeanOverflow(
        bad_manifest_names=bad_manifest_names,
        manifest_indices=manifest_indices,
        failing_draw_indices=failing_draw_indices,
        n_draws=int(linear_np.shape[0]),
        first_bad_time_index=first_bad_time_index,
        max_linear_predictor=max_linear_predictor,
        overflow_threshold=overflow_threshold,
        n_nonfinite=int((~np.isfinite(linear_np[..., log_link_mask])).sum()),
    )


def _sample_observations_for_draw(
    linear_predictors: jnp.ndarray,
    rng_key: jax.Array,
    *,
    observation_model: HeterogeneousObservation,
    observation_operator: ObservationOperator | None,
    observation_mask: jnp.ndarray | None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sample one observation trajectory from precomputed linear predictors."""
    _key_latent, key_point, key_interval_summary = random.split(rng_key, 3)

    def emit(key: jax.Array, predictor: jax.Array) -> tuple[jax.Array, jax.Array]:
        law = observation_model.at_predictor(predictor)
        return law.sample(key), law.response

    point_samples, responses = jax.vmap(emit)(
        random.split(key_point, linear_predictors.shape[0]), linear_predictors
    )

    if observation_operator is None or not observation_operator.requires_interval_summary_handling:
        effective_mask = _resolve_effective_observation_mask(
            point_samples.shape,
            None,
            observation_mask,
        )
        return (
            _apply_observation_mask(point_samples, None, observation_mask),
            effective_mask,
            _apply_observation_mask(responses, None, observation_mask),
        )

    expected_means, semantic_mask = observation_operator.project_response_trajectory(responses)

    interval_summary_indices = list(observation_operator.interval_summary_indices)
    interval_summary_idx = jnp.asarray(interval_summary_indices, dtype=jnp.int32)
    interval_sampler = build_interval_summary_sampler(
        observation_model.laws,
        observation_model.measurement.manifest_cov,
        interval_summary_indices,
    )
    sampled_interval_summary = interval_sampler.sample_mean_trajectory(
        key_interval_summary,
        expected_means[:, interval_summary_idx],
    )
    point_samples = jax.vmap(lambda y_t, sampled_t: y_t.at[interval_summary_idx].set(sampled_t))(
        point_samples,
        sampled_interval_summary,
    )

    effective_mask = _resolve_effective_observation_mask(
        point_samples.shape,
        semantic_mask,
        observation_mask,
    )
    return (
        _apply_observation_mask(point_samples, semantic_mask, observation_mask),
        effective_mask,
        _apply_observation_mask(expected_means, semantic_mask, observation_mask),
    )


def _predictive_observation_grid(
    times: jax.Array,
    n_manifest: int,
    observation_support: ObservationSupportRuntime | None,
    observation_mask: jax.Array | None,
    held_channels: tuple[int, ...],
) -> tuple[jax.Array | None, ObservationOperator | None]:
    observation_operator = compile_observation_operator(
        observation_support, held_channels=held_channels
    )
    mask = None if observation_mask is None else jnp.asarray(observation_mask, dtype=bool)
    if mask is not None and mask.shape != (times.shape[0], n_manifest):
        raise ValueError(
            "observation_mask must have shape (T, n_manifest) matching the predictive grid"
        )
    if observation_operator is not None and observation_operator.requires_interval_summary_handling:
        support = observation_operator.observation_support
        if support.anchor_times.shape != times.shape or not bool(
            jnp.allclose(support.anchor_times, times)
        ):
            raise ValueError("observation_support is not aligned to the predictive time grid")
    return mask, observation_operator


def sample_model_observations(
    dynamical_models: dsx.DynamicalModel,
    linear_predictors: jax.Array,
    times: jax.Array,
    *,
    rng_key: jax.Array,
    observation_support: ObservationSupportRuntime | None,
    observation_mask: jax.Array | None,
    manifest_names: Sequence[str] | None,
    held_channels: tuple[int, ...] = (),
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Draw point observations from the fitted model's law, then project interval summaries."""
    observation = dynamical_models.observation_model
    assert isinstance(
        observation, HeterogeneousObservation
    )  # The compiled nof1 model owns this native adapter.
    mask, operator = _predictive_observation_grid(
        times, linear_predictors.shape[-1], observation_support, observation_mask, held_channels
    )
    _raise_if_log_link_mean_overflow(
        linear_predictors,
        laws=observation.laws,
        manifest_names=manifest_names,
    )
    keys = random.split(rng_key, linear_predictors.shape[0])

    def emit(
        observation: HeterogeneousObservation, predictors: jax.Array, key: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        return _sample_observations_for_draw(
            predictors,
            key,
            observation_model=observation,
            observation_operator=operator,
            observation_mask=mask,
        )

    emit_draws: Callable[
        [HeterogeneousObservation, jax.Array, jax.Array], tuple[jax.Array, jax.Array, jax.Array]
    ] = eqx.filter_vmap(emit)
    return emit_draws(observation, linear_predictors, keys)
