"""Trajectory-level observation likelihoods for support-aware measurement structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, NamedTuple, overload

import jax
import jax.numpy as jnp

from grassmoth.models.ssm.compile.observation_windows import compile_observation_scan
from grassmoth.models.ssm.execution.contracts import NUMERICAL_EPSILON, MeasurementParams

if TYPE_CHECKING:
    from grassmoth.models.ssm.compile.observation_windows import (
        CompiledObservationScan,
        CompiledWindowPlan,
    )
    from grassmoth.models.ssm.execution.observation_model import CompiledObservationModel
    from grassmoth.models.ssm.preparation.observations import ObservationSupportRuntime

type ObservationGradient = Literal["full", "current"]

_POINT_SUPPORT_CODE = 0
_INTERVAL_SUPPORT_CODE = 1
_SUPPORT_KIND_TO_CODE = {
    None: _POINT_SUPPORT_CODE,
    "point": _POINT_SUPPORT_CODE,
    "interval": _INTERVAL_SUPPORT_CODE,
}

_FIRST_OPERATOR_CODE = 0
_LAST_OPERATOR_CODE = 1
_SUM_OPERATOR_CODE = 2
_COUNT_OPERATOR_CODE = 3
_MEAN_OPERATOR_CODE = 4
_STD_OPERATOR_CODE = 5
_SUMMARY_OPERATOR_TO_CODE = {
    None: _LAST_OPERATOR_CODE,
    "first": _FIRST_OPERATOR_CODE,
    "last": _LAST_OPERATOR_CODE,
    "sum": _SUM_OPERATOR_CODE,
    "count": _COUNT_OPERATOR_CODE,
    "mean": _MEAN_OPERATOR_CODE,
    "std": _STD_OPERATOR_CODE,
}


@dataclass(frozen=True)
class ObservationOperator:
    """Compiled observation-window semantics used by likelihoods and predictive paths."""

    observation_support: ObservationSupportRuntime
    support_kind_codes: jnp.ndarray
    summary_operator_codes: jnp.ndarray
    interval_summary_indices: tuple[int, ...]
    window_plan: CompiledWindowPlan
    scan: CompiledObservationScan

    @property
    def requires_interval_summary_handling(self) -> bool:
        """Whether this support plan includes observations summarized over intervals."""
        return bool(self.interval_summary_indices)

    def point_like_mask(self, dtype: jnp.dtype) -> jnp.ndarray:
        """Return a mask selecting point-like observation channels in the requested numeric dtype."""
        return get_point_like_mask(self.support_kind_codes, dtype)

    def interval_summary_mask(self, dtype: jnp.dtype) -> jnp.ndarray:
        """Return a mask selecting interval-summary observation channels in the requested dtype."""
        return get_interval_summary_mask(self.support_kind_codes, dtype)

    def project_response_trajectory(
        self,
        response_trajectory: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Project response trajectories onto this operator's recorded measurement supports."""
        return _project_response_trajectory_with_operator(response_trajectory, self)


class SupportObservationSummary(NamedTuple):
    """Expected responses and emission masks for observations whose support is ready to evaluate."""

    expected_mean: jnp.ndarray
    semantic_mask: jnp.ndarray


@overload
def compile_observation_operator(
    observation_support: ObservationSupportRuntime,
    *,
    held_channels: tuple[int, ...] = (),
) -> ObservationOperator: ...


@overload
def compile_observation_operator(
    observation_support: None = None, *, held_channels: tuple[int, ...] = ()
) -> None: ...


def compile_observation_operator(
    observation_support: ObservationSupportRuntime | None = None,
    *,
    held_channels: tuple[int, ...] = (),
) -> ObservationOperator | None:
    """Compile reusable observation-window semantics from runtime metadata."""
    if observation_support is None:
        return None

    interval_summary_indices = tuple(
        idx for idx, kind in enumerate(observation_support.support_kinds) if kind == "interval"
    )
    window_plan = observation_support.window_plan.with_held_channels(held_channels)
    return ObservationOperator(
        observation_support=observation_support,
        support_kind_codes=get_support_kind_codes(observation_support),
        summary_operator_codes=get_summary_operator_codes(observation_support),
        interval_summary_indices=interval_summary_indices,
        window_plan=window_plan,
        scan=compile_observation_scan(window_plan),
    )


def get_support_kind_codes(observation_support: ObservationSupportRuntime) -> jnp.ndarray:
    """Map support kinds to integer codes aligned with manifest order."""
    return jnp.asarray(
        [_SUPPORT_KIND_TO_CODE[kind] for kind in observation_support.support_kinds],
        dtype=jnp.int32,
    )


def get_summary_operator_codes(observation_support: ObservationSupportRuntime) -> jnp.ndarray:
    """Map summary operators to integer codes aligned with manifest order."""
    return jnp.asarray(
        [_SUMMARY_OPERATOR_TO_CODE[operator] for operator in observation_support.summary_operators],
        dtype=jnp.int32,
    )


def get_point_like_mask(support_kind_codes: jnp.ndarray, dtype: jnp.dtype) -> jnp.ndarray:
    """Mask manifests that still use point-observation semantics."""
    return (support_kind_codes == _POINT_SUPPORT_CODE).astype(dtype)


def get_interval_summary_mask(support_kind_codes: jnp.ndarray, dtype: jnp.dtype) -> jnp.ndarray:
    """Mask manifests that require interval-summary handling."""
    return 1.0 - get_point_like_mask(support_kind_codes, dtype)


def select_emitted_support_statistics(
    obs_sum: jnp.ndarray,
    obs_sumsq: jnp.ndarray,
    obs_weight: jnp.ndarray,
    emission_slot_indices: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Gather the active support slot referenced by each emitted observation."""
    safe_indices = jnp.clip(emission_slot_indices, 0, obs_sum.shape[-1] - 1)
    gather_idx = jnp.expand_dims(safe_indices, axis=-1)
    selected_sum = jnp.take_along_axis(obs_sum, gather_idx, axis=-1).squeeze(-1)
    selected_sumsq = jnp.take_along_axis(obs_sumsq, gather_idx, axis=-1).squeeze(-1)
    selected_weight = jnp.take_along_axis(obs_weight, gather_idx, axis=-1).squeeze(-1)
    valid = emission_slot_indices >= 0
    zeros = jnp.zeros_like(selected_sum)
    return (
        jnp.where(valid, selected_sum, zeros),
        jnp.where(valid, selected_sumsq, zeros),
        jnp.where(valid, selected_weight, zeros),
    )


def expected_observation_mean(
    response_t: jnp.ndarray,
    obs_sum: jnp.ndarray,
    obs_sumsq: jnp.ndarray,
    obs_weight: jnp.ndarray,
    summary_operator_codes: jnp.ndarray,
) -> jnp.ndarray:
    """Map accumulated response statistics to the expected observation mean."""
    safe_weight = jnp.maximum(obs_weight, NUMERICAL_EPSILON)
    window_mean = obs_sum / safe_weight
    window_var = jnp.maximum(obs_sumsq / safe_weight - window_mean**2, NUMERICAL_EPSILON)
    std_mean = jnp.sqrt(window_var)

    expected_mean = response_t
    expected_mean = jnp.where(
        jnp.logical_or(
            summary_operator_codes == _SUM_OPERATOR_CODE,
            summary_operator_codes == _COUNT_OPERATOR_CODE,
        ),
        obs_sum,
        expected_mean,
    )
    expected_mean = jnp.where(
        summary_operator_codes == _MEAN_OPERATOR_CODE, window_mean, expected_mean
    )
    return jnp.where(summary_operator_codes == _STD_OPERATOR_CODE, std_mean, expected_mean)


def summarize_support_observation(
    observation_operator: ObservationOperator,
    response_t: jnp.ndarray,
    statistics: jnp.ndarray,
    obs_mask_t: jnp.ndarray,
    emission_slot_indices: jnp.ndarray,
) -> SupportObservationSummary:
    """Summarize one support-aware observation emission at the current step."""
    obs_sum, obs_sumsq, obs_weight = statistics[..., 0], statistics[..., 1], statistics[..., 2]
    if not observation_operator.requires_interval_summary_handling:
        raise ValueError(
            "summarize_support_observation requires interval-summary observation support"
        )
    dtype = response_t.dtype
    point_like_mask = observation_operator.point_like_mask(dtype)
    interval_summary_mask = observation_operator.interval_summary_mask(dtype)
    selected_sum, selected_sumsq, selected_weight = select_emitted_support_statistics(
        obs_sum,
        obs_sumsq,
        obs_weight,
        emission_slot_indices,
    )
    expected_mean = expected_observation_mean(
        response_t,
        selected_sum,
        selected_sumsq,
        selected_weight,
        observation_operator.summary_operator_codes,
    )
    obs_mask_float = jnp.asarray(obs_mask_t, dtype=dtype)
    emitted_interval_summary_mask = (
        obs_mask_float * interval_summary_mask * (emission_slot_indices >= 0).astype(dtype)
    )
    semantic_mask = point_like_mask + emitted_interval_summary_mask
    return SupportObservationSummary(
        expected_mean=expected_mean,
        semantic_mask=semantic_mask,
    )


def _project_response_trajectory_with_operator(
    response_trajectory: jnp.ndarray,
    observation_operator: ObservationOperator,
    *,
    gradient: ObservationGradient = "full",
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Project response trajectories onto the operator's emitted measurement supports.

    Args:
        response_trajectory: Expected responses with shape ``(time, indicator)``.
        observation_operator: Resolved support schedule, integration weights, and
            summary operators for those time and indicator axes.
        gradient: Differentiate all responses or only each emission's current response.

    Returns:
        Expected observation means and a numeric emission mask, both with shape
        ``(time, indicator)``. Point-like channels emit on every row; interval
        summaries emit only at their designated anchor rows.
    """
    dtype = response_trajectory.dtype
    T, n_manifest = response_trajectory.shape

    if not observation_operator.requires_interval_summary_handling:
        return response_trajectory, jnp.ones((T, n_manifest), dtype=dtype)

    point_like_mask = observation_operator.point_like_mask(dtype)
    interval_summary_mask = observation_operator.interval_summary_mask(dtype)

    emission_slots = jnp.asarray(observation_operator.scan.emission_slots, dtype=jnp.int32)
    semantic_mask_0 = point_like_mask + interval_summary_mask * (emission_slots[0] >= 0).astype(
        dtype
    )
    expected_0 = response_trajectory[0]

    if T == 1:
        return expected_0[None, :], semantic_mask_0[None, :]

    prev_coeffs = jnp.asarray(observation_operator.scan.prev_coeffs, dtype=dtype)
    curr_coeffs = jnp.asarray(observation_operator.scan.curr_coeffs, dtype=dtype)
    interval_weights = jnp.asarray(observation_operator.scan.weights, dtype=dtype)
    full_obs_mask = jnp.ones((n_manifest,), dtype=dtype)
    previous, current = response_trajectory[:-1, :, None], response_trajectory[1:, :, None]
    increments = jnp.stack(
        (
            prev_coeffs[1:] * previous + curr_coeffs[1:] * current,
            prev_coeffs[1:] * previous**2 + curr_coeffs[1:] * current**2,
            interval_weights[1:],
        ),
        axis=-1,
    )
    slots = emission_slots[1:]
    slot_ids = jnp.arange(increments.shape[-2])
    reset = (
        (slots >= 0)[..., None]
        & interval_summary_mask[None, :, None].astype(bool)
        & (slot_ids == slots[..., None])
    )
    keep = (~reset).astype(dtype)[..., None]

    def compose(left, right):
        left_keep, left_value = left
        right_keep, right_value = right
        return right_keep * left_keep, right_keep * left_value + right_value

    # A reset is the affine map s -> 0, otherwise s -> s + increment.
    # Composition is associative, including reused slots and overlapping windows.
    _, after_reset = jax.lax.associative_scan(compose, (keep, keep * increments))
    before_reset = jnp.concatenate((jnp.zeros_like(after_reset[:1]), after_reset[:-1])) + increments
    if gradient == "current":
        # Filtering differentiates an observation at t only with respect to x_t.
        # Preserve the exact accumulated values, but hold every past contribution
        # fixed. This extracts the causal Jacobian diagonal in linear work instead
        # of materializing a time-by-time Jacobian.
        current_moments = jnp.stack(
            (
                curr_coeffs[1:] * current,
                curr_coeffs[1:] * current**2,
                jnp.zeros_like(current * curr_coeffs[1:]),
            ),
            axis=-1,
        )
        before_reset = jax.lax.stop_gradient(before_reset - current_moments) + current_moments

    def summarize(response, statistics, emission):
        summary = summarize_support_observation(
            observation_operator,
            response,
            statistics,
            full_obs_mask,
            emission,
        )
        return summary.expected_mean, summary.semantic_mask

    expected_rest, semantic_mask_rest = jax.vmap(summarize)(
        response_trajectory[1:], before_reset, slots
    )
    return (
        jnp.concatenate([expected_0[None, :], expected_rest], axis=0),
        jnp.concatenate([semantic_mask_0[None, :], semantic_mask_rest], axis=0),
    )


def project_response_trajectory(
    response_trajectory: jnp.ndarray,
    observation_support: ObservationSupportRuntime | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Project response-space trajectories into emitted observation means."""
    if observation_support is None:
        return response_trajectory, jnp.ones_like(response_trajectory)
    return _project_response_trajectory_with_operator(
        response_trajectory,
        compile_observation_operator(observation_support),
    )


def trajectory_observation_log_probs(
    latent_trajectory: jnp.ndarray,
    observations: jnp.ndarray,
    measurement: MeasurementParams,
    compiled: CompiledObservationModel,
    *,
    gradient: ObservationGradient = "full",
) -> jnp.ndarray:
    """Return per-timestep log-likelihood contributions for a latent trajectory."""
    H, d_meas, R = measurement.lambda_mat, measurement.manifest_means, measurement.manifest_cov
    obs_kernel, mean_log_prob_fn = compiled.kernel, compiled.mean_log_prob_fn
    obs_mask = ~jnp.isnan(observations)

    clean_obs = jnp.nan_to_num(observations, nan=0.0)
    mask_float = obs_mask.astype(jnp.float32)
    observation_operator = compiled.observation_operator

    if observation_operator is None or not observation_operator.requires_interval_summary_handling:
        return jax.vmap(
            lambda y_t, z_t, mask_t: obs_kernel.log_prob_fn(
                y_t,
                H @ z_t + d_meas,
                R,
                mask_t,
            )
        )(clean_obs, latent_trajectory, mask_float)

    if mean_log_prob_fn is None:
        raise ValueError("mean_log_prob_fn is required for interval-summary observation semantics")

    point_like_mask = observation_operator.point_like_mask(mask_float.dtype)
    interval_summary_mask = observation_operator.interval_summary_mask(mask_float.dtype)
    responses = jax.vmap(lambda z_t: obs_kernel.response_fn(H @ z_t + d_meas))(latent_trajectory)
    expected_means, semantic_mask = _project_response_trajectory_with_operator(
        responses, observation_operator, gradient=gradient
    )

    point_ll = jax.vmap(
        lambda y_t, z_t, mask_t: obs_kernel.log_prob_fn(
            y_t,
            H @ z_t + d_meas,
            R,
            mask_t * point_like_mask,
        )
    )(clean_obs, latent_trajectory, mask_float)
    interval_summary_ll = jax.vmap(
        lambda y_t, mean_t, mask_t, semantic_t: mean_log_prob_fn(
            y_t,
            mean_t,
            R,
            mask_t * interval_summary_mask * semantic_t,
        )
    )(clean_obs, expected_means, mask_float, semantic_mask)
    return point_ll + interval_summary_ll
