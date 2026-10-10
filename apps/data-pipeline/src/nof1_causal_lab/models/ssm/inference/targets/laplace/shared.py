"""Shared linear algebra and preprocessing helpers for MAP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
import numpy as np

from nof1_causal_lab.models.ssm.covariance_utils import (
    logdet_from_cholesky,
    symmetrize_with_jitter,
)
from nof1_causal_lab.models.ssm.inference.targets.laplace.tridiagonal import (
    factor_tridiagonal,
    inverse_subset_tridiagonal,
    solve_tridiagonal,
)
from nof1_causal_lab.models.ssm.inference.targets.transitions import build_discrete_transitions

if TYPE_CHECKING:
    from collections.abc import Callable

    from dynestyx import StochasticContinuousTimeStateEvolution

    from nof1_causal_lab.models.ssm.observation_support import ObservationSupportRuntime

_DENSE_SUPPORT_LAPLACE_MAX_FLAT_DIM = 160


@dataclass(frozen=True)
class SupportObservationWindowBatch:
    """Compiled anchored interval-summary windows padded to a common state length."""

    max_state_len: int
    state_lens: jnp.ndarray
    anchor_indices: jnp.ndarray
    start_indices: jnp.ndarray
    mask_full: jnp.ndarray
    prev_coeffs: jnp.ndarray
    curr_coeffs: jnp.ndarray
    weights: jnp.ndarray
    padded_state_indices: jnp.ndarray
    time_indices: jnp.ndarray
    valid_diag: jnp.ndarray
    component_windows: jnp.ndarray
    component_manifests: jnp.ndarray


@dataclass(frozen=True)
class GaussianTrajectoryPriorTerms:
    """Precomputed Gaussian factors for the latent trajectory prior."""

    init_mean: jnp.ndarray
    init_chol: jnp.ndarray
    init_logdet: jnp.ndarray
    transition_chol: jnp.ndarray
    transition_logdet: jnp.ndarray


def _transition_start_linearization_states(
    latent_trajectory: jnp.ndarray,
    init_mean: jnp.ndarray,
) -> jnp.ndarray:
    """Align the initial mean and preceding path states with transition intervals."""
    return jnp.concatenate((init_mean[None, :], latent_trajectory[:-1]), axis=0)


def _prepare_linearized_path(
    dynamics: StochasticContinuousTimeStateEvolution,
    time_intervals: jnp.ndarray,
    init_mean: jnp.ndarray,
    *,
    z_init: jnp.ndarray | None,
    dtype: jnp.dtype,
) -> tuple[
    Callable[[jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]],
    jnp.ndarray,
]:
    """Bind Dynestyx's Gaussian view and seed the warmup reference trajectory."""

    def transitions_at(path: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        transitions = build_discrete_transitions(
            dynamics,
            time_intervals,
            linearization_states=_transition_start_linearization_states(path, init_mean),
        )
        return transitions.A, transitions.cov, jnp.asarray(transitions.bias, dtype=dtype)

    if z_init is None:
        reference = jnp.broadcast_to(init_mean[None], (time_intervals.shape[0], init_mean.shape[0]))
        Ad, _Qd, cd = transitions_at(reference)
        path = _predictive_latent_init(Ad, cd, init_mean)
    else:
        path = jnp.asarray(z_init, dtype=dtype)
    return transitions_at, jnp.asarray(path, dtype=dtype)


def _should_use_dense_support_laplace(*, n_time: int, n_latent: int) -> bool:
    """Use the dense exact support-aware Newton system on short trajectories.

    The banded support-aware path pays substantial Python/autodiff overhead per
    anchored window. For small latent trajectories, a single dense exact Hessian
    over the full latent path is materially faster and preserves the same model.
    """
    return n_time * n_latent <= _DENSE_SUPPORT_LAPLACE_MAX_FLAT_DIM


def _symmetrize_psd(mats: jnp.ndarray, jitter: float = 0.0) -> jnp.ndarray:
    """Symmetrize square matrices and optionally add diagonal jitter."""
    eye = jnp.eye(mats.shape[-1], dtype=mats.dtype)
    return 0.5 * (mats + jnp.swapaxes(mats, -1, -2)) + jitter * eye


def _predictive_latent_init(
    Ad: jnp.ndarray,
    cd: jnp.ndarray,
    init_mean: jnp.ndarray,
) -> jnp.ndarray:
    """Deterministic latent rollout under the mean dynamics."""
    cd = jnp.asarray(cd, dtype=jnp.result_type(Ad, cd, init_mean))
    T = Ad.shape[0]
    z0 = Ad[0] @ init_mean + cd[0]
    if T == 1:
        return z0[None]

    def _step(z_prev, inputs):
        Ad_t, cd_t = inputs
        z_t = Ad_t @ z_prev + cd_t
        return z_t, z_t

    _, z_rest = jax.lax.scan(_step, z0, (Ad[1:], cd[1:]))
    return jnp.concatenate([z0[None], z_rest], axis=0)


def _batched_spd_solve(mats: jnp.ndarray, rhs: jnp.ndarray) -> jnp.ndarray:
    """Solve a batch of SPD linear systems with matching right-hand sides."""
    return jax.vmap(lambda mat, b: jla.solve(mat, b, assume_a="pos"))(mats, rhs)


def _gaussian_log_prob_from_cholesky(
    value: jnp.ndarray,
    mean: jnp.ndarray,
    chol: jnp.ndarray,
    logdet: jnp.ndarray,
) -> jnp.ndarray:
    """Exact Gaussian log density using a precomputed Cholesky factor."""
    diff = value - mean
    whitened = jla.solve_triangular(chol, diff, lower=True)
    dim_term = value.shape[-1] * jnp.log(2.0 * jnp.pi)
    return -0.5 * (dim_term + logdet + whitened @ whitened)


def build_gaussian_trajectory_prior_terms(
    Ad: jnp.ndarray,
    Qd: jnp.ndarray,
    cd: jnp.ndarray,
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    jitter: float = 1e-6,
) -> GaussianTrajectoryPriorTerms:
    """Precompute Gaussian factors for repeated latent-prior evaluations."""
    cd = jnp.asarray(cd, dtype=jnp.result_type(Ad, Qd, cd, init_mean, init_cov))
    T = Ad.shape[0]
    init_pred_mean = Ad[0] @ init_mean + cd[0]
    init_pred_cov = symmetrize_with_jitter(Ad[0] @ init_cov @ Ad[0].T + Qd[0], jitter=jitter)
    init_chol = jnp.linalg.cholesky(init_pred_cov)
    init_logdet = logdet_from_cholesky(init_chol)

    if T == 1:
        return GaussianTrajectoryPriorTerms(
            init_mean=init_pred_mean,
            init_chol=init_chol,
            init_logdet=init_logdet,
            transition_chol=jnp.zeros(
                (0, init_cov.shape[0], init_cov.shape[1]), dtype=init_cov.dtype
            ),
            transition_logdet=jnp.zeros((0,), dtype=init_cov.dtype),
        )

    transition_cov = symmetrize_with_jitter(Qd[1:], jitter=jitter)
    transition_chol = jax.vmap(jnp.linalg.cholesky)(transition_cov)
    transition_logdet = logdet_from_cholesky(transition_chol)
    return GaussianTrajectoryPriorTerms(
        init_mean=init_pred_mean,
        init_chol=init_chol,
        init_logdet=init_logdet,
        transition_chol=transition_chol,
        transition_logdet=transition_logdet,
    )


def trajectory_prior_log_prob_from_terms(
    latent_trajectory: jnp.ndarray,
    Ad: jnp.ndarray,
    cd: jnp.ndarray,
    prior_terms: GaussianTrajectoryPriorTerms,
) -> jnp.ndarray:
    """Return log p(z_{1:T}) using precomputed Gaussian factors."""
    cd = jnp.asarray(cd, dtype=jnp.result_type(latent_trajectory, Ad, cd))
    init_ll = _gaussian_log_prob_from_cholesky(
        latent_trajectory[0],
        prior_terms.init_mean,
        prior_terms.init_chol,
        prior_terms.init_logdet,
    )
    if latent_trajectory.shape[0] == 1:
        return init_ll

    transition_means = jax.vmap(lambda Ad_t, z_tm1, cd_t: Ad_t @ z_tm1 + cd_t)(
        Ad[1:],
        latent_trajectory[:-1],
        cd[1:],
    )
    transition_ll = jax.vmap(_gaussian_log_prob_from_cholesky)(
        latent_trajectory[1:],
        transition_means,
        prior_terms.transition_chol,
        prior_terms.transition_logdet,
    )
    return init_ll + jnp.sum(transition_ll)


def _build_prior_precision_blocks(
    Ad: jnp.ndarray,
    Qd: jnp.ndarray,
    cd: jnp.ndarray,
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    jitter: float = 1e-6,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute prior diagonal, adjacent lower blocks and RHS before layout jitter."""
    dtype = jnp.result_type(Ad, Qd, cd, init_mean, init_cov)
    Ad = jnp.asarray(Ad, dtype=dtype)
    Qd = jnp.asarray(Qd, dtype=dtype)
    cd = jnp.asarray(cd, dtype=dtype)
    init_mean = jnp.asarray(init_mean, dtype=dtype)
    init_cov = jnp.asarray(init_cov, dtype=dtype)
    T, D = Ad.shape[:2]
    eye = jnp.eye(D, dtype=dtype)

    diag_blocks = jnp.zeros((T, D, D), dtype=dtype)
    rhs = jnp.zeros((T, D), dtype=dtype)

    prior_mean = Ad[0] @ init_mean + cd[0]
    prior_cov = _symmetrize_psd(Ad[0] @ init_cov @ Ad[0].T + Qd[0], jitter=jitter)
    prior_inv = jla.solve(prior_cov, eye, assume_a="pos")

    diag_blocks = diag_blocks.at[0].add(prior_inv)
    rhs = rhs.at[0].add(prior_inv @ prior_mean)

    if T == 1:
        return diag_blocks, jnp.zeros((0, D, D), dtype=dtype), rhs

    q_reg = _symmetrize_psd(Qd[1:], jitter=jitter)
    eye_batch = jnp.broadcast_to(eye, q_reg.shape)
    q_inv = _batched_spd_solve(q_reg, eye_batch)
    q_inv_a = _batched_spd_solve(q_reg, Ad[1:])
    q_inv_c = _batched_spd_solve(q_reg, cd[1:])

    diag_blocks = diag_blocks.at[1:].add(q_inv)
    diag_blocks = diag_blocks.at[:-1].add(jnp.swapaxes(Ad[1:], -1, -2) @ q_inv_a)
    rhs = rhs.at[1:].add(q_inv_c)
    rhs = rhs.at[:-1].add(-jnp.einsum("tij,tj->ti", jnp.swapaxes(Ad[1:], -1, -2), q_inv_c))

    return diag_blocks, -q_inv_a, rhs


def _build_prior_tridiagonal_system(
    Ad: jnp.ndarray,
    Qd: jnp.ndarray,
    cd: jnp.ndarray,
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    jitter: float = 1e-6,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Lay out the prior blocks for IEKS, including its final diagonal jitter."""
    diag, adjacent_lower, rhs = _build_prior_precision_blocks(
        Ad, Qd, cd, init_mean, init_cov, jitter=jitter
    )
    lower = jnp.zeros_like(diag).at[1:].set(adjacent_lower)
    upper = jnp.zeros_like(diag).at[:-1].set(jnp.swapaxes(adjacent_lower, -1, -2))
    return lower, _symmetrize_psd(diag, jitter=jitter), upper, rhs


def _build_ieks_system_from_prior(
    prior_lower: jnp.ndarray,
    prior_diag: jnp.ndarray,
    prior_upper: jnp.ndarray,
    prior_rhs: jnp.ndarray,
    J_t: jnp.ndarray,
    tilde_y: jnp.ndarray,
    jitter: float = 1e-6,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Assemble IEKS normal equations from a precomputed prior system."""
    diag_blocks = prior_diag + J_t
    rhs = prior_rhs + tilde_y
    return prior_lower, _symmetrize_psd(diag_blocks, jitter=jitter), prior_upper, rhs


def _build_prior_banded_system(
    Ad: jnp.ndarray,
    Qd: jnp.ndarray,
    cd: jnp.ndarray,
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    bandwidth: int,
    jitter: float = 1e-6,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Assemble the Gaussian latent-prior contribution in block-banded form."""
    diag, adjacent_lower, rhs = _build_prior_precision_blocks(
        Ad, Qd, cd, init_mean, init_cov, jitter=jitter
    )
    T, D = diag.shape[:2]
    upper = jnp.zeros((bandwidth, T, D, D), dtype=diag.dtype)
    if bandwidth >= 1:
        upper = upper.at[0, :-1].set(jnp.swapaxes(adjacent_lower, -1, -2))

    return diag, upper, rhs


def _block_banded_logdet(chol_diag: jnp.ndarray) -> jnp.ndarray:
    """Log determinant from block-banded Cholesky factors."""
    return 2.0 * jnp.sum(jnp.log(jnp.clip(jnp.diagonal(chol_diag, axis1=1, axis2=2), 1e-12)))


def _compute_profile_lower_bandwidths(row_upper_bandwidths: np.ndarray) -> np.ndarray:
    """Return the realized lower profile widths implied by symmetric upper widths."""
    T = int(row_upper_bandwidths.shape[0])
    max_bandwidth = int(np.max(row_upper_bandwidths, initial=0))
    row_lower_bandwidths = np.zeros((T,), dtype=np.int64)
    for row_idx in range(T):
        max_offset = min(row_idx, max_bandwidth)
        lower_bandwidth = 0
        for offset in range(1, max_offset + 1):
            if row_upper_bandwidths[row_idx - offset] >= offset:
                lower_bandwidth = offset
        row_lower_bandwidths[row_idx] = lower_bandwidth
    return row_lower_bandwidths


def _factor_block_profile_cholesky(
    diag: jnp.ndarray,
    upper: jnp.ndarray,
    row_upper_bandwidths: jnp.ndarray,
    row_lower_bandwidths: jnp.ndarray,
    jitter: float = 1e-6,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Exact block-profile Cholesky factorization A = L L^T."""
    if upper.shape[0] == 1:
        off_diagonal = jnp.where(row_upper_bandwidths[:, None, None] > 0, upper[0], 0)
        chol, lower = factor_tridiagonal(diag, off_diagonal, jitter=jitter)
        return chol, lower[None]

    T, _D = diag.shape[:2]
    chol_diag = jnp.zeros_like(diag)
    lower = jnp.zeros_like(upper)

    def _factor_step(
        i: int | jnp.ndarray, state: tuple[jnp.ndarray, jnp.ndarray]
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        chol_diag_state, lower_state = state
        lower_bw_i = row_lower_bandwidths[i]
        upper_bw_i = row_upper_bandwidths[i]

        def _schur_cond(loop_state: tuple[int | jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
            offset, _schur = loop_state
            return offset <= lower_bw_i

        def _schur_body(
            loop_state: tuple[int | jnp.ndarray, jnp.ndarray],
        ) -> tuple[int | jnp.ndarray, jnp.ndarray]:
            offset, schur = loop_state
            schur = schur - lower_state[offset - 1, i] @ lower_state[offset - 1, i].T
            return offset + 1, schur

        _offset_final, schur = jax.lax.while_loop(_schur_cond, _schur_body, (1, diag[i]))
        l_ii = jnp.linalg.cholesky(_symmetrize_psd(schur, jitter=jitter))
        chol_diag_state = chol_diag_state.at[i].set(l_ii)

        def _future_cond(loop_state: tuple[int | jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
            offset_j, _lower_curr = loop_state
            return offset_j <= upper_bw_i

        def _future_body(
            loop_state: tuple[int | jnp.ndarray, jnp.ndarray],
        ) -> tuple[int | jnp.ndarray, jnp.ndarray]:
            offset_j, lower_curr = loop_state
            row_j = i + offset_j
            lower_bw_j = row_lower_bandwidths[row_j]

            def _cross_cond(cross_state: tuple[int | jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
                offset_k, _schur_off = cross_state
                return offset_k <= lower_bw_i

            def _cross_body(
                cross_state: tuple[int | jnp.ndarray, jnp.ndarray],
            ) -> tuple[int | jnp.ndarray, jnp.ndarray]:
                offset_k, schur_off = cross_state
                cross_idx = offset_j + offset_k - 1
                schur_off: jnp.ndarray = jax.lax.cond(
                    cross_idx < lower_bw_j,
                    lambda s: s - lower_curr[cross_idx, row_j] @ lower_curr[offset_k - 1, i].T,
                    lambda s: s,
                    schur_off,
                )
                return offset_k + 1, schur_off

            _cross_done, schur_off = jax.lax.while_loop(
                _cross_cond,
                _cross_body,
                (1, upper[offset_j - 1, i].T),
            )
            del _cross_done
            l_ji = jla.solve_triangular(l_ii, schur_off.T, lower=True).T
            lower_curr = lower_curr.at[offset_j - 1, row_j].set(l_ji)
            return offset_j + 1, lower_curr

        _future_done, lower_state = jax.lax.while_loop(
            _future_cond,
            _future_body,
            (1, lower_state),
        )
        del _future_done
        return chol_diag_state, lower_state

    result: tuple[jnp.ndarray, jnp.ndarray] = jax.lax.fori_loop(
        0, T, _factor_step, (chol_diag, lower)
    )
    return result


def _solve_block_profile_from_cholesky(
    chol_diag: jnp.ndarray,
    lower: jnp.ndarray,
    rhs: jnp.ndarray,
    row_upper_bandwidths: jnp.ndarray,
    row_lower_bandwidths: jnp.ndarray,
) -> jnp.ndarray:
    """Solve A x = rhs from exact block-profile Cholesky factors."""
    if lower.shape[0] == 1:
        return solve_tridiagonal(chol_diag, lower[0], rhs)

    T = rhs.shape[0]
    y = jnp.zeros_like(rhs)

    def _forward_step(i: int | jnp.ndarray, y_state: jnp.ndarray) -> jnp.ndarray:
        lower_bw_i = row_lower_bandwidths[i]

        def _forward_cond(loop_state: tuple[int | jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
            offset, _res = loop_state
            return offset <= lower_bw_i

        def _forward_body(
            loop_state: tuple[int | jnp.ndarray, jnp.ndarray],
        ) -> tuple[int | jnp.ndarray, jnp.ndarray]:
            offset, res = loop_state
            res = res - lower[offset - 1, i] @ y_state[i - offset]
            return offset + 1, res

        _offset_done, res = jax.lax.while_loop(_forward_cond, _forward_body, (1, rhs[i]))
        del _offset_done
        y_i = jla.solve_triangular(chol_diag[i], res, lower=True)
        return y_state.at[i].set(y_i)

    y = jax.lax.fori_loop(0, T, _forward_step, y)
    x = jnp.zeros_like(rhs)

    def _backward_step(rev_idx: int | jnp.ndarray, x_state: jnp.ndarray) -> jnp.ndarray:
        i = T - 1 - rev_idx
        upper_bw_i = row_upper_bandwidths[i]

        def _backward_cond(loop_state: tuple[int | jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
            offset, _res = loop_state
            return offset <= upper_bw_i

        def _backward_body(
            loop_state: tuple[int | jnp.ndarray, jnp.ndarray],
        ) -> tuple[int | jnp.ndarray, jnp.ndarray]:
            offset, res = loop_state
            res = res - lower[offset - 1, i + offset].T @ x_state[i + offset]
            return offset + 1, res

        _offset_done, res = jax.lax.while_loop(_backward_cond, _backward_body, (1, y[i]))
        del _offset_done
        x_i = jla.solve_triangular(chol_diag[i].T, res, lower=False)
        return x_state.at[i].set(x_i)

    result: jnp.ndarray = jax.lax.fori_loop(0, T, _backward_step, x)
    return result


def _selected_inverse_block(
    inv_diag: jnp.ndarray,
    inv_upper: jnp.ndarray,
    row_upper_bandwidths: jnp.ndarray,
    i: int | jnp.ndarray,
    j: int | jnp.ndarray,
) -> jnp.ndarray:
    """Return block (i, j) from the packed inverse subset."""
    zero = jnp.zeros_like(inv_diag[0])

    def _diag_branch(_: None) -> jnp.ndarray:
        return inv_diag[i]

    def _offdiag_branch(_: None) -> jnp.ndarray:
        def _upper_branch(_: None) -> jnp.ndarray:
            offset = j - i
            result: jnp.ndarray = jax.lax.cond(
                offset <= row_upper_bandwidths[i],
                lambda _: inv_upper[offset - 1, i],
                lambda _: zero,
                operand=None,
            )
            return result

        def _lower_branch(_: None) -> jnp.ndarray:
            offset = i - j
            result: jnp.ndarray = jax.lax.cond(
                offset <= row_upper_bandwidths[j],
                lambda _: jnp.swapaxes(inv_upper[offset - 1, j], -1, -2),
                lambda _: zero,
                operand=None,
            )
            return result

        result: jnp.ndarray = jax.lax.cond(j > i, _upper_branch, _lower_branch, operand=None)
        return result

    result: jnp.ndarray = jax.lax.cond(i == j, _diag_branch, _offdiag_branch, operand=None)
    return result


def _block_profile_inverse_subset_from_cholesky(
    chol_diag: jnp.ndarray,
    lower: jnp.ndarray,
    row_upper_bandwidths: jnp.ndarray,
    _row_lower_bandwidths: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return diagonal and upper-profile blocks of A^{-1} from A = L L^T."""
    if lower.shape[0] == 1:
        diagonal, upper = inverse_subset_tridiagonal(chol_diag, lower[0])
        return diagonal, upper[None]
    t_steps, block_dim = chol_diag.shape[:2]
    max_bandwidth = lower.shape[0]
    eye = jnp.eye(block_dim, dtype=chol_diag.dtype)
    inv_diag = jnp.zeros_like(chol_diag)
    inv_upper = jnp.zeros_like(lower)

    def _row_step(
        rev_i: int | jnp.ndarray, state: tuple[jnp.ndarray, jnp.ndarray]
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        inv_diag_state, inv_upper_state = state
        i = t_steps - 1 - rev_i
        l_ii = chol_diag[i]
        upper_bw_i = row_upper_bandwidths[i]

        def _column_sum(column: int | jnp.ndarray, inverse_upper: jnp.ndarray) -> jnp.ndarray:
            """Contract this Cholesky column against the current inverse blocks."""

            def _sum_step(offset_k_zero: int | jnp.ndarray, acc: jnp.ndarray) -> jnp.ndarray:
                offset_k = offset_k_zero + 1

                def _accumulate(a: jnp.ndarray) -> jnp.ndarray:
                    row_k = i + offset_k
                    l_ki = lower[offset_k - 1, row_k]
                    s_kj = _selected_inverse_block(
                        inv_diag_state,
                        inverse_upper,
                        row_upper_bandwidths,
                        row_k,
                        column,
                    )
                    return a + l_ki.T @ s_kj

                result: jnp.ndarray = jax.lax.cond(
                    offset_k <= upper_bw_i, _accumulate, lambda a: a, acc
                )
                return result

            result: jnp.ndarray = jax.lax.fori_loop(
                0,
                max_bandwidth,
                _sum_step,
                jnp.zeros((block_dim, block_dim), dtype=chol_diag.dtype),
            )
            return result

        def _offdiag_step(
            offset_j_zero: int | jnp.ndarray, inv_upper_curr: jnp.ndarray
        ) -> jnp.ndarray:
            offset_j = offset_j_zero + 1

            def _compute(curr: jnp.ndarray) -> jnp.ndarray:
                row_j = i + offset_j
                schur_term = _column_sum(row_j, curr)
                s_ij = -jla.solve_triangular(l_ii.T, schur_term, lower=False)
                return curr.at[offset_j - 1, i].set(s_ij)

            result: jnp.ndarray = jax.lax.cond(
                offset_j <= upper_bw_i,
                _compute,
                lambda curr: curr,
                inv_upper_curr,
            )
            return result

        inv_upper_state: jnp.ndarray = jax.lax.fori_loop(
            0, max_bandwidth, _offdiag_step, inv_upper_state
        )
        inv_l_ii = jla.solve_triangular(l_ii, eye, lower=True)
        diag_base = inv_l_ii.T @ inv_l_ii

        diag_schur = _column_sum(i, inv_upper_state)
        diag_i = diag_base - jla.solve_triangular(l_ii.T, diag_schur, lower=False)
        diag_i = 0.5 * (diag_i + diag_i.T)
        inv_diag_state = inv_diag_state.at[i].set(diag_i)
        return inv_diag_state, inv_upper_state

    result: tuple[jnp.ndarray, jnp.ndarray] = jax.lax.fori_loop(
        0, t_steps, _row_step, (inv_diag, inv_upper)
    )
    return result


def block_profile_logdet_packed_cotangent(
    chol_diag: jnp.ndarray,
    lower: jnp.ndarray,
    row_upper_bandwidths: jnp.ndarray,
    row_lower_bandwidths: jnp.ndarray,
    *,
    scale: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return cotangents for diag and packed upper blocks of log|A|."""
    inv_diag, inv_upper = _block_profile_inverse_subset_from_cholesky(
        chol_diag,
        lower,
        row_upper_bandwidths,
        row_lower_bandwidths,
    )
    return scale * inv_diag, 2.0 * scale * inv_upper


@jax.custom_jvp
def precision_logdet(
    diag: jax.Array, upper: jax.Array, row_upper: jax.Array, row_lower: jax.Array
) -> jax.Array:
    """Sparse log determinant; its derivative needs only the inverse's occupied bands."""
    chol, _lower = _factor_block_profile_cholesky(diag, upper, row_upper, row_lower)
    return _block_banded_logdet(chol)


@precision_logdet.defjvp
def _precision_logdet_jvp(primals, tangents):
    diag, upper, row_upper, row_lower = primals
    diag_dot, upper_dot, _row_upper_dot, _row_lower_dot = tangents
    chol, lower = _factor_block_profile_cholesky(diag, upper, row_upper, row_lower)
    diag_bar, upper_bar = block_profile_logdet_packed_cotangent(
        chol, lower, row_upper, row_lower, scale=jnp.asarray(1.0, dtype=diag.dtype)
    )
    return _block_banded_logdet(chol), jnp.sum(diag_bar * diag_dot) + jnp.sum(upper_bar * upper_dot)


def _infer_support_groups(
    observation_support: ObservationSupportRuntime,
    observed_mask: np.ndarray,
) -> tuple[SupportObservationWindowBatch, ...]:
    """Compile only observed interval factors into exact-preserving window buckets."""
    plan = observation_support.window_plan
    T, n_manifest = len(plan.times), plan.n_manifest
    observed_windows = tuple(
        window
        for window in plan.windows
        if observed_mask[window.anchor_index, window.manifest_index]
    )
    anchors = sorted({window.anchor_index for window in observed_windows})
    windows_by_anchor = {anchor: [] for anchor in anchors}
    for window in observed_windows:
        windows_by_anchor[window.anchor_index].append(window)

    compiled_windows: list[
        tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = []
    for anchor_idx, windows in windows_by_anchor.items():
        start_idx = min(window.start_index for window in windows)
        mask_full = np.zeros((n_manifest,), dtype=np.float64)
        segment_len = anchor_idx - start_idx
        group_prev = np.zeros((segment_len, n_manifest), dtype=np.float64)
        group_curr = np.zeros((segment_len, n_manifest), dtype=np.float64)
        group_weights = np.zeros((segment_len, n_manifest), dtype=np.float64)
        for window in windows:
            manifest_idx = window.manifest_index
            mask_full[manifest_idx] = 1.0
            indices = window.step_indices - start_idx - 1
            group_prev[indices, manifest_idx] = window.coefficients[:, 0]
            group_curr[indices, manifest_idx] = window.coefficients[:, 1]
            group_weights[indices, manifest_idx] = window.weights
        state_len = anchor_idx - start_idx + 1
        compiled_windows.append(
            (
                state_len,
                anchor_idx,
                start_idx,
                mask_full,
                group_prev,
                group_curr,
                group_weights,
            )
        )

    def _support_bucket_state_len(state_len: int) -> int:
        return 1 if state_len <= 1 else 1 << (state_len - 1).bit_length()

    def _compile_support_batch(
        batch_state_len: int,
        windows_for_state_len: list[
            tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ],
    ) -> SupportObservationWindowBatch:
        n_windows = len(windows_for_state_len)
        max_segment_len = batch_state_len - 1
        local_positions = np.arange(batch_state_len, dtype=np.int64)

        state_lens = np.zeros((n_windows,), dtype=np.int64)
        anchor_indices = np.zeros((n_windows,), dtype=np.int64)
        start_indices = np.zeros((n_windows,), dtype=np.int64)
        mask_full = np.zeros((n_windows, n_manifest), dtype=np.float64)
        padded_prev = np.zeros((n_windows, max_segment_len, n_manifest), dtype=np.float64)
        padded_curr = np.zeros((n_windows, max_segment_len, n_manifest), dtype=np.float64)
        padded_weights = np.zeros((n_windows, max_segment_len, n_manifest), dtype=np.float64)
        padded_state_indices = np.zeros((n_windows, batch_state_len), dtype=np.int64)
        time_indices = np.zeros((n_windows, batch_state_len), dtype=np.int64)
        valid_diag = np.zeros((n_windows, batch_state_len), dtype=bool)

        for window_idx, (
            state_len,
            anchor_idx,
            start_idx,
            mask_window,
            prev_window,
            curr_window,
            weights_window,
        ) in enumerate(windows_for_state_len):
            segment_len = state_len - 1
            state_lens[window_idx] = state_len
            anchor_indices[window_idx] = anchor_idx
            start_indices[window_idx] = start_idx
            mask_full[window_idx] = mask_window
            if segment_len > 0:
                padded_prev[window_idx, :segment_len] = prev_window
                padded_curr[window_idx, :segment_len] = curr_window
                padded_weights[window_idx, :segment_len] = weights_window

            raw_positions = start_idx + local_positions
            padded_state_indices[window_idx] = raw_positions
            time_indices[window_idx] = np.clip(raw_positions, 0, T - 1)
            valid_diag[window_idx, :state_len] = True

        component_windows, component_manifests = np.nonzero(mask_full)

        return SupportObservationWindowBatch(
            max_state_len=batch_state_len,
            state_lens=jnp.asarray(state_lens, dtype=jnp.int32),
            anchor_indices=jnp.asarray(anchor_indices, dtype=jnp.int32),
            start_indices=jnp.asarray(start_indices, dtype=jnp.int32),
            mask_full=jnp.asarray(mask_full),
            prev_coeffs=jnp.asarray(padded_prev),
            curr_coeffs=jnp.asarray(padded_curr),
            weights=jnp.asarray(padded_weights),
            padded_state_indices=jnp.asarray(padded_state_indices, dtype=jnp.int32),
            time_indices=jnp.asarray(time_indices, dtype=jnp.int32),
            valid_diag=jnp.asarray(valid_diag),
            component_windows=jnp.asarray(component_windows, dtype=jnp.int32),
            component_manifests=jnp.asarray(component_manifests, dtype=jnp.int32),
        )

    windows_by_state_len: dict[
        int,
        list[tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    ] = {}
    for window in compiled_windows:
        bucket_state_len = _support_bucket_state_len(window[0])
        windows_by_state_len.setdefault(bucket_state_len, []).append(window)

    return tuple(
        _compile_support_batch(state_len, windows_by_state_len[state_len])
        for state_len in sorted(windows_by_state_len)
    )
