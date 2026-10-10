"""Optimistix iteration and adjoints backed by the trajectory's sparse precision."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast, override

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import optimistix as optx
from optimistix._search import EvalGradHessianInv

from .interval_precision import (
    IntervalFactor,
    IntervalPrecision,
    factor_interval_precision,
    solve_interval_precision,
)
from .shared import _factor_block_profile_cholesky, _solve_block_profile_from_cholesky

if TYPE_CHECKING:
    from collections.abc import Callable

    from jaxtyping import PyTree
    from optimistix._search import Eval
    from optimistix._solver.backtracking import _BacktrackingState
    from optimistix._solver.gauss_newton import _NewtonDescentState

_MODE_DAMPING = 1e-6


type Solve = Callable[[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]


def solve_fixed_point_mode(
    update: Callable[[jax.Array, None], jax.Array],
    initial: jax.Array,
    *,
    max_steps: int,
) -> tuple[jax.Array, jax.Array]:
    """Iterate a local initialization mode with the common IEKS stopping policy."""

    def convergence_norm(value: PyTree[jax.Array]) -> jax.Array:
        # Optimistix initializes its scalar convergence state in the default
        # floating dtype, independently of the coordinate arrays' precision.
        return jnp.asarray(optx.max_norm(value), dtype=jnp.asarray(jnp.inf).dtype)

    solution = optx.fixed_point(
        update,
        optx.FixedPointIteration(rtol=1e-3, atol=1e-3, norm=convergence_norm),
        initial,
        max_steps=max(max_steps, 1),
        throw=False,
    )
    return jnp.asarray(solution.value), jnp.asarray(solution.stats["num_steps"])


def _apply_inverse(chol, lower, row_upper, row_lower, solve: Solve, vector) -> jax.Array:
    return solve(chol, lower, vector, row_upper, row_lower)


class _StructuredNewton(
    optx.AbstractQuasiNewton[jax.Array, None, EvalGradHessianInv[jax.Array], None]
):
    """Supply Newton curvature; Optimistix owns the search, iteration and termination."""

    inverse_at: Callable[[jax.Array], lx.AbstractLinearOperator] = eqx.field(kw_only=True)
    initial_inverse: lx.AbstractLinearOperator = eqx.field(kw_only=True)
    rtol: float = 1e-3
    atol: float = 1e-3
    norm: Callable[[PyTree[jax.Array]], jax.Array] = optx.two_norm
    use_inverse: bool = True  # noqa: V107 - Optimistix reads this inherited solver setting.
    descent: optx.AbstractDescent[  # noqa: V107 - Optimistix invokes this descent.
        jax.Array, EvalGradHessianInv[jax.Array], _NewtonDescentState[jax.Array]
    ] = cast(
        "optx.AbstractDescent[jax.Array, EvalGradHessianInv[jax.Array], _NewtonDescentState[jax.Array]]",
        optx.NewtonDescent(),
    )
    search: optx.AbstractSearch[
        jax.Array, EvalGradHessianInv[jax.Array], Eval, _BacktrackingState
    ] = cast(
        "optx.AbstractSearch[jax.Array, EvalGradHessianInv[jax.Array], Eval, _BacktrackingState]",
        optx.BacktrackingArmijo(),
    )
    verbose: Callable[..., None] = lambda **_kwargs: None  # noqa: V107 - Optimistix invokes this callback.

    @override  # noqa: V105 - Optimistix calls this initialization override.
    def init_hessian(self, y, f, grad):
        del y
        return optx.FunctionInfo.EvalGradHessianInv(f, grad, self.initial_inverse), None

    @override  # noqa: V105 - Optimistix calls this override.
    def update_hessian(self, y, y_eval, f_info, f_eval_info, hessian_update_state):
        del y, f_info, hessian_update_state
        return optx.FunctionInfo.EvalGradHessianInv(
            f_eval_info.f, f_eval_info.grad, self.inverse_at(y_eval)
        ), None

    @override
    def postprocess(self, fn, y, aux, args, options, state, tags, result):
        del fn, args, options, tags, result
        return y, aux, {"num_accepted_steps": state.num_accepted_steps - 1}


class _BandedAdjointSolver(lx.AbstractLinearSolver[tuple[jax.Array, jax.Array]]):
    """Recover Hessian bands by coloring JVPs, without forming a dense Jacobian."""

    bandwidth: int

    @override
    def init(self, operator, options):
        del options
        structure = operator.in_structure()
        n_time, n_state = structure.shape
        times, states = jnp.arange(n_time), jnp.arange(n_state)
        n_colors = min(n_time, 2 * self.bandwidth + 1)
        colors = (times[:, None] % n_colors) * n_state + states[None]
        probes = jnp.eye(n_colors * n_state, dtype=structure.dtype)[colors]
        products = jax.vmap(operator.mv)(jnp.moveaxis(probes, -1, 0))
        diag = products[colors[:, None, :], times[:, None, None], states[None, :, None]]
        upper = jnp.stack(
            tuple(
                jnp.where(
                    (times + offset < n_time)[:, None, None],
                    products[
                        colors[jnp.minimum(times + offset, n_time - 1), None, :],
                        times[:, None, None],
                        states[None, :, None],
                    ],
                    0.0,
                )
                for offset in range(1, self.bandwidth + 1)
            )
        )
        row_upper = jnp.minimum(self.bandwidth, n_time - 1 - times)
        row_lower = jnp.minimum(self.bandwidth, times)
        return _factor_block_profile_cholesky(diag, upper, row_upper, row_lower)

    @override  # noqa: V105 - Lineax calls this override.
    def compute(self, state, vector, options):
        del options
        chol, lower = state
        times = jnp.arange(chol.shape[0])
        row_upper = jnp.minimum(self.bandwidth, chol.shape[0] - 1 - times)
        row_lower = jnp.minimum(self.bandwidth, times)
        solution = _solve_block_profile_from_cholesky(chol, lower, vector, row_upper, row_lower)
        return solution, lx.RESULTS.successful, {}

    @override  # noqa: V105 - Lineax calls this adjoint override.
    def transpose(self, state, options):
        return state, options

    @override
    def conj(self, state, options):
        return jax.tree.map(jnp.conj, state), options

    @override  # noqa: V105 - Lineax calls this rank declaration.
    def assume_full_rank(self):
        return True


def solve_latent_mode(
    log_joint,
    system,
    initial: jax.Array,
    args,
    *,
    bandwidth,
    row_upper,
    row_lower,
    max_steps,
    factor=_factor_block_profile_cholesky,
    solve=_solve_block_profile_from_cholesky,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Solve an initialization mode with a sparse Newton step and sparse implicit AD."""
    curvature = eqx.filter_closure_convert(system, initial, args)
    diagonal = jnp.broadcast_to(
        jnp.eye(initial.shape[1], dtype=initial.dtype), (*initial.shape, initial.shape[1])
    )
    lower = jnp.zeros((bandwidth, *diagonal.shape), dtype=initial.dtype)
    template = _inverse_operator(
        eqx.Partial(_apply_inverse, diagonal, lower, row_upper, row_lower, solve), initial
    )
    return _minimize_mode(
        log_joint,
        initial,
        args,
        eqx.Partial(_banded_inverse_at, curvature, args, row_upper, row_lower, factor, solve),
        template,
        optx.ImplicitAdjoint(_BandedAdjointSolver(bandwidth)),
        max_steps,
    )


def _inverse_operator(apply, coordinates: jax.Array) -> lx.FunctionLinearOperator:
    return lx.FunctionLinearOperator(
        apply,
        jax.ShapeDtypeStruct(coordinates.shape, coordinates.dtype),
        tags=lx.positive_semidefinite_tag,
        closure_convert=False,
    )


def _banded_inverse_at(curvature, parameters, row_upper, row_lower, factor, solve, y):
    diag, upper, _rhs = curvature(y, parameters)
    diag, upper = jnp.asarray(diag, y.dtype), jnp.asarray(upper, y.dtype)
    diag = diag + _MODE_DAMPING * jnp.eye(y.shape[1], dtype=y.dtype)[None]
    chol, lower = factor(diag, upper, row_upper, row_lower)
    return _inverse_operator(
        eqx.Partial(_apply_inverse, chol, lower, row_upper, row_lower, solve), y
    )


def _interval_inverse_at(curvature, parameters, y):
    precision = jax.tree.map(lambda x: jnp.asarray(x, y.dtype), curvature(y, parameters))
    precision = IntervalPrecision(
        precision.diagonal + _MODE_DAMPING * jnp.eye(y.shape[1], dtype=y.dtype)[None],
        precision.upper,
        precision.jacobian,
        precision.information,
    )
    factor = factor_interval_precision(precision)
    return _inverse_operator(eqx.Partial(solve_interval_precision, factor), y)


def solve_interval_latent_mode(
    log_joint, system, initial, args, *, rank: int, max_steps: int
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Use Woodbury Newton steps and a matrix-free implicit differentiation solve."""
    curvature = eqx.filter_closure_convert(system, initial, args)
    diagonal = jnp.broadcast_to(
        jnp.eye(initial.shape[1], dtype=initial.dtype), (*initial.shape, initial.shape[1])
    )
    factor = IntervalFactor(
        diagonal,
        jnp.zeros((1, *diagonal.shape), dtype=initial.dtype),
        jnp.zeros((*initial.shape, rank), dtype=initial.dtype),
        jnp.zeros((rank, rank), dtype=initial.dtype),
        jnp.asarray(0.0, dtype=initial.dtype),
    )
    template = _inverse_operator(eqx.Partial(solve_interval_precision, factor), initial)
    return _minimize_mode(
        log_joint,
        initial,
        args,
        eqx.Partial(_interval_inverse_at, curvature, args),
        template,
        optx.ImplicitAdjoint(lx.GMRES(rtol=1e-6, atol=1e-6, max_steps=128)),
        max_steps,
    )


def _minimize_mode(
    log_joint, initial: jax.Array, args, inverse_at, initial_inverse, adjoint, max_steps
) -> tuple[jax.Array, dict[str, jax.Array]]:
    solution = optx.minimise(
        lambda y, parameters: -log_joint(y, parameters),
        _StructuredNewton(inverse_at=inverse_at, initial_inverse=initial_inverse),
        initial,
        args=args,
        max_steps=max(max_steps, 1) + 1,
        adjoint=adjoint,
        throw=False,
    )
    return solution.value, {
        "init_log_joint": log_joint(initial, args),
        "n_iterations": jnp.asarray(solution.stats["num_steps"] - 1, dtype=jnp.int32),
        "n_accepted_steps": solution.stats["num_accepted_steps"],
        "final_damping": jnp.asarray(_MODE_DAMPING, dtype=initial.dtype),
    }
