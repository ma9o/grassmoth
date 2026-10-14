"""Forward simulation of the declared nonlinear dynamics under interventions.

Dynestyx runs deterministic drift paths. The SDE adapter retains direct Diffrax
integration for indexed Brownian increments and seeded replay. Estimands and
orchestration over posterior draws live in the counterfactual package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import diffrax as dfx
import dynestyx as dsx
import equinox as eqx
import jax.numpy as jnp
import jax.random as random
import numpyro.distributions as dist
from dynestyx.solvers import solve_ode_state_path
from jax import Array

from nof1_causal_lab.models.ssm.dynamics.vector_field import VectorFieldArgs

if TYPE_CHECKING:
    from collections.abc import Mapping

    from nof1_causal_lab.models.ssm.dynamics.intervention import Intervention
    from nof1_causal_lab.models.ssm.dynamics.vector_field import VectorField


class ProcessNoise(eqx.Module):
    """A process diffusion factor and random key, optionally batched over draws."""

    key: Array
    diffusion: Array


class BrownianTreeSpec(eqx.Module):
    """Virtual Brownian tree with an optional fixed solver step size."""

    step_size: float | Array | None = None
    """``None`` selects ``(t1 - t0) / 200``."""
    tol: float = 1e-3


class IndexedBrownianSpec(eqx.Module):
    """Replayable integer-step Brownian increments on an explicit fixed grid."""

    step_size: float | Array


class SimulationConfig(eqx.Module):
    """Solver settings whose numerical step size can remain a traced array.

    This Equinox pytree lets per-draw CFL-capped step sizes pass through
    ``filter_jit`` as leaves, so changing a step size does not require a
    separate compiled program for each prior draw.
    """

    rtol: float = 1e-4
    atol: float = 1e-6
    max_steps: int = 4096
    brownian: BrownianTreeSpec | IndexedBrownianSpec = eqx.field(default_factory=BrownianTreeSpec)


class _IndexedBrownianPath(dfx.AbstractBrownianPath[Array | dfx.BrownianIncrement]):
    """Fast deterministic Brownian increments for a fixed-step forward solve.

    Every solver interval is keyed by its integer step index rather than the
    floating-point bit patterns of its endpoints. This has the fixed-step speed
    of ``UnsafeBrownianPath`` while making the same seed and schedule replay the
    same increments across process restarts.
    """

    # Diffrax declares these through Equinox AbstractVar. Ty currently models
    # AbstractVar as a class variable even though Diffrax requires instance fields.
    t0: Array  # pyright: ignore[reportIncompatibleVariableOverride] - Diffrax AbstractVar requires an instance field. # ty: ignore[invalid-attribute-override] - Diffrax requires instance fields; ty treats Equinox AbstractVar as a class variable.
    t1: Array  # pyright: ignore[reportIncompatibleVariableOverride] - Diffrax AbstractVar requires an instance field. # ty: ignore[invalid-attribute-override] - Diffrax requires instance fields; ty treats Equinox AbstractVar as a class variable.
    shape: tuple[int, ...] = eqx.field(static=True)
    key: Array
    step_size: Array
    levy_area: type[dfx.BrownianIncrement] = eqx.field(  # pyright: ignore[reportIncompatibleVariableOverride] - Equinox AbstractVar is implemented by this concrete static field.
        static=True, default=dfx.BrownianIncrement, kw_only=True
    )

    @eqx.filter_jit  # noqa: V105 - required by the Diffrax AbstractBrownianPath protocol
    @override
    def evaluate(self, t0, t1=None, left: bool = True, use_levy: bool = False):
        del left
        if t1 is None:
            raise ValueError("Indexed Brownian paths only support interval increments")
        dtype = jnp.result_type(t0, t1)
        start = jnp.asarray(t0, dtype=dtype)
        end = jnp.asarray(t1, dtype=dtype)
        step_index = jnp.rint((start - self.t0) / self.step_size).astype(jnp.int32)
        increment_key = random.fold_in(self.key, step_index)
        dt = end - start
        increment = random.normal(increment_key, self.shape, dtype=dtype) * jnp.sqrt(dt)
        if use_levy:
            return self.levy_area(dt=dt, W=increment)
        return increment


def simulate(
    vector_field: VectorField,
    params: tuple[Mapping[str, Array], ...],
    intervention: Intervention,
    initial_state: Array,
    time_grid: Array,
    config: SimulationConfig | None = None,
    *,
    noise: ProcessNoise | None = None,
) -> Array:
    """Bind a causal intervention, then simulate its declared Dynestyx evolution."""
    from nof1_causal_lab.models.ssm.execution.dynamical_model import continuous_state_evolution

    args = VectorFieldArgs(params=params, intervention=intervention)
    initial_state = vector_field.initial_condition(initial_state, args, time_grid[0])
    evolution = (
        vector_field.evolution(
            args,
        )
        if noise is None
        else continuous_state_evolution(
            vector_field, params, noise.diffusion, intervention=intervention
        )
    )
    dynamical_model = dsx.DynamicalModel(
        initial_condition=dist.Delta(initial_state, event_dim=1),
        state_evolution=evolution,
        observation_model=_latent_observation,
        t0=time_grid[0],
    )
    return simulate_model_path(
        dynamical_model,
        initial_state,
        time_grid,
        config=config,
        key=None if noise is None else noise.key,
    )


def simulate_model_path(
    dynamical_model: dsx.DynamicalModel,
    initial_state: Array,
    time_grid: Array,
    config: SimulationConfig | None = None,
    *,
    key: Array | None = None,
) -> Array:
    """Execute a declared model, preserving indexed Brownian replay.

    Both inference and prediction supply the same nonlinear state evolution.
    ODE paths use Dynestyx's solver. SDE paths use Diffrax directly because the
    Dynestyx solver does not accept the indexed Brownian path used by paired runs.
    """
    cfg = config or SimulationConfig()
    evolution = dynamical_model.state_evolution
    stochastic = isinstance(evolution, dsx.StochasticContinuousTimeStateEvolution)
    if stochastic != (key is not None):
        raise ValueError("Stochastic evolution requires a key; deterministic evolution does not")
    y0 = initial_state
    t0, t1 = time_grid[0], time_grid[-1]
    n_latent = dynamical_model.state_dim
    if time_grid.shape[0] == 1:
        return y0[None, :]
    if not stochastic:
        return solve_ode_state_path(
            dynamical_model,
            initial_state=y0,
            t0=t0,
            path_times=time_grid,
            diffeqsolve_settings={
                "solver": dfx.Tsit5(),
                "stepsize_controller": dfx.PIDController(rtol=cfg.rtol, atol=cfg.atol),
                "dt0": jnp.maximum((t1 - t0) / 256.0, 1e-6),
                "max_steps": cfg.max_steps,
                "throw": False,
            },
        )

    assert key is not None
    brownian_spec = cfg.brownian
    if isinstance(brownian_spec, IndexedBrownianSpec):
        brownian = _IndexedBrownianPath(
            t0=t0,
            t1=t1,
            shape=(n_latent,),
            key=key,
            step_size=jnp.asarray(brownian_spec.step_size),
        )
        adjoint = dfx.ForwardMode()
    else:
        brownian = dfx.VirtualBrownianTree(
            t0=t0,
            t1=t1,
            tol=brownian_spec.tol,
            shape=(n_latent,),
            key=key,
        )
        adjoint = dfx.RecursiveCheckpointAdjoint()

    def drift_at(t, y, runtime):
        return runtime.total_drift(x=y, u=None, t=t)

    ode_term = dfx.ODETerm(drift_at)
    diffusion_term = dfx.ControlTerm(
        lambda t, y, a: a.diffusion.as_matrix(x=y, u=None, t=t, state_dim=n_latent), brownian
    )
    term = dfx.MultiTerm(ode_term, diffusion_term)
    solver = dfx.Heun()
    dt0 = (
        brownian_spec.step_size if brownian_spec.step_size is not None else float((t1 - t0) / 200.0)
    )
    solution = dfx.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        args=evolution,
        saveat=dfx.SaveAt(ts=time_grid),
        max_steps=cfg.max_steps,
        throw=False,
        adjoint=adjoint,
    )
    path: Array = solution.ys  # pyright: ignore[reportAssignmentType] - Array y0 and SaveAt(ts=...) produce an array-valued solution.
    return path


def _latent_observation(x, u, t):
    del u, t
    return dist.Delta(x, event_dim=1)
