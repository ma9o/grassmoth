"""Bind additive scientific terms and causal interventions to Dynestyx evolution.

Directed terms read the state through per-edge input overrides. Node potentials
supply scalar energies for Dynestyx's negative-gradient drift. Hard interventions
replace the owning derivative and remove its natural potential contribution.
"""

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, cast, overload

import dynestyx as dsx
import equinox as eqx
import jax
import jax.numpy as jnp

from nof1_causal_lab.models.ssm.dynamics.edges import VectorFieldComponent
from nof1_causal_lab.models.ssm.dynamics.intervention import Intervention
from nof1_causal_lab.models.ssm.shapes import Array, Float

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.dynamics.expression import ExpressionComponent


type DriftProgram = Callable[[Array, Array, Array, tuple[Mapping[str, Array], ...]], Array]


class ComponentDrift(eqx.Module):
    """Compose numerical contributions with the same parameter and potential ownership."""

    components: tuple[VectorFieldComponent, ...]
    potential_indices: tuple[int, ...] = eqx.field(static=True, default=())

    def __call__(
        self,
        t: Array,
        eta: Array,
        eta_per_edge: Array,
        params: tuple[Mapping[str, Array], ...],
    ) -> Array:
        """Accumulate directed and intrinsic terms, leaving node energies to the potential."""
        value = jnp.zeros_like(eta)
        for index, (component, values) in enumerate(zip(self.components, params, strict=True)):
            if index not in self.potential_indices:
                value = component.contribute(value, eta, eta_per_edge, t, values)
        return value


class VectorFieldArgs(eqx.Module):
    """Arguments threaded through Diffrax / Optimistix to the field.

    ``params`` is a tuple of per-component pytrees (one slice per
    component, matched by position). ``intervention`` is the
    ``eqx.Module`` pytree carrying override structure.
    """

    params: tuple[Mapping[str, Array], ...]
    intervention: Intervention


def apply_variable_overrides_to_state(
    eta: Float[Array, " D"],
    t: Array,
    intervention: Intervention,
) -> Float[Array, " D"]:
    """Clamp ``eta[i] = u_i(t)`` for each variable override."""
    for ov in intervention.variable_overrides():
        eta = eta.at[ov.index].set(ov.value_fn(t))
    return eta


def _apply_edge_input_overrides(
    eta_eff: Float[Array, "D D"],
    t: Array,
    intervention: Intervention,
) -> Float[Array, "D D"]:
    """Replace ``eta_eff[target, source]`` with ``u(t)`` per edge override."""
    for ov in intervention.edge_input_overrides():
        eta_eff = eta_eff.at[ov.target, ov.source].set(ov.value_fn(t))
    return eta_eff


def _apply_variable_overrides_to_derivative(
    d_eta: Float[Array, " D"],
    t: Array,
    intervention: Intervention,
) -> Float[Array, " D"]:
    """Replace clamped state derivatives with the time derivatives of their prescribed values."""
    for ov in intervention.variable_overrides():
        du_dt = jax.grad(lambda tt, fn=ov.value_fn: jnp.sum(fn(tt)))(t)
        d_eta = d_eta.at[ov.index].set(du_dt)
    return d_eta


class VectorField(eqx.Module):
    """Exact additive field with node energies and explicit intervention ownership.

    ``args.params`` has one slice per component. Production terms are bound
    scalar expressions; linear numerical components share the same protocol.
    """

    n_latent: int = eqx.field(static=True)
    components: tuple[VectorFieldComponent, ...]
    drift_program: DriftProgram | ComponentDrift = eqx.field(static=True)
    potential_indices: tuple[int, ...] = eqx.field(static=True, default=())

    @overload
    def evolution(
        self, args: VectorFieldArgs, diffusion: None = None
    ) -> dsx.DeterministicContinuousTimeStateEvolution: ...

    @overload
    def evolution(
        self, args: VectorFieldArgs, diffusion: dsx.FullDiffusion
    ) -> dsx.StochasticContinuousTimeStateEvolution: ...

    def evolution(
        self, args: VectorFieldArgs, diffusion: dsx.FullDiffusion | None = None
    ) -> dsx.DeterministicContinuousTimeStateEvolution | dsx.StochasticContinuousTimeStateEvolution:
        """Bind scientific terms to Dynestyx's drift and native negative-gradient potential."""
        drift = StructuralDrift(
            self,
            args,
        )
        potential = StructuralPotential(self, args) if self.potential_indices else None
        if diffusion is None:
            return dsx.DeterministicContinuousTimeStateEvolution(
                drift=drift, potential=potential, use_negative_gradient=True
            )
        return dsx.StochasticContinuousTimeStateEvolution(
            drift=drift,
            potential=potential,
            use_negative_gradient=True,
            diffusion=diffusion,
        )

    def __call__(
        self, t: Array, eta: Float[Array, " D"], args: VectorFieldArgs
    ) -> Float[Array, " D"]:
        """Evaluate total drift at the supplied state and time using the bound intervention."""
        return self.evolution(args).total_drift(x=eta, u=None, t=t)

    def initial_condition(
        self, eta0: Float[Array, " D"], args: VectorFieldArgs, t0: Array | float = 0.0
    ) -> Float[Array, " D"]:
        """Apply whole-state interventions to the initial state at the requested start time."""
        return apply_variable_overrides_to_state(eta0, jnp.asarray(t0), args.intervention)

    @eqx.filter_jit(inline=jax.Inline.XLA_LATE)  # ty: ignore[no-matching-overload]  # pyright: ignore[reportCallIssue] -- Equinox forwards jitkwargs to JAX; its overloads omit them.
    def _natural_derivative(
        self, t: Array, eta: Float[Array, " D"], args: VectorFieldArgs
    ) -> Float[Array, " D"]:
        eta_eff = jnp.broadcast_to(eta[None, :], (self.n_latent, self.n_latent))
        eta_eff = _apply_edge_input_overrides(eta_eff, t, args.intervention)

        return self.drift_program(t, eta, eta_eff, args.params)


class StructuralDrift(eqx.Module):
    """Directed and intrinsic drift contributions, followed by hard interventions."""

    vector_field: VectorField
    args: VectorFieldArgs

    def __call__(self, x: Array, u: Array | None, t: float | int | Array) -> Array:
        """Evaluate structural drift and replace derivatives of intervention-clamped states."""
        del u
        t = jnp.asarray(t)
        value = self.vector_field._natural_derivative(t, x, self.args)
        return _apply_variable_overrides_to_derivative(value, t, self.args.intervention)


class StructuralPotential(eqx.Module):
    """Sum node energies; intervened nodes have their natural dynamics removed."""

    vector_field: VectorField
    args: VectorFieldArgs

    def __call__(self, x: Array, u: Array | None, t: float | int | Array) -> Array:
        """Sum node-potential energies, excluding states fixed by whole-state interventions."""
        del u, t
        clamped = {override.index for override in self.args.intervention.variable_overrides()}
        energy = jnp.zeros((), dtype=x.dtype)
        for index in self.vector_field.potential_indices:
            component = cast("ExpressionComponent", self.vector_field.components[index])
            if component.target not in clamped:
                energy = energy + component.evaluate(x, self.args.params[index])
        return energy
