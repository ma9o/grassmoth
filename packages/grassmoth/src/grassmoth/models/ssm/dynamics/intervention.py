"""Intervention DSL for counterfactual simulation.

Two override kinds, both targeted at a vector field:

- ``VariableOverride``: hard clamp ``eta[index] = value_fn(t)``. Sets the
  initial condition and the drift component so the latent tracks the
  callable. Equivalent to Pearl's ``do(η_index = u(t))``.

- ``EdgeInputOverride``: surgical replacement of the source value as seen
  by a specific target edge. Other consumers of the source are unchanged.
  This is the primitive that makes the non-linear edge vocabulary
  (Hill, multiplicative, effect-compartment) cleanly interveneable later.

Value functions are ``eqx.Module`` pytrees: array fields traced through
``vmap`` / ``jit``, structure carried in the treedef. ``ConstantValueFn``
covers the immediate ``set`` / ``shift`` cases; ``LinearRampValueFn``
covers piecewise-linear protocols (e.g., dose tapers). Adding a new value
family is a new ``eqx.Module`` class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp

if TYPE_CHECKING:
    from jax import Array


class ConstantValueFn(eqx.Module):
    """Time-invariant value function ``u(t) = value``."""

    value: Array

    def __call__(self, _t: Array) -> Array:
        """Return the intervention's fixed value at any model time."""
        return self.value


class LinearRampValueFn(eqx.Module):
    """A linear intervention ramp that holds its endpoint values outside the ramp interval.

    Interpolates between ``(t_start, value_start)`` and ``(t_end, value_end)``
    for dose tapers and other time-varying intervention protocols.
    """

    t_start: Array
    t_end: Array
    value_start: Array
    value_end: Array

    def __call__(self, t: Array) -> Array:
        """Interpolate linearly within the ramp interval and hold endpoint values outside it."""
        frac = jnp.clip(
            (t - self.t_start) / jnp.maximum(self.t_end - self.t_start, 1e-12), 0.0, 1.0
        )
        return self.value_start + frac * (self.value_end - self.value_start)


class PrecomputedValueFn(eqx.Module):
    """Piecewise-linear interpolation of ``values`` sampled at ``times``.

    Holds the endpoints outside ``[times[0], times[-1]]`` (``jnp.interp``
    semantics). Used for ``trajectory`` clamps where the caller supplies an
    explicit list of values across a window.
    """

    times: Array
    values: Array

    def __call__(self, t: Array) -> Array:
        """Interpolate the recorded intervention values onto the requested model time."""
        return jnp.interp(t, self.times, self.values)


ValueFn = ConstantValueFn | LinearRampValueFn | PrecomputedValueFn


class VariableOverride(eqx.Module):
    """Hard clamp ``eta[index]`` to ``value_fn(t)`` for all simulated time."""

    index: int = eqx.field(static=True)
    value_fn: ValueFn


class EdgeInputOverride(eqx.Module):
    """A replacement source value seen by one target's drift calculation.

    Replaces ``eta[source]`` with ``value_fn(t)`` for contributions to
    ``eta[target]``. Other edges from the source see the natural state.
    """

    source: int = eqx.field(static=True)
    target: int = eqx.field(static=True)
    value_fn: ValueFn


Override = VariableOverride | EdgeInputOverride


class Intervention(eqx.Module):
    """Set of overrides active for the entire simulation horizon.

    Time-windowed activation is expressed inside ``value_fn`` rather than
    at the ``Intervention`` level (use ``LinearRampValueFn`` or compose
    new ``ValueFn`` modules).
    """

    overrides: tuple[Override, ...]

    @classmethod
    def none(cls) -> Intervention:
        """Construct the natural-course intervention with no state or edge overrides."""
        return cls(overrides=())

    def variable_overrides(self) -> tuple[VariableOverride, ...]:
        """Select whole-state overrides in their authored order."""
        return tuple(o for o in self.overrides if isinstance(o, VariableOverride))

    def edge_input_overrides(self) -> tuple[EdgeInputOverride, ...]:
        """Select overrides affecting a source state's input to a particular edge."""
        return tuple(o for o in self.overrides if isinstance(o, EdgeInputOverride))
