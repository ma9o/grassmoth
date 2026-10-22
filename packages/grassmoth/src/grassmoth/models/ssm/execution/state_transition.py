"""Dynestyx Euler–Maruyama transitions with compiler-owned Gaussian support."""

from typing import override

import dynestyx as dsx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from dynestyx.models import DiscreteTimeStateEvolution
from dynestyx.solvers import euler_maruyama_loc_cov

from grassmoth.models.ssm.execution.supported_normal import SupportedNormal


class SupportedEulerMaruyama(DiscreteTimeStateEvolution):
    """Keep exact deterministic coordinates outside the transition Cholesky factor.

    Dynestyx owns the numerical step. Its full-rank Gaussian discretizer cannot
    represent deterministic coordinates without adding noise; this adapter factors
    only the compiler's stochastic submatrix and embeds its samples in state space.
    """

    continuous: dsx.StochasticContinuousTimeStateEvolution
    indices: tuple[int, ...] = eqx.field(static=True)

    @override
    def __call__(
        self, x: jax.Array, u: jax.Array | None, t_now: float | jax.Array, t_next: float | jax.Array
    ) -> dist.MultivariateNormal:
        moments = euler_maruyama_loc_cov(
            self.continuous, x, u, jnp.asarray(t_now), jnp.asarray(t_next)
        )
        return SupportedNormal(moments["loc"], moments["cov"], self.indices)
