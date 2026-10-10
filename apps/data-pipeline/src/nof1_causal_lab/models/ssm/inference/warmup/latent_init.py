"""Data-conditioned latent reference-path initialization for particle MCMC.

Unconditional predictive simulation can diverge at data-informed parameter
positions when the vector field is nonlinear: finite posterior density does
not imply forward-simulation stability. A diverged predictive path poisons
the particle-MCMC reference trajectory with non-finite values that no move
can recover from (the kernel fails fast on that). The IEKS mode path is
data-conditioned and finite whenever the position's Laplace objective is,
so it is the correct reference-path init wherever positions come from
warmup rather than from the prior.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.autoreparam import Strategy
    from nof1_causal_lab.models.ssm.parameterization import PriorRuntimeBundle
    from nof1_causal_lab.models.ssm.runtime import BoundPanel

import jax.numpy as jnp

from nof1_causal_lab.models.ssm.inference.backend_factory import build_laplace_backend


def compute_ieks_latent_paths(
    priors: PriorRuntimeBundle,
    panel: BoundPanel,
    *,
    positions: jnp.ndarray,
    trace_key: jnp.ndarray,
    reparam: Strategy | None,
    n_ieks_iters: int,
) -> jnp.ndarray:
    """Initialize particle reference paths with data-conditioned IEKS modes.

    Args:
        priors: Prior laws bound to the compiled model's sampling sites.
        panel: Model, observations, time grid, and resolved measurement supports.
        positions: Flat unconstrained parameter positions with shape
            ``(chain, dimension)``, using the warmup's site layout.
        trace_key: JAX key used to establish that parameter-site layout.
        reparam: Reparameterization used by the warmup positions, if any.
        n_ieks_iters: Number of iterated smoothing passes for each position.

    Returns:
        Initial latent paths with shape ``(chain, time, state)``. These approximate
        modes seed the particle sampler and are not posterior trajectory draws.

    Raises:
        ValueError: The Laplace solver supplies no latent mode or returns a
            non-finite path for any chain.
    """
    observations, times = panel.observations, panel.times
    from nof1_causal_lab.models.ssm.inference.warmup.map import _build_map_laplace_bundle

    backend = build_laplace_backend(panel, n_ieks_iters)
    bundle = _build_map_laplace_bundle(priors, panel, trace_key, backend, reparam)
    aux_fn = bundle["neg_log_posterior_with_aux_fn"]
    dtype = bundle["flat_example"].dtype

    paths = []
    for chain_idx, position in enumerate(jnp.asarray(positions, dtype=dtype)):
        _, aux = aux_fn(position, observations, times)
        mode = aux["inner"].state.latent_mode
        if mode is None:
            raise ValueError(
                "Laplace backend returned no latent mode; cannot build a "
                "data-conditioned latent init path."
            )
        path = jnp.asarray(mode)
        if not bool(jnp.all(jnp.isfinite(path))):
            raise ValueError(
                f"IEKS latent init path for chain {chain_idx} is non-finite; the "
                "position lies outside the Laplace objective's stable region."
            )
        paths.append(path)
    return jnp.stack(paths, axis=0)
