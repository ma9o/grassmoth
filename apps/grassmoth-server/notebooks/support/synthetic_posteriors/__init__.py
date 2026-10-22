"""Synthetic posteriors — compose pathological densities from named geometric primitives.

Three layers:

- **Bases** (``Gaussian``, ``StudentT``, ``Cauchy``): set tail behaviour.
- **Bijectors** (``Shift``, ``Shear``, ``Bend``, ``Funnel``, ``Softplus``, ``Logit``): invertible warps with
  analytical log-det-jacobian. Compose with ``Chain``.
- **Structural combinators** (``Mixture``, ``Mirror``, ``SoftConstraint``, ``invariance``):
  non-diffeomorphic modifications — multimodality, symmetry, and soft factors.

Build a target with ``TransformedTarget(base, chain)`` and wrap it in combinators to add
multimodality or non-identifiability. See ``notebooks/pedagogical_2d_posterior.ipynb``.
"""

from notebooks.support.synthetic_posteriors.bases import Base, Cauchy, Gaussian, StudentT
from notebooks.support.synthetic_posteriors.bijectors import (
    Bend,
    Bijector,
    Funnel,
    Identity,
    Logit,
    Shear,
    Shift,
    Softplus,
)
from notebooks.support.synthetic_posteriors.combinators import Chain
from notebooks.support.synthetic_posteriors.targets import (
    Mirror,
    Mixture,
    SoftConstraint,
    Target,
    TransformedTarget,
    invariance,
)

__all__ = [
    "Base",
    "Bend",
    "Bijector",
    "Cauchy",
    "Chain",
    "Funnel",
    "Gaussian",
    "Identity",
    "Logit",
    "Mirror",
    "Mixture",
    "Shear",
    "Shift",
    "SoftConstraint",
    "Softplus",
    "StudentT",
    "Target",
    "TransformedTarget",
    "invariance",
]
