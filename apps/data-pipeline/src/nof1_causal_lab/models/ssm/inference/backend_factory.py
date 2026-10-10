"""Leaf factory for concrete marginal likelihood backend construction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from nof1_causal_lab.artifacts.likelihood import DeltaLawSpec, NormalLawSpec
from nof1_causal_lab.models.ssm import numerics as numeric
from nof1_causal_lab.models.ssm.dynamics.expression import BoundExpression

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.likelihood import Law
    from nof1_causal_lab.models.ssm.execution.contracts import ObservationLaws
    from nof1_causal_lab.models.ssm.runtime import BoundPanel


def build_laplace_backend(
    panel: BoundPanel,
    n_ieks_iters: int,
):
    """Construct a Laplace likelihood backend for a compiled spec."""
    from nof1_causal_lab.models.ssm.inference.targets.laplace import LaplaceLikelihood

    return LaplaceLikelihood(
        n_latent=numeric.n_states(panel.compiled_dynamical_model),
        n_manifest=numeric.n_observations(panel.compiled_dynamical_model),
        n_ieks_iters=n_ieks_iters,
        observation_support=panel.observation_support,
        observed_mask=~np.isnan(np.asarray(panel.observations)),
    )


def initialization_observation_laws(laws: ObservationLaws) -> ObservationLaws:
    """Gaussian initialization view only; particle targets retain the original Delta."""

    def gaussian(law: Law[BoundExpression]) -> Law[BoundExpression]:
        if not isinstance(law, DeltaLawSpec):
            return law
        scale = BoundExpression(
            law.v.expression,
            lambda _eta, scale, _values: scale,
            (),
            (),
            0,
            0,
            lambda eta: eta,
            law.v.link,
        )
        return NormalLawSpec(loc=law.v, scale=scale)

    return tuple(gaussian(law) for law in laws)
