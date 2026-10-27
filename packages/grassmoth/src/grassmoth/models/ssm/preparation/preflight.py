"""Check prior reach against model-bound observations before numerical fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import numpy as np
import numpyro.distributions as dist

from grassmoth.artifacts.model.likelihood import (
    DeltaLawSpec,
    NormalLawSpec,
    StudentTLawSpec,
)
from grassmoth.artifacts.probability.families import DistributionFamily
from grassmoth.artifacts.sampling import DsmcSpec, NutsDASpec, ParticleMGradSpec, SamplerSpec
from grassmoth.models.ssm import compiled_access as numeric
from grassmoth.models.ssm.compile.state_space import SequentialRecurrences

if TYPE_CHECKING:
    from grassmoth.models.ssm.compile.inputs import CompiledDynamicalModel
    from grassmoth.models.ssm.parameterization import PriorRuntimeBundle
    from grassmoth.models.ssm.preparation.binding import BoundPanel

LOCATION_REACH_SIGMAS = 6.0


@dataclass(frozen=True)
class ObservationPreflightFailure:
    """Observed data is inconsistent with the spec/prior configuration."""

    message: str

    @classmethod
    def rejected(cls, message: str) -> Self:
        """Own the rejection value shared by observation preparation and fitting."""
        return cls(message)


def validate_sampler_support(
    compiled: CompiledDynamicalModel,
    sampler: SamplerSpec,
) -> ObservationPreflightFailure | None:
    """Own sampler-specific restrictions, including checks before remote allocation."""
    if DistributionFamily.STUDENT_T in numeric.diffusion_families(compiled):
        return ObservationPreflightFailure.rejected(
            "Exact inference currently requires Gaussian latent diffusion for every state."
        )
    if isinstance(sampler.algorithm, NutsDASpec):
        return None
    if not isinstance(sampler.algorithm.path_update, DsmcSpec):
        if isinstance(sampler.algorithm.path_update, ParticleMGradSpec) and (
            compiled.state_space.recurrence
            or any(item.support.support_kind == "interval" for item in compiled.observations)
        ):
            return ObservationPreflightFailure.rejected(
                "Particle-mGRAD supports point readings without deterministic recurrences."
            )
        if sampler.algorithm.tempering is not None:
            return ObservationPreflightFailure.rejected(
                "Sequential guided path updates do not support tempering."
            )
        return None
    unsupported = tuple(
        ", ".join(compiled.states[index].name for index in stage.indices)
        for stage in compiled.state_space.recurrence
        if isinstance(stage, SequentialRecurrences)
    )
    if unsupported:
        return ObservationPreflightFailure.rejected(
            "DSMC requires each deterministic dependency stage to be affine in its own "
            "coordinates to preserve O(log T) parallel depth; unsupported components: "
            + "; ".join(unsupported)
        )
    return None


def validate_observation_support_for_fit(panel: BoundPanel) -> ObservationPreflightFailure | None:
    """Reject observation semantics the particle target cannot represent."""
    intervals = [
        indicator.name
        for indicator in panel.compiled_dynamical_model.observations
        if indicator.support.support_kind == "interval"
        and not panel.compiled_dynamical_model.states[indicator.state_index].is_input
        and isinstance(indicator.law, DeltaLawSpec)
    ]
    if intervals:
        names = ", ".join(intervals)
        return ObservationPreflightFailure.rejected(
            "Exact endogenous interval constraints are not supported by particle inference; "
            f"use a noisy interval observation law for: {names}."
        )
    compiled = panel.compiled_dynamical_model
    observed = np.asarray(panel.observations)
    for index, state in enumerate(compiled.states):
        columns = tuple(
            column
            for column, indicator in enumerate(compiled.observations)
            if indicator.state_index == index and isinstance(indicator.law, DeltaLawSpec)
        )
        if state.is_input or not columns:
            continue
        exact = observed[:, columns]
        readings = exact[np.isfinite(exact)]
        if state.time_invariant and readings.size and np.any(readings != readings[0]):
            return ObservationPreflightFailure.rejected(
                f"Conflicting exact observations for static construct {state.name!r}."
            )
        initial = readings if state.time_invariant else exact[0][np.isfinite(exact[0])]
        if initial.size and index not in compiled.state_space.initial:
            if compiled.initial_mean_block.free_support[index]:
                return ObservationPreflightFailure.rejected(
                    f"Exact readings of {state.name!r} constrain its uncertain initial mean; "
                    "inference on parameter equality constraints is not supported."
                )
            if np.any(initial != np.asarray(compiled.initial_mean_block.template)[index]):
                return ObservationPreflightFailure.rejected(
                    f"Exact readings of {state.name!r} disagree with its fixed initial value."
                )
    for column, indicator in enumerate(compiled.observations):
        state = compiled.states[indicator.state_index]
        if (
            isinstance(indicator.law, DeltaLawSpec)
            and not state.is_input
            and not state.time_invariant
            and indicator.state_index not in compiled.state_space.transition
            and np.any(np.isfinite(observed[1:, column]))
        ):
            return ObservationPreflightFailure.rejected(
                f"Exact readings of deterministic evolving construct {state.name!r} after the "
                "initial time require constrained inference, which is not supported. "
                "Its declared deterministic dynamics have been preserved."
            )
    return None


def _prior_loc_scale(
    prior: dist.Distribution, n_free: int, free_idx: int
) -> tuple[str, float, float] | None:
    if isinstance(prior, dist.MixtureGeneral):
        prior = prior.component_distributions[free_idx]
    while isinstance(prior, (dist.ExpandedDistribution, dist.MaskedDistribution)):
        prior = prior.base_dist
    family: str = type(prior).__name__
    if isinstance(prior, dist.TwoSidedTruncatedDistribution):
        prior = prior.base_dist
        family = "TruncatedNormal"
    if not isinstance(prior, dist.Normal):
        return None
    mu = np.broadcast_to(np.asarray(prior.loc, dtype=np.float64), (n_free,))
    sigma = np.broadcast_to(np.asarray(prior.scale, dtype=np.float64), (n_free,))
    return family, float(mu[free_idx]), float(sigma[free_idx])


def validate_observations_for_fit(
    priors: PriorRuntimeBundle, panel: BoundPanel
) -> ObservationPreflightFailure | None:
    """Validate (spec, priors, observations) consistency before fitting.

    Return an expected scientific rejection listing every violating channel.
    """
    failure = validate_observation_support_for_fit(panel)
    if failure is not None:
        return failure
    compiled_dynamical_model = panel.compiled_dynamical_model
    obs = np.asarray(panel.observations, dtype=np.float64)
    standardized = numeric.observation_standardized(compiled_dynamical_model)
    names = numeric.observation_names(compiled_dynamical_model)

    means_block = compiled_dynamical_model.observation_mean_block
    free_support = np.asarray(means_block.free_support, dtype=bool)
    n_free = int(free_support.sum())
    free_prior = priors.priors[means_block.free_site_name] if n_free else None

    problems: list[str] = []
    for j in range(numeric.n_observations(compiled_dynamical_model)):
        # Interval sums and variability are not on the point-intercept scale.
        # Their reach depends on the whole path and its recorded support.
        if compiled_dynamical_model.observations[j].support.support_kind == "interval":
            continue
        finite = obs[:, j][np.isfinite(obs[:, j])]
        if finite.size == 0:
            continue
        mean_j = float(finite.mean())

        if bool(standardized[j]):
            continue

        if not isinstance(
            compiled_dynamical_model.observations[j].law, (NormalLawSpec, StudentTLawSpec)
        ):
            continue
        if not bool(free_support[j]):
            continue
        if free_prior is None:
            continue

        free_idx = int(free_support[:j].sum())
        prior_summary = _prior_loc_scale(free_prior, n_free, free_idx)
        if prior_summary is None:
            continue
        prior_family, mu_j, sigma_j = prior_summary
        if sigma_j <= 0.0:
            continue
        z = abs(mean_j - mu_j) / sigma_j
        if z > LOCATION_REACH_SIGMAS:
            problems.append(
                f"{names[j]}: observed mean {mean_j:.4g} lies {z:.1f} prior sd from its free "
                f"manifest-mean prior {prior_family}(mu={mu_j:.4g}, sigma={sigma_j:.4g}); "
                "the posterior cannot reach the data location — mark the indicator standardized "
                "(and standardize the data) or author the prior on the data scale"
            )

    if problems:
        return ObservationPreflightFailure.rejected(
            "Observation/prior preflight failed:\n- " + "\n- ".join(problems)
        )
    return None
