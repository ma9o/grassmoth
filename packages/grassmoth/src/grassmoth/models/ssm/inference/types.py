"""Inference result and artifact types."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jax.stages import Compiled
    from pydantic import Field
    from xarray import DataTree

    from grassmoth.artifacts.continuation import SamplerCheckpoint
    from grassmoth.artifacts.identity import ConstructId
    from grassmoth.artifacts.reports.profiling import FitProfileReport
    from grassmoth.artifacts.sampling import SamplerSpec
    from grassmoth.models.ssm.inference.diagnostics.latent import LatentChainEvidence
    from grassmoth.models.ssm.inference.mcmc_state import TrajectoryMCMCResult


from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Annotated, Self

import jax.numpy as jnp
from numpy.typing import NDArray

from grassmoth.artifacts.immutability import freeze_fields
from grassmoth.artifacts.reports.posterior_diagnostics import (
    ChainDiagnostics,
    LOODiagnostics,
    ParameterDiagnostics,
    ParetoKPoint,
    RankHistogram,
    SamplerDiagnostics,
    TraceSeries,
)
from grassmoth.models.ssm.inference.diagnostics.plots import (
    ParameterReferences,
    pareto_k_points,
    sample_coordinates,
)
from grassmoth.models.ssm.inference.diagnostics.plots import (
    build_rank_histograms as _build_rank_histograms,
)
from grassmoth.models.ssm.inference.diagnostics.plots import (
    build_trace_data as _build_trace_data,
)
from grassmoth.models.ssm.inference.sample_projection import _filter_public_samples

type ArviZArrayMapping = Mapping[str, jnp.ndarray | NDArray]

_PARETO_K_WARNING: Annotated[
    float,
    Field(
        description=(
            "Vehtari, Gelman & Gabry (2017), https://arxiv.org/abs/1507.04544, section 2.1: warn "
            "above 0.5."
        )
    ),
] = 0.5
_PARETO_K_BAD: Annotated[
    float,
    Field(
        description=(
            "Vehtari, Gelman & Gabry (2017), section 2.1: good performance observed up to 0.7; "
            "above it they advise sampling directly, K-fold cross-validation or a more robust "
            "model."
        )
    ),
] = 0.7


def _arviz_idata_from_posterior(
    posterior: ArviZArrayMapping,
    *,
    log_likelihood: ArviZArrayMapping | None = None,
) -> DataTree:
    """Build ArviZ inference data from array-valued posterior dictionaries."""
    import arviz_base as az_base
    import numpy as np

    payload = {"posterior": {name: np.asarray(values) for name, values in posterior.items()}}
    if log_likelihood is not None:
        payload["log_likelihood"] = {
            name: np.asarray(values) for name, values in log_likelihood.items()
        }
    return az_base.from_dict(payload)


@dataclass(frozen=True)
class UnretainedRun:
    """A completed compilation or warmup with no retained posterior and no checkpoint."""

    settings: SamplerSpec
    compiled_step: Compiled
    phase_extra_fields: Mapping[str, Mapping[str, jnp.ndarray]]
    profile: FitProfileReport | None = None

    @classmethod
    def completed(
        cls,
        settings: SamplerSpec,
        compiled_step: Compiled,
        phase_extra_fields: Mapping[str, Mapping[str, jnp.ndarray]],
        profile: FitProfileReport | None = None,
    ) -> UnretainedRun:
        """Own a completed interval whose settings retain no posterior samples."""
        if settings.num_samples_per_chain != 0:
            raise ValueError("Unretained runs require zero requested posterior samples")
        return cls(
            settings,
            compiled_step,
            phase_extra_fields,
            profile,
        )

    def __post_init__(self) -> None:
        """Freeze telemetry at the execution boundary."""
        freeze_fields(self)


@dataclass(frozen=True, kw_only=True)
class ProductionDiagnostics:
    """The exact producer's native buffers, separate from serialized report values."""

    mcmc: TrajectoryMCMCResult
    checkpoint: SamplerCheckpoint | None = None
    profile: FitProfileReport | None = None
    observation_log_probs: jnp.ndarray | NDArray
    latent_chains: LatentChainEvidence | None = None
    compiled_step: Compiled | None = None
    public_sites: tuple[str, ...] | None = None
    exact_observation_rows: jnp.ndarray | NDArray | None = None
    sampler_diagnostics: SamplerDiagnostics | None = None
    phase_extra_fields: Mapping[str, Mapping[str, jnp.ndarray | NDArray]] | None = None
    warmup_complete_log_posterior_history: jnp.ndarray | None = None
    all_complete_log_posterior_history: jnp.ndarray | NDArray | None = None

    def __post_init__(self) -> None:
        """Freeze phase-specific diagnostic mappings after construction."""
        if self.phase_extra_fields is not None:
            object.__setattr__(
                self,
                "phase_extra_fields",
                MappingProxyType(
                    {
                        phase: MappingProxyType(dict(values))
                        for phase, values in self.phase_extra_fields.items()
                    }
                ),
            )
        freeze_fields(self)


@dataclass(frozen=True)
class JointPosteriorDraws:
    """Aligned parameter and latent-trajectory draws; the leading axis identifies one joint draw."""

    parameters: Mapping[str, jnp.ndarray | NDArray]
    latent_paths: jnp.ndarray | NDArray | None = None
    state_ids: tuple[ConstructId, ...] = ()

    def __post_init__(self) -> None:
        """Require aligned parameter and latent-path draws with uniquely labeled state coordinates."""
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))
        counts = set()
        for values in self.parameters.values():
            if values.ndim < 1:
                raise ValueError("Posterior parameters require a leading draw axis")
            counts.add(values.shape[0])
        if self.latent_paths is not None:
            if self.latent_paths.ndim != 3:
                raise ValueError("Posterior latent paths require draw, time, and state axes")
            counts.add(self.latent_paths.shape[0])
            if self.state_ids and (
                len(self.state_ids) != self.latent_paths.shape[2]
                or len(set(self.state_ids)) != len(self.state_ids)
            ):
                raise ValueError("Latent state IDs must label the state axis exactly")
        if len(counts) > 1:
            raise ValueError("Posterior parameters and latent paths must share the draw axis")
        freeze_fields(self)


@dataclass(frozen=True)
class MCMCPosterior:
    """Joint posterior draws produced by an exact state and parameter MCMC engine."""

    draws: JointPosteriorDraws
    diagnostics: ProductionDiagnostics
    initial_latent_delta: jnp.ndarray | None = None
    final_latent_delta: jnp.ndarray | None = None

    @classmethod
    def from_run(
        cls,
        *,
        draws: JointPosteriorDraws,
        diagnostics: ProductionDiagnostics,
        initial_latent_delta: jnp.ndarray | None = None,
        final_latent_delta: jnp.ndarray | None = None,
    ) -> Self:
        """Own the resolved joint draws and exact diagnostics."""
        return cls(
            draws=draws,
            diagnostics=diagnostics,
            initial_latent_delta=initial_latent_delta,
            final_latent_delta=final_latent_delta,
        )

    def get_samples(self) -> Mapping[str, jnp.ndarray | NDArray]:
        """Return parameter draws aligned with the retained latent trajectories."""
        return self.draws.parameters

    def get_inference_diagnostics(self, references: ParameterReferences) -> ChainDiagnostics:
        """Compute chain diagnostics with scientific identities supplied for sampled coordinates."""
        return self.get_mcmc_diagnostics(references)

    def get_mcmc_diagnostics(self, references: ParameterReferences) -> ChainDiagnostics:
        """Reduce native chain estimators directly into scientifically referenced values."""
        import math

        from grassmoth.models.ssm.inference.diagnostics.chain_stats import chain_statistics

        mcmc = self.diagnostics.mcmc
        chain_samples = mcmc.get_samples(group_by_chain=True)
        public_sites = self.diagnostics.public_sites
        if public_sites is not None:
            chain_samples = _filter_public_samples(chain_samples, set(public_sites))
        parameters: list[ParameterDiagnostics] = []
        for coordinate, values in sample_coordinates(chain_samples, sample_dims=2):
            reference = references[coordinate]
            if reference is not None:
                label, subject = reference
                metrics = chain_statistics(jnp.asarray(values)[..., None])[0]
                r_hat, bulk, tail, mcse_mean = (
                    float(value) if math.isfinite(value) else None for value in metrics
                )
                parameters.append(
                    ParameterDiagnostics(
                        parameter=label,
                        subject=subject,
                        r_hat=r_hat,
                        ess_bulk=bulk,
                        ess_tail=tail,
                        mcse_mean=mcse_mean,
                    )
                )
        extra = mcmc.get_extra_fields()

        def average(name: str) -> float | None:
            return float(jnp.mean(jnp.asarray(extra[name]))) if name in extra else None

        return ChainDiagnostics(
            num_chains=int(mcmc.num_chains),
            num_samples_per_chain=int(mcmc.num_samples),
            per_parameter=tuple(parameters),
            per_latent=self.diagnostics.latent_chains.diagnostics
            if self.diagnostics.latent_chains is not None
            else None,
            latent_accept_prob_mean=average("latent_accept_prob"),
            parameter_accept_prob_mean=average("parameter_accept_prob"),
        )

    def get_chain_detail(
        self, references: ParameterReferences
    ) -> tuple[tuple[TraceSeries, ...], tuple[RankHistogram, ...]]:
        """Build trace series and rank histograms for retained public parameter sites."""
        samples = self.diagnostics.mcmc.get_samples(group_by_chain=True)
        if (sites := self.diagnostics.public_sites) is not None:
            samples = _filter_public_samples(samples, set(sites))
        return _build_trace_data(samples, references), _build_rank_histograms(samples, references)

    def get_loo_diagnostics(
        self, *, observed_rows: jnp.ndarray | NDArray
    ) -> tuple[LOODiagnostics, tuple[ParetoKPoint, ...]] | None:
        """PSIS leave-one-measurement-row-out from the joint particle posterior.

        The exact emission factors condition on each sampled latent state. The
        held-out row is interpolated using all other rows, including future
        measurements. This does not estimate leave-future-out forecast skill.
        """
        exact_rows = self.diagnostics.exact_observation_rows
        if exact_rows is not None and bool(jnp.any(exact_rows)):
            # Removing an equality increases the support dimension. Reweighting
            # draws confined to that equality cannot recover the held-out target.
            return None
        import numpy as np
        from arviz_stats.loo import loo
        from arviz_stats.utils import ELPDDataLOO

        factors = np.asarray(self.diagnostics.observation_log_probs)
        factors = factors[:, :, observed_rows]
        if factors.shape[1] == 0 or factors.shape[2] == 0:
            return None
        mcmc = self.diagnostics.mcmc
        posterior = mcmc.get_samples(group_by_chain=True)
        idata = _arviz_idata_from_posterior(
            posterior,
            log_likelihood={"measurement_row": factors},
        )
        estimate = loo(
            idata, pointwise=True
        )  # ArviZ returns ELPDDataLOO for the ordinary, non-moment-match path.
        assert isinstance(estimate, ELPDDataLOO)
        values = tuple(float(value) for value in estimate.pareto_k.values)
        points = pareto_k_points(
            values, tuple(int(index) + 1 for index in np.flatnonzero(observed_rows))
        )
        return LOODiagnostics(
            elpd_loo=float(estimate.elpd),
            p_loo=float(estimate.p),
            se=float(estimate.se),
            n_data_points=int(estimate.n_data_points),
            n_bad_k=sum(value > _PARETO_K_BAD for value in values) if values else None,
            n_warn_k=sum(_PARETO_K_WARNING < value <= _PARETO_K_BAD for value in values)
            if values
            else None,
        ), points
