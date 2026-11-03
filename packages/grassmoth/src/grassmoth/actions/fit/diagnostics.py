"""Apply production diagnostics to one retained fit segment without altering the saved report."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from grassmoth.artifacts.identity import ParameterRef
from grassmoth.artifacts.model.parameter import ParameterCoordinate
from grassmoth.artifacts.probability.serialization import empirical_atoms
from grassmoth.models.ssm.inference.mcmc_state import TrajectoryMCMCResult
from grassmoth.models.ssm.inference.types import (
    JointPosteriorDraws,
    MCMCPosterior,
    ProductionDiagnostics,
)

if TYPE_CHECKING:
    from grassmoth.artifacts.identity import DistributionId
    from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec
    from grassmoth.artifacts.reports.inference import InferenceEvidence
    from grassmoth.artifacts.reports.posterior_diagnostics import ChainDiagnostics
    from grassmoth.models.ssm.inference.diagnostics.latent import LatentChainEvidence


def segment_diagnostics(
    model: DynamicalModelSpec,
    distribution: DistributionId,
    evidence: InferenceEvidence,
    chains: int,
    latent_chains: LatentChainEvidence | None,
) -> ChainDiagnostics:
    """Diagnose the current execution before earlier draws and telemetry are appended."""
    layout = model.law_layouts[distribution]
    atoms = empirical_atoms(model.distributions[distribution])
    count = atoms.shape[0] // chains
    parameters = {
        str(element): atoms[:, column] for element, column in layout.parameter_columns.items()
    }
    references = {
        ParameterCoordinate(site_name=str(element), indices=()): (
            layout.labels[element],
            ParameterRef(parameter_id=parameter, element_id=element),
        )
        for parameter, elements in layout.parameters
        for element in elements
    }
    # This is the current execution's telemetry, before any cross-kernel concatenation.
    extra = evidence.chain_extra_fields
    posterior = MCMCPosterior.from_run(
        draws=JointPosteriorDraws(parameters=parameters),
        diagnostics=ProductionDiagnostics(
            mcmc=TrajectoryMCMCResult(
                chain_samples={
                    name: values.reshape(chains, count) for name, values in parameters.items()
                },
                chain_extra_fields={name: value.values for name, value in extra.items()},
                num_chains=chains,
                num_samples=count,
            ),
            observation_log_probs=np.empty((chains, count, 0), dtype=np.float32),
            latent_chains=latent_chains,
        ),
    )
    diagnostics = posterior.get_inference_diagnostics(references)
    return diagnostics.revised(
        per_latent=tuple(
            entry
            for identity in layout.constructs
            for entry in diagnostics.per_latent
            if entry.subject.id == identity
        )
        if diagnostics.per_latent is not None
        else None,
    )
