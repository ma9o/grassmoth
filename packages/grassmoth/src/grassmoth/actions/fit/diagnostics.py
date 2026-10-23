"""Apply production diagnostics to one retained fit segment without altering the saved report."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from grassmoth.artifacts.identity import ConstructRef, ParameterRef
from grassmoth.artifacts.model.parameter import ParameterCoordinate
from grassmoth.artifacts.probability.serialization import empirical_atoms
from grassmoth.models.ssm.inference.diagnostics.latent import LatentChainEvidence
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
    from grassmoth.artifacts.reports.posterior_diagnostics import (
        ChainDiagnostics,
        MCMCEvidence,
    )


def segment_diagnostics(
    model: DynamicalModelSpec,
    distribution: DistributionId,
    evidence: InferenceEvidence,
    chains: int,
    engine: MCMCEvidence,
) -> ChainDiagnostics:
    """Diagnose the current execution before earlier draws and telemetry are appended."""
    layout = model.law_layouts[distribution]
    atoms = empirical_atoms(model.distributions[distribution])
    count = atoms.shape[0] // chains
    parameters = {
        str(element): jnp.asarray(atoms[:, column])
        for element, column in layout.parameter_columns.items()
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
                backend=engine.method,
            ),
            observation_log_probs=jnp.empty((chains, count, 0)),
            latent_chains=LatentChainEvidence(
                paths=np.stack(
                    tuple(atoms[:, span] for span in layout.trajectory_slices.values()), axis=-1
                ).reshape(chains, count, len(layout.time_points), len(layout.constructs)),
                free_mask=np.asarray(evidence.latent_free_mask.values),
                states=tuple(
                    (ConstructRef(id=identity), model.get_construct(identity).name)
                    for identity in layout.constructs
                ),
            )
            if evidence.latent_free_mask is not None
            else None,
        ),
        evidence=engine,
    )
    return posterior.get_inference_diagnostics(references)
