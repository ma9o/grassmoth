"""Apply production diagnostics to one retained fit segment without altering the saved report."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from evaluation.comparison import _posterior_draws
from nof1_causal_lab.artifacts.identity import ConstructRef, ParameterRef
from nof1_causal_lab.artifacts.model.parameter import ParameterCoordinate
from nof1_causal_lab.models.ssm.inference.diagnostics.convergence import chain_convergence
from nof1_causal_lab.models.ssm.inference.diagnostics.latent import LatentChainEvidence
from nof1_causal_lab.models.ssm.inference.mcmc_state import TrajectoryMCMCResult
from nof1_causal_lab.models.ssm.inference.types import (
    JointPosteriorDraws,
    MCMCPosterior,
    ProductionDiagnostics,
)

if TYPE_CHECKING:
    from nof1_causal_lab.actions.contracts.bodies import FitOutput
    from nof1_causal_lab.artifacts.reports.posterior_diagnostics import ChainDiagnostics
    from nof1_causal_lab.json_types import JsonObject


def segment_diagnostics(result: FitOutput) -> ChainDiagnostics | None:
    """Use the exact producer's estimators on the latest segment's scientific coordinates."""
    metadata = result.inference.core.inference_metadata
    segment = metadata.segments[-1]
    count = segment.stop - segment.start
    if count == 0:
        return None
    layout, atoms = _posterior_draws(result, segment)
    model = result.dynamical_model_spec
    assert model is not None
    evidence = result.inference.evidence
    chains = metadata.num_chains
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
    # Phase buffers preserve diagnostics introduced by a switched kernel, even when
    # cumulative evidence cannot concatenate them with the earlier kernel's fields.
    extra = evidence.phase_extra_fields["post_warmup"]
    posterior = MCMCPosterior.from_run(
        draws=JointPosteriorDraws(parameters=parameters),
        diagnostics=ProductionDiagnostics(
            mcmc=TrajectoryMCMCResult(
                chain_samples={
                    name: values.reshape(chains, count) for name, values in parameters.items()
                },
                chain_extra_fields={
                    name: jnp.asarray(value.values) for name, value in extra.items()
                },
                num_chains=chains,
                num_samples=count,
                backend=metadata.engine.method,
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
        evidence=metadata.engine,
    )
    return posterior.get_inference_diagnostics(references)


def segment_summary(result: FitOutput) -> JsonObject:
    """Pair this segment's ESS and convergence with this segment's sampling interval."""
    diagnostics = segment_diagnostics(result)
    gate = chain_convergence(diagnostics)
    segment = result.inference.core.inference_metadata.segments[-1]
    seconds = sum(p.end_seconds - p.start_seconds for p in segment.phases if p.stage == "sampling")
    rows = (
        () if diagnostics is None else (*diagnostics.per_parameter, *(diagnostics.per_latent or ()))
    )
    bulk = [row.ess_bulk for row in rows if row.ess_bulk is not None]
    tail = [row.ess_tail for row in rows if row.ess_tail is not None]
    hats = [row.r_hat for row in rows if row.r_hat is not None]
    minimum = min((*bulk, *tail)) if bulk and tail else None
    return {
        "gate": gate.status,
        "warnings": list(gate.warnings),
        "sampling_seconds": seconds,
        "min_bulk_ess": min(bulk) if bulk else None,
        "min_tail_ess": min(tail) if tail else None,
        "max_r_hat": max(hats) if hats else None,
        "min_ess_per_sampling_second": minimum / seconds
        if minimum is not None and seconds > 0
        else None,
        "min_ess_per_sweep": minimum / (segment.stop - segment.start)
        if minimum is not None
        else None,
        "num_divergences": diagnostics.num_divergences if diagnostics is not None else None,
        "bfmi": list(diagnostics.energy.bfmi)
        if diagnostics is not None and diagnostics.energy is not None
        else None,
        "segment_diagnostics": None if diagnostics is None else diagnostics.model_dump(mode="json"),
    }
