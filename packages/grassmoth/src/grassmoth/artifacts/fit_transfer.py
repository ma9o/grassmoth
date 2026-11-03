"""A bounded transfer view of a complete fit; retained cache values stay intact."""

from __future__ import annotations

from typing import TYPE_CHECKING

from grassmoth.artifacts.arrays import ArrayVector, NumericalArray
from grassmoth.artifacts.probability.serialization import empirical_atoms, empirical_distribution

if TYPE_CHECKING:
    from grassmoth.actions.contracts.io import FitOutput

# A heuristic: about 2.5 times the 4M-value, 22 MB probe results that downloaded
# and decoded without trouble. About 2,000 draws remain for France-sized models,
# leaving the 400-ESS gate reachable even in the transferred sample.
MAX_TRANSFER_DRAW_VALUES = 10_000_000


def transfer_fit(result: FitOutput) -> FitOutput:
    """Select one evenly spaced subset, preserving the full-chain certificate."""
    report = result.inference
    metadata = report.core.inference_metadata
    model = result.dynamical_model_spec
    if (
        model is None
        or metadata.distribution is None
        or not metadata.segments
        or not metadata.segments[-1].settings.thinning
    ):
        return result
    identity = metadata.distribution
    atoms = empirical_atoms(model.distributions[identity])
    chains = metadata.num_chains
    count = atoms.shape[0] // chains
    capacity = max(1, MAX_TRANSFER_DRAW_VALUES // (chains * atoms.shape[1]))
    # One draw per chain is the irreducible transfer, even when it exceeds the heuristic.
    factor = max(1, (count + capacity - 1) // capacity)
    if factor == 1:
        return result
    selected = atoms.reshape(chains, count, -1)[:, ::factor]
    kept = selected.shape[1]
    flat = selected.reshape(chains * kept, -1)
    law = empirical_distribution(flat)
    conditioned = model.with_entities(distributions={**model.distributions, identity: law})
    owned = NumericalArray.from_numpy(flat)
    layout = model.law_layouts[identity]
    evidence = report.evidence

    def thin(value: NumericalArray | None) -> NumericalArray | None:
        return None if value is None else NumericalArray.from_numpy(value.values[:, ::factor])

    def phases(name: str, values: NumericalArray) -> NumericalArray:
        # Starts are one state per chain, with no draw axis.
        phase = name.rsplit("/", 1)[-1]
        if phase in {"initialization", "continued_start"}:
            return values
        segment = metadata.segments[-1 - name.count("previous/")]
        origin = (
            segment.start if phase == "post_warmup" else segment.start - segment.settings.num_warmup
        )
        start = (-origin) % factor
        return NumericalArray.from_numpy(values.values[:, start::factor])

    return result.revised(
        dynamical_model_spec=conditioned,
        inference=report.revised(
            evidence=evidence.revised(
                chain_extra_fields={
                    name: NumericalArray.from_numpy(value.values[:, ::factor])
                    for name, value in evidence.chain_extra_fields.items()
                },
                observation_log_probs=thin(evidence.observation_log_probs),
                phase_extra_fields={
                    phase: {name: phases(phase, value) for name, value in fields.items()}
                    for phase, fields in evidence.phase_extra_fields.items()
                },
                warmup_complete_log_posterior_history=phases(
                    "warmup", evidence.warmup_complete_log_posterior_history
                )
                if evidence.warmup_complete_log_posterior_history is not None
                else None,
                all_complete_log_posterior_history=phases(
                    "all", evidence.all_complete_log_posterior_history
                )
                if evidence.all_complete_log_posterior_history is not None
                else None,
            ),
            core=report.core.revised(
                inference_metadata=metadata.revised(
                    thinning_factor=metadata.thinning_factor * factor,
                    draws_kept_per_chain=kept,
                    num_samples_total=chains * kept,
                ),
            ),
            detail=report.detail.revised(
                trace_data=tuple(
                    trace.revised(
                        chains=tuple(
                            ArrayVector(
                                array=owned,
                                indices=(None, layout.parameter_columns[trace.subject.element_id]),
                                start=chain * kept,
                                stop=(chain + 1) * kept,
                            )
                            for chain in range(chains)
                        )
                    )
                    for trace in report.detail.trace_data
                ),
            ),
        ),
    )
