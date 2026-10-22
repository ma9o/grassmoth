"""Recorded plot fixtures for the single comprehensive workbench story."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from grassmoth.actions.contracts.bodies import SimulateOutput
from grassmoth.actions.simulate.summaries import simulation_summary
from grassmoth.artifacts.arrays import NumericalArray
from grassmoth.artifacts.reports.display_frames import central_frame
from grassmoth.artifacts.scenarios import CausalEffectResult
from grassmoth.artifacts.simulation import (
    PairedArmSimulation,
    SimulationArm,
    SimulationReport,
)
from grassmoth.models.ssm.counterfactual.estimands import summarize_draws
from grassmoth.study.projections.visuals import simulation_observation_histories
from grassmoth_server.study.persistence.codec import result_payload

if TYPE_CHECKING:
    from scripts.fixtures.reader import ModelReader


def workbench_visuals(reader: ModelReader, template):
    """Generate explicit illustrative draws, then use production projection code.

    The demonstration's archived fit has no original joint draws. Its observation
    history stays source grounded; new simulation paths are explicitly illustrative,
    and their summaries are derived from these paths, never inverted into fake draws.
    """
    dynamical_model_spec = reader.dynamical_model_spec
    assert dynamical_model_spec is not None
    assert reader.data_metadata is not None
    report = SimulationReport.model_validate(restore_fixture(template)["report"])
    rng = np.random.default_rng(41)
    time = np.asarray(report.evidence.times)
    phase = rng.uniform(-0.5, 0.5, (report.evidence.draws, 1, 1))
    modes = np.where(np.arange(report.evidence.draws) % 2, 1, -1)[:, None, None]
    dimensions = np.arange(len(report.evidence.state_ids))[None, None, :]
    reference = modes * (1.2 + np.sin(time[None, :, None] * 0.9 + phase + dimensions))
    differences = modes * 0.4 + 0.3 * (1 - np.exp(-time[None, :, None] / 2))
    action = reference + differences
    variables = report.evidence.observation_layout.variables
    observations = np.stack(
        [action[:, :, index % len(report.evidence.state_ids)] for index in range(len(variables))],
        axis=-1,
    )
    reference_observations = np.stack(
        [
            reference[:, :, index % len(report.evidence.state_ids)]
            for index in range(len(variables))
        ],
        axis=-1,
    )
    mask = np.ones_like(observations, dtype=bool)
    mask[:, 2, :] = False
    retain = NumericalArray.from_numpy

    starts = np.broadcast_to(time[:, None] - 1, (len(time), len(variables)))
    ends = np.broadcast_to(time[:, None], starts.shape)
    arms = report.evidence.arms
    assert isinstance(arms, PairedArmSimulation)
    assert isinstance(arms.causal, CausalEffectResult)
    outcome = report.evidence.state_ids.index(arms.causal.outcome)
    contrasts = action[:, :, outcome] - reference[:, :, outcome]
    evidence = report.evidence.revised(
        arms=PairedArmSimulation(
            action=SimulationArm(latent_paths=retain(action), observations=retain(observations)),
            reference=SimulationArm(
                latent_paths=retain(reference), observations=retain(reference_observations)
            ),
            causal=arms.causal.revised(
                differences=retain(contrasts),
                frame=central_frame(contrasts),
                summary=summarize_draws(contrasts[:, -1]),
                reference_mean=float(reference[:, -1, outcome].mean()),
                manifest_effects={
                    variable.id: float((observations - reference_observations)[:, -1, index].mean())
                    for index, variable in enumerate(variables)
                },
            ),
        ),
        observation_layout=report.evidence.observation_layout.revised(
            mask=retain(mask),
            support_start_times=retain(starts),
            support_end_times=retain(ends),
        ),
    )
    report = report.revised(
        evidence=evidence,
        summary=simulation_summary(
            evidence, action, observations, mask, reference, reference_observations
        ),
    )
    result = SimulateOutput(
        report=report,
        data=simulation_observation_histories(evidence, observations, mask),
    )
    observations = {}
    for variable in reader.data_metadata.variables:
        history = reader.observation_history(variable.id)
        assert history is not None
        observations[variable.id] = history.model_dump(mode="json")
    return {
        "note": "Simulation paths are explicitly illustrative, not a production fit or scientific evidence. Summaries are derived from these exact paths.",
        "simulation": result_payload(result),
        "observations": observations,
    }


def render_fixture(value):
    """Spell binary fixture bytes once; subsequent views select their earlier buffer."""
    buffers = {}

    def visit(item):
        if isinstance(item, bytes):
            if item in buffers:
                return {"buffer": buffers[item]}
            buffers[item] = len(buffers)
            return list(item)
        if isinstance(item, dict):
            return {key: visit(child) for key, child in item.items()}
        if isinstance(item, (tuple, list)):
            return [visit(child) for child in item]
        return item

    return visit(value)


def restore_fixture(value):
    """Restore the explicit JSON spelling before the production owner parses a fixture."""
    buffers = []

    def visit(item):
        if isinstance(item, dict):
            if set(item) == {"npy"}:
                value = item["npy"]
                if isinstance(value, dict):
                    return {"npy": buffers[value["buffer"]]}
                payload = bytes(value)
                buffers.append(payload)
                return {"npy": payload}
            return {key: visit(child) for key, child in item.items()}
        if isinstance(item, list):
            return [visit(child) for child in item]
        return item

    return visit(value)
