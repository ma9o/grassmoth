"""Inference evidence and attempt history for server tests."""

from typing import TYPE_CHECKING

from grassmoth.artifacts.probability.serialization import empirical_atoms
from grassmoth_server.actions.contracts.effects import ActionEffects
from grassmoth_server.study.records import Applied
from grassmoth_server.study.state import ArtifactRecord
from grassmoth_test_support.git import git_oid

if TYPE_CHECKING:
    from grassmoth.artifacts.identity import GitOid
    from grassmoth_server.actions.contracts.identity import ArtifactId

_PRIOR_REVISION, _FITTED_REVISION = git_oid(1), git_oid(2)


def inference_log(
    model,
    *,
    revision=_FITTED_REVISION,
    prior_revision=_PRIOR_REVISION,
    seq=3,
    workspace_id="workspace",
):
    pins: dict[ArtifactId, GitOid] = {"model": prior_revision, "panel": git_oid(1)}
    from grassmoth_server.study.records import AttemptRecord, FitAttempt, StudyRevision

    record = AttemptRecord(
        seq=seq,
        ts="2026-07-03T00:00:00+00:00",
        attempt=FitAttempt(
            action="fit",
            request=None,
            outcome=Applied(
                result=git_oid(500 + seq),
                effects=ActionEffects(
                    produced=(
                        ArtifactRecord(
                            artifact_id="model",
                            revision=revision,
                            produced_by="fit",
                            derived_from=pins,
                        ),
                    ),
                ),
            ),
        ),
    )
    return StudyRevision(commit_id=git_oid(100 + seq), parent_ids=(), record=record)


def inference_metadata(model):
    """Metadata for an explicit test-owned joint law with one retained chain."""
    from grassmoth.artifacts.reports.inference import InferenceMetadata
    from grassmoth.artifacts.reports.posterior_diagnostics import MCMCEvidence

    identity = next(iter(model.law_layouts))
    return InferenceMetadata(
        distribution=identity,
        engine=MCMCEvidence(),
        duration_seconds=0,
        num_samples_total=len(empirical_atoms(model.distributions[identity])),
        num_chains=1,
        sampler_diagnostics=None,
    )
