"""Inference lineage is a journal query, never a field of the scientific model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import TypeAdapter

from nof1_causal_lab.artifacts.observations.preparation import PreparedDataMetadata
from nof1_causal_lab.artifacts.reports.predictive_provenance import (
    AuthoredLawProvenance,
    FittedLawProvenance,
    MixedLawProvenance,
    PredictiveLawProvenance,
    UnknownLawProvenance,
)
from nof1_causal_lab.study.artifact_files import json_filename
from nof1_causal_lab.study.persistence.store import read_model
from nof1_causal_lab.study.records import StudyRevision, inference_record

if TYPE_CHECKING:
    from collections.abc import Iterable

    from nof1_causal_lab.artifacts.identity import GitOid
    from nof1_causal_lab.artifacts.model.dynamical_model_spec import DynamicalModelSpec
    from nof1_causal_lab.artifacts.observations.reference import DataRef
    from nof1_causal_lab.artifacts.reports.inference import InferenceReportCore
    from nof1_causal_lab.study.persistence.store import ArtifactStore
    from nof1_causal_lab.study.state import ArtifactRecord


def read_data_metadata(store: ArtifactStore, revision: GitOid) -> PreparedDataMetadata:
    """Parse the preparation metadata owned by an exact panel artifact revision."""
    return TypeAdapter[PreparedDataMetadata](PreparedDataMetadata).validate_python(
        store.read_json_file("panel", revision, json_filename("panel", "metadata"))
    )


def fitted_law_report(
    store: ArtifactStore, records: Iterable[StudyRevision], revision: GitOid
) -> InferenceReportCore:
    """Read the committed fit that owns inherited laws and their model coordinates."""
    fitted = inference_record(records, revision)
    if fitted is None:
        raise ValueError("Fitted model laws require their committed inference report")
    assert fitted.record.attempt.action == "fit"
    assert fitted.record.attempt.outcome.status == "applied"
    from nof1_causal_lab.actions.contracts.bodies import FitOutput

    result = store.read_result(fitted.record.attempt.outcome.result, FitOutput)
    return result.inference.core


def law_provenance(
    store: ArtifactStore,
    record: ArtifactRecord,
    dynamical_model_spec: DynamicalModelSpec,
    data: DataRef[GitOid, int] | None,
) -> PredictiveLawProvenance:
    """Follow authored ancestry; a native law family alone never establishes fitting."""
    uncertain = {
        quantity.distribution
        for quantity in (
            *dynamical_model_spec.parameters,
            *(
                construct
                for construct in dynamical_model_spec.constructs
                if construct.role == "endogenous"
            ),
        )
        if quantity.distribution is not None
    }
    laws = dynamical_model_spec.model_dump(mode="json")["distributions"]
    current = record
    while True:
        if current.produced_by == "fit":
            fitted = read_model(store, current.revision).model_dump(mode="json")["distributions"]
            inherited = {
                key
                for key, value in laws.items()
                if key in uncertain
                and key in dynamical_model_spec.law_layouts
                and fitted.get(key) == value
            }
            if inherited:
                from pathlib import Path

                from nof1_causal_lab.study.persistence.history import StudyRepository
                from nof1_causal_lab.study.records import Applied

                fit = inference_record(
                    StudyRepository(
                        store.workspace_id, repository_path=Path(store.repo.path)
                    ).attempts(),
                    current.revision,
                )
                if fit is None:
                    raise ValueError("Fitted laws require their recorded data selection")
                assert fit.record.attempt.action == "fit"
                assert isinstance(fit.record.attempt.outcome, Applied)
                assert fit.record.attempt.request is not None
                fitted_data = fit.record.attempt.request.input.data_ref
                if inherited != uncertain or any(
                    fitted.get(key) != value for key, value in laws.items() if key not in uncertain
                ):
                    return MixedLawProvenance(
                        fitted_data=fitted_data,
                        fitted_model_revision=current.revision,
                    )
                return FittedLawProvenance(
                    fitted_data=fitted_data,
                    fitted_model_revision=current.revision,
                    interpretation=(
                        "in_sample_posterior_predictive"
                        if fitted_data == data
                        # A different panel revision does not prove held-out observations.
                        else "posterior_predictive"
                    ),
                )
        parent = current.derived_from.get("model")
        if parent is None:
            break
        current = store.read_meta("model", parent)
    # Imported joint laws can contain externally conditioned draws. Their family
    # does not establish a training panel, so their interpretation stays unknown.
    from nof1_causal_lab.artifacts.probability.serialization import distribution_shape

    joint = any(
        any(distribution_shape(dynamical_model_spec.distributions[key])) for key in uncertain
    )
    return UnknownLawProvenance() if joint else AuthoredLawProvenance()
