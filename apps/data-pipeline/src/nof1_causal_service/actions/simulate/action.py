"""Resolve simulation provenance and delegate the calculation to the library."""

from nof1_causal_lab.actions.contracts.bodies import SimulateOutput
from nof1_causal_lab.actions.simulate.action import simulate
from nof1_causal_lab.artifacts.identity import GitOid
from nof1_causal_lab.artifacts.reports.predictive_provenance import (
    FittedLawProvenance,
    MixedLawProvenance,
)
from nof1_causal_lab.models.ssm.preparation.preflight import ObservationPreflightFailure
from nof1_causal_lab.models.structure import StructuralSelection
from nof1_causal_service.actions.contracts.inputs import SimulateInput
from nof1_causal_service.bootstrap import initialize_numerical_runtime
from nof1_causal_service.compute.resources import read_compute_budget
from nof1_causal_service.study.persistence.history import StudyRepository
from nof1_causal_service.study.persistence.store import (
    ArtifactStore,
    model_values,
    read_model,
    read_question,
)
from nof1_causal_service.study.queries.lineage import fitted_law_report, law_provenance
from nof1_causal_service.study.records import inference_record


def run_simulation(
    store: ArtifactStore, request: SimulateInput[GitOid], question_revision: GitOid
) -> SimulateOutput | ObservationPreflightFailure:
    """Read the pinned model and its retained fit evidence before scientific execution."""
    initialize_numerical_runtime()
    revision = request.dynamical_model_spec_ref
    model = model_values(store, read_model(store, revision))
    selection = StructuralSelection.for_question(model, read_question(store, question_revision))
    records = StudyRepository(store.workspace_id).attempts()
    law = law_provenance(store, store.read_meta("model", revision), model, None)
    law_report = (
        fitted_law_report(store, records, law.fitted_model_revision)
        if isinstance(law, (FittedLawProvenance, MixedLawProvenance))
        else None
    )
    inference = inference_record(records, revision)
    fitted = (
        None
        if inference is None
        else (
            store.dynamical_model_spec_ref(revision),
            fitted_law_report(store, [inference], revision),
        )
    )
    return simulate(
        selection,
        request.simulation,
        seed=request.seed,
        num_replicates=request.num_replicates,
        budget=read_compute_budget(),
        law=law,
        law_report=law_report,
        fitted=fitted,
    )
