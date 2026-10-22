"""Resolve simulation provenance and delegate the calculation to the library."""

from grassmoth.public import (
    FittedLawProvenance,
    GitOid,
    MixedLawProvenance,
    ObservationPreflightFailure,
    SimulateOutput,
    StructuralSelection,
    simulate,
)
from grassmoth_server.actions.contracts.inputs import SimulateInput
from grassmoth_server.bootstrap import initialize_numerical_runtime
from grassmoth_server.compute.resources import read_compute_budget
from grassmoth_server.study.persistence.history import StudyRepository
from grassmoth_server.study.persistence.store import (
    ArtifactStore,
    model_values,
    read_model,
    read_question,
)
from grassmoth_server.study.queries.lineage import fitted_law_report, law_provenance
from grassmoth_server.study.records import inference_record


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
