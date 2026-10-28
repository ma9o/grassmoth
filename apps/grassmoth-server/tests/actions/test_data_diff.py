"""Stored histories compare symmetrically without generation or scientific state changes."""

from datetime import UTC, date, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from grassmoth.artifacts.arrays import NumericalArray
from grassmoth.artifacts.identity import GitOid
from grassmoth.artifacts.observations.comparison import Dataset
from grassmoth.artifacts.observations.dataset import ObservationDataset
from grassmoth.artifacts.observations.preparation import PreparedDataMetadata
from grassmoth.artifacts.observations.reference import DataRef
from grassmoth.artifacts.observations.semantics import AnchorPolicy
from grassmoth.artifacts.observations.spec import (
    LabeledMeasurement,
    NumericMeasurement,
    ResolvedObservationSpec,
)
from grassmoth.artifacts.reports.checks import NotEvaluated
from grassmoth.artifacts.reports.posterior_diagnostics import PosteriorPredictiveChecks
from grassmoth.artifacts.reports.predictive_provenance import AuthoredLawProvenance
from grassmoth.artifacts.simulation import (
    SimulationArm,
    SimulationEvidence,
    SimulationObservationLayout,
    SimulationReport,
    SimulationSpec,
    SingleArmSimulation,
)
from grassmoth.models.diagnostics.history_comparison import compare_data_variables
from grassmoth_server.actions.contracts.effects import ActionEffects
from grassmoth_server.actions.contracts.inputs import DataDiffInput, SimulateInput
from grassmoth_server.actions.contracts.requests import DataDiffRequest, SimulateRequest
from grassmoth_server.actions.data_diff import read_data_diff
from grassmoth_server.study.errors import StudyLookupError
from grassmoth_server.study.persistence.history import StudyRepository
from grassmoth_server.study.persistence.store import ArtifactStore
from grassmoth_server.study.records import Applied
from grassmoth_server_test_support.action import applied_record, empty_simulation_summary
from grassmoth_server_test_support.git import artifact_revisions
from grassmoth_test_support.git import git_oid


def _dataset(values, *, number=1, times=None, variable=None):
    variable = variable or ResolvedObservationSpec(
        id="indicator:y",
        name="Y",
        measurement=NumericMeasurement(dtype="continuous"),
        aggregation="last",
        observation_window="1d",
    )
    times = list(range(len(values))) if times is None else times
    origin = datetime(1970, 1, 1, tzinfo=UTC)
    anchors = [origin + timedelta(days=day) for day in times]
    width = timedelta(days=1) if variable.support_kind.value == "interval" else timedelta()
    frame = pl.DataFrame(
        {
            "indicator_id": [variable.id] * len(values),
            "value": values,
            "anchor_time": anchors,
            "support_start": [time - width for time in anchors],
            "support_end": anchors,
            "support_kind": [variable.support_kind.value] * len(values),
            "summary_operator": [variable.summary_operator.value] * len(values),
            "anchor_policy": [variable.anchor_policy.value] * len(values),
            "observation_window": [variable.observation_window.source] * len(values),
        },
        schema_overrides={"value": pl.Float64},
    )
    return Dataset.from_observations(
        DataRef[GitOid, int](replicate_index=0, revision=git_oid(number)),
        ObservationDataset.from_frame(frame, (variable,), time_origin=origin),
    )


@pytest.mark.contract
def test_selection_contracts_require_nonempty_explicit_sources():
    source = {"revision": git_oid(1)}
    request = DataDiffRequest[GitOid](
        input=DataDiffInput[GitOid](left_ref=(source,), right_ref=(source,))
    )
    assert DataDiffRequest[GitOid].model_validate_json(request.model_dump_json()) == request
    for invalid in (
        [],
        source,
        [{**source, "replicate": 0}],
        [{**source, "time_origin": "2026-01-01T00:00:00Z"}],
    ):
        with pytest.raises(ValidationError):
            DataDiffRequest[GitOid].model_validate(
                {"action": "data_diff", "input": {"left_ref": invalid, "right_ref": [source]}}
            )
    for invalid in ({"replicate_index": -1}, {"time_origin": "2026-01-01"}):
        with pytest.raises(ValidationError):
            DataRef[GitOid, int | None].model_validate({**source, **invalid})


@pytest.mark.inference(concern="predictive")
def test_single_histories_report_values_missingness_and_schedule_changes():
    left = _dataset([1, None, 3], times=[0, 1, 2])
    right = _dataset([1, 4, 8], times=[0, 1, 3], number=2)
    variable = compare_data_variables((left,), (right,))[0]
    assert variable.predictive is None
    assert [item.kind for item in variable.changes] == ["revised", "removed", "added"]
    revised = variable.changes[0]
    assert revised.kind == "revised"
    assert revised.before.value is None
    assert revised.after.value == 4
    assert [finding.code for finding in variable.findings] == ["observation_schedule"]
    missing = next(item for item in variable.statistics if item.statistic == "missing_count")
    assert (missing.left, missing.right) == ((1,), (0,))
    reverse = compare_data_variables((right,), (left,))[0]
    assert [item.kind for item in reverse.changes] == ["revised", "added", "removed"]
    assert compare_data_variables((left,), (left,))[0].changes == ()


@pytest.mark.inference(concern="predictive")
def test_replica_checks_are_symmetric_and_preserve_whole_history_statistics():
    observed = _dataset([0, 1, None, 3])
    replicas = [
        _dataset([offset, 1 + offset, 1000, 3 + offset, 999], number=offset + 2)
        for offset in range(3)
    ]
    forward = compare_data_variables(replicas, (observed,))[0]
    backward = compare_data_variables((observed,), replicas)[0]
    assert forward.predictive is not None
    assert backward.predictive is not None
    assert forward.predictive.reference_side == "right"
    assert backward.predictive.reference_side == "left"
    assert forward.predictive.evaluation == backward.predictive.evaluation
    checks = forward.predictive.evaluation
    assert isinstance(checks, PosteriorPredictiveChecks)
    stats = {item.stat_name: item for item in checks.test_stats}
    assert stats["mean"].observed_value == pytest.approx(4 / 3)
    assert stats["mean"].rep_values == pytest.approx([4 / 3, 7 / 3, 10 / 3])
    assert checks.n_subsample == 3
    assert {item.code for item in forward.findings} == {"observation_schedule"}
    many = compare_data_variables(replicas, [observed, _dataset([0, 1, 2, 3], number=9)])[0]
    assert many.predictive is None
    means = next(item for item in many.statistics if item.statistic == "mean")
    assert len(means.left) == 3
    assert len(means.right) == 2


@pytest.mark.inference(concern="predictive")
@pytest.mark.parametrize("aggregation", ["first", "last", "mean", "sum", "count", "std"])
def test_predictive_support_follows_the_model_observation_semantics(aggregation):
    variable = ResolvedObservationSpec(
        id="indicator:y",
        name="Y",
        measurement=NumericMeasurement(dtype="count" if aggregation == "count" else "continuous"),
        aggregation=aggregation,
        observation_window="1d",
    )
    observed = _dataset([1, 2, 3], variable=variable, times=[1, 2, 3])
    series = observed.series[variable.id]
    width = timedelta(days=1)
    points = tuple(
        point.revised(support_end=point.anchor_time + width)
        if variable.anchor_policy == AnchorPolicy.SUPPORT_START
        else point.revised(
            support_start=point.anchor_time
            - width * (2 if variable.requires_interval_summary_measurement else 1)
        )
        for point in series.points
    )
    observed = observed.revised(series={variable.id: series.revised(points=points)})
    replicas = tuple(
        _dataset([1, 2, 3], number=n, variable=variable, times=[1, 2, 3]) for n in (2, 3)
    )
    forward = compare_data_variables((observed,), replicas)[0]
    reverse = compare_data_variables(replicas, (observed,))[0]
    if variable.requires_interval_summary_measurement:
        assert forward.predictive is None
        assert reverse.predictive is None
        assert {finding.code for finding in forward.findings} == {
            "observation_schedule",
            "observation_support",
        }
        assert forward.findings == reverse.findings
    else:
        assert forward.predictive is not None
        assert reverse.predictive is not None
        assert isinstance(forward.predictive.evaluation, PosteriorPredictiveChecks)
        assert forward.predictive.evaluation == reverse.predictive.evaluation
        assert forward.findings == reverse.findings == ()


@pytest.mark.inference(concern="predictive")
def test_measurement_mismatches_and_uncovered_times_do_not_produce_predictive_checks():
    observed = _dataset([1, 2, 3])
    replicas = [_dataset([1, 2, 3], number=n, times=[0, 1, 2.5]) for n in (2, 3)]
    mismatch = compare_data_variables((observed,), replicas)[0]
    assert mismatch.predictive is None
    assert {finding.code for finding in mismatch.findings} == {
        "observation_schedule",
        "observation_support",
    }
    missing = compare_data_variables(
        (observed,), [_dataset([1, None, 3], number=n) for n in (2, 3)]
    )[0]
    assert missing.predictive is not None
    assert isinstance(missing.predictive.evaluation, NotEvaluated)
    assert missing.predictive.evaluation.reason == "MISSING_REPLICATE_VALUES"
    assert missing.findings == ()
    original = next(iter(observed.series.values())).variable
    replicas = [
        _dataset([1, 2, 3], number=n, variable=original.revised(aggregation="mean")) for n in (2, 3)
    ]
    mismatch = compare_data_variables((observed,), replicas)[0]
    assert mismatch.predictive is None
    assert "measurement_definition" in {finding.code for finding in mismatch.findings}
    renamed = _dataset(
        [1, 2, 3], variable=original.revised(name="Renamed", observation_window="24h")
    )
    assert not compare_data_variables((observed,), (renamed,))[0].findings
    first = _dataset([1, 2, 3], variable=original.revised(aggregation="first"))
    assert compare_data_variables((first,), (first,))[0].changes == ()
    with pytest.raises(ValidationError):
        observed.revised(time_origin=None)


@pytest.mark.inference(concern="predictive")
def test_discrete_codebooks_compare_frequencies_and_keep_absent_variables_explicit():
    variable = ResolvedObservationSpec(
        id="indicator:category",
        name="Category",
        measurement=LabeledMeasurement(dtype="categorical", levels=("a", "b", "c")),
        aggregation="last",
        observation_window="1d",
    )
    a = _dataset([0, 1, 1], variable=variable)
    b = [_dataset([2, 2, 1], number=n, variable=variable) for n in (2, 3)]
    result = compare_data_variables((a,), b)[0]
    assert result.predictive is None
    proportions = {item.level: item for item in result.statistics if item.statistic == "proportion"}
    assert proportions["c"].left == (0,)
    assert proportions["c"].right == pytest.approx((2 / 3, 2 / 3))
    assert not any(item.statistic == "mean" for item in result.statistics)
    result = compare_data_variables((a,), (_dataset([1, 2, 3], number=4),))
    assert len(result) == 2
    assert all(item.findings[0].code == "indicator_presence" for item in result)
    for item in result:
        count = next(stat for stat in item.statistics if stat.statistic == "observed_count")
        assert (None,) in (count.left, count.right)


@pytest.mark.contract
def test_invalid_histories_and_duplicate_sources_are_rejected_before_comparison():
    data = _dataset([1, 2, 3])
    with pytest.raises(ValueError, match="at least one"):
        compare_data_variables([], (data,))
    with pytest.raises(ValueError, match="counted twice"):
        compare_data_variables([data, data], (data,))
    with pytest.raises(ValueError, match="duplicate anchors"):
        compare_data_variables((_dataset([1, 2], times=[0, 0]),), (data,))
    with pytest.raises(ValueError, match="finite"):
        compare_data_variables((_dataset([1, np.inf]),), (data,))


@pytest.mark.inference(concern="predictive")
def test_reads_saved_draws_without_generation_or_writing_models(tmp_path, monkeypatch):
    from grassmoth.artifacts.observations.preparation import (
        DataPreparationSpec,
        DataVariableSpec,
        SemanticExtractionSpec,
    )
    from grassmoth_server.storage import paths as data_module
    from grassmoth_test_support.helpers import make_model

    monkeypatch.setattr(data_module, "_DATA_URI", str(tmp_path))
    store, history = ArtifactStore("DIFF"), StudyRepository("DIFF")
    model = store.write_artifact(
        "model",
        produced_by="edit_model",
        derived_from={},
        json_files={"model.json": make_model(["X", "Y"], [("X", "Y")]).model_dump(mode="json")},
    )
    observed = _dataset([1, 2, 3])
    values = np.asarray([[[0], [1], [2]], [[1], [2], [3]], [[2], [3], [4]]], dtype=float)
    report = SimulationReport(
        summary=empty_simulation_summary(),
        fit_reliability="not_fitted",
        law=AuthoredLawProvenance(),
        evidence=SimulationEvidence(
            time_origin=datetime(2026, 1, 1, tzinfo=UTC),
            times=(0, 1, 2),
            draws=3,
            seed=0,
            state_ids=(),
            parameter_draws={},
            arms=SingleArmSimulation(
                action=SimulationArm(
                    latent_paths=NumericalArray.from_numpy(np.zeros((3, 3, 0))),
                    observations=NumericalArray.from_numpy(values),
                ),
            ),
            observation_layout=SimulationObservationLayout(
                variables=tuple(series.variable for series in observed.series.values()),
                support_start_times=NumericalArray.from_numpy(np.arange(3.0)[:, None]),
                support_end_times=NumericalArray.from_numpy(np.arange(3.0)[:, None]),
                mask=NumericalArray.from_numpy(np.ones_like(values, dtype=bool)),
            ),
            assignments=SimulationSpec(start=date(2026, 1, 1), horizon="2d").assignments(
                datetime(2026, 1, 1, tzinfo=UTC)
            ),
        ),
    )
    commit = history.append(
        applied_record(
            store.workspace_id,
            Applied(
                result=report.evidence,
                effects=ActionEffects(reports={"simulation": store.write_result(report)}),
            ),
            request=SimulateRequest(
                input=SimulateInput(
                    dynamical_model_spec_ref=model.revision,
                    simulation=SimulationSpec(start="2026-01-01", horizon="2d"),
                )
            ),
            seq=1,
            ts="2026-09-28T00:00:00Z",
            trace_ids=[],
        )
    ).commit_id
    origin = datetime(2026, 1, 1, tzinfo=UTC)
    shifted = pl.DataFrame(
        {
            "indicator_id": ["indicator:y"] * 3,
            "value": [1.0, 2.0, 3.0],
            "anchor_time": [origin + timedelta(days=n) for n in range(3)],
            "support_start": [origin + timedelta(days=n) for n in range(3)],
            "support_end": [origin + timedelta(days=n) for n in range(3)],
            "support_kind": ["point"] * 3,
            "summary_operator": ["last"] * 3,
            "anchor_policy": ["support_end"] * 3,
            "observation_window": ["1d"] * 3,
        }
    )
    panel = store.write_artifact(
        "panel",
        produced_by="prepare_data",
        derived_from={},
        json_files={
            "metadata.json": PreparedDataMetadata(
                preparation=DataPreparationSpec(
                    default_window="1d",
                    variables=tuple(
                        DataVariableSpec(
                            observation=series.variable,
                            extraction=SemanticExtractionSpec(how_to_measure="Read Y"),
                        )
                        for series in observed.series.values()
                    ),
                ),
                time_origin=origin,
            ).model_dump(mode="json")
        },
        parquet_files={"panel.parquet": shifted},
    )
    request = DataDiffRequest[GitOid](
        input=DataDiffInput[GitOid](
            left_ref=(DataRef[GitOid, int | None](replicate_index=None, revision=commit),),
            right_ref=(DataRef[GitOid, int](replicate_index=0, revision=panel.revision),),
        )
    )
    refs_before = sorted(store.repo.references)
    output = read_data_diff("DIFF", request).model_dump(mode="json")
    assert set(output) == {"report"}
    result = output["report"]
    assert len(result["left"]) == 3
    assert result["left"] == [{"revision": commit, "replicate_index": index} for index in range(3)]
    assert result["variables"][0]["predictive"]["evaluation"]["n_subsample"] == 3
    assert history.head() == commit
    assert sorted(store.repo.references) == refs_before
    assert artifact_revisions(store, "model") == [model.revision]
    assert len(request.input.left_ref) == 1
    one = request.revised(
        input=request.input.revised(
            left_ref=(request.input.left_ref[0].revised(replicate_index=1),)
        )
    )
    assert read_data_diff("DIFF", one).report.variables[0].changes == ()
    bad = request.revised(
        input=request.input.revised(
            left_ref=(DataRef[GitOid, int | None](revision=commit, replicate_index=3),)
        )
    )
    with pytest.raises(StudyLookupError, match="replicate"):
        read_data_diff("DIFF", bad)
    bad = request.revised(
        input=request.input.revised(
            right_ref=(DataRef[GitOid, int](replicate_index=0, revision=git_oid(98)),)
        )
    )
    with pytest.raises(StudyLookupError):
        read_data_diff("DIFF", bad)

    # Saved histories keep absolute model days, and materialization keeps their dates.
    from grassmoth_server.study.queries.data import read_simulation_observations

    origin = datetime(2026, 1, 1, tzinfo=UTC)
    commits = []
    for start in (5, 6):
        times = np.arange(start, start + 3.0)
        saved = report.evidence.revised(
            time_origin=origin,
            times=tuple(times),
            observation_layout=report.evidence.observation_layout.revised(
                support_start_times=NumericalArray.from_numpy(times[:, None]),
                support_end_times=NumericalArray.from_numpy(times[:, None]),
            ),
        )
        commits.append(
            history.append(
                applied_record(
                    store.workspace_id,
                    Applied(
                        result=saved,
                        effects=ActionEffects(
                            reports={
                                "simulation": store.write_result(report.revised(evidence=saved))
                            }
                        ),
                    ),
                    request=SimulateRequest(
                        input=SimulateInput(
                            dynamical_model_spec_ref=model.revision,
                            simulation=SimulationSpec(start=date(2026, 1, 1 + start), horizon="2d"),
                        )
                    ),
                    seq=2 + start - 5,
                    ts="2026-09-28T00:00:00Z",
                    trace_ids=[],
                )
            ).commit_id
        )
    aligned = read_data_diff(
        "DIFF",
        DataDiffRequest[GitOid](
            input=DataDiffInput[GitOid](
                left_ref=(DataRef[GitOid, int | None](revision=commits[0], replicate_index=1),),
                right_ref=(DataRef[GitOid, int | None](revision=commits[1], replicate_index=1),),
            )
        ),
    ).report.variables[0]
    assert [
        (
            (change.before.anchor_time if change.kind == "removed" else change.after.anchor_time),
            change.kind,
        )
        for change in aligned.changes
    ] == [
        (origin + timedelta(days=5), "removed"),
        (origin + timedelta(days=6), "revised"),
        (origin + timedelta(days=7), "revised"),
        (origin + timedelta(days=8), "added"),
    ]
    materialized = read_simulation_observations(saved, 1)
    assert materialized["anchor_time"][0] == (origin + timedelta(days=6)).replace(tzinfo=None)
