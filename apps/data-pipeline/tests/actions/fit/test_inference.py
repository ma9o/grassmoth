"""Tests for Stage 5 inference task logging and orchestration."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from nof1_causal_lab.actions.fit import execution as stage5_inference
from nof1_causal_lab.artifacts.joint_chart import CentredChartRecord
from nof1_causal_lab.artifacts.model.construct import replace_constructs
from nof1_causal_lab.artifacts.observations.dataset import ObservationDataset
from nof1_causal_lab.artifacts.sampling import SamplerSpec
from nof1_causal_lab.models.ssm.compile import inputs as compilation
from nof1_causal_lab.models.ssm.inference.types import JointPosteriorDraws
from nof1_causal_lab.models.ssm.preparation import panel as runtime_module
from nof1_causal_lab.models.structure import StructuralSelection
from tests.support.inference import (
    bind_panel_fixture,
    compile_fit_fixture,
    make_observation_support_runtime,
)
from tests.support.model import load_model_fixture


def _sleep_state_model() -> DynamicalModelSpec:
    dynamical_model_spec = load_model_fixture("inference/sleep_state_model.json")
    first = dynamical_model_spec.constructs[0]
    indicator = first.indicators[0]
    return dynamical_model_spec.with_entities(
        edges=replace_constructs(
            dynamical_model_spec.edges,
            (
                first.revised(
                    indicators=(
                        indicator.revised(
                            observation=indicator.observation.revised(aggregation="mean")
                        ),
                        *first.indicators[1:],
                    )
                ),
            ),
        )
    )


if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.model.dynamical_model_spec import DynamicalModelSpec
    from nof1_causal_lab.models.ssm.preparation.observations import ObservationSupportRuntime


pytestmark = pytest.mark.contract


def _fake_result():
    from tests.support.inference import particle_posterior

    return particle_posterior(
        JointPosteriorDraws(parameters={"theta": jnp.zeros((4, 1), dtype=jnp.float32)})
    )


def _make_observation_support_runtime() -> ObservationSupportRuntime:
    return make_observation_support_runtime(
        manifest_names=("manifest_0", "manifest_1"),
        anchor_times=np.array([0.0, 1.5]),
        support_kinds=("interval", "point"),
        summary_operators=("mean", "last"),
        anchor_policies=("support_end", "support_end"),
        observation_windows=("1d", "1d"),
        support_start_times=np.array([[np.nan, 0.0], [0.0, 1.5]]),
        support_end_times=np.array([[np.nan, 0.0], [1.5, 1.5]]),
    )


def _make_panel(inputs: CompiledFitInputs) -> BoundPanel:
    return bind_panel_fixture(
        inputs.compiled_dynamical_model,
        jnp.array([[jnp.nan, 0.8], [0.2, 0.5]], dtype=jnp.float32),
        jnp.array([0.0, 1.5], dtype=jnp.float32),
        support=_make_observation_support_runtime(),
    )


def test_fit_model_logs_runtime_summary_and_diagnostic_boundaries(monkeypatch, caplog):
    fake_result = _fake_result()
    dynamical_model_spec = _sleep_state_model()
    fake_model = compile_fit_fixture(dynamical_model_spec)
    runtime = _make_panel(fake_model)

    monkeypatch.setattr(compilation, "compile_ssm_inputs_from_model", lambda _selection: fake_model)
    monkeypatch.setattr(runtime_module, "bind_panel", lambda *_args, **_kwargs: runtime)
    monkeypatch.setattr(
        stage5_inference, "fit_prepared_model", lambda _prepared, **_kwargs: fake_result
    )

    data_for_model = ObservationDataset.from_frame(
        pl.DataFrame(runtime.rows),
        tuple(item.observation for item in runtime.compiled_dynamical_model.observations),
        time_origin=runtime.time_origin,
    )

    with caplog.at_level(logging.INFO, logger=stage5_inference.logger.name):
        result = stage5_inference.fit_model(
            StructuralSelection(dynamical_model_spec, None),
            data_for_model,
            time_origin=datetime(2024, 1, 1, tzinfo=UTC),
            sampler=SamplerSpec(),
        )

    assert result["fitted"]
    assert result["result"] is fake_result
    assert result["panel"] is runtime
    assert "Prepared runtime in" in caplog.text
    assert "support=interval(1: manifest_0) max_active_windows=1" in caplog.text
    assert "Manifest order: manifest_0, manifest_1" in caplog.text
    assert "Starting inference kernel..." in caplog.text
    assert "Retaining native posterior and sampler telemetry" in caplog.text
    assert "inference_diagnostics" not in result


if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.compile.inputs import CompiledFitInputs
    from nof1_causal_lab.models.ssm.preparation.panel import BoundPanel


def test_continued_fit_appends_within_chains_and_zero_draws_keep_existing_evidence(monkeypatch):
    from nof1_causal_lab.actions.contracts.bodies import FitOutput
    from nof1_causal_lab.actions.fit.action import fit
    from nof1_causal_lab.artifacts.arrays import NumericalArray
    from nof1_causal_lab.artifacts.continuation import JointTuning, SamplerCheckpoint
    from nof1_causal_lab.artifacts.probability.serialization import empirical_atoms
    from nof1_causal_lab.artifacts.reports.data_quality import DataProfileReport
    from nof1_causal_lab.artifacts.reports.inference import (
        FitCheckReport,
        InferenceReport,
        InferenceReportCore,
        InferenceReportDetail,
    )
    from nof1_causal_lab.artifacts.reports.model_checks import QuestionCheckReport
    from nof1_causal_lab.artifacts.reports.posterior_diagnostics import MCMCEvidence
    from nof1_causal_lab.models.ssm.inference.diagnostics.convergence import chain_convergence
    from nof1_causal_lab.models.ssm.inference.mcmc_state import TrajectoryMCMCResult
    from nof1_causal_lab.models.ssm.inference.types import (
        MCMCPosterior,
        ProductionDiagnostics,
        UnretainedRun,
    )
    from tests.support.inference import parameter_draws

    model = _sleep_state_model()
    panel = _make_panel(compile_fit_fixture(model))
    dataset = ObservationDataset.from_frame(
        pl.DataFrame(panel.rows),
        tuple(item.observation for item in panel.compiled_dynamical_model.observations),
        time_origin=panel.time_origin,
    )
    settings = SamplerSpec(num_chains=2, num_warmup=0, num_samples_per_chain=2, hardware="local")
    arrays = {}

    def write(values):
        from nof1_causal_lab.storage.arrays import encode_array

        identity, _ = encode_array(np.asarray(values))
        arrays[identity] = values
        return identity

    def posterior(count, offset):
        parameters = {
            name: values + offset for name, values in parameter_draws(model, 2 * count).items()
        }
        paths = (
            jnp.arange(
                2 * count * 2 * len(panel.compiled_dynamical_model.states), dtype=jnp.float32
            ).reshape(2 * count, 2, -1)
            + offset
        )
        checkpoint = SamplerCheckpoint(
            settings=settings.revised(num_samples_per_chain=count),
            parameters=NumericalArray.from_numpy(np.zeros((2, 1))),
            paths=NumericalArray.from_numpy(np.asarray(paths.reshape(2, count, 2, -1)[:, -1])),
            random_key=NumericalArray.from_numpy(np.array([0, offset], dtype=np.uint32)),
            native_state=(),
            tuning=JointTuning(chart=CentredChartRecord()),
        )
        return MCMCPosterior.from_run(
            draws=JointPosteriorDraws(parameters, paths),
            diagnostics=ProductionDiagnostics(
                checkpoint=checkpoint,
                mcmc=TrajectoryMCMCResult(
                    chain_samples={
                        name: values.reshape(2, count, *values.shape[1:])
                        for name, values in parameters.items()
                    },
                    chain_extra_fields={},
                    num_chains=2,
                    num_samples=count,
                ),
                observation_log_probs=jnp.full((2, count, 2), offset),
            ),
        )

    def execute(result, count, previous=None):
        monkeypatch.setattr(
            stage5_inference,
            "fit_model",
            lambda *_args, **_kwargs: {
                "fitted": True,
                "duration_seconds": 1.0,
                "result": result,
                "panel": panel,
            },
        )
        return fit(
            selection=StructuralSelection(model, None),
            data_for_model=dataset,
            time_origin=panel.time_origin,
            sampler=settings.revised(num_samples_per_chain=count),
            array_writer=write,
            array_loader=arrays.__getitem__,
            compute_loo_diagnostics=True,
            previous=previous,
        )

    def output(result):
        return FitOutput(
            dynamical_model_spec=result["_dynamical_model_spec"],
            checks=FitCheckReport(
                data=DataProfileReport(indicators={}, findings=()),
                preflight=(),
                question=QuestionCheckReport(findings=()),
            ),
            inference=InferenceReport(
                evidence=result["evidence"],
                core=InferenceReportCore(
                    inference_metadata=result["metadata"],
                    inference_diagnostics=None,
                    convergence=chain_convergence(None),
                    posterior_marginals=(),
                    prior_densities={},
                ),
                detail=InferenceReportDetail(),
            ),
        )

    first = execute(posterior(2, 1), 2)
    first["evidence"] = first["evidence"].revised(
        phase_extra_fields={
            "initialization": {"position": NumericalArray.from_numpy(np.array([1.0, 2.0]))}
        }
    )
    standalone = execute(posterior(1, 5), 1)
    combined = execute(posterior(1, 5), 1, output(first))
    identity = first["metadata"].distribution
    old = empirical_atoms(first["_dynamical_model_spec"].distributions[identity]).reshape(2, 2, -1)
    new = empirical_atoms(standalone["_dynamical_model_spec"].distributions[identity]).reshape(
        2, 1, -1
    )
    atoms = empirical_atoms(combined["_dynamical_model_spec"].distributions[identity]).reshape(
        2, 3, -1
    )
    np.testing.assert_array_equal(atoms, np.concatenate((old, new), axis=1))
    assert combined["metadata"].num_samples_total == 6
    assert [(segment.start, segment.stop) for segment in combined["metadata"].segments] == [
        (0, 2),
        (2, 3),
    ]
    np.testing.assert_array_equal(
        combined["evidence"].observation_log_probs.values[0, :, 0], [1, 1, 5]
    )
    from nof1_causal_lab.actions.fit import remote
    from nof1_causal_lab.artifacts.sampling import InferenceHardware
    from nof1_causal_lab.study.persistence.codec import unpack_result

    def dispatch_without_history_buffers(payload, *, hardware):
        assert hardware is InferenceHardware.L40S
        prior = FitOutput.model_validate(unpack_result(payload.previous))
        assert not prior.inference.evidence.phase_extra_fields
        return remote.execute_fit_compute(payload)

    monkeypatch.setattr(remote, "_dispatch_fit", dispatch_without_history_buffers)

    def remotely(count, previous):
        return remote.fit_on_modal(
            InferenceHardware.L40S,
            selection=StructuralSelection(model, None),
            data_for_model=dataset,
            time_origin=panel.time_origin,
            sampler=settings.revised(num_samples_per_chain=count),
            array_writer=write,
            array_loader=arrays.__getitem__,
            compute_loo_diagnostics=True,
            previous=previous,
        )

    transported = remotely(1, output(first))
    np.testing.assert_array_equal(
        empirical_atoms(transported["_dynamical_model_spec"].distributions[identity]),
        empirical_atoms(combined["_dynamical_model_spec"].distributions[identity]),
    )
    np.testing.assert_array_equal(
        transported["evidence"].phase_extra_fields["previous/initialization"]["position"].values,
        [1.0, 2.0],
    )
    from unittest.mock import Mock

    from jax.stages import Compiled

    zero = execute(
        UnretainedRun(settings, MCMCEvidence(), Mock(spec=Compiled), {}), 0, output(combined)
    )
    assert zero["_dynamical_model_spec"] is None
    assert zero["metadata"].num_samples_total == 6
    assert zero["metadata"].segments[-1].start == zero["metadata"].segments[-1].stop == 3
    assert zero["evidence"].checkpoint == combined["evidence"].checkpoint
    np.testing.assert_array_equal(
        zero["evidence"].observation_log_probs.values,
        combined["evidence"].observation_log_probs.values,
    )
    remote_zero = remotely(0, output(combined))
    assert remote_zero["_dynamical_model_spec"] is None
    assert remote_zero["metadata"].num_samples_total == 6
    np.testing.assert_array_equal(
        remote_zero["evidence"]
        .phase_extra_fields["previous/previous/initialization"]["position"]
        .values,
        [1.0, 2.0],
    )
