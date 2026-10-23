"""Tests for Stage 5 inference task logging and orchestration."""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from grassmoth.actions.fit import execution as stage5_inference
from grassmoth.actions.fit.execution import FittedModelResult
from grassmoth.artifacts.joint_parameterization import CentredParameterizationRecord
from grassmoth.artifacts.model.construct import replace_constructs
from grassmoth.artifacts.observations.dataset import ObservationDataset
from grassmoth.artifacts.sampling import SamplerSpec
from grassmoth.compute.budget import ComputeBudget, SharedMemoryBudget
from grassmoth.models.ssm.compile import inputs as compilation
from grassmoth.models.ssm.inference.types import JointPosteriorDraws
from grassmoth.models.ssm.preparation import panel as runtime_module
from grassmoth.models.structure import StructuralSelection
from grassmoth_server.actions.fit.runtime import fit_runtime
from grassmoth_server.compute.compilation_cache import compile_staged, prepare_program
from grassmoth_server_test_support.model import load_study_model_fixture
from grassmoth_test_support.inference import (
    bind_panel_fixture,
    compile_fit_fixture,
    make_observation_support_runtime,
)


def _sleep_state_model() -> DynamicalModelSpec:
    dynamical_model_spec = load_study_model_fixture("inference/sleep_state_model.json")
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
    from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec
    from grassmoth.models.ssm.preparation.observations import ObservationSupportRuntime


pytestmark = pytest.mark.contract


def _fake_result():
    from grassmoth_test_support.inference import particle_posterior

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

    with caplog.at_level(logging.INFO, logger=__name__):
        result = stage5_inference.fit_model(
            StructuralSelection(dynamical_model_spec, None),
            data_for_model,
            time_origin=datetime(2024, 1, 1, tzinfo=UTC),
            sampler=SamplerSpec(),
            budget=ComputeBudget(memory=SharedMemoryBudget(available_bytes=2**30)),
            clock=time.monotonic,
            logger=logging.getLogger(__name__),
            wall_clock=lambda: datetime.now(UTC),
            compiler=compile_staged,
            prepare_program=prepare_program,
        )

    assert isinstance(result, FittedModelResult)
    assert result.result is fake_result
    assert result.panel is runtime
    assert "Prepared runtime in" in caplog.text
    assert "support=interval(1: manifest_0) max_active_windows=1" in caplog.text
    assert "Manifest order: manifest_0, manifest_1" in caplog.text
    assert "Starting inference kernel..." in caplog.text
    assert "Retaining native posterior and sampler telemetry" in caplog.text
    assert not hasattr(result, "inference_diagnostics")


if TYPE_CHECKING:
    from grassmoth.models.ssm.compile.inputs import CompiledFitInputs
    from grassmoth.models.ssm.preparation.binding import BoundPanel


def test_continued_fit_appends_within_chains_and_zero_draws_keep_existing_evidence(monkeypatch):
    from grassmoth.actions.contracts.bodies import FitOutput
    from grassmoth.actions.fit.action import _fit as fit
    from grassmoth.artifacts.reports.posterior_diagnostics import ChainDiagnostics

    segment_diagnostics = ChainDiagnostics(num_chains=2, num_samples_per_chain=1, per_parameter=())
    monkeypatch.setattr(
        "grassmoth.actions.fit.action.segment_diagnostics",
        lambda *_args: segment_diagnostics,
    )
    from grassmoth.artifacts.arrays import NumericalArray
    from grassmoth.artifacts.continuation import JointTuning, SamplerCheckpoint
    from grassmoth.artifacts.probability.serialization import empirical_atoms
    from grassmoth.artifacts.reports.data_quality import DataProfileReport
    from grassmoth.artifacts.reports.inference import (
        FitCheckReport,
        InferenceReport,
        InferenceReportCore,
        InferenceReportDetail,
    )
    from grassmoth.artifacts.reports.model_checks import QuestionCheckReport
    from grassmoth.artifacts.reports.posterior_diagnostics import MCMCEvidence
    from grassmoth.models.ssm.inference.diagnostics.convergence import chain_convergence
    from grassmoth.models.ssm.inference.mcmc_state import TrajectoryMCMCResult
    from grassmoth.models.ssm.inference.types import (
        MCMCPosterior,
        ProductionDiagnostics,
        UnretainedRun,
    )
    from grassmoth_test_support.inference import parameter_draws

    monkeypatch.setattr(MCMCPosterior, "get_loo_diagnostics", lambda *_args, **_kwargs: None)
    model = _sleep_state_model()
    panel = _make_panel(compile_fit_fixture(model))
    dataset = ObservationDataset.from_frame(
        pl.DataFrame(panel.rows),
        tuple(item.observation for item in panel.compiled_dynamical_model.observations),
        time_origin=panel.time_origin,
    )
    settings = SamplerSpec(num_chains=2, num_warmup=0, num_samples_per_chain=2, hardware="local")

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
            tuning=JointTuning(parameterization=CentredParameterizationRecord()),
        )
        telemetry = {
            "dsmc_weight_ess": jnp.arange(2 * count * 3, dtype=jnp.float32).reshape(2, count, 3)
            + offset,
            "dsmc_weight_entropy": jnp.full((2, count, 3), offset, dtype=jnp.float32),
        }
        return MCMCPosterior.from_run(
            draws=JointPosteriorDraws(parameters, paths),
            diagnostics=ProductionDiagnostics(
                checkpoint=checkpoint,
                mcmc=TrajectoryMCMCResult(
                    chain_samples={
                        name: values.reshape(2, count, *values.shape[1:])
                        for name, values in parameters.items()
                    },
                    chain_extra_fields=telemetry,
                    num_chains=2,
                    num_samples=count,
                ),
                observation_log_probs=jnp.full((2, count, 2), offset),
                phase_extra_fields={"post_warmup": telemetry},
            ),
        )

    def execute(result, count, previous=None):
        monkeypatch.setattr(
            stage5_inference,
            "fit_model",
            lambda *_args, **_kwargs: FittedModelResult.from_run(
                duration_seconds=1.0, result=result, panel=panel
            ),
        )
        return fit(
            selection=StructuralSelection(model, None),
            data_for_model=dataset,
            time_origin=panel.time_origin,
            sampler=settings.revised(num_samples_per_chain=count),
            budget=ComputeBudget(memory=SharedMemoryBudget(available_bytes=2**30)),
            runtime=fit_runtime(settings),
            clock=time.monotonic,
            logger=logging.getLogger(__name__),
            wall_clock=lambda: datetime.now(UTC),
            compiler=compile_staged,
            prepare_program=prepare_program,
            compute_loo_diagnostics=True,
            previous=previous,
        )

    def output(result):
        return FitOutput(
            dynamical_model_spec=result.dynamical_model_spec,
            checks=FitCheckReport(
                data=DataProfileReport(indicators={}, findings=()),
                preflight=(),
                question=QuestionCheckReport(findings=()),
            ),
            inference=InferenceReport(
                evidence=result.evidence,
                core=InferenceReportCore(
                    inference_metadata=result.metadata,
                    inference_diagnostics=None,
                    convergence=chain_convergence(None),
                    posterior_marginals=(),
                    prior_densities={},
                ),
                detail=InferenceReportDetail(),
            ),
        )

    first = execute(posterior(2, 1), 2)
    for name, retained in first.evidence.chain_extra_fields.items():
        assert retained is first.evidence.phase_extra_fields["post_warmup"][name]
        np.testing.assert_array_equal(
            retained.values, posterior(2, 1).diagnostics.mcmc.chain_extra_fields[name]
        )

    from grassmoth_server.study.persistence.codec import pack_result, unpack_result

    restored = FitOutput.model_validate(unpack_result(pack_result(output(first))))
    for name, retained in restored.inference.evidence.chain_extra_fields.items():
        assert (
            retained.npy is restored.inference.evidence.phase_extra_fields["post_warmup"][name].npy
        )
    first = replace(
        first,
        evidence=first.evidence.revised(
            phase_extra_fields={
                "initialization": {"position": NumericalArray.from_numpy(np.array([1.0, 2.0]))}
            }
        ),
    )
    standalone = execute(posterior(1, 5), 1)
    combined = execute(posterior(1, 5), 1, output(first))
    identity = first.metadata.distribution
    old = empirical_atoms(first.dynamical_model_spec.distributions[identity]).reshape(2, 2, -1)
    new = empirical_atoms(standalone.dynamical_model_spec.distributions[identity]).reshape(2, 1, -1)
    atoms = empirical_atoms(combined.dynamical_model_spec.distributions[identity]).reshape(2, 3, -1)
    np.testing.assert_array_equal(atoms, np.concatenate((old, new), axis=1))
    for name, retained in combined.evidence.chain_extra_fields.items():
        np.testing.assert_array_equal(
            retained.values,
            np.concatenate(
                (
                    first.evidence.chain_extra_fields[name].values,
                    standalone.evidence.chain_extra_fields[name].values,
                ),
                axis=1,
            ),
        )
    assert combined.metadata.num_samples_total == 6
    assert combined.metadata.segments[-1].diagnostics == segment_diagnostics
    assert [(segment.start, segment.stop) for segment in combined.metadata.segments] == [
        (0, 2),
        (2, 3),
    ]
    np.testing.assert_array_equal(
        combined.evidence.observation_log_probs.values[0, :, 0], [1, 1, 5]
    )
    from grassmoth.artifacts.sampling import InferenceHardware
    from grassmoth_server.actions.fit import remote

    def dispatch_without_history_buffers(payload, *, hardware):
        assert hardware is InferenceHardware.L40S
        prior = FitOutput.model_validate(unpack_result(payload.previous))
        assert not prior.inference.evidence.phase_extra_fields
        return remote.execute_fit_compute(payload)

    monkeypatch.setattr(remote, "_dispatch_fit", dispatch_without_history_buffers)

    from grassmoth.artifacts.model.question import QuestionSpec

    def remotely(count, previous):
        return remote.fit_on_modal(
            InferenceHardware.L40S,
            selection=StructuralSelection(model, None),
            data_for_model=dataset,
            time_origin=panel.time_origin,
            sampler=settings.revised(num_samples_per_chain=count),
            question=QuestionSpec(text="Test", outcome=model.constructs[0].id),
            compute_loo_diagnostics=True,
            previous=previous,
        )

    transported = remotely(1, output(first))
    np.testing.assert_array_equal(
        empirical_atoms(transported.dynamical_model_spec.distributions[identity]),
        empirical_atoms(combined.dynamical_model_spec.distributions[identity]),
    )
    np.testing.assert_array_equal(
        transported.inference.evidence.phase_extra_fields["previous/initialization"][
            "position"
        ].values,
        [1.0, 2.0],
    )
    from unittest.mock import Mock

    from jax.stages import Compiled

    zero = execute(
        UnretainedRun.completed(
            settings.revised(num_samples_per_chain=0), MCMCEvidence(), Mock(spec=Compiled), {}
        ),
        0,
        output(combined),
    )
    assert zero.dynamical_model_spec is None
    assert zero.metadata.num_samples_total == 6
    assert zero.metadata.segments[-1].start == zero.metadata.segments[-1].stop == 3
    assert zero.metadata.segments[-1].diagnostics is None
    assert zero.metadata.segments[-1].convergence.status != "passed"
    assert zero.evidence.checkpoint == combined.evidence.checkpoint
    np.testing.assert_array_equal(
        zero.evidence.observation_log_probs.values,
        combined.evidence.observation_log_probs.values,
    )
    remote_zero = remotely(0, output(combined))
    assert remote_zero.dynamical_model_spec is not None
    assert remote_zero.inference.core.inference_metadata.num_samples_total == 6
    np.testing.assert_array_equal(
        remote_zero.inference.evidence.phase_extra_fields["previous/previous/initialization"][
            "position"
        ].values,
        [1.0, 2.0],
    )
