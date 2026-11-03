"""Tests for Stage 5 inference task logging and orchestration."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from grassmoth.actions.contracts.io import FitInputs
from grassmoth.actions.fit import execution as stage5_inference
from grassmoth.actions.fit.execution import FittedModelResult
from grassmoth.artifacts.model.construct import replace_constructs
from grassmoth.artifacts.model.question import QuestionSpec
from grassmoth.artifacts.observations.dataset import ObservationDataset
from grassmoth.artifacts.sampling import SamplerSpec
from grassmoth.compute.budget import ComputeBudget, SharedMemoryBudget
from grassmoth.models.ssm.compile import inputs as compilation
from grassmoth.models.ssm.inference import fit as inference_fit
from grassmoth.models.ssm.inference.types import JointPosteriorDraws
from grassmoth.models.ssm.preparation import panel as runtime_module
from grassmoth.models.structure import StructuralSelection
from grassmoth_server.actions.fit.runtime import fit_runtime
from grassmoth_server.compute.compilation_cache import compile_program, prepare_program
from grassmoth_server_test_support.model import load_study_model_fixture
from grassmoth_test_support.compute import EXECUTION_CONTEXT
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
    monkeypatch.setattr(inference_fit, "fit", lambda _priors, _panel, **_kwargs: fake_result)

    data_for_model = ObservationDataset.from_frame(
        pl.DataFrame(runtime.rows),
        tuple(item.observation for item in runtime.compiled_dynamical_model.observations),
        time_origin=runtime.time_origin,
    )

    with caplog.at_level(logging.INFO, logger=__name__):
        result = stage5_inference.fit_model(
            FitInputs(
                selection=StructuralSelection(dynamical_model_spec, None),
                question=QuestionSpec(text="Test", outcome=dynamical_model_spec.constructs[0].id),
                data_for_model=data_for_model,
                sampler=SamplerSpec(),
                compute_loo_diagnostics=False,
            ),
            continuation=None,
            execution_context=replace(
                EXECUTION_CONTEXT,
                compiler=compile_program,
                prepare_program=prepare_program,
                budget=ComputeBudget(memory=SharedMemoryBudget(available_bytes=2**30)),
                logger=logging.getLogger(__name__),
            ),
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


@pytest.mark.inference(concern="sampling")
def test_continued_fit_appends_within_chains_and_zero_draws_keep_existing_evidence(
    monkeypatch, tmp_path
):
    from grassmoth.actions.contracts.io import FitOutput
    from grassmoth.actions.fit.action import _fit as fit
    from grassmoth.artifacts.arrays import NumericalArray
    from grassmoth.artifacts.continuation import SamplerCheckpoint, TransferredTuning
    from grassmoth.artifacts.probability.serialization import empirical_atoms
    from grassmoth.artifacts.reports.data_quality import DataProfileReport
    from grassmoth.artifacts.reports.inference import (
        FitCheckReport,
        InferenceReport,
        InferenceReportCore,
        InferenceReportDetail,
    )
    from grassmoth.artifacts.reports.model_checks import QuestionCheckReport
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
    settings = SamplerSpec(
        num_chains=2,
        num_warmup=0,
        num_samples_per_chain=2,
        hardware="local",
    )

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
            tuning=TransferredTuning(),
        )
        telemetry = {
            "particle_weight_ess": jnp.arange(2 * count * 3, dtype=jnp.float32).reshape(2, count, 3)
            + offset,
            "particle_weight_entropy": jnp.full((2, count, 3), offset, dtype=jnp.float32),
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
            FitInputs(
                selection=StructuralSelection(model, None),
                question=QuestionSpec(text="Test", outcome=model.constructs[0].id),
                data_for_model=dataset,
                sampler=settings.revised(num_samples_per_chain=count),
                compute_loo_diagnostics=True,
                previous=previous,
            ),
            continuation=None,
            runtime=fit_runtime(settings),
            execution_context=replace(
                EXECUTION_CONTEXT,
                compiler=compile_program,
                prepare_program=prepare_program,
                budget=ComputeBudget(memory=SharedMemoryBudget(available_bytes=2**30)),
            ),
        )

    def output(result, *, render=False):
        from grassmoth.actions.fit.action import inference_report

        return FitOutput(
            dynamical_model_spec=result.dynamical_model_spec,
            checks=FitCheckReport(
                data=DataProfileReport(indicators={}, findings=()),
                preflight=(),
                question=QuestionCheckReport(findings=()),
            ),
            inference=inference_report(result, model, (), execution_context=EXECUTION_CONTEXT)
            if render
            else InferenceReport(
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
    assert first.metadata.segments[0].settings.num_samples_per_chain == 2
    # Post-sampling timing is log-only; the stubbed sampler recorded no phases.
    assert first.metadata.segments[0].phases == ()
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
    assert combined.metadata.segments[-1].settings.num_samples_per_chain == 1
    assert combined.metadata.num_samples_total == 6
    assert (
        combined.metadata.segments[-1].diagnostics == standalone.metadata.segments[-1].diagnostics
    )
    assert [(segment.start, segment.stop) for segment in combined.metadata.segments] == [
        (0, 2),
        (2, 3),
    ]
    np.testing.assert_array_equal(
        combined.evidence.observation_log_probs.values[0, :, 0], [1, 1, 5]
    )
    from grassmoth.artifacts.sampling import InferenceHardware
    from grassmoth_server.actions.contracts.identity import CallId
    from grassmoth_server.actions.fit import remote
    from grassmoth_server.compute.analysis_cache import AnalysisCall, RetainedFit, cache_fit_result
    from grassmoth_server.compute.fit_results import RemoteFitResult
    from grassmoth_server.study.persistence.arrays import owned_arrays

    monkeypatch.setattr(remote, "ANALYSIS_CACHE_ROOT", tmp_path)

    def dispatch_retained(payload, *, hardware):
        assert hardware is InferenceHardware.L40S
        # A remote continuation sends only the retained identity, including prior traces.
        assert not isinstance(payload.previous, bytes)
        fitted = remote.execute_fit_compute(payload)
        assert fitted.inference.evidence.checkpoint is not None
        body = pack_result(fitted)
        entry = cache_fit_result(tmp_path / "retained", payload.analysis_call, unpack_result(body))
        return remote.FitComputeResult(
            pack_result(
                RemoteFitResult(
                    full_fit=RetainedFit(
                        entry=entry, settings=fitted.inference.evidence.checkpoint.settings
                    ),
                    transfer=entry,
                    has_model=fitted.dynamical_model_spec is not None,
                    law_arrays=tuple(owned_arrays(unpack_result(body)["dynamical_model_spec"])),
                )
            )
        )

    monkeypatch.setattr(remote, "_dispatch_fit", dispatch_retained)
    analysis_call = AnalysisCall(workspace_id="TEST", call_id=CallId("call:" + "a" * 64))

    def remotely(count, previous):
        parent = cache_fit_result(
            tmp_path / "retained", analysis_call, unpack_result(pack_result(previous))
        )
        receipt = remote.fit_on_modal(
            FitInputs(
                selection=StructuralSelection(model, None),
                question=QuestionSpec(text="Test", outcome=model.constructs[0].id),
                data_for_model=dataset,
                sampler=settings.revised(
                    hardware=InferenceHardware.L40S,
                    num_samples_per_chain=count,
                ),
                compute_loo_diagnostics=True,
            ),
            analysis_call=analysis_call,
            previous_cache=parent,
        )
        return FitOutput.model_validate(
            unpack_result(
                remote.retained_fit_result(
                    tmp_path / "retained", receipt.full_fit.entry
                ).read_bytes()
            )
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
        UnretainedRun.completed(settings.revised(num_samples_per_chain=0), Mock(spec=Compiled), {}),
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

    from grassmoth.artifacts import fit_transfer
    from grassmoth.artifacts.arrays import resolve_vector
    from grassmoth_server.study.persistence.codec import result_payload

    full = output(execute(posterior(65, 2), 100), render=True)
    identity = full.inference.core.inference_metadata.distribution
    atoms = empirical_atoms(full.dynamical_model_spec.distributions[identity])
    # Use an actual shorter run than requested, as at an early retained checkpoint.
    cap = 2 * atoms.shape[1] * 9
    monkeypatch.setattr(fit_transfer, "MAX_TRANSFER_DRAW_VALUES", cap)

    def transferred(value):
        return FitOutput.model_validate(result_payload(value, thin_fit_draws=True))

    small = transferred(full)
    meta = small.inference.core.inference_metadata
    assert meta.draws_sampled_per_chain == 65
    assert meta.draws_kept_per_chain == 9
    assert meta.thinning_factor == 8
    selected = empirical_atoms(small.dynamical_model_spec.distributions[identity])
    assert selected.size <= cap
    np.testing.assert_array_equal(selected.reshape(2, 9, -1), atoms.reshape(2, 65, -1)[:, ::8])
    assert small.inference.evidence.checkpoint == full.inference.evidence.checkpoint
    assert small.inference.core.inference_diagnostics == full.inference.core.inference_diagnostics
    assert small.inference.core.convergence == full.inference.core.convergence
    assert small.inference.core.posterior_marginals == full.inference.core.posterior_marginals
    assert small.inference.detail.rank_histograms == full.inference.detail.rank_histograms
    for before, after in zip(
        full.inference.detail.trace_data, small.inference.detail.trace_data, strict=True
    ):
        for full_chain, thin_chain in zip(before.chains, after.chains, strict=True):
            assert resolve_vector(thin_chain) == resolve_vector(full_chain)[::8]
    for name, value in small.inference.evidence.chain_extra_fields.items():
        np.testing.assert_array_equal(
            value.values, full.inference.evidence.chain_extra_fields[name].values[:, ::8]
        )
    assert transferred(small) == small
    disabled = full.revised(
        inference=full.inference.revised(
            core=full.inference.core.revised(
                inference_metadata=full.inference.core.inference_metadata.revised(
                    segments=tuple(
                        segment.revised(settings=segment.settings.revised(thinning=False))
                        for segment in full.inference.core.inference_metadata.segments
                    )
                )
            )
        )
    )
    assert transferred(disabled) == disabled
    extended = output(execute(posterior(38, 4), 38, full), render=True)
    extended_small = transferred(extended)
    extended_atoms = empirical_atoms(extended.dynamical_model_spec.distributions[identity]).reshape(
        2, 103, -1
    )
    extended_meta = extended_small.inference.core.inference_metadata
    assert extended_meta.thinning_factor == 12
    np.testing.assert_array_equal(
        empirical_atoms(extended_small.dynamical_model_spec.distributions[identity]).reshape(
            2, 9, -1
        ),
        extended_atoms[:, ::12],
    )
    assert extended.inference.core.inference_diagnostics.num_samples_per_chain == 103
    assert (
        extended_small.inference.core.inference_diagnostics
        == extended.inference.core.inference_diagnostics
    )
    assert extended_small.inference.core.convergence == extended.inference.core.convergence
    assert (
        extended_small.inference.core.posterior_marginals
        == extended.inference.core.posterior_marginals
    )
    assert (
        extended_small.inference.detail.rank_histograms == extended.inference.detail.rank_histograms
    )
    layout = extended.dynamical_model_spec.law_layouts[identity]
    for marginal in extended.inference.core.posterior_marginals:
        column = layout.parameter_columns[marginal.subject.element_id]
        assert marginal.mean == pytest.approx(float(extended_atoms[..., column].mean()))
    assert all(
        sum(chain) == 103
        for hist in extended.inference.detail.rank_histograms
        for chain in hist.chains
    )
    monkeypatch.setattr(fit_transfer, "MAX_TRANSFER_DRAW_VALUES", 1)
    irreducible = transferred(full)
    assert irreducible.inference.core.inference_metadata.draws_kept_per_chain == 1
    assert irreducible.inference.core.inference_metadata.thinning_factor == 65
    np.testing.assert_array_equal(
        empirical_atoms(irreducible.dynamical_model_spec.distributions[identity]),
        atoms.reshape(2, 65, -1)[:, 0],
    )
