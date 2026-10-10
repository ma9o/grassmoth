"""Tests for Stage 5 inference task logging and orchestration."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from nof1_causal_lab.actions.inference import fit as stage5_inference
from nof1_causal_lab.artifacts.construct import replace_constructs
from nof1_causal_lab.artifacts.observation_data import ObservationDataset
from nof1_causal_lab.models.model_structure import StructuralSelection
from nof1_causal_lab.models.ssm import runtime as runtime_module
from nof1_causal_lab.models.ssm.compile import inputs as compilation
from nof1_causal_lab.models.ssm.inference.types import JointPosteriorDraws
from nof1_causal_lab.sampler_config import SamplerSpec
from tests.inference_fixtures import (
    bind_panel_fixture,
    compile_fit_fixture,
    make_observation_support_runtime,
)
from tests.model_fixtures import load_model_fixture


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
    from nof1_causal_lab.artifacts.dynamical_model_spec import DynamicalModelSpec
    from nof1_causal_lab.models.ssm.observation_support import ObservationSupportRuntime


pytestmark = pytest.mark.contract


def _fake_result():
    from tests.inference_fixtures import particle_posterior

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
    from nof1_causal_lab.models.ssm.runtime import BoundPanel
