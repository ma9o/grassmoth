"""Explicit data schemas accompanying test-owned observations."""

from datetime import UTC, datetime

import numpy as np

from nof1_causal_lab.artifacts.observations.preparation import (
    DataPreparationSpec,
    DataVariableSpec,
    PreparedDataMetadata,
    SemanticExtractionSpec,
)
from nof1_causal_lab.artifacts.simulation import SimulationObservationLayout
from nof1_causal_service.actions.contracts.preparation import FileSourceRef
from tests.support.inference import compile_model_fixture


def metadata_for_model(model):
    preparation = DataPreparationSpec(
        default_window=model.measurement_clock,
        variables=tuple(
            DataVariableSpec(
                observation=indicator.observation,
                extraction=SemanticExtractionSpec(
                    how_to_measure="Read " + indicator.observation.name
                ),
            )
            for indicator in model.indicators
        ),
    )
    return PreparedDataMetadata(
        time_origin=datetime(2024, 1, 1, tzinfo=UTC),
        preparation=preparation,
    )


def simulation_layout(model, times, mask, write_array):
    from nof1_causal_lab.models.ssm.preparation.observations import simulation_observation_support

    compiled_dynamical_model = compile_model_fixture(model)
    support = simulation_observation_support(
        compiled_dynamical_model, np.asarray(times), time_origin=datetime(2024, 1, 1, tzinfo=UTC)
    )
    return SimulationObservationLayout(
        variables=tuple(
            model.indicator(observation.id).observation.resolved(observation.observation_window)
            for observation in compiled_dynamical_model.observations
        ),
        support_start_times=write_array(support.support_start_times),
        support_end_times=write_array(support.support_end_times),
        mask=write_array(mask),
    )


def preparation_input(store, preparation):
    """Publish the model owning a test recipe before submitting extraction instructions."""
    from nof1_causal_lab.artifacts.identity import GitOid
    from nof1_causal_lab.artifacts.model.construct import replace_constructs
    from nof1_causal_lab.artifacts.model.indicator import IndicatorSpec
    from nof1_causal_service.actions.contracts.inputs import PrepareDataInput
    from tests.support.helpers import make_model

    variables = preparation.definition.variables
    dynamical_model_spec = make_model([item.observation.name for item in variables])
    dynamical_model_spec = dynamical_model_spec.revised(
        measurement_clock=preparation.definition.default_window
    ).with_entities(
        edges=replace_constructs(
            dynamical_model_spec.edges,
            tuple(
                construct.revised(
                    indicators=(
                        IndicatorSpec(observation=item.observation, construct_polarity="positive"),
                    )
                )
                for construct in dynamical_model_spec.constructs
                for item in variables
                if construct.name == item.observation.name
            ),
        ),
    )
    info = store.write_artifact(
        "model",
        derived_from={},
        produced_by="edit_model",
        json_files={"model.json": dynamical_model_spec.model_dump(mode="json")},
    )
    return PrepareDataInput[GitOid, FileSourceRef](
        dynamical_model_spec_ref=info.revision,
        source=preparation.source,
        extraction={item.observation.id: item.extraction for item in variables},
        context=preparation.definition.context,
    )
