"""Models used by server study and action tests."""

from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec
from grassmoth.artifacts.model.expressions import coefficient, state
from grassmoth.artifacts.model.mechanism import DriftMechanismSpec
from grassmoth_server_test_support.fixtures import study_fixture_json
from grassmoth_test_support.model import construct_named


def load_study_model_fixture(name: str) -> DynamicalModelSpec:
    return DynamicalModelSpec.model_validate_json(study_fixture_json(name)).materialized()


def stress_sleep_model() -> DynamicalModelSpec:
    return load_study_model_fixture("common/stress_sleep_model.json")


def stress_sleep_causal_model() -> DynamicalModelSpec:
    """The two measured states with Sleep as the question's downstream outcome."""
    from grassmoth.artifacts.identity import MechanismId

    dynamical_model_spec = stress_sleep_model()
    stress = construct_named(dynamical_model_spec, "Stress")
    return dynamical_model_spec.with_entities(
        edges=(
            dynamical_model_spec.edges[0].revised(
                effect=construct_named(dynamical_model_spec, "Sleep"),
                mechanisms=(
                    DriftMechanismSpec(
                        id=MechanismId("mechanism:stress_to_sleep"),
                        expression=coefficient(0.1, role="weight") * state(stress.id),
                    ),
                ),
            ),
        )
    )
