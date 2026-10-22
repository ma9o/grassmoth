"""Directional comparison of two supplied scientific documents."""

from grassmoth.actions.contracts.bodies import ModelDiffOutput
from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec


def model_diff(before: DynamicalModelSpec, after: DynamicalModelSpec) -> ModelDiffOutput:
    """Compare documents without compiling or evaluating scientific findings."""
    return ModelDiffOutput(changes=after.changes_from(before))
