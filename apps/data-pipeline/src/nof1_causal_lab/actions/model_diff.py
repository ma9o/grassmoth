"""Directional comparison of two supplied scientific documents."""

from nof1_causal_lab.actions.contracts.bodies import ModelDiffOutput
from nof1_causal_lab.artifacts.model.dynamical_model_spec import DynamicalModelSpec


def model_diff(before: DynamicalModelSpec, after: DynamicalModelSpec) -> ModelDiffOutput:
    """Compare documents without compiling or evaluating scientific findings."""
    return ModelDiffOutput(changes=after.changes_from(before))
