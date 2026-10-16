"""Broad invariants of action-built experiments without executing any case or fit."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from evaluation.comparison import differing_controls
from evaluation.declarations import CaseRecipe, Experiment
from evaluation.run import assigned_settings
from nof1_causal_lab.artifacts.model.dynamical_model_spec import DynamicalModelSpec

pytestmark = pytest.mark.contract


def declaration():
    return {
        "name": "particles", "case": "tiny.json", "base": {"hardware": "local"},
        "factors": ["num_particles"],
        "arms": [{"name": "small", "settings": {"num_particles": 8}},
                 {"name": "large", "settings": {"num_particles": 16}}],
        "seeds": [{"generation": 9, "sampler": 13}],
    }


def test_declared_treatment_preserves_controls_and_rejects_accidental_changes():
    experiment = Experiment.model_validate(declaration())
    left, right = [assigned_settings(experiment, arm, 13).model_dump(mode="json") for arm in experiment.arms]
    assert not differing_controls(left, right, experiment.factors)
    right["num_warmup"] += 1
    assert differing_controls(left, right, experiment.factors) == ("num_warmup",)
    right["runtime"] = {"source_digest": "new code"}
    assert "runtime.source_digest" in differing_controls(left, right, experiment.factors)
    payload = declaration()
    payload["arms"][1]["settings"]["num_warmup"] = 0
    with pytest.raises(ValidationError, match="exactly the declared factors"):
        Experiment.model_validate(payload)
    payload["factors"].append("num_warmup")
    with pytest.raises(ValidationError, match="interaction"):
        Experiment.model_validate(payload)


def test_action_recipe_models_are_complete_authored_documents():
    root = Path(__file__).parents[2] / "evaluation/cases"
    for path in root.glob("*.json"):
        recipe = CaseRecipe.model_validate_json(path.read_text())
        model = DynamicalModelSpec.model_validate_json((root / recipe.fit_model).read_text()).materialized()
        assert model.get_construct(recipe.question.outcome)
        assert model.measurement_clock is not None
        if recipe.data.kind == "simulate":
            truth = DynamicalModelSpec.model_validate_json((root / recipe.data.truth_model).read_text()).materialized()
            assert not truth.parameters
        else:
            assert set(recipe.data.extraction) == {item.observation.id for item in model.indicators}
            assert all((root / file).is_file() for file in recipe.data.files)
        assert json.loads(path.read_text())["provenance"]


def test_unavailable_generation_checks_do_not_accept_failed_scientific_checks():
    from evaluation.recipes import require_findings
    from nof1_causal_lab.artifacts.reports.checks import Evaluated, NotEvaluated

    missing = NotEvaluated(code="C5a location reach", subject="y", reason="NO_OBSERVATIONS", detail="No comparison observations")
    failed = Evaluated(code="C5a location reach", subject="y", outcome="failed", evidence="Outside support")
    allowed = {"NO_OBSERVATIONS": "One synthetic history has no external calibration data"}
    assert require_findings((missing,), {}, allowed)
    with pytest.raises(ValueError, match="Unaccepted structured finding"):
        require_findings((failed,), {}, allowed)
