"""Broad invariants of action-built experiments without executing any case or fit."""

import json
from pathlib import Path

import pytest
from evaluation.comparison import differing_controls
from evaluation.declarations import CaseFile, CaseRecipe, Experiment, StudyModel
from evaluation.run import assigned_settings
from evaluation.sources import model_document, read_source
from pydantic import ValidationError

from nof1_causal_lab.artifacts.model.dynamical_model_spec import DynamicalModelSpec

pytestmark = pytest.mark.contract


def declaration():
    return {
        "name": "particles",
        "case": "tiny.json",
        "base": {"hardware": "local"},
        "factors": ["num_particles"],
        "arms": [
            {"name": "small", "settings": {"num_particles": 8}},
            {"name": "large", "settings": {"num_particles": 16}},
        ],
        "seeds": [{"generation": 9, "sampler": 13}],
    }


def test_declared_treatment_preserves_controls_and_rejects_accidental_changes():
    experiment = Experiment.model_validate(declaration())
    left, right = [
        assigned_settings(experiment, arm, 13).model_dump(mode="json") for arm in experiment.arms
    ]
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
        if isinstance(recipe.fit_model, StudyModel):
            assert recipe.fit_model.repository.startswith("../../../../data/")
            assert recipe.data.kind == "prepare_data"
            assert all(file.path.startswith("../../../../data/") for file in recipe.data.files)
            continue
        model = model_document(recipe.fit_model, read_source(recipe.fit_model, root)).materialized()
        assert model.get_construct(recipe.question.outcome)
        assert model.measurement_clock is not None
        if recipe.data.kind == "simulate":
            truth = DynamicalModelSpec.model_validate_json(
                read_source(recipe.data.truth_model, root)
            ).materialized()
            assert not truth.parameters
        else:
            assert set(recipe.data.extraction) == {item.observation.id for item in model.indicators}
            for file in (*recipe.data.files, *recipe.data.provenance_files):
                assert isinstance(file, CaseFile)
                assert read_source(file, root)
        assert json.loads(path.read_text())["provenance"]


def test_unavailable_generation_checks_do_not_accept_failed_scientific_checks():
    from evaluation.recipes import require_findings

    from nof1_causal_lab.artifacts.reports.checks import Evaluated, NotEvaluated, NotEvaluatedReason

    missing = NotEvaluated(
        code="C5a location reach",
        subject="y",
        reason="NO_OBSERVATIONS",
        detail="No comparison observations",
    )
    failed = Evaluated(
        code="C5a location reach", subject="y", outcome="failed", evidence="Outside support"
    )
    allowed: dict[NotEvaluatedReason, str] = {
        "NO_OBSERVATIONS": "One synthetic history has no external calibration data"
    }
    assert require_findings((missing,), {}, allowed)
    with pytest.raises(ValueError, match="Unaccepted structured finding"):
        require_findings((failed,), {}, allowed)


def test_t512_variants_share_data_and_compile_the_accumulator_for_replay():
    from nof1_causal_lab.models.ssm.compile.inputs import compile_executable_model
    from nof1_causal_lab.models.structure import StructuralSelection

    root = Path(__file__).parents[2] / "evaluation/cases"
    recipes = tuple(
        CaseRecipe.model_validate_json((root / (name + ".json")).read_text())
        for name in ("t512", "t512_unknown_diffusion")
    )
    assert recipes[0].data == recipes[1].data
    models = tuple(
        model_document(recipe.fit_model, read_source(recipe.fit_model, root)).materialized()
        for recipe in recipes
    )
    for model, recipe in zip(models, recipes, strict=True):
        compiled = compile_executable_model(StructuralSelection(model, recipe.question.outcome))
        accumulator = next(
            index for index, state in enumerate(compiled.states) if state.name == "accumulator"
        )
        assert accumulator in compiled.state_space.replayed_coordinates
        assert accumulator not in compiled.state_space.transition
        assert accumulator not in compiled.state_space.initial
    assert {parameter.name for parameter in models[0].parameters} == {
        "input_effect",
        "cross_effect",
    }
    assert {parameter.name for parameter in models[1].parameters} == {
        "input_effect",
        "cross_effect",
        "sigma",
    }
