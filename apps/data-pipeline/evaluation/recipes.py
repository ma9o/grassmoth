"""Build benchmark studies using the seven production actions and their findings."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from pydantic import TypeAdapter

from evaluation.declarations import CaseRecipe, GeneratedData
from evaluation.sources import model_document, read_source
from nof1_causal_service.public.contracts import (
    ActionInput,
    Assessment,
    DataRef,
    Evaluated,
    GitOid,
    NotEvaluatedReason,
    Value,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from evaluation.client import ActionClient
    from evaluation.declarations import ModelSource

_REQUEST = TypeAdapter(ActionInput)


class PreparedCase(Value):
    """Scientific content, revisions and reviewed exceptions captured before fitting."""

    model_ref: GitOid
    data_ref: DataRef[GitOid, int]
    model_digest: str
    data_digest: str
    recipe_digest: str
    provenance: str
    findings: tuple[str, ...]


def require_findings[Subject, Evidence](
    findings: Sequence[Assessment[Subject, Evidence]],
    allowed: Mapping[str, str],
    unavailable: Mapping[NotEvaluatedReason, str],
) -> tuple[str, ...]:
    """Read structured outcomes; exceptions require a predeclared code and reason."""
    issues = []
    for item in findings:
        if isinstance(item, Evaluated):
            if item.outcome == "passed":
                continue
            explanation = allowed.get(item.code)
        else:
            explanation = unavailable.get(item.reason)
        if explanation is None:
            raise ValueError(f"Unaccepted structured finding: {item.model_dump_json()}")
        issues.append(f"{item.code}: {explanation}")
    return tuple(issues)


def fingerprint(value: object) -> str:
    """Hash an evaluation declaration or mapping of verified content digests."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def prepare_case(recipe: CaseRecipe, root: Path, client: ActionClient, seed: int) -> PreparedCase:
    """Pin the model and one history; no sampler or alternate data generator runs here."""
    sources = (
        (recipe.fit_model, recipe.data.truth_model)
        if isinstance(recipe.data, GeneratedData)
        else (recipe.fit_model, *recipe.data.files, *recipe.data.provenance_files)
    )
    content = {source.sha256: read_source(source, root) for source in sources}
    recipe_digest = fingerprint(recipe.model_dump(mode="json"))
    question = client.call(
        _REQUEST.validate_python(
            {
                "action": "edit_question",
                "input": {"question": recipe.question.model_dump(mode="json")},
            }
        )
    )
    question_revision = client.artifact_revision(question.call_id, "question")

    def model(source: ModelSource, parent: GitOid):
        result = client.call(
            _REQUEST.validate_python(
                {
                    "action": "edit_model",
                    "input": {
                        "parent_ref": parent,
                        "dynamical_model_spec": model_document(content[source.sha256]).model_dump(
                            mode="json"
                        ),
                    },
                }
            )
        )
        if result.action != "edit_model":
            raise TypeError("edit_model returned a different action body")
        findings = require_findings(
            result.body.checks.specification, recipe.allowed_findings, recipe.unavailable_findings
        )
        findings += require_findings(
            result.body.checks.question.findings,
            recipe.allowed_findings,
            recipe.unavailable_findings,
        )
        identified = result.body.identification.estimable_treatments
        if not set(recipe.expected_identified) <= set(identified):
            raise ValueError("The case's predeclared treatment is not identified by edit_model")
        return result, findings

    if isinstance(recipe.data, GeneratedData):
        truth, truth_findings = model(recipe.data.truth_model, question_revision)
        generated = client.call(
            _REQUEST.validate_python(
                {
                    "action": "simulate",
                    "input": {
                        "dynamical_model_spec_ref": client.artifact_revision(
                            truth.call_id, "model"
                        ),
                        "simulation": recipe.data.simulation.model_dump(mode="json"),
                        "seed": seed,
                        "num_replicates": 1,
                    },
                }
            )
        )
        if generated.action != "simulate":
            raise TypeError("simulate returned a different action body")
        generation_findings = require_findings(
            generated.body.report.findings, recipe.allowed_findings, recipe.unavailable_findings
        )
        data = generated.body.data[0]
        data_revision = generated.commit_id
        fitted, findings = model(recipe.fit_model, question_revision)
        findings += truth_findings + generation_findings
    else:
        fitted, findings = model(recipe.fit_model, question_revision)
        for file in recipe.data.files:
            client.upload((root / file.path).name, content[file.sha256])
        prepared = client.call(
            _REQUEST.validate_python(
                {
                    "action": "prepare_data",
                    "input": {
                        "dynamical_model_spec_ref": client.artifact_revision(
                            fitted.call_id, "model"
                        ),
                        "source": "input",
                        "extraction": {
                            key: rule.model_dump(mode="json")
                            for key, rule in recipe.data.extraction.items()
                        },
                        "context": recipe.data.context,
                    },
                }
            )
        )
        if prepared.action != "prepare_data":
            raise TypeError("prepare_data returned a different action body")
        findings += require_findings(
            prepared.body.profile.findings, recipe.allowed_findings, recipe.unavailable_findings
        )
        for audit in prepared.body.profile.indicators.values():
            findings += require_findings(
                audit.findings, recipe.allowed_findings, recipe.unavailable_findings
            )
        data = prepared.body.data
        data_revision = client.artifact_revision(prepared.call_id, "panel")
    return PreparedCase(
        model_ref=client.artifact_revision(fitted.call_id, "model"),
        data_ref=DataRef(revision=data_revision, replicate_index=0),
        model_digest=fitted.body.dynamical_model_spec.content_digest(),
        data_digest=fingerprint({key: value.content_digest() for key, value in data.items()}),
        recipe_digest=recipe_digest,
        provenance=recipe.provenance,
        findings=findings,
    )
