"""Build benchmark studies using the seven production actions and their findings."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import TypeAdapter

from evaluation.client import ActionClient
from evaluation.declarations import CaseRecipe, GeneratedData
from nof1_causal_lab.actions.contracts.bodies import EditModelOutput, PrepareDataOutput, SimulateOutput
from nof1_causal_lab.actions.contracts.requests import ActionInput
from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.artifacts.identity import GitOid
from nof1_causal_lab.artifacts.observations.reference import DataRef
from nof1_causal_lab.artifacts.reports.checks import Assessment, Evaluated, NotEvaluatedReason
from nof1_causal_lab.study.persistence.codec import pack_payload, result_payload

_REQUEST = TypeAdapter(ActionInput)


class PreparedCase(Value):
    """Scientific content, revisions and reviewed exceptions captured before fitting."""

    model_ref: GitOid
    data_ref: DataRef[GitOid, int]
    model_digest: str
    data_digest: str
    recipe_digest: str
    findings: tuple[str, ...]


def require_findings[Subject, Evidence](
    findings: Sequence[Assessment[Subject, Evidence]], allowed: Mapping[str, str],
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
    """Hash content including native array bytes, independently of workspace and Git IDs."""
    return hashlib.sha256(pack_payload(value)).hexdigest()


def prepare_case(recipe: CaseRecipe, root: Path, client: ActionClient, seed: int) -> PreparedCase:
    """Pin the model and one history; no sampler or alternate data generator runs here."""
    files = (recipe.fit_model, recipe.data.truth_model) if isinstance(recipe.data, GeneratedData) else (recipe.fit_model, *recipe.data.files)
    content = {name: (root / name).read_bytes() for name in files}
    recipe_digest = fingerprint({"recipe": recipe.model_dump(mode="json"), "files": content})
    question = client.call(_REQUEST.validate_python({"action": "edit_question", "input": {"question": recipe.question.model_dump(mode="json")}}))

    def model(path: str, parent: GitOid):
        result = client.call(_REQUEST.validate_python({"action": "edit_model", "input": {
            "parent_ref": parent, "dynamical_model_spec": json.loads(content[path]),
        }}))
        if result.action != "edit_model":
            raise TypeError("edit_model returned a different action body")
        findings = require_findings(result.body.checks.specification, recipe.allowed_findings, recipe.unavailable_findings)
        findings += require_findings(result.body.checks.question.findings, recipe.allowed_findings, recipe.unavailable_findings)
        identified = result.body.identification.estimable_treatments
        if not set(recipe.expected_identified) <= set(identified):
            raise ValueError("The case's predeclared treatment is not identified by edit_model")
        return result, findings

    if isinstance(recipe.data, GeneratedData):
        truth, truth_findings = model(recipe.data.truth_model, question.commit_id)
        generated = client.call(_REQUEST.validate_python({"action": "simulate", "input": {
            "dynamical_model_spec_ref": truth.commit_id, "simulation": recipe.data.simulation.model_dump(mode="json"),
            "seed": seed, "num_replicates": 1,
        }}))
        if generated.action != "simulate":
            raise TypeError("simulate returned a different action body")
        generation_findings = require_findings(generated.body.report.findings, recipe.allowed_findings, recipe.unavailable_findings)
        data = generated.body.data[0]
        data_revision = generated.commit_id
        fitted, findings = model(recipe.fit_model, question.commit_id)
        findings += truth_findings + generation_findings
    else:
        fitted, findings = model(recipe.fit_model, question.commit_id)
        for file in recipe.data.files:
            client.upload(root / file)
        prepared = client.call(_REQUEST.validate_python({"action": "prepare_data", "input": {
            "dynamical_model_spec_ref": fitted.commit_id, "source": "input",
            "extraction": {key: rule.model_dump(mode="json") for key, rule in recipe.data.extraction.items()},
            "context": recipe.data.context,
        }}))
        if prepared.action != "prepare_data":
            raise TypeError("prepare_data returned a different action body")
        findings += require_findings(prepared.body.profile.findings, recipe.allowed_findings, recipe.unavailable_findings)
        for audit in prepared.body.profile.indicators.values():
            findings += require_findings(audit.findings, recipe.allowed_findings, recipe.unavailable_findings)
        data, data_revision = prepared.body.data, prepared.commit_id
    return PreparedCase(
        model_ref=fitted.commit_id, data_ref=DataRef(revision=data_revision, replicate_index=0),
        model_digest=fingerprint(result_payload(fitted.body.dynamical_model_spec)),
        data_digest=fingerprint({key: result_payload(value) for key, value in data.items()}),
        recipe_digest=recipe_digest, findings=findings,
    )
