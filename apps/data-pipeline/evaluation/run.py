"""Run a declared experiment through public actions; importing this module executes nothing."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import batched
from pathlib import Path
from typing import TYPE_CHECKING, cast

from evaluation.client import ActionClient
from evaluation.comparison import (
    _posterior_draws,
    differing_controls,
    initializer_scores,
    posterior_equivalence,
    qualify_reference,
)
from evaluation.declarations import Arm, CaseRecipe, Experiment
from evaluation.recipes import PreparedCase, fingerprint, prepare_case, require_findings
from evaluation.report import BillingReport, write_report
from evaluation.segments import segment_summary
from evaluation.telemetry import sampler_outcomes
from evaluation.workspaces import copy_anchor_workspace
from nof1_causal_lab.actions.contracts.bodies import (
    ContinueFitTarget,
    FitInput,
    FitOutput,
    NewFitTarget,
)
from nof1_causal_lab.actions.contracts.requests import FitRequest
from nof1_causal_lab.artifacts.sampling import FitSettingsSpec, SamplerSpec
from nof1_causal_lab.study.persistence.codec import result_payload

if TYPE_CHECKING:
    from typing import Literal

    from nof1_causal_lab.actions.contracts.polling import SuccessfulPoll
    from nof1_causal_lab.artifacts.identity import GitOid
    from nof1_causal_lab.json_types import JsonObject


def assigned_settings(experiment: Experiment, arm: Arm, seed: int) -> FitSettingsSpec:
    """Apply exactly the declared assigned values, then parse at the production owner."""
    values = experiment.base.model_dump(mode="json", exclude={"retain_latent_paths"})
    for path, value in arm.settings.items():
        current = values
        parts = path.split(".")
        for part in parts[:-1]:
            current = current[part]
        current[parts[-1]] = value
    values["seed"] = seed
    if experiment.probe_samples is not None:
        values["num_warmup"] = experiment.probe_warmup
        values["num_samples_per_chain"] = experiment.probe_samples
    return FitSettingsSpec.model_validate(values)


def compile_seconds(result: FitOutput) -> float:
    """Compute the union of overlapping compile intervals in the latest segment."""
    intervals = sorted(
        (p.start_seconds, p.end_seconds)
        for p in result.inference.core.inference_metadata.segments[-1].phases
        if "compil" in p.stage
    )
    stop, elapsed = 0.0, 0.0
    for left, right in intervals:
        elapsed += max(0.0, right - max(left, stop))
        stop = max(stop, right)
    return elapsed


def _fit(
    client: ActionClient,
    prepared: PreparedCase,
    settings: FitSettingsSpec,
    repeat: int,
    previous: GitOid | None = None,
):
    target = (
        ContinueFitTarget(fit_ref=previous)
        if previous is not None
        else NewFitTarget(dynamical_model_spec_ref=prepared.model_ref, data_ref=prepared.data_ref)
    )
    result = client.call(
        FitRequest(input=FitInput(target=target, settings=settings, repeat=repeat))
    )
    if result.action != "fit":
        raise TypeError("fit returned a different action body")
    require_findings(result.body.checks.preflight, {}, {})
    return result


def _controls(result: FitOutput) -> JsonObject:
    segment = result.inference.core.inference_metadata.segments[-1]
    return cast(
        "JsonObject",
        {
            **segment.settings.model_dump(mode="json"),
            "runtime": segment.runtime.model_dump(mode="json"),
        },
    )


def run(
    experiment: Experiment,
    declaration_root: Path,
    endpoint: str,
    output: Path,
    billing: BillingReport | None,
    workspace_root: Path | None = None,
) -> None:
    """Prepare equal cases, schedule isolated arms, compare only valid pairs and retain failures."""
    if experiment.probe_samples is not None and workspace_root is None:
        raise ValueError("Shared probes require --workspace-root for the isolated local facade")
    case_path = (declaration_root / experiment.case).resolve()
    recipe = CaseRecipe.model_validate_json(case_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    (output / "experiment.json").write_text(experiment.model_dump_json(indent=2))
    identity = hashlib.sha256(
        experiment.model_dump_json(
            exclude={"repeats", "posterior", "reservation_usd_per_fit", "concurrent_arms"}
        ).encode()
    ).hexdigest()[:12]
    rows: list[JsonObject] = []
    try:
        for seed in experiment.seeds:
            for repeat in range(experiment.repeats):
                clients = tuple(
                    ActionClient(
                        arm.endpoint or endpoint,
                        f"bench-{identity}-g{seed.generation}-s{seed.sampler}-{i}-r{repeat}",
                        output / f"g{seed.generation}-s{seed.sampler}" / arm.name / f"r{repeat}",
                    )
                    for i, arm in enumerate(experiment.arms)
                )
                with ThreadPoolExecutor(max_workers=experiment.concurrent_arms) as pool:
                    previous: tuple[GitOid | None, ...] = tuple(None for _ in clients)
                    if experiment.probe_samples is not None:
                        assert workspace_root is not None
                        source = experiment.probe_anchor
                        anchor_client = ActionClient(
                            endpoint,
                            source.workspace
                            if source is not None
                            else clients[0].workspace + "-anchor",
                            output / f"anchor-s{seed.sampler}-r{repeat}",
                        )
                        if source is None:
                            anchor_case = prepare_case(
                                recipe, case_path.parent, anchor_client, seed.generation
                            )
                            base = FitSettingsSpec.model_validate(
                                experiment.base.revised(seed=seed.sampler).model_dump(
                                    mode="json", exclude={"retain_latent_paths"}
                                )
                            )
                            anchor = _fit(anchor_client, anchor_case, base, repeat)
                        else:
                            anchor_case = PreparedCase.model_validate_json(
                                (declaration_root / source.case_manifest).read_text()
                            )
                            anchor = anchor_client.saved("fit", source.call_id)
                            if anchor.action != "fit":
                                raise TypeError("Anchor must be a fit")
                            inputs = anchor_client.fit_source(source.call_id)
                            if (
                                inputs.dynamical_model_spec_ref != anchor_case.model_ref
                                or inputs.data_ref != anchor_case.data_ref
                            ):
                                raise ValueError("Anchor journal differs from its case manifest")
                            verified = prepare_case(
                                recipe,
                                case_path.parent,
                                ActionClient(
                                    endpoint,
                                    clients[0].workspace + "-verify",
                                    output / "verified-case",
                                ),
                                seed.generation,
                            )
                            if (
                                verified.model_digest,
                                verified.data_digest,
                                verified.recipe_digest,
                            ) != (
                                anchor_case.model_digest,
                                anchor_case.data_digest,
                                anchor_case.recipe_digest,
                            ):
                                raise ValueError("Anchor belongs to a different scientific case")
                        (anchor_client.output / "case.json").write_text(
                            anchor_case.model_dump_json(indent=2)
                        )
                        checkpoint = anchor.body.inference.evidence.checkpoint
                        if (
                            checkpoint is None
                            or checkpoint.settings.seed != seed.sampler
                            or checkpoint.settings.num_chains != experiment.base.num_chains
                        ):
                            raise ValueError(
                                "Anchor must retain the requested chains and random stream"
                            )
                        digest = fingerprint(result_payload(checkpoint))
                        for client in clients:
                            hashes = copy_anchor_workspace(
                                workspace_root, anchor_client, client.workspace
                            )
                            copied = client.saved("fit", anchor.call_id)
                            copied_checkpoint = (
                                copied.body.inference.evidence.checkpoint
                                if copied.action == "fit"
                                else None
                            )
                            if (
                                copied_checkpoint is None
                                or fingerprint(result_payload(copied_checkpoint)) != digest
                            ):
                                raise ValueError("Copied anchor checkpoint differs")
                            (client.output / "anchor.json").write_text(
                                json.dumps(
                                    {
                                        "workspace": anchor_client.workspace,
                                        "call_id": anchor.call_id,
                                        "revision": anchor.commit_id,
                                        "checkpoint_sha256": digest,
                                        "ledger_files": hashes,
                                    },
                                    indent=2,
                                )
                            )
                        cases = tuple(anchor_case for _ in clients)
                        previous = tuple(anchor.commit_id for _ in clients)
                    else:
                        cases = tuple(
                            pool.map(
                                partial(
                                    prepare_case, recipe, case_path.parent, seed=seed.generation
                                ),
                                clients,
                            )
                        )
                    for client, case in zip(clients, cases, strict=True):
                        (client.output / "case.json").write_text(case.model_dump_json(indent=2))
                    if (
                        len(
                            {
                                (case.model_digest, case.data_digest, case.recipe_digest)
                                for case in cases
                            }
                        )
                        != 1
                    ):
                        raise ValueError(
                            "Prepared arms have different scientific model or data content"
                        )
                    reference = None
                    if experiment.posterior is not None:
                        rule = experiment.posterior.references[seed.generation]
                        reference_case = PreparedCase.model_validate_json(
                            (declaration_root / rule.case_manifest).read_text()
                        )
                        if (reference_case.model_digest, reference_case.data_digest) != (
                            cases[0].model_digest,
                            cases[0].data_digest,
                        ):
                            raise ValueError("Reference was fitted to a different model or dataset")
                        reference_client = ActionClient(
                            rule.endpoint or endpoint, rule.workspace, output / "reference"
                        )
                        saved = reference_client.saved("fit", rule.call_id)
                        if saved.action != "fit":
                            raise TypeError("Reference must be a fit")
                        source_inputs = reference_client.fit_source(rule.call_id)
                        if (
                            source_inputs.dynamical_model_spec_ref != reference_case.model_ref
                            or source_inputs.data_ref != reference_case.data_ref
                        ):
                            raise ValueError(
                                "Reference journal does not match its supplied case manifest"
                            )
                        reference = saved.body
                        qualify_reference(reference, rule)
                    settings = tuple(
                        assigned_settings(experiment, arm, seed.sampler) for arm in experiment.arms
                    )
                    jobs = tuple(zip(clients, cases, settings, previous, strict=True))
                    results_by_batch: list[SuccessfulPoll[Literal["fit"], FitOutput]] = []
                    for batch in batched(jobs, experiment.concurrent_arms):
                        fit_futures = tuple(
                            pool.submit(_fit, client, case, setting, repeat, prior)
                            for client, case, setting, prior in batch
                        )
                        # A lost HTTP response can leave accepted work running.
                        # Resolve this batch before admitting any further arms.
                        results_by_batch.extend(future.result() for future in fit_futures)
                    results = tuple(results_by_batch)
                    controls = tuple(_controls(result.body) for result in results)
                    factors = (
                        (*experiment.factors, "runtime.devices")
                        if "hardware" in experiment.factors
                        else experiment.factors
                    )
                    for control in controls[1:]:
                        if mismatch := differing_controls(controls[0], control, factors):
                            raise ValueError(f"Pair differs beyond assigned treatment: {mismatch}")
                    for arm, client, case, setting, result, control in zip(
                        experiment.arms, clients, cases, settings, results, controls, strict=True
                    ):
                        segment = result.body.inference.core.inference_metadata.segments[-1]
                        assigned = SamplerSpec.model_validate(setting.model_dump(mode="json"))
                        if segment.settings != assigned:
                            raise ValueError(
                                "Resolved fit settings differ from the assigned controls"
                            )
                        from evaluation.comparison import flattened

                        runtime = flattened(
                            cast("JsonObject", segment.runtime.model_dump(mode="json"))
                        )
                        if any(runtime.get(key) != value for key, value in arm.runtime.items()):
                            raise ValueError(
                                "Returned runtime differs from the declared code/compiler treatment"
                            )
                        row: JsonObject = {
                            "arm": arm.name,
                            "generation_seed": seed.generation,
                            "sampler_seed": seed.sampler,
                            "repeat": repeat,
                            "workspace": client.workspace,
                            "call_id": result.call_id,
                            "reused": client.reused[result.call_id],
                            "revision": result.commit_id,
                            "compute_seconds": result.body.inference.core.inference_metadata.duration_seconds,
                            "compile_seconds": compile_seconds(result.body),
                            **segment_summary(result.body),
                            **initializer_scores(result.body, reference),
                            **sampler_outcomes(result.body),
                        }
                        if (
                            reference is not None
                            and experiment.posterior is not None
                            and segment.stop > segment.start
                        ):
                            row.update(
                                posterior_equivalence(
                                    result.body, reference, experiment.posterior.tolerance, segment
                                )
                            )
                        if segment.stop > segment.start:
                            layout, atoms = _posterior_draws(result.body, segment)
                            row["truth_absolute_errors"] = {
                                layout.labels[element]: abs(
                                    float(atoms[:, column].mean())
                                    - recipe.truth[layout.labels[element]]
                                )
                                for element, column in layout.parameter_columns.items()
                                if layout.labels[element] in recipe.truth
                            }
                        attempt = {
                            "case": case.model_dump(mode="json"),
                            "controls": control,
                            "summary": row,
                        }
                        (client.output / f"repeat-{repeat}.json").write_text(
                            json.dumps(attempt, indent=2)
                        )
                        rows.append(row)
    except Exception as exc:
        (output / "failure.json").write_text(
            json.dumps({"type": type(exc).__name__, "detail": str(exc)}, indent=2)
        )
        rows.append({"status": "failed", "detail": str(exc)})
        raise
    finally:
        write_report(output / "report.html", experiment, rows, billing)


def main() -> None:
    """Execute an explicitly supplied declaration against an already running study facade."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("declaration", type=Path)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--workspace-root", type=Path, help="Local isolated facade data root for shared probes"
    )
    parser.add_argument(
        "--billing-report",
        type=Path,
        help="Saved output of modal billing report; preserved verbatim",
    )
    args = parser.parse_args()
    billing = BillingReport.read(args.billing_report) if args.billing_report else None
    run(
        Experiment.model_validate_json(args.declaration.read_text()),
        args.declaration.parent,
        args.endpoint,
        args.output,
        billing,
        args.workspace_root,
    )


if __name__ == "__main__":
    main()
