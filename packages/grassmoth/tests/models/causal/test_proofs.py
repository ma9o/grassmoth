"""Proof-carrying boundaries for numeric causal analysis."""

from __future__ import annotations

import subprocess
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING

import jax.numpy as jnp
import pytest

from grassmoth.artifacts.identity import GitRef
from grassmoth.artifacts.reports.identification import (
    IdentificationReport,
    IdentifiedTreatmentStatus,
    NonIdentifiableTreatmentStatus,
)
from grassmoth.models.causal.proofs import (
    CausalCertificationError,
    CertifiedCausalAnalysis,
    certify_identified_estimand,
)
from grassmoth.models.ssm.inference.types import JointPosteriorDraws
from grassmoth_test_support.git import git_oid
from grassmoth_test_support.model import load_model_fixture
from grassmoth_test_support.values import revise_fixture


def _treatment_outcome() -> DynamicalModelSpec:
    return load_model_fixture("causal_proofs/treatment_outcome.json")


def _conditioned_treatment_outcome() -> DynamicalModelSpec:
    return load_model_fixture("causal_proofs/conditioned_treatment_outcome.json")


if TYPE_CHECKING:
    from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec


pytestmark = pytest.mark.contract


def _identification():
    dynamical_model_spec = _treatment_outcome()
    return IdentificationReport(
        outcome=dynamical_model_spec.constructs[1].id,
        treatments={
            dynamical_model_spec.constructs[0].id: IdentifiedTreatmentStatus(
                estimand="E[outcome | do(treatment)]"
            )
        },
    )


def test_identification_proof_is_estimand_specific() -> None:
    design = _treatment_outcome()
    design_ref = GitRef(workspace_id="workspace", revision=git_oid(1), path="model.json")

    proof = certify_identified_estimand(
        design,
        _identification(),
        dynamical_model_spec_ref=design_ref,
        treatment="treatment",
        outcome="outcome",
    )

    assert proof.treatment == "treatment"
    assert proof.outcome == "outcome"
    assert proof.estimand == "E[outcome | do(treatment)]"


def test_identification_contract_rejects_linear_iv_evidence_for_nonlinear_models():
    from pydantic import ValidationError

    payload = _identification().model_dump(mode="json")
    finding = next(iter(payload["treatments"].values()))
    finding.update(method="instrumental_variable", estimand="IV(Z) [requires linearity]")
    with pytest.raises(ValidationError, match="Extra inputs"):
        IdentificationReport.model_validate(payload)


@pytest.mark.parametrize(
    ("explicit_finding", "_design_payload"),
    [
        pytest.param(
            False,
            _treatment_outcome,
            id="False",
        ),
        pytest.param(
            True,
            _treatment_outcome,
            id="True",
        ),
    ],
)
def test_identification_proof_rejects_unidentified_treatment(
    explicit_finding, _design_payload
) -> None:
    model = _design_payload()
    report = IdentificationReport(
        outcome=model.constructs[1].id,
        treatments={model.constructs[0].id: NonIdentifiableTreatmentStatus()}
        if explicit_finding
        else {},
    )
    with pytest.raises(CausalCertificationError, match="is not identified"):
        certify_identified_estimand(
            model,
            report,
            dynamical_model_spec_ref=GitRef(
                workspace_id="workspace", revision=git_oid(1), path="model.json"
            ),
            treatment="treatment",
            outcome="outcome",
        )


def test_conditioning_rejects_warmup_statically(tmp_path):
    probe = tmp_path / "conditioning_types.py"
    probe.write_text(
        dedent("""\
            from datetime import UTC, datetime
            from jax import Array
            from grassmoth.models.ssm.compile.inputs import CompiledDynamicalModel
            from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec
            from grassmoth.actions.fit.conditioning import condition_model
            from grassmoth.models.ssm.inference.types import MCMCPosterior
            from grassmoth.models.ssm.inference.warmup.initialization import ParticleInitialization

            def condition(model: DynamicalModelSpec, compiled: CompiledDynamicalModel, posterior: MCMCPosterior, warmup: ParticleInitialization, times: Array):
                condition_model(model, compiled, posterior, times=times, time_origin=datetime(2024, 1, 1, tzinfo=UTC))
                condition_model(model, compiled, warmup, times=times, time_origin=datetime(2024, 1, 1, tzinfo=UTC))
            """)
    )
    checked = subprocess.run(
        [
            "ty",
            "check",
            str(probe),
            "--project",
            str(Path(__file__).parents[3]),
            "--output-format",
            "concise",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert checked.returncode == 1
    assert checked.stdout.count("error[") == 1, checked.stdout + checked.stderr
    assert "invalid-argument-type" in checked.stdout
    assert "MCMCPosterior" in checked.stdout
    assert "ParticleInitialization" in checked.stdout


def test_causal_reporting_requires_retained_uncertainty_and_matching_model():
    from grassmoth.models.causal.proofs import certify_conditioned_model

    dynamical_model_spec = _conditioned_treatment_outcome()
    certify_conditioned_model(dynamical_model_spec, dynamical_model_spec)
    with pytest.raises(CausalCertificationError, match="differs from"):
        certify_conditioned_model(_treatment_outcome(), dynamical_model_spec)
    prior = _treatment_outcome()
    with pytest.raises(CausalCertificationError, match="retained joint uncertainty"):
        certify_conditioned_model(prior, prior)


def test_causal_analysis_joins_matching_proofs():
    design = _conditioned_treatment_outcome()
    design_ref = GitRef(workspace_id="workspace", revision=git_oid(2), path="model.json")
    analysis = CertifiedCausalAnalysis(
        dynamical_model_spec=design,
        identification=_identification(),
        dynamical_model_spec_ref=design_ref,
        estimands=(
            certify_identified_estimand(
                design,
                _identification(),
                dynamical_model_spec_ref=design_ref,
                treatment="treatment",
                outcome="outcome",
            ),
        ),
        fitted_dynamical_model_spec=design,
    )
    assert analysis.treatments == ["treatment"]
    assert analysis.outcome == "outcome"


@pytest.mark.parametrize(
    ("parameters", "paths", "message"),
    [
        ({"beta": jnp.zeros((3, 2))}, jnp.zeros((2, 4, 2)), "share the draw axis"),
        ({"beta": jnp.zeros(3), "gamma": jnp.zeros(4)}, None, "share the draw axis"),
        ({"beta": jnp.zeros(3)}, jnp.zeros((1, 3, 4, 2)), "draw, time, and state axes"),
        ({"beta": jnp.array(1.0)}, None, "leading draw axis"),
    ],
)
def test_joint_posterior_rejects_misaligned_or_ambiguous_draw_axes(parameters, paths, message):
    with pytest.raises(ValueError, match=message):
        JointPosteriorDraws(parameters=parameters, latent_paths=paths)


def test_unconverged_causal_summary_keeps_effect_and_failed_criteria():
    from datetime import date

    import numpy as np

    from grassmoth.actions.contracts.io import SimulationInputs
    from grassmoth.actions.simulate.effects import summarize_causal_simulation
    from grassmoth.artifacts.arrays import NumericalArray
    from grassmoth.artifacts.reports.predictive_provenance import AuthoredLawProvenance
    from grassmoth.artifacts.scenarios import CausalEffectResult, InterventionSpec
    from grassmoth.artifacts.simulation import SimulationArm, SimulationSpec
    from grassmoth.models.ssm.inference.diagnostics.convergence import chain_convergence
    from grassmoth.models.structure import StructuralSelection
    from grassmoth_test_support.inference import _report

    model = _conditioned_treatment_outcome()
    treatment, outcome = (construct.id for construct in model.constructs)
    report = _report(model).core
    assert report.inference_diagnostics is not None
    convergence = chain_convergence(
        revise_fixture(
            report.inference_diagnostics,
            per_parameter=tuple(
                revise_fixture(row, ess_bulk=264.0)
                for row in report.inference_diagnostics.per_parameter
            ),
        )
    )
    assert convergence.status == "failed"
    result = summarize_causal_simulation(
        SimulationInputs(
            selection=StructuralSelection(model, outcome),
            design=SimulationSpec(
                start=date(2024, 1, 1),
                horizon="1d",
                interventions=(InterventionSpec(target=treatment, value=1.0),),
            ),
            seed=0,
            num_replicates=2,
            law=AuthoredLawProvenance(),
            fitted=(
                GitRef(workspace_id="workspace", revision=git_oid(2), path="model.json"),
                revise_fixture(report, convergence=convergence),
            ),
        ),
        action=SimulationArm(
            latent_paths=NumericalArray.from_numpy(np.ones((2, 2, 2))),
            observations=NumericalArray.from_numpy(np.ones((2, 2, 0))),
        ),
        reference=SimulationArm(
            latent_paths=NumericalArray.from_numpy(np.zeros((2, 2, 2))),
            observations=NumericalArray.from_numpy(np.zeros((2, 2, 0))),
        ),
        state_ids=(treatment, outcome),
        indicator_ids=(),
    )
    assert isinstance(result, CausalEffectResult)
    np.testing.assert_array_equal(result.differences.values, np.ones((2, 2)))
    assert result.summary.mean == 1.0
    from grassmoth.actions.contracts.io import FittedLaw
    from grassmoth.actions.simulate.findings import _fit_findings
    from grassmoth.artifacts.observations.reference import DataRef
    from grassmoth.artifacts.reports.predictive_provenance import FittedLawProvenance

    (finding,) = _fit_findings(
        FittedLaw(
            provenance=FittedLawProvenance(
                fitted_data=DataRef(revision=git_oid(3), replicate_index=0),
                fitted_model_revision=git_oid(2),
                interpretation="posterior_predictive",
            ),
            report=revise_fixture(report, convergence=convergence),
        )
    )
    assert finding.kind == "evaluated"
    assert finding.code == "fit_reliability"
    assert finding.outcome == "failed"
    assert [(item.criterion, item.value, item.lower) for item in finding.evidence] == [
        ("ess_bulk", 264.0, 400.0)
    ]


def test_fit_findings_distinguish_authored_laws_from_unknown_fit_history():
    from grassmoth.actions.simulate.findings import _fit_findings
    from grassmoth.artifacts.reports.predictive_provenance import (
        AuthoredLawProvenance,
        UnknownLawProvenance,
    )

    assert _fit_findings(AuthoredLawProvenance()) == ()
    (finding,) = _fit_findings(UnknownLawProvenance())
    assert finding.kind == "not_evaluated"
    assert finding.code == "fit_reliability"
    assert finding.reason == "NO_RETAINED_CHAINS"
