"""Convergence criteria measured directly on producer-owned chain diagnostics."""

from __future__ import annotations

from grassmoth.artifacts.reports.checks import (
    ConvergenceCriterion,
    ConvergenceSubject,
    Evaluated,
    LatentConvergenceSubject,
    NotEvaluated,
    NumericCriterionEvidence,
)
from grassmoth.artifacts.reports.posterior_diagnostics import (
    ChainDiagnostics,
    ConvergenceReport,
)

# Practical cutoff documented by RStan: https://mc-stan.org/rstan/reference/Rhat.html.
# The warning retains Vehtari et al. (2021)'s stricter diagnostic signal:
# https://doi.org/10.1214/20-BA1221. Neither is a familywise error guarantee.
R_HAT_LIMIT = 1.05
R_HAT_WARNING = 1.01
ESS_PER_CHAIN = 100


def chain_convergence(chains: ChainDiagnostics | None) -> ConvergenceReport:
    """Assess retained chain convergence or report explicitly that chain evidence is unavailable."""
    if chains is None:
        return ConvergenceReport(
            findings=(
                NotEvaluated(
                    code="parameter_chains",
                    subject="recorded_parameter_chains",
                    reason="NO_RETAINED_CHAINS",
                    detail="No retained chain measurements are available.",
                ),
            ),
        )
    findings = []
    minimum = ESS_PER_CHAIN * chains.num_chains
    coordinates = tuple(
        (
            ConvergenceSubject(parameter=p.subject, label=p.parameter),
            p.r_hat,
            p.ess_bulk,
            p.ess_tail,
        )
        for p in chains.per_parameter
    ) + tuple(
        (LatentConvergenceSubject(state=p.subject, label=p.label), p.r_hat, p.ess_bulk, p.ess_tail)
        for p in chains.per_latent or ()
    )
    if chains.per_latent is None:
        findings.append(
            NotEvaluated(
                code="latent_chains",
                subject="recorded_latent_chains",
                reason="NO_RETAINED_CHAINS",
                detail="Latent chain convergence was not measured.",
            )
        )
    for subject, r_hat, ess_bulk, ess_tail in coordinates:
        for criterion, value, lower, upper in (
            (ConvergenceCriterion.R_HAT, r_hat, None, R_HAT_LIMIT),
            (ConvergenceCriterion.ESS_BULK, ess_bulk, float(minimum), None),
            (ConvergenceCriterion.ESS_TAIL, ess_tail, float(minimum), None),
        ):
            if value is None:
                findings.append(
                    NotEvaluated(
                        code=criterion.value,
                        subject=subject,
                        reason="INSUFFICIENT_CHAIN_SAMPLES",
                        detail=f"{subject.label}: {criterion.value} is unavailable.",
                    )
                )
            else:
                passes = value < upper if upper is not None else value >= minimum
                evidence = NumericCriterionEvidence(
                    criterion=criterion.value,
                    value=value,
                    lower=lower,
                    upper=upper,
                    upper_inclusive=upper is None,
                    note=subject.label,
                )
                findings.append(
                    Evaluated(
                        code=criterion.value,
                        subject=subject,
                        outcome="passed" if passes else "failed",
                        evidence=evidence,
                    )
                )
    return ConvergenceReport(
        findings=tuple(findings),
        warnings=tuple(
            f"{subject.label}: r_hat is {r_hat:g}; inspect mixing before relying on this fit."
            for subject, r_hat, _, _ in coordinates
            if r_hat is not None and R_HAT_WARNING <= r_hat < R_HAT_LIMIT
        ),
    )


def convergence_failures(report: ConvergenceReport) -> tuple[str, ...]:
    """Action messages use the same measured verdict, without reconstruction."""
    return report.messages
