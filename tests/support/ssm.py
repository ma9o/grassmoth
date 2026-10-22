"""Shared fixtures for SSM contracts, inference, and predictive test suites."""

from __future__ import annotations

from grassmoth.artifacts.model.likelihood import LinkFunction

# ══════════════════════════════════════════════════════════════════════════════
# OBSERVATION FAMILY MATRIX
# ══════════════════════════════════════════════════════════════════════════════


def complex_mixed_family_config() -> tuple[list[str], list[str], list[int], list[str]]:
    manifest_dists = [
        "gaussian",
        "bernoulli",
        "poisson",
        "student_t",
        "gamma",
        "beta",
        "ordered_logistic",
        "categorical",
        "negative_binomial",
        "gaussian",
    ]
    manifest_links = [
        LinkFunction.IDENTITY.value,
        LinkFunction.LOGIT.value,
        LinkFunction.LOG.value,
        LinkFunction.IDENTITY.value,
        LinkFunction.LOG.value,
        LinkFunction.LOGIT.value,
        LinkFunction.CUMULATIVE_LOGIT.value,
        LinkFunction.SOFTMAX.value,
        LinkFunction.LOG.value,
        LinkFunction.IDENTITY.value,
    ]
    manifest_level_counts = [0, 0, 0, 0, 0, 0, 4, 4, 0, 0]
    manifest_names = [
        "stress_cont",
        "adherence_flag",
        "steps_count",
        "fatigue_t",
        "screen_gap",
        "sleep_efficiency",
        "symptom_severity",
        "coping_style",
        "rumination_count",
        "focus_cont",
    ]
    return manifest_dists, manifest_links, manifest_level_counts, manifest_names


# ══════════════════════════════════════════════════════════════════════════════
# PRIOR-PREDICTIVE RUNTIME SPEC
# ══════════════════════════════════════════════════════════════════════════════
