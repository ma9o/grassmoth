"""Documentation heuristics for authoring prior parameters."""

from dataclasses import dataclass
from typing import Final, Literal

from nof1_causal_lab.artifacts.probability.families import PriorDistributionFamily


@dataclass(frozen=True)
class PriorParameterGuidanceRow:
    """Parameter-level prior heuristics reused across model-spec prompts."""

    parameter_type: str
    typical_distribution: str
    typical_range: str
    scale: str


LAGGED_BETA_AUTHORED_INTERVAL_SCALE: Final[str] = (
    "Authored interval effect (`transform.interval_days` names a positive duration "
    "or explicitly selects `model_clock`)"
)


PRIOR_PARAMETER_GUIDANCE_ROWS: Final[tuple[PriorParameterGuidanceRow, ...]] = (
    PriorParameterGuidanceRow(
        "beta (causal effect)",
        "Normal(0, 0.5)",
        "[-2, 2]",
        LAGGED_BETA_AUTHORED_INTERVAL_SCALE,
    ),
    PriorParameterGuidanceRow(
        "rho (AR coefficient)",
        "Beta(2, 2) or Uniform(0, 1)",
        "[0, 1]",
        "Baseline discrete-time persistence absent feedback",
    ),
    PriorParameterGuidanceRow("sigma (residual SD)", "HalfNormal(1)", "[0, 5]", "Data scale"),
    PriorParameterGuidanceRow(
        "t0_mean (initial-state mean)",
        "Normal(0, 1)",
        "[-3, 3]",
        "Latent state scale; do not copy raw indicator means or log-means unless the construct is explicitly identified on that observed scale",
    ),
    PriorParameterGuidanceRow(
        "t0_sd (initial-state SD)",
        "HalfNormal(1)",
        "[0, 3]",
        "Latent state scale",
    ),
    PriorParameterGuidanceRow(
        "lambda (loading)",
        "HalfNormal(1) if positive, TruncatedNormal(-1, 0.5, -5, 0) if negative",
        "[-3, 3]",
        "Data scale with sign fixed by indicator polarity",
    ),
    PriorParameterGuidanceRow(
        "obs_sd (measurement error SD)",
        "HalfNormal(0.5) or HalfNormal(1)",
        "[0, 3]",
        "Manifest observation-noise scale; larger values attribute more variation to indicator noise instead of the latent state",
    ),
    PriorParameterGuidanceRow(
        "obs_df (Student-t tails)",
        "Gamma(5, 1) or LogNormal(log(5), 0.3)",
        "[2, 30]",
        "Observation-tail heaviness; smaller values mean heavier tails",
    ),
    PriorParameterGuidanceRow(
        "obs_shape (Gamma shape)",
        "Gamma(2, 1)",
        "[0.5, 10]",
        "Observation overdispersion/shape for Gamma-family emissions",
    ),
    PriorParameterGuidanceRow(
        "obs_r (negative-binomial dispersion)",
        "Gamma(2, 0.5)",
        "[0.5, 20]",
        "Observation overdispersion; smaller values imply heavier count overdispersion",
    ),
    PriorParameterGuidanceRow(
        "obs_concentration (Beta concentration)",
        "Gamma(5, 0.5)",
        "[1, 50]",
        "Observation concentration around the latent mean on (0, 1)",
    ),
    PriorParameterGuidanceRow(
        "obs_ordered_base_<indicator> (ordered thresholds)",
        "Normal(0, 1)",
        "[-3, 3]",
        "Indicator-specific ordered-logistic threshold location on the latent predictor scale",
    ),
    PriorParameterGuidanceRow(
        "obs_ordered_gaps_<indicator> (ordered threshold gaps)",
        "HalfNormal(1)",
        "[0, 3]",
        "Indicator-specific positive spacing between adjacent ordered-logistic thresholds",
    ),
    PriorParameterGuidanceRow(
        "obs_cat_intercepts (categorical logits)",
        "Normal(0, 1)",
        "[-4, 4]",
        "Baseline category-logit offsets on the latent predictor scale",
    ),
    PriorParameterGuidanceRow(
        "obs_cat_slopes (categorical logits)",
        "Normal(0, 1)",
        "[-4, 4]",
        "Category-specific slope adjustments on the latent predictor scale; when every "
        "indicator of a construct is categorical, the reference channel's first "
        "non-baseline slope is compiler-pinned to +1 as the scale/sign anchor and the "
        "prior applies to the remaining slopes",
    ),
    PriorParameterGuidanceRow(
        "cor (correlation)",
        "Uniform(-1, 1) or TruncatedNormal(0, 0.3, -1, 1)",
        "[-1, 1]",
        "Innovation correlation",
    ),
    PriorParameterGuidanceRow("tau (random SD)", "HalfNormal(0.5)", "[0, 2]", "Data scale"),
)


def render_prior_parameter_guidance_markdown_table() -> str:
    """Render a markdown table for common parameter-level prior defaults."""
    lines = [
        "| Type | Typical Distribution | Typical Range | Scale |",
        "|---|---|---|---|",
    ]
    for row in PRIOR_PARAMETER_GUIDANCE_ROWS:
        lines.append(
            f"| {row.parameter_type} | {row.typical_distribution} | {row.typical_range} | {row.scale} |"
        )
    return "\n".join(lines)


@dataclass(frozen=True)
class PriorFamilySpec:
    """Central prior-family metadata shared across prompts, docs, and runtime."""

    family: PriorDistributionFamily
    summary: str
    support: Literal["real", "positive", "unit_interval", "bounded"]

    @property
    def signature(self) -> str:
        """Render the approved NumPyro constructor in authoring vocabulary."""
        from nof1_causal_lab.artifacts.probability.native import authored_argument_names

        arguments = ", ".join(authored_argument_names(self.family).values())
        return f"{self.family.value}({arguments})"


PRIOR_FAMILY_SPECS: Final[tuple[PriorFamilySpec, ...]] = (
    PriorFamilySpec(
        family=PriorDistributionFamily.NORMAL,
        summary="Unconstrained effects that can be positive or negative.",
        support="real",
    ),
    PriorFamilySpec(
        family=PriorDistributionFamily.HALF_NORMAL,
        summary="Positive-only parameters such as standard deviations and scales.",
        support="positive",
    ),
    PriorFamilySpec(
        family=PriorDistributionFamily.BETA,
        summary="Parameters constrained to the unit interval [0, 1].",
        support="unit_interval",
    ),
    PriorFamilySpec(
        family=PriorDistributionFamily.UNIFORM,
        summary="Hard-bounded parameters when only plausible limits are known.",
        support="bounded",
    ),
    PriorFamilySpec(
        family=PriorDistributionFamily.TRUNCATED_NORMAL,
        summary="Bounded parameters when both a center and hard limits are meaningful.",
        support="bounded",
    ),
    PriorFamilySpec(
        family=PriorDistributionFamily.GAMMA,
        summary="Positive-only parameters when right-skewed uncertainty is plausible.",
        support="positive",
    ),
    PriorFamilySpec(
        family=PriorDistributionFamily.LOG_NORMAL,
        summary="Positive-only parameters when uncertainty is multiplicative on the log scale.",
        support="positive",
    ),
    PriorFamilySpec(
        family=PriorDistributionFamily.EXPONENTIAL,
        summary="Positive-only parameters with mass near zero and a single decay rate.",
        support="positive",
    ),
)
