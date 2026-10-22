"""Export generated docs from the central distribution catalog.

Usage:
    cd apps/grassmoth-server
    uv run python -m scripts.codegen.export_distribution_docs

Overwrites generated sections in:
  - docs/reference/statistical-model-spec/parameters.md  (Supported Prior Families, Common Defaults)
  - docs/reference/statistical-model-spec/likelihoods.md (Dtype-to-Distribution Mapping,
    Distribution Families, Link Functions)

Hand-written prose surrounding these sections is preserved.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from grassmoth.artifacts.model.likelihood import (
    OBSERVATION_FAMILY_SPECS,
    OBSERVATION_LINK_VALUES_BY_DISTRIBUTION,
)
from grassmoth.artifacts.observations.dtypes import (
    VALID_LIKELIHOODS_FOR_DTYPE,
)
from grassmoth.artifacts.probability.families import DistributionFamily
from scripts.codegen.distribution_guidance import (
    PRIOR_FAMILY_SPECS,
    render_prior_parameter_guidance_markdown_table,
)


@dataclass(frozen=True)
class ParameterRoleSpec:
    """Metadata for a parameter role used by docs codegen and validation."""

    role: str
    symbol: str
    count: str
    constraint: str
    ssm_location: str
    note: str = ""

    @property
    def domain(self) -> str:
        return {
            "unit_interval": "[0, 1]",
            "none": "(-inf, +inf)",
            "positive": "(0, +inf)",
            "negative": "(-inf, 0)",
            "correlation": "[-1, 1]",
        }[self.constraint]


PARAMETER_ROLE_SPECS: Final[tuple[ParameterRoleSpec, ...]] = (
    ParameterRoleSpec(
        role="ar_coefficient",
        symbol="rho",
        count="One per endogenous time-varying construct",
        constraint="unit_interval",
        ssm_location="State-decay dynamics site",
        note="model-spec elicits baseline discrete-time persistence absent feedback; "
        "compilation binds it to the owning decay component "
        "and converts to continuous-time decay scale",
    ),
    ParameterRoleSpec(
        role="fixed_effect",
        symbol="beta",
        count="One per causal edge",
        constraint="none",
        ssm_location="Dynamics edge or input-effect site",
        note="Causal effects can be positive or negative; compiler binds each "
        "coefficient to the owning edge component or known-input effect site",
    ),
    ParameterRoleSpec(
        role="dynamics_parameter",
        symbol="theta",
        count="One per real-valued component-owned dynamics parameter",
        constraint="none",
        ssm_location="Component dynamics site",
        note="Used for component-owned dynamics parameters that are not authored "
        "as interval-scale effect coefficients.",
    ),
    ParameterRoleSpec(
        role="dynamics_parameter_positive",
        symbol="theta+",
        count="One per positive component-owned dynamics parameter",
        constraint="positive",
        ssm_location="Component dynamics site",
        note="Used for positive component-owned dynamics parameters such as Hill Emax and EC50.",
    ),
    ParameterRoleSpec(
        role="residual_sd",
        symbol="sigma",
        count="One per construct",
        constraint="positive",
        ssm_location="Diffusion diagonal",
    ),
    ParameterRoleSpec(
        role="state_intercept",
        symbol="cint",
        count="One per eligible dynamic construct when equilibrium forcing is enabled",
        constraint="none",
        ssm_location="Continuous-time state intercept",
    ),
    ParameterRoleSpec(
        role="observation_intercept",
        symbol="manifest_mean",
        count="One per manifest channel whose observation family requires a baseline intercept",
        constraint="none",
        ssm_location="Manifest intercept vector",
    ),
    ParameterRoleSpec(
        role="initial_state_mean",
        symbol="t0_mean",
        count="One per latent construct",
        constraint="none",
        ssm_location="Initial-state mean vector",
    ),
    ParameterRoleSpec(
        role="initial_state_sd",
        symbol="t0_sd",
        count="One per latent construct",
        constraint="positive",
        ssm_location="Initial-state covariance diagonal",
    ),
    ParameterRoleSpec(
        role="static_state_sd",
        symbol="tau",
        count="One per compiled baseline factor induced by marginalized time-invariant confounders",
        constraint="positive",
        ssm_location="Static baseline-factor covariance",
        note="Used to build low-rank initial-state covariance contributions of the form "
        "`B diag(tau^2) B^T`.",
    ),
    ParameterRoleSpec(
        role="loading",
        symbol="lambda",
        count="One per non-reference indicator in multi-indicator constructs",
        constraint="positive",
        ssm_location="Observation model",
        note="measurement-structure indicator polarity fixes each loading sign as either "
        "`positive` or `negative`; model-spec no longer chooses loading orientation",
    ),
    ParameterRoleSpec(
        role="measurement_error_sd",
        symbol="obs_sd",
        count="One per free manifest measurement-error SD",
        constraint="positive",
        ssm_location="Manifest variance diagonal",
        note="Surfaced only when measurement error is separately estimated "
        "(multi-indicator constructs).",
    ),
    ParameterRoleSpec(
        role="observation_hyperparameter",
        symbol="obs_*",
        count="One per active real-valued observation-family hyperparameter site",
        constraint="none",
        ssm_location="Observation-family auxiliary site",
        note="Examples include ordered-threshold bases and categorical logit offsets.",
    ),
    ParameterRoleSpec(
        role="observation_hyperparameter_positive",
        symbol="obs_*",
        count="One per active positive observation-family hyperparameter site",
        constraint="positive",
        ssm_location="Observation-family auxiliary site",
        note="Examples include Student-t degrees of freedom, Gamma shape, and NB dispersion.",
    ),
    ParameterRoleSpec(
        role="correlation",
        symbol="cor",
        count="One per construct-pair with marginalized confounder",
        constraint="correlation",
        ssm_location="Diffusion covariance",
    ),
)


_REPO_ROOT = Path(__file__).resolve().parents[4]
_DOCS_DIR = _REPO_ROOT / "docs" / "reference" / "statistical-model-spec"
_PARAMETERS_PATH = _DOCS_DIR / "parameters.md"
_LIKELIHOODS_PATH = _DOCS_DIR / "likelihoods.md"

_DTYPE_ALTERNATIVE_NOTES: dict[tuple[str, DistributionFamily], str] = {
    ("categorical", DistributionFamily.ORDERED_LOGISTIC): (
        "when categories are substantively ordered"
    ),
}


def _render_prior_distribution_markdown_table() -> str:
    lines = [
        "| Family | Signature | Support | Use When |",
        "|---|---|---|---|",
    ]
    for spec in PRIOR_FAMILY_SPECS:
        lines.append(
            f"| `{spec.family.value}` | `{spec.signature}` | `{spec.support}` | {spec.summary} |"
        )
    return "\n".join(lines)


def _format_dist_with_links(dist: DistributionFamily) -> str:
    links = OBSERVATION_LINK_VALUES_BY_DISTRIBUTION[dist]
    if len(links) == 1:
        return f"`{dist.value}` (`{links[0]}`)"
    link_str = " or ".join(f"`{link}`" for link in links)
    return f"`{dist.value}` ({link_str})"


def _render_dtype_likelihood_markdown_table() -> str:
    lines = [
        "| `measurement_dtype` | Default distribution | Link | Alternatives |",
        "|---|---|---|---|",
    ]
    for dtype, valid_dists in VALID_LIKELIHOODS_FOR_DTYPE.items():
        default_dist = valid_dists[0]
        default_links = OBSERVATION_LINK_VALUES_BY_DISTRIBUTION[default_dist]
        alternatives = [f"`{default_dist.value}` with `{link}`" for link in default_links[1:]]
        for dist in valid_dists[1:]:
            entry = _format_dist_with_links(dist)
            note = _DTYPE_ALTERNATIVE_NOTES.get((dtype, dist))
            alternatives.append(f"{entry} {note}" if note else entry)
        alt_str = ", ".join(alternatives) if alternatives else "None"
        lines.append(f"| `{dtype}` | `{default_dist.value}` | `{default_links[0]}` | {alt_str} |")
    return "\n".join(lines)


def _render_distribution_families_prose() -> str:
    names = [f"`{spec.family.value}`" for spec in OBSERVATION_FAMILY_SPECS]
    return (
        "`DistributionFamily` names the internal numerical emission kernels: "
        f"{', '.join(names[:-1])}, and {names[-1]}."
    )


def _render_link_functions_prose() -> str:
    links = list(
        dict.fromkeys(
            f"`{link}`"
            for links in OBSERVATION_LINK_VALUES_BY_DISTRIBUTION.values()
            for link in links
        )
    )
    return (
        "`LinkFunction` names the internal responses derived from conditional expressions: "
        f"{', '.join(links[:-1])}, and {links[-1]}."
    )


def _render_parameter_roles_markdown_table() -> str:
    lines = [
        "| Role | Symbol | Count | Constraint | SSM location |",
        "|---|---|---|---|---|",
    ]
    for spec in PARAMETER_ROLE_SPECS:
        constraint_cell = f"`{spec.constraint}` `{spec.domain}`"
        if spec.role == "loading":
            constraint_cell = "`positive` or `negative`"
        lines.append(
            f"| `{spec.role}` | `{spec.symbol}` "
            f"| {spec.count} | {constraint_cell} | {spec.ssm_location} |"
        )
    return "\n".join(lines)


def _render_parameter_constraint_notes() -> str:
    notes = [spec for spec in PARAMETER_ROLE_SPECS if spec.note]
    return "\n".join(f"- `{spec.role}`: {spec.note}" for spec in notes)


def _replace_section(text: str, heading: str, new_body: str, path: Path) -> str:
    """Replace everything between *heading* and the next ``## `` heading."""
    pattern = re.compile(
        rf"({re.escape(heading)}\n)(.*?)(?=\n## |\Z)",
        re.DOTALL,
    )
    replacement = rf"\1\n{new_body}\n"
    result, n = pattern.subn(replacement, text)
    if n == 0:
        raise ValueError(f"Section '{heading}' not found in {path}")
    return result


def _export_parameters(*, check: bool) -> bool:
    """Regenerate the generated sections of parameters.md in-place."""
    original = _PARAMETERS_PATH.read_text()

    roles_body = "\n".join(
        [
            "A parameter's role comes from the coefficient slots that reference it. "
            "A fixed coefficient is a literal in its slot and needs no parameter; "
            "roles and constraints are not stored again on the parameter definition:",
            "",
            _render_parameter_roles_markdown_table(),
            "",
            "Constraint notes:",
            "",
            _render_parameter_constraint_notes(),
        ]
    )

    families_body = "\n".join(
        [
            _render_prior_distribution_markdown_table(),
            "",
            "The `Family` values are the exact canonical strings accepted by model-spec prior schemas; aliases are not supported.",
            "The `Use When` column is the authoritative short guidance reused by the model-spec prompts.",
        ]
    )

    defaults_body = render_prior_parameter_guidance_markdown_table()

    updated = _replace_section(original, "## Parameter Roles", roles_body, _PARAMETERS_PATH)
    updated = _replace_section(
        updated, "## Supported Prior Families", families_body, _PARAMETERS_PATH
    )
    updated = _replace_section(updated, "## Common Defaults", defaults_body, _PARAMETERS_PATH)

    if check:
        changed = updated != original
        if changed:
            print(f"  {_PARAMETERS_PATH.relative_to(_REPO_ROOT)}")
        return changed

    _PARAMETERS_PATH.write_text(updated)
    print(f"  parameters.md  -> {_PARAMETERS_PATH}")
    return updated != original


def _export_likelihoods(*, check: bool) -> bool:
    """Regenerate the generated sections of likelihoods.md in-place."""
    original = _LIKELIHOODS_PATH.read_text()

    dtype_body = "\n".join(
        [
            "The model author writes each indicator's law and coefficients through `edit_model`. "
            "Its `measurement_dtype` limits the families that law may use, and an indicator "
            "with any other family is rejected. The first column is the usual choice; "
            "the family and link names describe the law's numerical lowering.",
            "",
            _render_dtype_likelihood_markdown_table(),
        ]
    )

    updated = _replace_section(
        original, "## Dtype-to-Distribution Mapping", dtype_body, _LIKELIHOODS_PATH
    )
    updated = _replace_section(
        updated,
        "## Distribution Families",
        _render_distribution_families_prose(),
        _LIKELIHOODS_PATH,
    )
    updated = _replace_section(
        updated, "## Link Functions", _render_link_functions_prose(), _LIKELIHOODS_PATH
    )

    if check:
        changed = updated != original
        if changed:
            print(f"  {_LIKELIHOODS_PATH.relative_to(_REPO_ROOT)}")
        return changed

    _LIKELIHOODS_PATH.write_text(updated)
    print(f"  likelihoods.md -> {_LIKELIHOODS_PATH}")
    return updated != original


def export_prior_distribution_docs(*, check: bool = False) -> bool:
    """Write the authoritative statistical-model-spec reference pages."""
    parameters_changed = _export_parameters(check=check)
    likelihoods_changed = _export_likelihoods(check=check)
    return parameters_changed or likelihoods_changed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true", help="Verify generated docs without writing files."
    )
    args = parser.parse_args()

    has_changes = export_prior_distribution_docs(check=args.check)
    if args.check:
        if has_changes:
            print(
                "Distribution docs codegen is out of date. Run `bun run docs:codegen`.",
                file=sys.stderr,
            )
            sys.exit(1)
        print("Distribution docs checked.")
