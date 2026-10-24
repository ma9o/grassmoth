"""Assemble parameter blocks after establishing required execution choices."""

from __future__ import annotations

from typing import TYPE_CHECKING

from grassmoth.artifacts.probability.families import DistributionFamily
from grassmoth.compilation_errors import IncompleteModelError
from grassmoth.models.ssm.compile.layout.axes import observed_indicators, state_ids
from grassmoth.models.ssm.compile.layout.observations import (
    loading_block,
    observation_mean_block,
    observation_noise_block,
)
from grassmoth.models.ssm.compile.layout.states import (
    diffusion_block,
    initial_covariance_block,
    initial_mean_block,
)
from grassmoth.models.ssm.compile.layout.static_factors import static_scale_block

if TYPE_CHECKING:
    from grassmoth.models.ssm.structure.blocks import (
        DiffusionBlockSpec,
        ManifestCholBlockSpec,
        SparseBlockSpec,
        T0CholBlockSpec,
    )
    from grassmoth.models.structure import StructuralSelection


def parameter_blocks(
    selection: StructuralSelection,
) -> tuple[
    DiffusionBlockSpec,
    SparseBlockSpec[tuple[int, int]],
    SparseBlockSpec[int],
    ManifestCholBlockSpec,
    SparseBlockSpec[int],
    T0CholBlockSpec,
    SparseBlockSpec[int],
]:
    """Compile all parameter blocks in the order expected by SSM assembly.

    Args:
        selection: Scientific model and outcome determining the execution structure.

    Returns:
        Diffusion, loadings, observation means, observation noise, initial means,
        initial covariance, and static-factor scale blocks, in that order.
    """
    return (
        diffusion_block(selection),
        loading_block(selection),
        observation_mean_block(selection),
        observation_noise_block(selection),
        initial_mean_block(selection),
        initial_covariance_block(selection),
        static_scale_block(selection),
    )


def _require_execution_choices(selection: StructuralSelection) -> None:
    """Check numerical execution requirements."""
    from grassmoth.models.structure import validate_execution_structure

    validate_execution_structure(selection)
    dynamical_model_spec = selection.dynamical_model_spec

    for identity in state_ids(selection):
        construct = dynamical_model_spec.get_construct(identity)
        if construct.role == "exogenous":
            if construct.distribution is None:
                raise IncompleteModelError(
                    f"Exogenous construct {construct.name!r} requires a deterministic trajectory law"
                )
            continue
        if any(construct.coefficient(role) is None for role in ("initial_mean", "initial_scale")):
            raise IncompleteModelError(
                f"Construct {construct.name!r} requires initial-state coefficients"
            )
        if (
            construct.temporal_status == "time_varying"
            and construct.coefficient("diffusion_scale") is None
        ):
            raise IncompleteModelError(
                f"Construct {construct.name!r} requires an innovation distribution"
            )
        if (
            construct.innovation_family == DistributionFamily.STUDENT_T
            and construct.coefficient("process_degrees_of_freedom") is None
        ):
            raise IncompleteModelError(
                f"{construct.name}.process_degrees_of_freedom requires an explicit coefficient"
            )
    for indicator in observed_indicators(selection):
        likelihood = indicator.likelihood
        if likelihood is None:
            raise IncompleteModelError(
                f"Retained indicator {indicator.observation.name!r} requires a likelihood"
            )
        terms = likelihood.parsed
        missing = [operand.role for operand in terms.operands if operand.value is None]
        if missing:
            raise IncompleteModelError(
                f"Indicator {indicator.observation.name!r} requires explicit measurement coefficients: {missing}"
            )
