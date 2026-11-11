"""Launch choices for the exact point-program time loops."""

from typing import Annotated, Literal

from pydantic import Field

PERSISTENT_WIDE_ENTRIES: Annotated[
    int,
    Field(
        description=(
            "Heuristic: use the padded N*M*D workload to choose a block. Scalar SV L40S C80/T3000: "
            "at N2048/M2, sixteen forward warps reduced the stacked sweep 94.91 to 82.17 ms and "
            "registers 255 to 124; at N256/M2 sixteen regressed 22.21 to 27.59 ms. "
            "Higher dimensions require their own correctness/resource gates; their speed is unmeasured."
        )
    ),
] = 4096
PERSISTENT_NARROW_WARPS: Annotated[
    int,
    Field(
        description=(
            "Eight forward warps won the N256/M2 scalar stack comparison, 22.21 against 27.59 ms "
            "for sixteen. Bistable C8/N256/M2, padded D4: 8/8 forward/backward warps won at 4.865 ms; 4/4 took 6.365, 8/4 5.148, 16/8 5.165, and 16/16 5.171 ms. Eight backward warps passed the streamed exactness gates."
        )
    ),
] = 8
PERSISTENT_WIDE_WARPS: Annotated[
    int,
    Field(
        description=(
            "Sixteen forward warps won N2048/M2 scalar stack timing, 82.17 against 94.91 ms for eight."
        )
    ),
] = 16


def persistent_warps(
    particles: int, labels: int, dimension: int, direction: Literal["forward", "backward"]
) -> int:
    """Scale the forward block by its padded workload; keep the measured backward block."""
    padded = 1 << max(dimension - 1, 0).bit_length()
    return (
        PERSISTENT_WIDE_WARPS
        if direction == "forward" and particles * labels * padded >= PERSISTENT_WIDE_ENTRIES
        else PERSISTENT_NARROW_WARPS
    )


STREAMED_BACKWARD_HISTORY_ENTRIES: Annotated[
    int,
    Field(
        description=(
            "Heuristic crossover bracket from paired L40S sweeps, persistent forward held fixed: "
            "bistable T360/N256/M2/C8: history 3.747 versus streamed 4.732 ms; "
            "SV T3000/N256/M2/C80: history 24.120 versus streamed 25.894 ms; "
            "SV T3000/N2048/M2/C80: streamed 79.179 versus history 86.287 ms. "
            "Choose streaming from 2**22 history entries per chain, between the measured "
            "1536000 and 12288000 endpoints. The exact crossover within the bracket is unmeasured."
        )
    ),
] = 1 << 22


def stream_backward_workload(steps: int, particles: int, labels: int) -> bool:
    """Retain cached moments for small histories; recompute when the history is large."""
    return steps * particles * labels >= STREAMED_BACKWARD_HISTORY_ENTRIES


PERSISTENT_PROGRAM_ENTRIES: Annotated[
    int,
    Field(
        description=(
            "Finished scalar L40S gates cover 4096 padded particle-label-coordinate entries. "
            "The D9/N128/M16 full probe (32768 entries) did not finish within the roughly "
            "400-second bounded attempt, and its candidate-only compile did not finish within a 240-second app cap. Larger blocks retain the existing kernels; this changes "
            "no accepted request setting. Speed above two latent states is unmeasured."
        )
    ),
] = 4096


def compiled_point_workload(particles: int, labels: int, dimension: int) -> bool:
    """Statically retain the existing kernels above the bounded compilation envelope."""
    padded = 1 << max(dimension - 1, 0).bit_length()
    return particles * labels * padded <= PERSISTENT_PROGRAM_ENTRIES
