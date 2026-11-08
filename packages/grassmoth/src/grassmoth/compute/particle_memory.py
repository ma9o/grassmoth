"""Analytical storage envelope for the current marginalized particle implementation.

This is a census of owned initialization and sampling arrays, not an XLA peak
prediction. Named intermediates overlap only within their consuming stages; no fitted
coefficient, allocator allowance or empirical workspace multiplier is applied.
The resource owner reserves hardware headroom and compiled admission remains
authoritative. Arbitrary authored-expression temporaries and backend workspaces
cannot be determined from these structural dimensions alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from grassmoth.artifacts.base import Value
from grassmoth.artifacts.reports.posterior_diagnostics import ParticleWeightMetric
from grassmoth.artifacts.sampling import (
    MGRAD_STATE_LIMIT,
    GuidedMethodSpec,
    InferenceHardware,
    MarginalParticleGibbsSpec,
    SamplerSpec,
)
from grassmoth.compute.budget import MemoryDemand

type ParticleMemoryPhase = Literal["initialization", "sampling"]
type _MemoryStage = Literal[
    "initialization", "forward", "concatenation", "backward", "projection", "retention"
]

_KERNEL_STAGES: tuple[_MemoryStage, ...] = ("forward", "concatenation", "backward")
_SAMPLING_STAGES: tuple[_MemoryStage, ...] = (*_KERNEL_STAGES, "projection", "retention")


class ParticleMgradMemoryShape(Value):
    """Compiled model and observation extents; no sampled numerical values."""

    time_points: int = Field(ge=1)
    state_dimensions: int = Field(ge=1)
    initial_dimensions: int = Field(ge=0)
    transition_dimensions: int = Field(ge=0)
    parameter_dimensions: int = Field(ge=0)
    observation_dimensions: int = Field(ge=0)
    free_path_coordinates: int = Field(ge=0)
    initial_covariance: Literal["diagonal", "dense"]
    transition_covariance: Literal["diagonal", "dense"]
    sweep_block_size: int = Field(ge=1)


class ParticleMgradMemoryFailure(Value):
    """Metadata or implementation support prevents analytical admission."""

    message: str


class ParticleMgradMemoryBuffer(Value):
    """One named group of simultaneously budgeted arrays and its source."""

    name: str
    phase: ParticleMemoryPhase
    demand: MemoryDemand
    source: str


class ParticleMgradMemoryReport(Value):
    """Owned storage envelope, excluding opaque compiler/expression workspaces."""

    shape: ParticleMgradMemoryShape
    initialization: MemoryDemand
    sampling: MemoryDemand
    peak: MemoryDemand
    buffers: tuple[ParticleMgradMemoryBuffer, ...]
    interpretation: str = (
        "Analytical owned-buffer envelope, taking the maximum of stage-specific live buffers. "
        "No calibrated multiplier or extra compiler-workspace allowance is included. "
        "Authored-expression temporaries, allocator fragmentation and backend workspaces are "
        "not shape-determined. Hardware reserve and compiled memory admission still apply. "
        "Remote hardware uses the CUDA branch; LOCAL conservatively includes CPU reference "
        "noise buffers even when its actual device is a GPU. "
        "Diagnostics do not enlarge this initialization/sampling device envelope."
    )


@dataclass(frozen=True)
class _BufferLifetime:
    """Private liveness policy; public reports retain their existing buffer fields."""

    buffer: ParticleMgradMemoryBuffer
    stages: tuple[_MemoryStage, ...]


def _buffer(
    name: str,
    stages: tuple[_MemoryStage, ...],
    source: str,
    *,
    device: int = 0,
    host: int = 0,
) -> _BufferLifetime:
    return _BufferLifetime(
        buffer=ParticleMgradMemoryBuffer(
            name=name,
            phase="initialization" if "initialization" in stages else "sampling",
            demand=MemoryDemand(host_bytes=host, device_bytes=device),
            source=f"{source}. Live stages: {', '.join(stages)}.",
        ),
        stages=stages,
    )


def _history_elements(
    shape: ParticleMgradMemoryShape, labels: int, metrics: tuple[str, ...]
) -> int:
    """The runner's ten scalar fields, D frozen fractions, M label scores and selected groups.

    Every retained telemetry item is float32 or int32. Weight telemetry comprises
    the eight ParticleWeightMetric histories, each with T filter coordinates.
    This follows mpgibbs.runner.sweep and particle.diagnostics, not a benchmark.
    """
    time = shape.time_points
    return (
        10
        + shape.state_dimensions
        + labels
        + (time + 2 if "particle_identity" in metrics else 0)
        + (1 if "parameter_movement" in metrics else 0)
        + (time if "regional_movement" in metrics else 0)
        + (len(ParticleWeightMetric) * time if "particle_weights" in metrics else 0)
    )


def estimate_particle_mgrad_memory(
    shape: ParticleMgradMemoryShape, sampler: SamplerSpec
) -> ParticleMgradMemoryReport | ParticleMgradMemoryFailure:
    """Count current float32/int32 buffers with concrete M, N and C settings.

    Four bytes per float/int and one per boolean are the production GPU dtypes.
    The vector backward pass retains two T*N*M*Q whitened transition banks plus
    the gathered T*N*Q input while preparing them, never an N*N table. Q is the
    transition dimension; the forward bank and normal draws still use all D.
    Matrix counts below enumerate owned curvature/solve operands; their overlap
    is conservative and is deliberately separate from unknown XLA scratch.
    Initialization conservatively includes a fresh Gaussian/IEKS start even for
    continuation. The application owns retained checkpoint storage and reserves
    its actual cached size separately in the occupied host-memory budget.
    """
    options = sampler.algorithm
    if (
        not isinstance(options, MarginalParticleGibbsSpec)
        or not isinstance(options.path_update.method, GuidedMethodSpec)
        or options.path_update.method.kind != "particle_mgrad"
    ):
        return ParticleMgradMemoryFailure(
            message="Analytical sizing supports MPGibbs with Particle-mGRAD only."
        )
    if (failure := sampler.support_failure) is not None:
        return ParticleMgradMemoryFailure(message=failure.message)
    if shape.state_dimensions > MGRAD_STATE_LIMIT:
        return ParticleMgradMemoryFailure(
            message=f"Particle-mGRAD supports at most {MGRAD_STATE_LIMIT} compiled state dimensions; got {shape.state_dimensions}."
        )
    t, d, q = shape.time_points, shape.state_dimensions, shape.transition_dimensions
    u, y = shape.parameter_dimensions, shape.observation_dimensions
    c, m, n = (
        sampler.num_chains,
        options.n_parameter_particles,
        options.path_update.method.num_particles,
    )
    w, s, block = sampler.num_warmup, sampler.num_samples_per_chain, shape.sweep_block_size
    padded_t = (
        (t - 1 + options.path_update.method.scan_unroll - 1)
        // options.path_update.method.scan_unroll
    ) * options.path_update.method.scan_unroll
    padded_n = 1 << (n - 1).bit_length()
    padded_m = 1 << (m - 1).bit_length()
    diagonal = shape.initial_covariance == shape.transition_covariance == "diagonal"
    covariance_width = d if diagonal else d * d
    transition_width = q if shape.transition_covariance == "diagonal" else q * q
    panel = 4 * (t * (y + d + 1)) + t * (y + d)
    # Carry owns paths, gradients/parameters, Cholesky, DA state, path scales,
    # curvature moments and covariance moments. Old/new copies overlap a move.
    carry = 4 * (c * (4 * t * d + 8 * t + 2 * u + u * u + 8) + 2 * u * u + 2 * u)
    context_size = 4 * (3 * d * d + y * d + y * y + 3 * d + 3 * y + t + u)
    context = c * (m + 1) * context_size
    histories = _history_elements(shape, m, tuple(options.diagnostic_metrics))
    mandatory_outputs = _history_elements(
        shape,
        m,
        ("particle_identity", "parameter_movement", "regional_movement", "particle_weights"),
    )
    # ParticleTarget sums channels before returning its T observation factors.
    posterior = 4 * c * s * (t * d + u + histories + t)
    warmup = 4 * c * w * histories
    combined = 4 * c * (w + s) * histories
    transfer = 4 * c * block * (t * d + u + mandatory_outputs)
    projection_transfer = 4 * c * block * t
    # The diagonal CUDA tail draws killing/labels inline from (T-1,4,2) keys.
    # Its CPU interpreter retains the four reference draw histories. Initial
    # killing/label draws and a shared dummy row remain on the CUDA branch.
    inline_random = diagonal and sampler.hardware is not InferenceHardware.LOCAL
    resampling_noise = 4 * c * ((5 * n if inline_random else 4 * t * n) + 8 * max(1, t - 1))
    # Scalar backward selection consumes unwhitened means and one point bank;
    # vector selection first gathers points, then whitens both banks by label.
    backward_width = n * (m + 1) * q if q <= 1 else n * (2 * m + 1) * q
    buffers = (
        _buffer(
            "panel",
            ("initialization",),
            "BoundPanel values, times, input values and masks",
            device=panel,
            host=panel,
        ),
        _buffer(
            "affine_chart",
            ("initialization",),
            "joint_coordinates.affine_chart: path, covariance, embedded covariance and Cholesky",
            device=4 * t * (d + 3 * d * d) + t * d + 4 * shape.free_path_coordinates,
        ),
        _buffer(
            "ieks_curvature",
            ("initialization",),
            "PathCurvature: four state matrices, reading Jacobian/precision/gradient, predictors and fixed mask",
            device=4 * t * (4 * d * d + y * d + y * y + 2 * y) + t * d,
        ),
        _buffer(
            "ieks_smoother",
            ("initialization",),
            "smoothing_mean/future_information: information, affine solve and associative-scan matrix/vector operands",
            device=4 * t * (12 * d * d + 10 * d),
        ),
        _buffer(
            "gaussian_starts",
            ("initialization",),
            "initialization.gaussian_initialization: noise, decoded parameters/paths and parameter Cholesky",
            device=4 * (c * (max(1, u + shape.free_path_coordinates) + u + t * d) + u * u),
        ),
        _buffer(
            "initial_population",
            ("initialization",),
            "runner particle_chain_initializer, curvature geometry and adaptation population",
            device=carry + c * context_size,
        ),
        _buffer(
            "panel",
            _SAMPLING_STAGES,
            "BoundPanel values, times, input values and masks",
            device=panel,
            host=panel,
        ),
        _buffer(
            "population",
            _SAMPLING_STAGES,
            "Old/new TrajectoryMCMCState, adaptation and parameter-label contexts",
            device=2 * carry + context,
        ),
        _buffer(
            "particle_banks",
            _KERNEL_STAGES,
            "guided forward state, prefix/observation label scores and weights, with unroll padding; production requests neither proposal-centre traces nor backward genealogy",
            device=4 * c * (padded_t + 1) * n * (d + 2 * m + 1),
        ),
        _buffer(
            "concatenation_outputs",
            ("concatenation",),
            "guided memory_bank and weights concatenate their first row with the scan outputs; preserve both inputs and new outputs without assuming XLA aliasing",
            device=4 * c * t * n * (d + 1),
        ),
        _buffer(
            "forward_normal_noise",
            ("forward",),
            "guided first and subsequent normal draws use N-1 particles in all D compiled coordinates",
            device=4 * c * t * (n - 1) * d,
        ),
        _buffer(
            "resampling_noise",
            ("forward",),
            "diagonal CUDA TailRandomKeys replaces four reference histories with inline draws, initial draws and dummy rows; LOCAL/dense retains reference histories",
            device=resampling_noise,
        ),
        _buffer(
            "backward_gumbels",
            ("backward",),
            "select_path generates (T-1,N) Gumbels after the forward scan",
            device=4 * c * (t - 1) * n,
        ),
        _buffer(
            "backward_moments",
            ("backward",),
            "guided_selection: vector whitened means/points plus raw gathered point input; scalar means/points stay unwhitened",
            device=4 * c * (t - 1) * backward_width,
        ),
        _buffer(
            "transition_density_factors",
            _KERNEL_STAGES,
            "ParticleNoiseFactors transition whiteners and normalizers serve both forward scoring and backward selection",
            device=4 * c * t * m * (transition_width + 1),
        ),
        _buffer(
            "observation_density_factors",
            ("forward",),
            "prepare_grouped_density per-time Gaussian whiteners/normalizers and fixed scalar event terms per label; disjoint group squares sum to at most Y², and group normalizers plus scalar terms to at most Y",
            device=4 * c * t * m * (y * y + y),
        ),
        _buffer(
            "initial_density_factors",
            _KERNEL_STAGES,
            "initial covariance factor, inverse factor and normalizer per label; backward suffix selection scores the initial density again",
            device=4
            * c
            * m
            * (
                2
                * (
                    shape.initial_dimensions
                    if shape.initial_covariance == "diagonal"
                    else shape.initial_dimensions**2
                )
                + 1
            ),
        ),
        _buffer(
            "proposal_geometry",
            ("forward",),
            "guided_geometry: prior map, auxiliary map, centre, Cholesky, whitener and normalizer",
            device=4 * c * t * m * (4 * covariance_width + d + 1),
        ),
        _buffer(
            "forward_workspace",
            ("forward",),
            "One forward step's label means, reading derivatives, selected bank and padded tile reductions",
            device=4
            * c
            * (n * (m * (2 * d + y + 3) + 6 * covariance_width + 6 * d) + padded_n * padded_m * 4),
        ),
        _buffer(
            "retained_histories",
            _SAMPLING_STAGES,
            "run_sweeps assembles posterior on host; completed warmup telemetry stays on device; host combined diagnostics have their own storage",
            device=warmup,
            host=posterior + warmup + combined,
        ),
        _buffer(
            "posterior_device_history",
            ("retention",),
            "run_sweeps uploads posterior paths/positions/telemetry and per-time observation factors after the last block and projection have completed",
            device=posterior,
        ),
        _buffer(
            "sweep_output_block",
            (*_KERNEL_STAGES, "projection"),
            "sweep_block fixed-width outputs and one host device_get transfer; disabled metrics still have compiled outputs",
            device=transfer,
            host=transfer,
        ),
        _buffer(
            "projection_output_block",
            ("projection",),
            "compile_projected_sweeps returns (block,C,T) channel-summed observation factors and transfers them once to host",
            device=projection_transfer,
            host=projection_transfer,
        ),
        _buffer(
            "observation_projection",
            ("projection",),
            "projected_sweeps per-channel predictor, reading and reduction intermediates; paths/positions already belong to sweep_output_block; particle workspaces have finished",
            device=4 * c * block * 3 * t * y,
        ),
    )

    def total(stage: _MemoryStage) -> MemoryDemand:
        return MemoryDemand(
            host_bytes=sum(
                item.buffer.demand.host_bytes for item in buffers if stage in item.stages
            ),
            device_bytes=sum(
                item.buffer.demand.device_bytes for item in buffers if stage in item.stages
            ),
        )

    initialization = total("initialization")
    sampling_stages = tuple(total(stage) for stage in _SAMPLING_STAGES)
    sampling = MemoryDemand(
        host_bytes=max(stage.host_bytes for stage in sampling_stages),
        device_bytes=max(stage.device_bytes for stage in sampling_stages),
    )
    return ParticleMgradMemoryReport(
        shape=shape,
        initialization=initialization,
        sampling=sampling,
        peak=MemoryDemand(
            host_bytes=max(initialization.host_bytes, sampling.host_bytes),
            device_bytes=max(initialization.device_bytes, sampling.device_bytes),
        ),
        buffers=tuple(item.buffer for item in buffers),
    )
