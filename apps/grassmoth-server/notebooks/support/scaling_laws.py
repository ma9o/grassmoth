"""Scaling laws for GPU fits: the evidence table, its fitted constants and the per-stage estimate.

Every constant is refitted from ``scaling_laws_runs.json`` when the notebook loads, so a new row
changes the laws without touching code. Each law says what backs it: a chain ladder, one run, a spec
sheet, one pair of models, or the code's structure alone.
"""

from __future__ import annotations

import itertools
import json
import math
import statistics
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

type Method = Literal["mgrad", "kalman_whole", "kalman_blocks"]

GIB = 2**30
MIB = 2**20
FLOAT32_BYTES = 4
REFERENCE_T = 3000
REFERENCE_N = 128
REFERENCE_K = 2
PLANNING_FRACTION = 0.8  # MEMORY_BUDGET_FRACTION in compute/resources.py
HOST_REQUEST_GIB = 32  # _FIT_HOST_MEMORY_MIB in actions/fit/remote.py
TRANSFER_BLOCK = 25  # SWEEP_BLOCK_SIZE in samplers/sweep_plan.py
RETAINED_SCALARS = 15  # Mandatory per-sweep diagnostics in a saved SV fit
SV_PARAMETERS = 3

METHOD_LABELS: Mapping[Method, str] = {
    "mgrad": "mGRAD",
    "kalman_whole": "aux Kalman, whole path",
    "kalman_blocks": "aux Kalman, blocks",
}
LANE_LABELS: Mapping[Method, str] = {
    "mgrad": "mGRAD",
    "kalman_whole": "Kalman, whole",
    "kalman_blocks": "Kalman, blocks",
}
_SENTENCE_NAMES: Mapping[Method, str] = {
    "mgrad": "mGRAD",
    "kalman_whole": "whole-path Kalman",
    "kalman_blocks": "blocked Kalman",
}
METHODS: tuple[Method, ...] = ("mgrad", "kalman_whole", "kalman_blocks")
_LADDER_PREFIX: Mapping[Method, str] = {
    "mgrad": "sv-mgrad",
    "kalman_whole": "sv-kalman-whole",
    "kalman_blocks": "sv-kalman-blocks",
}
_REFERENCE_GPU: Mapping[Method, str] = {
    "mgrad": "H200",
    "kalman_whole": "L40S",
    "kalman_blocks": "L40S",
}


@dataclass(frozen=True)
class Run:
    """One saved fit, or one failed allocation, as a row of the evidence table.

    The file keeps more columns than the laws use yet, such as power and DRAM counters.
    """

    id: str
    model: str
    T: int
    D: int
    method: str
    block: int | None
    N: int | None
    K: int | None
    gpu: str
    C: int
    W: int | None
    S: int | None
    warmup_ms: float | None
    sampling_ms: float | None
    warmup_s: float | None
    compile_s: float | None
    mem_compile_gib: float | None
    mem_sampling_gib: float | None
    mem_conditioning_gib: float | None
    mem_diagnostics_gib: float | None
    peak_gib: float | None
    failed_alloc_gib: float | None
    conditioning_s: float | None
    diagnostics_s: float | None
    busy_pct: float | None
    clock_mhz: float | None
    pool_gib: float | None
    flops: float | None
    ladder: str | None
    build: str | None
    note: str | None
    source: str | None

    @property
    def variant(self) -> Method:
        """Split Kalman by whether it updates the whole path or blocks of it."""
        if self.method == "mgrad":
            return "mgrad"
        return "kalman_whole" if self.block is None else "kalman_blocks"

    @property
    def sweep_memory_gib(self) -> float | None:
        """Prefer the sweep's own peak; older runs recorded only the whole fit's."""
        return self.mem_sampling_gib if self.mem_sampling_gib is not None else self.peak_gib


@dataclass(frozen=True)
class Gpu:
    """Capacity and published speed of one GPU type."""

    name: str
    architecture: str
    vram_gb: float
    memory: str
    cores: int
    clock_mhz: float
    clock_assumed: bool
    dram_gb_s: float
    pool_gib: float
    pool_source: str
    offered: bool
    note: str | None

    @property
    def compute_power(self) -> float:
        """Cores times clock, the spec-sheet proxy for per-chain arithmetic speed."""
        return self.cores * self.clock_mhz


@dataclass(frozen=True)
class Evidence:
    """Everything the laws are fitted to and checked against."""

    gpus: tuple[Gpu, ...]
    runs: tuple[Run, ...]
    traces: Mapping[str, tuple[tuple[tuple[float, float], ...], ...]]

    @property
    def offered_gpus(self) -> tuple[str, ...]:
        """The GPUs the calculator offers; the others only supply evidence."""
        return tuple(gpu.name for gpu in self.gpus if gpu.offered)

    def gpu(self, name: str) -> Gpu:
        """Look a GPU up by name."""
        return next(gpu for gpu in self.gpus if gpu.name == name)

    def run(self, run_id: str) -> Run:
        """Look a run up by its identifier."""
        return next(run for run in self.runs if run.id == run_id)


def load_evidence() -> Evidence:
    """Read the evidence table and the allocator traces beside this module."""
    here = Path(__file__)
    raw = json.loads(here.with_name("scaling_laws_runs.json").read_text())
    traces = json.loads(here.with_name("scaling_laws_traces.json").read_text())
    used = {field.name for field in fields(Run)}
    return Evidence(
        tuple(Gpu(**record) for record in raw["gpus"]),
        tuple(
            Run(**{name: value for name, value in record.items() if name in used})
            for record in raw["runs"]
        ),
        {
            run_id: tuple(tuple((share, gib) for share, gib in stage) for stage in stages)
            for run_id, stages in traces.items()
        },
    )


@dataclass(frozen=True)
class Curve:
    """Sweep time a + b*C: fixed latency per sweep plus cost per chain."""

    a_ms: float
    b_ms: float

    def time_ms(self, chains: float) -> float:
        """Milliseconds per sweep at this chain count."""
        return self.a_ms + self.b_ms * chains

    def fill(self, chains: float) -> float:
        """Share of this model's throughput ceiling on this GPU that these chains reach."""
        return self.b_ms * chains / self.time_ms(chains)

    @property
    def half_chains(self) -> float:
        """Chains at which fill reaches 50%."""
        return self.a_ms / self.b_ms

    def chains_at(self, fill: float) -> float:
        """Chains needed to reach this fill."""
        return fill / (1 - fill) * self.half_chains

    @property
    def ceiling_per_second(self) -> float:
        """Chain-sweeps per second as chains grow without bound."""
        return 1000 / self.b_ms


def fit_line(points: Iterable[tuple[float, float]]) -> tuple[float, float]:
    """Least-squares intercept and slope."""
    xs, ys = zip(*points, strict=True)
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / sum(
        (x - mean_x) ** 2 for x in xs
    )
    return mean_y - slope * mean_x, slope


def fit_curve(points: Iterable[tuple[float, float]]) -> Curve:
    """Fit a + b*C to (chains, milliseconds per sweep) points."""
    a_ms, b_ms = fit_line(points)
    return Curve(a_ms, b_ms)


@dataclass(frozen=True)
class Laws:
    """Constants fitted to the evidence table; nothing here is typed in by hand."""

    ladders: Mapping[str, Curve]
    warmup_overhead_s: float
    projection_ms_per_chain: float
    bank_bytes: float
    kalman_bytes: float
    kalman_bytes_low: float
    kalman_bytes_high: float
    conditioning_units: float
    diagnostics_units: float
    diagnostics_offset_gib: float
    high_water_mib_per_chain: float
    compile_mib_per_chain: float
    post_intercept_s: float
    post_ns_per_coordinate: float
    kalman_d_exponent: float
    mgrad_d_exponent: float
    flops_constant: float
    flops_cubic: float


def pair_bytes(chains: int, labels: int, time_points: int, particles: int, states: int) -> float:
    """The mGRAD pair workspace; at one state it coincides with the scalar pair table."""
    return FLOAT32_BYTES * chains * labels * (time_points - 1) * particles**2 * states


def _retained_gib(run: Run) -> float:
    """Device bytes per float32 of retained path, in GiB, for this run's chains and draws."""
    assert run.S is not None
    return run.C * run.S * run.T * FLOAT32_BYTES / GIB


def fit_laws(evidence: Evidence) -> Laws:
    """Refit every constant from the evidence table."""
    runs = evidence.runs
    sv = tuple(run for run in runs if run.model == "SV")
    names = sorted({run.ladder for run in runs if run.ladder})
    ladders = {
        name: fit_curve(
            (run.C, run.sampling_ms)
            for run in runs
            if run.ladder == name and run.sampling_ms is not None
        )
        for name in names
    }

    overheads, projections = [], []
    for name, curve in ladders.items():
        warmups = [run for run in runs if run.ladder == name and run.warmup_s is not None]
        if len({run.C for run in warmups}) < 2:
            continue
        schedule = warmups[0].W
        assert schedule is not None
        intercept, slope = fit_line((run.C, run.warmup_s or 0.0) for run in warmups)
        overheads.append(intercept - schedule * curve.a_ms / 1000)
        projections.append(curve.b_ms - 1000 * slope / schedule)

    mgrad_memory = [
        run
        for run in sv
        if run.variant == "mgrad" and run.sweep_memory_gib is not None and run.N is not None
    ]
    bank = statistics.median(
        ((run.sweep_memory_gib or 0.0) * GIB - pair_bytes(run.C, 2, run.T, run.N or 0, run.D))
        / (run.C * (run.N or 0) * run.T)
        for run in mgrad_memory
    )
    kalman_per_time = [
        run.mem_sampling_gib * GIB / (run.C * run.T)
        for run in sv
        if run.method == "kalman" and run.mem_sampling_gib is not None
    ]

    post = [run for run in sv if run.mem_conditioning_gib is not None and run.S is not None]
    conditioning_units = sum(
        (run.mem_conditioning_gib or 0.0) * _retained_gib(run) for run in post
    ) / sum(_retained_gib(run) ** 2 for run in post)
    diagnostics_offset, diagnostics_units = fit_line(
        (_retained_gib(run), run.mem_diagnostics_gib or 0.0) for run in post
    )
    high_water = statistics.median(
        ((run.peak_gib or 0.0) - (run.mem_conditioning_gib or 0.0)) * 1024 / run.C
        for run in post
        if run.method == "kalman"
    )
    compile_mib = max(
        (run.mem_compile_gib or 0.0) * 1024 / run.C for run in sv if run.mem_compile_gib is not None
    )
    post_intercept, post_seconds = fit_line(
        (run.C * (run.S or 0) * run.T, (run.conditioning_s or 0.0) + (run.diagnostics_s or 0.0))
        for run in post
    )

    whole_reference = ladders["sv-kalman-whole-l40s"]
    sv_whole = evidence.run("profile/kalman-whole")
    lorenz_whole = evidence.run("lorenz96/kalman-whole")
    kalman_exponent = math.log(
        _kalman_per_chain_time_us(lorenz_whole, whole_reference)
        / _kalman_per_chain_time_us(sv_whole, whole_reference)
    ) / math.log(lorenz_whole.D)
    mgrad_reference = ladders["sv-mgrad-h200"]
    lorenz_mgrad = evidence.run("lorenz96/mgrad-full-c2")
    mgrad_exponent = math.log(
        _mgrad_per_pair_ns(lorenz_mgrad, mgrad_reference)
        / (1e6 * mgrad_reference.b_ms / (REFERENCE_T * REFERENCE_N**2 * REFERENCE_K))
    ) / math.log(lorenz_mgrad.D)

    sv_flops = flops_per_chain_time(sv_whole)
    lorenz_flops = flops_per_chain_time(lorenz_whole)
    cubic = (lorenz_flops - sv_flops) / (lorenz_whole.D**3 - 1)

    return Laws(
        ladders=ladders,
        warmup_overhead_s=statistics.fmean(overheads),
        projection_ms_per_chain=statistics.fmean(projections),
        bank_bytes=bank,
        kalman_bytes=statistics.median(kalman_per_time),
        kalman_bytes_low=min(kalman_per_time),
        kalman_bytes_high=max(kalman_per_time),
        conditioning_units=conditioning_units,
        diagnostics_units=diagnostics_units,
        diagnostics_offset_gib=diagnostics_offset,
        high_water_mib_per_chain=high_water,
        compile_mib_per_chain=compile_mib,
        post_intercept_s=post_intercept,
        post_ns_per_coordinate=post_seconds * 1e9,
        kalman_d_exponent=kalman_exponent,
        mgrad_d_exponent=mgrad_exponent,
        flops_constant=sv_flops - cubic,
        flops_cubic=cubic,
    )


def flops_per_chain_time(run: Run) -> float:
    """XLA's count for one sweep, per chain and time point; loops are counted once."""
    return (run.flops or 0.0) / (run.C * run.T)


def _latency_ms(reference: Curve, method: str, time_points: int) -> float:
    """Latency grows with the sequential scan for mGRAD, parallel levels for Kalman."""
    if method == "mgrad":
        return reference.a_ms * time_points / REFERENCE_T
    return reference.a_ms * math.log2(time_points) / math.log2(REFERENCE_T)


def _kalman_per_chain_time_us(run: Run, reference: Curve) -> float:
    """Per-chain sweep time per time point, assuming SV's latency carries over."""
    assert run.sampling_ms is not None
    per_chain = (run.sampling_ms - _latency_ms(reference, "kalman", run.T)) / run.C
    return 1000 * per_chain / run.T


def _mgrad_per_pair_ns(run: Run, reference: Curve) -> float:
    """Per-chain sweep time per scored particle pair, assuming SV's latency carries over."""
    assert run.sampling_ms is not None
    assert run.N is not None
    per_chain = (run.sampling_ms - _latency_ms(reference, "mgrad", run.T)) / run.C
    return 1e6 * per_chain / (run.T * run.N**2 * (run.K or REFERENCE_K))


def kalman_points(evidence: Evidence, laws: Laws) -> tuple[tuple[Run, float], ...]:
    """Per-chain time per time point for every Kalman run usable at its own state dimension."""
    chosen = (
        "profile/kalman-whole",
        "lorenz96/kalman-whole",
        "profile/kalman-blocked",
        "tail/seed-19",
        "france/block-100",
        "blocked-speed/lorenz-v4",
    )
    points = []
    for run_id in chosen:
        run = evidence.run(run_id)
        reference = laws.ladders[
            "sv-kalman-whole-l40s" if run.block is None else "sv-kalman-blocks-l40s"
        ]
        points.append((run, _kalman_per_chain_time_us(run, reference)))
    return tuple(points)


def mgrad_points(evidence: Evidence, laws: Laws) -> tuple[tuple[str, float, float], ...]:
    """Per-chain time per scored pair: SV from its ladder, Lorenz-96 from its one matched fit."""
    reference = laws.ladders["sv-mgrad-h200"]
    lorenz = evidence.run("lorenz96/mgrad-full-c2")
    return (
        ("SV", 1.0, 1e6 * reference.b_ms / (REFERENCE_T * REFERENCE_N**2 * REFERENCE_K)),
        ("Lorenz-96", float(lorenz.D), _mgrad_per_pair_ns(lorenz, reference)),
    )


@dataclass(frozen=True)
class Scenario:
    """The inputs of one estimate."""

    gpu: str
    method: Method
    chains: int
    particles: int = REFERENCE_N
    labels: int = REFERENCE_K
    block: int = 300
    windowed: bool = False
    time_points: int = REFERENCE_T
    states: int = 1
    warmup: int = 2000
    draws: int = 1000


def each_method(scenario: Scenario) -> tuple[Scenario, ...]:
    """The same settings once per lane; only the method differs."""
    return tuple(replace(scenario, method=method) for method in METHODS)


def scrub_steps(start: int, stop: int, *extra: int) -> list[int]:
    """Slider stops: every integer at the low end, then coarser, so dragging feels logarithmic."""
    grid = (
        *range(1, 101),
        *range(105, 505, 5),
        *range(520, 2020, 20),
        *range(2100, 10100, 100),
        *range(10500, 50500, 500),
        *range(52000, 202000, 2000),
        *range(210000, 1010000, 10000),
    )
    return sorted({value for value in (*grid, *extra) if start <= value <= stop})


def compile_bytes_per_chain(scenario: Scenario, laws: Laws) -> float:
    """Chain state plus autotuning scratch, scaled from SV by path size."""
    return laws.compile_mib_per_chain * MIB * scenario.time_points * scenario.states / REFERENCE_T


def sweep_bytes_per_chain(scenario: Scenario, laws: Laws) -> float:
    """Device bytes per chain while warmup and sampling sweeps run."""
    if scenario.method == "mgrad":
        return (
            pair_bytes(
                1, scenario.labels, scenario.time_points, scenario.particles, scenario.states
            )
            + laws.bank_bytes * scenario.particles * scenario.time_points * scenario.states
        )
    return laws.kalman_bytes * scenario.time_points * scenario.states**2


def post_bytes_per_chain(scenario: Scenario, laws: Laws) -> float:
    """Device high-water per chain after sampling. At one state, three path-sized arrays."""
    units = laws.conditioning_units * (2 * scenario.states + 1) / 3
    return (
        units * scenario.draws * scenario.time_points * FLOAT32_BYTES
        + laws.high_water_mib_per_chain * MIB
    )


def host_bytes_per_chain(scenario: Scenario) -> float:
    """Host bytes the planner reserves per chain: history, combined diagnostics and one transfer."""
    block = scenario.draws + TRANSFER_BLOCK
    scalars = 2 * (scenario.warmup + scenario.draws) + TRANSFER_BLOCK
    return FLOAT32_BYTES * (
        block * scenario.time_points * (scenario.states + 1)
        + block * SV_PARAMETERS
        + scalars * RETAINED_SCALARS
    )


def planner_device_overhead_per_chain(scenario: Scenario) -> float:
    """Device bytes the planner adds to the sweep: retained history and observation log-probs."""
    return FLOAT32_BYTES * (
        scenario.warmup * RETAINED_SCALARS
        + scenario.draws
        * (scenario.time_points * scenario.states + SV_PARAMETERS + RETAINED_SCALARS)
        + scenario.draws * scenario.time_points
    )


def post_seconds(scenario: Scenario, laws: Laws) -> float:
    """Conditioning plus diagnostics, which grow with the retained coordinates."""
    coordinates = scenario.chains * scenario.draws * scenario.time_points
    return (
        laws.post_intercept_s
        + laws.post_ns_per_coordinate * 1e-9 * coordinates * (scenario.states + 1) / 2
    )


def spec_curve(method: Method, gpu: Gpu, laws: Laws, evidence: Evidence) -> Curve:
    """Scale the reference ladder: latency stays, arithmetic follows cores times clock.

    mGRAD also streams its pair table once per sweep, so part of its per-chain cost follows DRAM
    bandwidth instead.
    """
    source = evidence.gpu(_REFERENCE_GPU[method])
    reference = laws.ladders[f"{_LADDER_PREFIX[method]}-{source.name.lower()}"]
    speed = source.compute_power / gpu.compute_power
    if method != "mgrad":
        return Curve(reference.a_ms, reference.b_ms * speed)
    table_gb = pair_bytes(1, REFERENCE_K, REFERENCE_T, REFERENCE_N, 1) / 1e9
    source_dram_ms = 1000 * table_gb / source.dram_gb_s
    target_dram_ms = 1000 * table_gb / gpu.dram_gb_s
    return Curve(reference.a_ms, target_dram_ms + (reference.b_ms - source_dram_ms) * speed)


@dataclass(frozen=True)
class SpeedEstimate:
    """A curve at this scenario's T, D, N and K, and what backs it."""

    curve: Curve
    basis: str
    measured: bool


def _base_curve(method: Method, gpu: Gpu, laws: Laws, evidence: Evidence) -> SpeedEstimate:
    """Best evidence first: a chain ladder, one run on this GPU, then the spec sheet."""
    ladder = laws.ladders.get(f"{_LADDER_PREFIX[method]}-{gpu.name.lower()}")
    if ladder is not None:
        return SpeedEstimate(ladder, f"chain ladder on {gpu.name}", measured=True)
    source = _REFERENCE_GPU[method]
    reference = laws.ladders[f"{_LADDER_PREFIX[method]}-{source.lower()}"]
    single = next(
        (
            run
            for run in evidence.runs
            if run.model == "SV"
            and run.variant == method
            and run.gpu == gpu.name
            and run.sampling_ms is not None
            and run.N in (None, REFERENCE_N)
        ),
        None,
    )
    if single is not None:
        assert single.sampling_ms is not None
        return SpeedEstimate(
            Curve(reference.a_ms, (single.sampling_ms - reference.a_ms) / single.C),
            f"one run on {gpu.name}, latency from {source}",
            measured=True,
        )
    clock = ", clock assumed" if gpu.clock_assumed else ""
    return SpeedEstimate(
        spec_curve(method, gpu, laws, evidence),
        f"no run on {gpu.name}: spec sheet from {source}{clock}",
        measured=False,
    )


def block_factor(block: int, evidence: Evidence) -> float:
    """France's measured sweep time at this block size, relative to its fastest."""
    measured = sorted(
        (run.block, run.sampling_ms or 0.0)
        for run in evidence.runs
        if run.id.startswith("france/block-") and run.block is not None
    )
    fastest = min(time for _, time in measured)
    if block <= measured[0][0]:
        return measured[0][1] / fastest
    if block >= measured[-1][0]:
        return measured[-1][1] / fastest
    for (left, left_time), (right, right_time) in itertools.pairwise(measured):
        if left <= block <= right:
            share = (math.log(block) - math.log(left)) / (math.log(right) - math.log(left))
            return (left_time + share * (right_time - left_time)) / fastest
    raise AssertionError(block)


def scenario_speed(scenario: Scenario, laws: Laws, evidence: Evidence) -> SpeedEstimate:
    """The reference curve at this GPU, scaled to this scenario's T, D, N, K and blocks."""
    base = _base_curve(scenario.method, evidence.gpu(scenario.gpu), laws, evidence)
    grain = scenario.time_points / REFERENCE_T
    latency = _latency_ms(base.curve, scenario.method, scenario.time_points)
    notes = [base.basis]
    if scenario.time_points != REFERENCE_T:
        notes.append("T from code")
    if scenario.states != 1:
        notes.append("D from one model pair")
    if scenario.method == "mgrad":
        per_chain = (
            base.curve.b_ms
            * grain
            * (scenario.particles / REFERENCE_N) ** 2
            * (scenario.labels / REFERENCE_K)
            * scenario.states**laws.mgrad_d_exponent
        )
        if scenario.particles != REFERENCE_N:
            notes.append("N rule")
        if scenario.labels != REFERENCE_K:
            notes.append("K from code")
    else:
        per_chain = base.curve.b_ms * grain * scenario.states**laws.kalman_d_exponent
        if scenario.labels != REFERENCE_K:
            notes.append("K not modelled")
    if scenario.method == "kalman_blocks" and scenario.windowed:
        factor = block_factor(scenario.block, evidence)
        latency, per_chain = latency * factor, per_chain * factor
        notes.append("France block curve")
    return SpeedEstimate(
        Curve(latency, per_chain), "; ".join(notes), measured=base.measured and len(notes) == 1
    )


def warmup_curve(scenario: Scenario, speed: Curve, laws: Laws) -> Curve:
    """Warmup skips the observation projection and path copy that sampling pays per chain."""
    saving = (
        laws.projection_ms_per_chain
        * scenario.time_points
        / REFERENCE_T
        * (scenario.states + 1)
        / 2
    )
    return Curve(speed.a_ms, max(speed.b_ms - saving, speed.b_ms / 2))


@dataclass(frozen=True)
class StageRow:
    """One stage of the estimate. Stages without sweeps describe the GPU instead of a fill."""

    stage: str
    memory_gib: float
    pool_share: float
    fill: float | None
    idle: str
    time: str


@dataclass(frozen=True)
class Limits:
    """Chain counts at which each resource runs out, for one method on one GPU."""

    half_fill: float
    high_fill: float
    planner: int
    sweep_pool: int
    post_pool: int

    @property
    def pool(self) -> int:
        """The first point at which any stage stops fitting the GPU's pool."""
        return min(self.sweep_pool, self.post_pool)


def scenario_limits(scenario: Scenario, laws: Laws, evidence: Evidence) -> Limits:
    """Find where each limit falls for this scenario's settings."""
    gpu = evidence.gpu(scenario.gpu)
    sweep = sweep_bytes_per_chain(scenario, laws)
    budget = PLANNING_FRACTION * gpu.pool_gib * GIB
    device = int(budget // (sweep + planner_device_overhead_per_chain(scenario)))
    host = int(PLANNING_FRACTION * HOST_REQUEST_GIB * GIB // host_bytes_per_chain(scenario))
    speed = scenario_speed(scenario, laws, evidence).curve
    return Limits(
        half_fill=speed.chains_at(0.5),
        high_fill=speed.chains_at(0.9),
        planner=min(device, host),
        sweep_pool=int(gpu.pool_gib * GIB // sweep),
        post_pool=int(gpu.pool_gib * GIB // post_bytes_per_chain(scenario, laws)),
    )


def _seen_compile_seconds(scenario: Scenario, evidence: Evidence) -> str:
    """The range saved runs actually took; it depends on the model, not on these inputs."""
    family = "mgrad" if scenario.method == "mgrad" else "kalman"
    near = [
        run.compile_s
        for run in evidence.runs
        if run.method == family
        and run.compile_s is not None
        and (run.D == 1) == (scenario.states == 1)
    ]
    return f"{min(near):.0f}–{max(near):.0f} s seen"


def estimate(scenario: Scenario, laws: Laws, evidence: Evidence) -> tuple[StageRow, ...]:
    """Memory, processing fill and time for each stage."""
    gpu = evidence.gpu(scenario.gpu)
    speed = scenario_speed(scenario, laws, evidence).curve
    chains = scenario.chains
    sweep = sweep_bytes_per_chain(scenario, laws) * chains / GIB
    post = post_bytes_per_chain(scenario, laws) * chains / GIB
    device = compile_bytes_per_chain(scenario, laws) * chains / GIB
    warmup = warmup_curve(scenario, speed, laws)
    warmup_seconds = laws.warmup_overhead_s + scenario.warmup * warmup.time_ms(chains) / 1000
    sampling_seconds = scenario.draws * speed.time_ms(chains) / 1000
    return (
        StageRow(
            "Compilation",
            device,
            device / gpu.pool_gib,
            None,
            "GPU idle",
            _seen_compile_seconds(scenario, evidence),
        ),
        StageRow(
            "Warmup",
            sweep,
            sweep / gpu.pool_gib,
            warmup.fill(chains),
            "",
            format_seconds(warmup_seconds),
        ),
        StageRow(
            "Sampling",
            sweep,
            sweep / gpu.pool_gib,
            speed.fill(chains),
            "",
            format_seconds(sampling_seconds),
        ),
        StageRow(
            "After sampling",
            post,
            post / gpu.pool_gib,
            None,
            "Mostly idle",
            format_seconds(post_seconds(scenario, laws)),
        ),
    )


def format_seconds(seconds: float) -> str:
    """Readable duration."""
    if seconds < 90:
        return f"{seconds:.1f} s" if seconds < 10 else f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


def format_share(share: float) -> str:
    """A percentage with a decimal only where it's small enough to need one."""
    return f"{100 * share:.1f}%" if share < 0.1 else f"{100 * share:.0f}%"


def memory_is_measured(scenario: Scenario) -> bool:
    """Memory demand follows runs only at the settings they covered; it doesn't depend on the GPU."""
    return (
        scenario.time_points == REFERENCE_T
        and scenario.states == 1
        and scenario.labels == REFERENCE_K
        and not (scenario.windowed and scenario.method == "kalman_blocks")
    )


def memory_basis(scenario: Scenario) -> str:
    """What backs the memory estimate at these settings."""
    mgrad = scenario.method == "mgrad"
    if scenario.states == 1:
        parts = ["measured" if mgrad else "measured, within about 20%"]
    else:
        parts = ["one check at 40 states" if mgrad else "upper bound from code"]
    if scenario.time_points != REFERENCE_T:
        parts.append("T from code")
    if scenario.labels != REFERENCE_K:
        parts.append("K from code" if mgrad else "K not modelled")
    if scenario.windowed and scenario.method == "kalman_blocks":
        parts.append("window memory not modelled")
    return "; ".join(parts)


@dataclass(frozen=True)
class Trace:
    """A saved run's allocator bytes per stage, as (share of the stage, GiB) points."""

    run_id: str
    stages: tuple[tuple[tuple[float, float], ...], ...]


def matching_trace(scenario: Scenario, evidence: Evidence) -> Trace | None:
    """The telemetry of a run at exactly these memory-relevant settings, if one was saved."""
    if not memory_is_measured(scenario):
        return None
    for run in evidence.runs:
        if (
            run.id in evidence.traces
            and run.gpu == scenario.gpu
            and run.variant == scenario.method
            and scenario.chains == run.C
            and scenario.draws == run.S
            and run.N in (None, scenario.particles)
            and (scenario.method != "kalman_blocks" or run.block == scenario.block)
        ):
            return Trace(run.id, evidence.traces[run.id])
    return None


type FitState = Literal["fits", "rejected", "after_sampling", "over_pool"]


def fit_state(scenario: Scenario, limits: Limits) -> FitState:
    """What happens to this chain count: the planner stops a fit before the pool would."""
    if scenario.chains > limits.sweep_pool:
        return "over_pool"
    if scenario.chains > limits.planner:
        return "rejected"
    if scenario.chains > limits.post_pool:
        return "after_sampling"
    return "fits"


_STATE_PHRASES: Mapping[FitState, tuple[str, str]] = {
    "fits": ("fits", "fit"),
    "rejected": ("is rejected by the planner", "are rejected by the planner"),
    "after_sampling": (
        "runs out of GPU memory after sampling",
        "run out of GPU memory after sampling",
    ),
    "over_pool": ("doesn't fit the GPU", "don't fit the GPU"),
}


def headline(scenario: Scenario, laws: Laws, evidence: Evidence) -> str:
    """The answer across all three methods in one sentence."""
    groups: dict[FitState, list[Method]] = {}
    for variant in each_method(scenario):
        state = fit_state(variant, scenario_limits(variant, laws, evidence))
        groups.setdefault(state, []).append(variant.method)
    start = f"At {scenario.chains:,} chains on {scenario.gpu}, "
    if len(groups) == 1:
        state = next(iter(groups))
        return f"{start}all three methods {_STATE_PHRASES[state][1]}."
    clauses = []
    for state, methods in groups.items():
        names = " and ".join(_SENTENCE_NAMES[method] for method in methods)
        clauses.append(f"{names} {_STATE_PHRASES[state][len(methods) > 1]}")
    return f"{start}{'; '.join(clauses)}."


def detail(scenario: Scenario) -> str:
    """Host RAM is the one limit the lanes share, because retained draws don't depend on method."""
    host = host_bytes_per_chain(scenario) * scenario.chains / GIB
    budget = PLANNING_FRACTION * HOST_REQUEST_GIB
    return f"Host demand is {host:.1f} of {budget:.1f} GiB in all three."


def lane_basis(scenario: Scenario, laws: Laws, evidence: Evidence) -> str:
    """What backs one lane's numbers, in words, because hatching alone can't say."""
    speed = scenario_speed(scenario, laws, evidence)
    trace = matching_trace(scenario, evidence)
    note = f"Memory: {memory_basis(scenario)}. Time: {speed.basis}."
    if trace is None:
        return note
    return f"{note} Thin line: saved run {trace.run_id.removeprefix('profile/')}."


type Backing = Literal["measured", "pair", "code"]


@dataclass(frozen=True)
class ScalingCell:
    """How one quantity scales with one setting; empty text means flat."""

    text: str
    backing: Backing


@dataclass(frozen=True)
class ScalingRow:
    """One setting across memory, latency, per-chain cost, after sampling and compile, per method."""

    setting: str
    cells: Mapping[Method, tuple[ScalingCell, ...]]


_FLAT = ScalingCell("", "measured")
SCALING_COLUMNS = ("Mem", "a", "b", "After", "Comp")


def scaling_rows(scenario: Scenario, laws: Laws) -> tuple[ScalingRow, ...]:
    """Every setting's scaling per method; a method missing from a row doesn't use that setting."""

    def cell(text: str, backing: Backing) -> ScalingCell:
        return ScalingCell(text, backing)

    def everywhere(*cells: ScalingCell) -> Mapping[Method, tuple[ScalingCell, ...]]:
        return dict.fromkeys(METHODS, cells)

    def both_kalman(*cells: ScalingCell) -> dict[Method, tuple[ScalingCell, ...]]:
        return {"kalman_whole": cells, "kalman_blocks": cells}

    slight = cell("slight", "code")
    block = (
        (cell("rises", "code"), cell("rises", "measured"), cell("rises", "measured"), _FLAT, _FLAT)
        if scenario.windowed
        else (_FLAT, cell("log L", "code"), _FLAT, _FLAT, _FLAT)
    )
    return (
        ScalingRow(
            "Chains", everywhere(cell("C", "measured"), _FLAT, _FLAT, cell("C", "measured"), _FLAT)
        ),
        ScalingRow("Retained draws", everywhere(_FLAT, _FLAT, _FLAT, cell("S", "measured"), _FLAT)),
        ScalingRow(
            "Time points",
            {
                "mgrad": (
                    cell("T", "pair"),
                    cell("T", "code"),
                    cell("T", "code"),
                    cell("T", "code"),
                    _FLAT,
                ),
                **both_kalman(
                    cell("T", "code"),
                    cell("log T", "code"),
                    cell("T", "code"),
                    cell("T", "code"),
                    _FLAT,
                ),
            },
        ),
        ScalingRow(
            "States",
            {
                "mgrad": (
                    cell("D", "pair"),
                    cell("?", "code"),
                    cell(f"D<sup>{laws.mgrad_d_exponent:.1f}</sup>", "pair"),
                    cell("D", "code"),
                    cell("grows", "pair"),
                ),
                **both_kalman(
                    cell("D<sup>2</sup>", "code"),
                    cell("?", "code"),
                    cell(f"D<sup>{laws.kalman_d_exponent:.1f}</sup>", "pair"),
                    cell("D", "code"),
                    cell("grows", "pair"),
                ),
            },
        ),
        ScalingRow(
            "MPGibbs particles",
            {
                "mgrad": (cell("K", "code"), _FLAT, cell("K", "code"), _FLAT, _FLAT),
                **both_kalman(slight, _FLAT, slight, _FLAT, _FLAT),
            },
        ),
        ScalingRow(
            "mGRAD particles",
            {
                "mgrad": (
                    cell("N<sup>2</sup>", "measured"),
                    _FLAT,
                    cell("N<sup>2</sup>", "measured"),
                    _FLAT,
                    _FLAT,
                )
            },
        ),
        ScalingRow("Block size", {"kalman_blocks": block}),
        ScalingRow(
            "GPU", everywhere(_FLAT, cell("same?", "code"), cell("spec", "pair"), _FLAT, _FLAT)
        ),
    )


def formula_rows(laws: Laws) -> list[dict[str, str]]:
    """The fitted laws as formulas."""
    return [
        {
            "Quantity": "mGRAD memory while sampling",
            "Law": "C·(K·(T−1)·N²·D·4 B + r·N·T·D)",
            "Fitted": f"r = {laws.bank_bytes:.0f} bytes",
        },
        {
            "Quantity": "Kalman memory while sampling",
            "Law": "m·C·T·D²",
            "Fitted": (
                f"m = {laws.kalman_bytes / 1000:.2f} KB "
                f"(runs: {laws.kalman_bytes_low / 1000:.2f}-{laws.kalman_bytes_high / 1000:.2f})"
            ),
        },
        {
            "Quantity": "Compile, device",
            "Law": "at most c·C at T=3000",
            "Fitted": f"c = {laws.compile_mib_per_chain:.1f} MiB per chain",
        },
        {
            "Quantity": "After sampling, conditioning",
            "Law": "u·C·S·T·4 B, plus h·C for the allocator high-water",
            "Fitted": f"u = {laws.conditioning_units:.2f}; h = {laws.high_water_mib_per_chain:.1f} MiB",
        },
        {
            "Quantity": "After sampling, diagnostics",
            "Law": "v·C·S·T·4 B + w",
            "Fitted": f"v = {laws.diagnostics_units:.2f}; w = {laws.diagnostics_offset_gib:.2f} GiB",
        },
        {
            "Quantity": "After sampling, time",
            "Law": "t₀ + τ·C·S·T",
            "Fitted": f"t₀ = {laws.post_intercept_s:.1f} s; τ = {laws.post_ns_per_coordinate:.1f} ns",
        },
        {
            "Quantity": "Warmup, time",
            "Law": "O + W·(a + (b − p)·C)",
            "Fitted": f"O = {laws.warmup_overhead_s:.2f} s; p = {laws.projection_ms_per_chain:.3f} ms",
        },
        {
            "Quantity": "Kalman flops per chain-time",
            "Law": "c₀ + c₃·D³",
            "Fitted": f"c₀ = {laws.flops_constant:,.0f}; c₃ = {laws.flops_cubic:.0f} (two points)",
        },
        *(
            {
                "Quantity": f"Sweep time, {name}",
                "Law": "a + b·C",
                "Fitted": (
                    f"a = {curve.a_ms:.2f} ms; b = {curve.b_ms:.3f} ms; "
                    f"50% fill at {curve.half_chains:.0f} chains"
                ),
            }
            for name, curve in laws.ladders.items()
        ),
    ]


def _check(law: str, run: str, predicted: float, observed: float, unit: str) -> dict[str, str]:
    return {
        "Law": law,
        "Run": run,
        "Predicted": f"{predicted:,.2f} {unit}",
        "Observed": f"{observed:,.2f} {unit}",
        "Error": f"{100 * (predicted / observed - 1):+.1f}%",
    }


def check_rows(laws: Laws, evidence: Evidence) -> list[dict[str, str]]:
    """Every law against every run it applies to. Ladder rows leave the checked run out of the fit."""
    rows = []
    for run in evidence.runs:
        if run.model == "SV" and run.variant == "mgrad" and run.sweep_memory_gib is not None:
            scenario = Scenario(run.gpu, "mgrad", run.C, particles=run.N or REFERENCE_N)
            predicted = sweep_bytes_per_chain(scenario, laws) * run.C / GIB
            rows.append(_check("mGRAD memory", run.id, predicted, run.sweep_memory_gib, "GiB"))
        if run.failed_alloc_gib is not None:
            predicted = pair_bytes(run.C, run.K or 2, run.T, run.N or 0, run.D) / GIB
            rows.append(
                _check("mGRAD pair workspace", run.id, predicted, run.failed_alloc_gib, "GiB")
            )
        if run.model == "SV" and run.method == "kalman" and run.mem_sampling_gib is not None:
            predicted = laws.kalman_bytes * run.C * run.T / GIB
            rows.append(_check("Kalman memory", run.id, predicted, run.mem_sampling_gib, "GiB"))
        if run.mem_conditioning_gib is not None:
            predicted = laws.conditioning_units * _retained_gib(run)
            rows.append(_check("Conditioning", run.id, predicted, run.mem_conditioning_gib, "GiB"))
        if run.mem_diagnostics_gib is not None:
            predicted = laws.diagnostics_offset_gib + laws.diagnostics_units * _retained_gib(run)
            rows.append(_check("Diagnostics", run.id, predicted, run.mem_diagnostics_gib, "GiB"))
    for name in laws.ladders:
        members = [
            run for run in evidence.runs if run.ladder == name and run.sampling_ms is not None
        ]
        for held in members:
            kept = [(run.C, run.sampling_ms or 0.0) for run in members if run is not held]
            predicted = fit_curve(kept).time_ms(held.C)
            rows.append(
                _check(
                    f"Sweep time, left out of {name}",
                    held.id,
                    predicted,
                    held.sampling_ms or 0.0,
                    "ms",
                )
            )
    mgrad = laws.ladders["sv-mgrad-h200"]
    for run_id in ("profile/mgrad-noprofile", "p128/mgrad_p256", "round4/binary-particles512"):
        run = evidence.run(run_id)
        predicted = mgrad.a_ms + run.C * mgrad.b_ms * ((run.N or REFERENCE_N) / REFERENCE_N) ** 2
        rows.append(_check("mGRAD time, N rule", run.id, predicted, run.sampling_ms or 0.0, "ms"))
    for check in cross_gpu_checks(laws, evidence):
        rows.append(
            _check(
                f"Spec sheet, from {check.source_gpu}",
                check.run.id,
                check.spec_ms,
                check.observed_ms,
                "ms",
            )
        )
    return rows


@dataclass(frozen=True)
class CrossGpuCheck:
    """One run on a GPU other than its ladder's: measured, spec-scaled and GPU-blind."""

    label: str
    run: Run
    source_gpu: str
    observed_ms: float
    spec_ms: float
    blind_ms: float


def cross_gpu_checks(laws: Laws, evidence: Evidence) -> tuple[CrossGpuCheck, ...]:
    """Predict other GPUs from the reference ladder, with and without their spec sheets."""
    cases: tuple[tuple[str, Method, str], ...] = (
        ("Kalman whole, H200", "kalman_whole", "profile/kalman-whole"),
        ("Kalman whole, L4", "kalman_whole", "profile/kalman-whole-l4"),
        ("Kalman blocks, H200", "kalman_blocks", "profile/kalman-blocked"),
        ("mGRAD, L40S", "mgrad", "profile/mgrad-l40s-40"),
    )
    checks = []
    for label, method, run_id in cases:
        run = evidence.run(run_id)
        source = _REFERENCE_GPU[method]
        reference = laws.ladders[f"{_LADDER_PREFIX[method]}-{source.lower()}"]
        spec = spec_curve(method, evidence.gpu(run.gpu), laws, evidence)
        checks.append(
            CrossGpuCheck(
                label,
                run,
                source,
                run.sampling_ms or 0.0,
                spec.time_ms(run.C),
                reference.time_ms(run.C),
            )
        )
    return tuple(checks)


def run_rows(evidence: Evidence) -> list[dict[str, object]]:
    """The evidence table for display."""
    return [
        {
            "Run": run.id,
            "Model": run.model,
            "T": run.T,
            "D": run.D,
            "Method": METHOD_LABELS[run.variant],
            "Block": run.block,
            "N": run.N,
            "GPU": run.gpu,
            "Chains": run.C,
            "W": run.W,
            "S": run.S,
            "Sampling ms": run.sampling_ms,
            "Warmup ms": run.warmup_ms,
            "Compile s": run.compile_s,
            "Sampling GiB": run.mem_sampling_gib,
            "Peak GiB": run.peak_gib,
            "Failed alloc GiB": run.failed_alloc_gib,
            "Ladder": run.ladder,
            "Build": run.build,
            "Note": run.note,
            "Source": run.source,
        }
        for run in evidence.runs
    ]


def gpu_rows(evidence: Evidence) -> list[dict[str, object]]:
    """The GPU table for display, with how many runs back each one."""
    return [
        {
            "GPU": gpu.name,
            "In calculator": "yes" if gpu.offered else "no, evidence only",
            "Runs": sum(run.gpu == gpu.name for run in evidence.runs),
            "Architecture": gpu.architecture,
            "VRAM GB": gpu.vram_gb,
            "Memory": gpu.memory,
            "CUDA cores": gpu.cores,
            "Clock MHz": f"{gpu.clock_mhz:,.0f}" + (" (assumed)" if gpu.clock_assumed else ""),
            "DRAM GB/s": gpu.dram_gb_s,
            "JAX pool GiB": gpu.pool_gib,
            "Pool": gpu.pool_source,
            "Note": gpu.note,
        }
        for gpu in evidence.gpus
    ]
