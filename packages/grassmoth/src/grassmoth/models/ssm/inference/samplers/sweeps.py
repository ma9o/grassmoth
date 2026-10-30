"""Host orchestration for compiled sweeps with one history transfer per block."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, overload

import jax
import jax.numpy as jnp
import numpy as np

from grassmoth.artifacts.identity import ConstructRef, ParameterRef
from grassmoth.artifacts.immutability import freeze_fields
from grassmoth.compute.batching import compiled_memory
from grassmoth.compute.budget import NO_MEMORY_DEMAND, MemoryDemand
from grassmoth.models.ssm.inference.samplers.progress import (
    inference_phase,
    log_inference_event,
)
from grassmoth.models.ssm.inference.samplers.sweep_plan import (
    SWEEP_BLOCK_SIZE,
    CompiledSweepPhase,
    EmptySweepPhase,
    SweepPlan,
    sweep_block,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from concurrent.futures import Future

    from jax.stages import Compiled
    from numpy.typing import NDArray

    from grassmoth.artifacts.reports.posterior_diagnostics import ConvergenceReport
    from grassmoth.artifacts.reports.profiling import FitProfileReport
    from grassmoth.artifacts.sampling import ConvergenceStoppingSpec, SamplerSpec
    from grassmoth.compute.context import ModelExecutionContext
    from grassmoth.compute.programs import ProgramCompiler
    from grassmoth.models.ssm.compile.inputs import CompiledDynamicalModel
    from grassmoth.models.ssm.inference.problem import ParticleProblem
    from grassmoth.models.ssm.inference.samplers.sweep_plan import SweepProgram
    from grassmoth.models.ssm.inference.targets.particle import ParticleTarget
    from grassmoth.models.ssm.inference.warmup.outcomes import InitializationFailure


@dataclass(frozen=True)
class SweepHistory:
    """Immutable chain-major phase outputs, detached from host assembly buffers."""

    warmup: Mapping[str, jax.Array]
    posterior: Mapping[str, jax.Array]

    def __post_init__(self) -> None:
        """Freeze phase mappings while leaving device arrays intact."""
        freeze_fields(self)


@dataclass(frozen=True)
class SweepRun[Carry]:
    """Final native JAX state, histories and actual execution durations."""

    state: Carry
    history: SweepHistory
    first_step_seconds: float
    sampling_loop_seconds: float
    retained_draws: int
    stopping_checkpoint: ConvergenceCheckpoint | None
    profile: FitProfileReport | None = None


@dataclass(frozen=True)
class ConvergenceCheckpoint:
    """The first eligible passing checkpoint, counted within this execution."""

    draws: int
    report: ConvergenceReport


@dataclass(frozen=True)
class RetainedConvergenceCheck:
    """A declared policy and its native production measurement on retained buffers."""

    settings: ConvergenceStoppingSpec
    evaluate: Callable[[Mapping[str, NDArray]], ConvergenceReport]


def retained_convergence_check(
    settings: ConvergenceStoppingSpec | None,
    *,
    bundle: ParticleProblem,
    compiled_dynamical_model: CompiledDynamicalModel,
    free_mask: jax.Array,
    compiler: ProgramCompiler,
) -> RetainedConvergenceCheck | None:
    """Compose the existing public projection, native chain estimators and production gate."""
    if settings is None:
        return None
    from grassmoth.models.ssm.inference.diagnostics.convergence import chain_convergence
    from grassmoth.models.ssm.inference.diagnostics.latent import LatentChainEvidence
    from grassmoth.models.ssm.inference.extraction import extract_grouped_public_samples
    from grassmoth.models.ssm.inference.mcmc_state import TrajectoryMCMCResult
    from grassmoth.models.ssm.inference.types import (
        JointPosteriorDraws,
        MCMCPosterior,
        ProductionDiagnostics,
    )

    references = MappingProxyType(
        {
            **{
                coordinate: (
                    binding.elements[element],
                    ParameterRef(parameter_id=binding.parameter_id, element_id=element),
                )
                for binding in compiled_dynamical_model.bindings
                for element, coordinate in binding.coordinates.items()
            },
            **dict.fromkeys(compiled_dynamical_model.auxiliary_coordinates),
        }
    )
    states = tuple(
        (ConstructRef(id=state.id), state.name) for state in compiled_dynamical_model.states
    )
    mask = np.asarray(free_mask)

    def evaluate(history: Mapping[str, NDArray]) -> ConvergenceReport:
        chains, draws = history["positions"].shape[:2]
        samples = extract_grouped_public_samples(
            jnp.asarray(history["positions"]),
            bundle=bundle,
            num_chains=chains,
            num_samples=draws,
            compiler=compiler,
        )
        mcmc = TrajectoryMCMCResult(
            chain_samples=samples,
            chain_extra_fields={},
            num_chains=chains,
            num_samples=draws,
            backend="convergence_checkpoint",
        )
        posterior = MCMCPosterior.from_run(
            draws=JointPosteriorDraws(parameters=mcmc.get_samples()),
            diagnostics=ProductionDiagnostics(
                mcmc=mcmc,
                public_sites=tuple(sorted(bundle.public_sites)),
                observation_log_probs=np.empty((chains, draws, 0)),
                latent_chains=LatentChainEvidence(
                    paths=history["paths"], free_mask=mask, states=states
                ),
            ),
        )
        return chain_convergence(posterior.get_inference_diagnostics(references))

    return RetainedConvergenceCheck(settings=settings, evaluate=evaluate)


def compile_sweeps[Carry](
    program: SweepProgram[Carry],
    initial: Carry,
    *,
    execution_context: ModelExecutionContext,
    sampler: SamplerSpec,
    additional_overhead: MemoryDemand = NO_MEMORY_DEMAND,
) -> SweepPlan:
    """Check compiled peak memory plus histories, without timing or executing a sweep.

    Compile the maximum node/root width first and halve it until compiled memory
    fits the budget. Independent chains remain vectorized. The common cap of
    25 sequential sweeps bounds output buffers and host progress latency.
    """
    block_size = SWEEP_BLOCK_SIZE
    num_warmup, num_samples = sampler.num_warmup, sampler.num_samples_per_chain
    max_batch_size = program.max_batch_size

    def prepare_phase(
        batch: int,
    ) -> tuple[
        Callable[[], Compiled],
        tuple[jax.Array, ...],
        Mapping[str, jax.ShapeDtypeStruct],
    ]:
        compile_program, arguments, (_, block_shapes), _ = execution_context.prepare_program(
            sweep_block(program.step_factory(batch), block_size),
            initial,
            jax.ShapeDtypeStruct((), jnp.int32),
            jax.ShapeDtypeStruct((), jnp.bool_),
        )
        shapes = {
            name: jax.ShapeDtypeStruct(shape.shape[1:], shape.dtype)
            for name, shape in block_shapes.items()
            if name not in program.excluded_history
        }
        return compile_program, arguments, shapes

    maximum = prepare_phase(max_batch_size)
    _, _, history_shapes = maximum
    warmup_shapes = {
        name: shape for name, shape in history_shapes.items() if name not in {"positions", "paths"}
    }

    def compile_plan(batch: int) -> tuple[SweepPlan, MemoryDemand]:
        compile_program, arguments, shapes = (
            maximum if batch == max_batch_size else prepare_phase(batch)
        )
        executable = compile_program()
        warmup = (
            CompiledSweepPhase(num_warmup, executable, warmup_shapes, arguments)
            if num_warmup or num_warmup + num_samples == 0
            else EmptySweepPhase(warmup_shapes)
        )
        posterior = (
            CompiledSweepPhase(num_samples, executable, shapes, arguments)
            if num_samples or num_warmup + num_samples == 0
            else EmptySweepPhase(shapes)
        )
        return (
            SweepPlan(warmup, posterior, batch, executable, program.label),
            compiled_memory(executable),
        )

    history_bytes = sum(
        steps * sum(shape.size * shape.dtype.itemsize for shape in shapes.values())
        for steps, shapes in ((num_warmup, warmup_shapes), (num_samples, history_shapes))
    )
    transfer_bytes = block_size * sum(
        shape.size * shape.dtype.itemsize for shape in history_shapes.values()
    )
    # Host assembly buffers coexist with one device_get block, including its
    # padded tail. XLA accounts for device block buffers; returned phases and
    # combined diagnostics can also coexist on device. These do not scale with
    # node concurrency and must be reserved only once.
    overhead = MemoryDemand(
        host_bytes=history_bytes + transfer_bytes + additional_overhead.host_bytes,
        device_bytes=2 * history_bytes + additional_overhead.device_bytes,
    )
    batch = max_batch_size
    while True:
        plan, memory = compile_plan(batch)
        if execution_context.budget.planning_capacity(memory, limit=1, overhead=overhead):
            if not execution_context.budget.capacity(memory, limit=1, overhead=overhead):
                raise MemoryError("Compiled sweep exceeds available memory")
            return replace(
                plan,
                memory=MemoryDemand(
                    host_bytes=memory.host_bytes + overhead.host_bytes,
                    device_bytes=memory.device_bytes + overhead.device_bytes,
                ),
            )
        if batch == 1:
            raise MemoryError("One sweep lane exceeds the memory budget")
        batch //= 2


def compile_projected_sweeps[Carry](
    program: SweepProgram[Carry],
    initial: Carry,
    *,
    execution_context: ModelExecutionContext,
    sampler: SamplerSpec,
    target: ParticleTarget,
) -> SweepPlan:
    """Compile independent exact observation projection while the sampler compiles."""
    block_size = SWEEP_BLOCK_SIZE

    def observation_projection(position: jax.Array, path: jax.Array) -> jax.Array:
        return jnp.asarray(
            target.observation_log_probs(
                target.context(position, target.times), path, target.observations
            )
        )

    positions = jax.ShapeDtypeStruct(
        (block_size, sampler.num_chains, target.initial_position.size),
        target.initial_position.dtype,
    )
    paths = jax.ShapeDtypeStruct(
        (block_size, sampler.num_chains, len(target.times), target.independent_mask.shape[-1]),
        target.observations.dtype,
    )
    compile_program, projection_arguments, shape, estimate = execution_context.prepare_program(
        jax.vmap(jax.vmap(observation_projection)), positions, paths
    )
    transfer = shape.size * shape.dtype.itemsize
    retained = transfer // block_size * sampler.num_samples_per_chain
    overhead = MemoryDemand(
        host_bytes=estimate.host_bytes + retained + transfer,
        device_bytes=estimate.device_bytes + 2 * retained,
    )

    def compile_projection() -> Compiled:
        with inference_phase(
            program.label, "compiling_observation_projection", execution_context=execution_context
        ):
            return compile_program()

    with ThreadPoolExecutor(max_workers=1) as executor:
        context = copy_context()
        future: Future[Compiled] = executor.submit(lambda: context.run(compile_projection))
        plan = compile_sweeps(
            program,
            initial,
            execution_context=execution_context,
            sampler=sampler,
            additional_overhead=overhead,
        )
        projection = future.result()
    actual = compiled_memory(projection)
    demand = MemoryDemand(
        host_bytes=plan.memory.host_bytes - estimate.host_bytes + actual.host_bytes,
        device_bytes=plan.memory.device_bytes - estimate.device_bytes + actual.device_bytes,
    )
    if not execution_context.budget.capacity(demand, limit=1):
        raise MemoryError(
            "Sweep and observation projection exceed the numerical execution memory budget"
        )
    return replace(
        plan, projection=projection, projection_arguments=projection_arguments, memory=demand
    )


@overload
def run_sweeps[Carry](
    plan: SweepPlan,
    initial: Carry,
    *,
    execution_context: ModelExecutionContext,
    convergence: RetainedConvergenceCheck | None = None,
    profile: Callable[[Carry, CompiledSweepPhase], FitProfileReport] | None = None,
    warmup_failure: None = None,
) -> SweepRun[Carry]: ...


@overload
def run_sweeps[Carry](
    plan: SweepPlan,
    initial: Carry,
    *,
    execution_context: ModelExecutionContext,
    convergence: RetainedConvergenceCheck | None = None,
    profile: Callable[[Carry, CompiledSweepPhase], FitProfileReport] | None = None,
    warmup_failure: Callable[[Carry], InitializationFailure | None],
) -> SweepRun[Carry] | InitializationFailure: ...


def run_sweeps[Carry](
    plan: SweepPlan,
    initial: Carry,
    *,
    execution_context: ModelExecutionContext,
    convergence: RetainedConvergenceCheck | None = None,
    profile: Callable[[Carry, CompiledSweepPhase], FitProfileReport] | None = None,
    warmup_failure: Callable[[Carry], InitializationFailure | None] | None = None,
) -> SweepRun[Carry] | InitializationFailure:
    """Keep only requested phase outputs; padded block slots execute no transition."""
    clock, label = execution_context.clock, plan.label
    state: Carry = initial
    started = clock()
    first_step_seconds = 0.0
    completed = 0
    histories: list[dict[str, jax.Array]] = []
    stopping_checkpoint = None
    retained_draws = 0
    profiling = None

    def profile_copy(phase: CompiledSweepPhase) -> FitProfileReport:
        assert profile is not None
        with inference_phase(label, "profiling", execution_context=execution_context):
            copied = jax.block_until_ready(jax.tree.map(jnp.copy, state))
            return profile(copied, phase)

    for stage, phase in (("warmup", plan.warmup), ("sampling", plan.posterior)):
        # Assemble directly in the final chain-major layout. Each live block is
        # copied once; no retained chunks or full-history concatenation coexist.
        shapes = dict(phase.shapes)
        if stage == "sampling" and plan.projection is not None:
            projected_shape = plan.projection.out_info
            shapes["observation_log_probs"] = jax.ShapeDtypeStruct(
                projected_shape.shape[1:], projected_shape.dtype
            )
        buffers = {
            name: np.empty((shape.shape[0], phase.steps, *shape.shape[1:]), dtype=shape.dtype)
            for name, shape in shapes.items()
        }
        phase_draws = 0
        if isinstance(phase, CompiledSweepPhase):
            phase_started = clock()
            with inference_phase(
                label,
                stage,
                execution_context=execution_context,
                total=phase.steps,
                block_size=SWEEP_BLOCK_SIZE,
                particle_batch_size=plan.batch_size,
            ):
                remaining = phase.steps
                while remaining:
                    count = 1 if completed == 0 else min(SWEEP_BLOCK_SIZE, remaining)
                    if stage == "sampling" and convergence is not None:
                        interval = convergence.settings.checkpoint_draws
                        count = min(count, interval - phase_draws % interval)
                    transition: tuple[Carry, dict[str, jax.Array]] = phase.executable(
                        phase.arguments,
                        state,
                        jnp.asarray(count),
                        jnp.asarray(stage == "sampling"),
                    )
                    state, block = transition
                    if stage == "sampling" and plan.projection is not None:
                        block = {
                            **block,
                            "observation_log_probs": plan.projection(
                                plan.projection_arguments, block["positions"], block["paths"]
                            ),
                        }
                    host_block = jax.device_get({name: block[name] for name in buffers})
                    if stage == "warmup" and warmup_failure is not None:
                        failure = warmup_failure(state)
                        if failure is not None:
                            return failure
                    start = phase.steps - remaining
                    for name in buffers:
                        buffers[name][:, start : start + count] = host_block[name][:count].swapaxes(
                            0, 1
                        )
                    del transition, block, host_block
                    completed += count
                    remaining -= count
                    phase_draws += count
                    if completed == 1:
                        first_step_seconds = clock() - started
                        if profile is not None and plan.warmup.steps == 0:
                            profiling = profile_copy(phase)
                    log_inference_event(
                        label,
                        stage,
                        "progress",
                        execution_context=execution_context,
                        completed=phase.steps - remaining,
                        total=phase.steps,
                        elapsed_seconds=clock() - phase_started,
                    )
                    if (
                        stage == "sampling"
                        and convergence is not None
                        and phase_draws % convergence.settings.checkpoint_draws == 0
                    ):
                        report = convergence.evaluate(
                            MappingProxyType(
                                {name: value[:, :phase_draws] for name, value in buffers.items()}
                            )
                        )
                        log_inference_event(
                            label,
                            stage,
                            "convergence_checkpoint",
                            execution_context=execution_context,
                            completed=phase_draws,
                            total=phase.steps,
                            convergence_status=report.status,
                        )
                        if (
                            phase_draws >= convergence.settings.min_draws
                            and report.status == "passed"
                        ):
                            stopping_checkpoint = ConvergenceCheckpoint(phase_draws, report)
                            break
            if stage == "warmup" and phase.steps and profile is not None:
                profiling = profile_copy(phase)
        else:
            log_inference_event(
                label, stage, "skipped", execution_context=execution_context, total=0
            )
        history = {name: jnp.asarray(value[:, :phase_draws]) for name, value in buffers.items()}
        if stage == "sampling":
            retained_draws = phase_draws
        histories.append(history)
        del buffers
    return SweepRun(
        state,
        SweepHistory(histories[0], histories[1]),
        first_step_seconds,
        clock() - started,
        retained_draws,
        stopping_checkpoint,
        profiling,
    )
