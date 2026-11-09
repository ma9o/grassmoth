"""GPU compute for local fits; the caller still owns the study and publication."""

from __future__ import annotations

import hashlib
import io
import logging
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from grassmoth.public import (
    DynamicalModelSpec,
    FitInputs,
    FitOutput,
    ObservationDataset,
)
from grassmoth_server.compute.analysis_cache import (
    ANALYSIS_CACHE_ROOT,
    ANALYSIS_CACHE_VOLUME,
    AnalysisCacheEntry,
    AnalysisCall,
    RetainedFit,
    cache_fit_result,
    fit_profile_directory,
    retained_fit_result,
)
from grassmoth_server.compute.fit_results import RemoteFitResult
from grassmoth_server.compute.resources import (
    FIT_DEVICE_MEMORY_FRACTION,
    FIT_HOST_MEMORY_MIB,
    HOST_MEMORY_ALLOCATION_ENV,
)
from grassmoth_server.study.persistence.codec import pack_payload, pack_result, unpack_result

if TYPE_CHECKING:
    from collections.abc import Iterator

    import modal

    from grassmoth.public import (
        InferenceHardware,
        SamplerSpec,
    )

_FIT_CPU_CORES = 8
FIT_CACHE_VOLUME_ENV = "GRASSMOTH_FIT_CACHE_VOLUME"
_RESULT_HEADER_BYTES = 8 + hashlib.sha256().digest_size
_RESULT_CHUNK_BYTES = 8 * 1024**2


@dataclass(frozen=True)
class FitComputeInput:
    """Pinned, self-contained inputs; no remote workspace or storage credentials."""

    dynamical_model_spec_bytes: bytes
    observations: bytes
    sampler: SamplerSpec
    compute_loo_diagnostics: bool
    analysis_call: AnalysisCall
    profile: bool = False
    previous: bytes | AnalysisCacheEntry | None = None


@dataclass(frozen=True)
class FitComputeResult:
    """Verified full-result and transfer pointers returned to the caller."""

    output: bytes


def encode_fit_result(result: FitComputeResult) -> bytes:
    """Losslessly compress the complete function return for transport."""
    import pyarrow as pa

    raw = pickle.dumps(result, protocol=5)
    compressed = bytes(pa.Codec("zstd", compression_level=1).compress(raw))
    return len(raw).to_bytes(8, "big") + hashlib.sha256(raw).digest() + compressed


def decode_fit_result(frame: bytes) -> FitComputeResult:
    """Decode a worker return and verify its exact bytes before unpickling."""
    import pyarrow as pa

    if len(frame) <= _RESULT_HEADER_BYTES:
        raise ValueError("Invalid fit result transport size")
    size = int.from_bytes(frame[:8], "big")
    with pa.CompressedInputStream(pa.BufferReader(frame[_RESULT_HEADER_BYTES:]), "zstd") as stream:
        raw = stream.read()
    if len(raw) != size:
        raise ValueError("Fit result transport length does not match")
    if hashlib.sha256(raw).digest() != frame[8:_RESULT_HEADER_BYTES]:
        raise ValueError("Fit result transport checksum does not match")
    result = pickle.loads(raw)
    if not isinstance(result, FitComputeResult):
        raise ValueError("Unexpected fit result transport value")
    return result


def execute_fit_compute(payload: FitComputeInput) -> FitOutput:
    """Use the ordinary fit/report/conditioning code with an in-memory array store."""
    from grassmoth_server.bootstrap import initialize_numerical_runtime

    initialize_numerical_runtime()

    from grassmoth.public import StructuralSelection
    from grassmoth_server.actions.fit.action import run_fit

    dynamical_model_spec = DynamicalModelSpec.model_validate(
        unpack_result(payload.dynamical_model_spec_bytes)
    ).materialized()
    previous = (
        retained_fit_result(ANALYSIS_CACHE_ROOT / "retained", payload.previous).read_bytes()
        if isinstance(payload.previous, AnalysisCacheEntry)
        else payload.previous
    )
    return run_fit(
        FitInputs(
            selection=StructuralSelection.for_model(dynamical_model_spec),
            data_for_model=ObservationDataset.model_validate(unpack_result(payload.observations)),
            sampler=payload.sampler,
            compute_loo_diagnostics=payload.compute_loo_diagnostics,
            previous=FitOutput.model_validate(unpack_result(previous))
            if previous is not None
            else None,
        ),
        profile_dir=fit_profile_directory(ANALYSIS_CACHE_ROOT, payload.analysis_call)
        if payload.profile
        else None,
    )


def _claim_fit_attempt(attempt: modal.Dict) -> None:
    """Fence platform crash redelivery before any expensive computation starts.

    Modal reschedules crashed containers independently of ``retries=0``. The
    durable, atomic claim outlives a killed runner and stays claimed through
    result delivery; a replacement reports failure instead of repeating a fit.
    """
    if not attempt.put("started", True, skip_if_exists=True):
        raise RuntimeError("Modal fit runner was interrupted; refusing to repeat the fit")


def _result_chunks(frame: bytes) -> Iterator[bytes]:
    """Bound each SDK upload, which otherwise copies the full return per part."""
    for start in range(0, len(frame), _RESULT_CHUNK_BYTES):
        yield frame[start : start + _RESULT_CHUNK_BYTES]


def _fit_gpu(payload: FitComputeInput, attempt: modal.Dict) -> Iterator[bytes]:
    _claim_fit_attempt(attempt)

    import jax
    import modal

    from grassmoth_server.bootstrap import initialize_numerical_runtime
    from grassmoth_server.compute.compilation_cache import local_compilation_cache

    logging.basicConfig(level=logging.INFO)
    initialize_numerical_runtime()
    if jax.default_backend() != "gpu":
        raise RuntimeError("Modal fit requires a GPU backend")
    logger = logging.getLogger(__name__)
    logger.info("Fit compute devices: %s", jax.devices())
    volume = modal.Volume.from_name(os.environ[FIT_CACHE_VOLUME_ENV])
    with local_compilation_cache(Path("/cache/jax-bundles"), commit=volume.commit):
        result = execute_fit_compute(payload)
        from grassmoth_server.study.persistence.arrays import owned_arrays
        from grassmoth_server.study.persistence.codec import result_payload

        started = time.perf_counter()
        values = result_payload(result)
        logger.info(
            "Fit result_payload complete: elapsed_seconds=%.6f", time.perf_counter() - started
        )
        started = time.perf_counter()
        entry = cache_fit_result(ANALYSIS_CACHE_ROOT / "retained", payload.analysis_call, values)
        logger.info(
            "Fit streamed cache write complete: elapsed_seconds=%.6f", time.perf_counter() - started
        )
        del values
        # The full-result pointer serves analysis and continuation.
        # The 14-day temporary cache outlives the four-hour fit activity and
        # transport retries. Later checks/publication retries already read Git.
        transfer_values = result_payload(result, thin_fit_draws=True)
        transfer = cache_fit_result(ANALYSIS_CACHE_ROOT, payload.analysis_call, transfer_values)
        started = time.perf_counter()
        checkpoint = result.inference.evidence.checkpoint
        completion = RemoteFitResult(
            full_fit=RetainedFit(
                entry=entry, settings=None if checkpoint is None else checkpoint.settings
            ),
            transfer=transfer,
            law_arrays=tuple(owned_arrays(transfer_values.get("dynamical_model_spec"))),
            has_model=result.dynamical_model_spec is not None,
        )
        logger.info(
            "Fit result_pointers complete: elapsed_seconds=%.6f", time.perf_counter() - started
        )
        started = time.perf_counter()
        modal.Volume.from_name(ANALYSIS_CACHE_VOLUME).commit()
        logger.info(
            "Fit volume_commit complete: elapsed_seconds=%.6f", time.perf_counter() - started
        )
        started = time.perf_counter()
        frame = encode_fit_result(FitComputeResult(pack_result(completion)))
        logger.info(
            "Fit result_frame_encode complete: elapsed_seconds=%.6f", time.perf_counter() - started
        )
        del result, transfer_values
    yield from _result_chunks(frame)


def _dispatch_fit(payload: FitComputeInput, *, hardware: InferenceHardware) -> FitComputeResult:
    import modal

    from grassmoth_server.compute.compilation_cache import FIT_CACHE_MIN_COMPILE_SECONDS
    from grassmoth_server.compute.modal_images import gpu_image

    # A separate ephemeral app per call supports concurrent local study workers
    # and ships their current source; no deployed production function is replaced.
    app = modal.App("grassmoth-server")
    cache_volume = os.environ.get(FIT_CACHE_VOLUME_ENV, "grassmoth-fit-cache")
    remote_fit = app.function(
        image=gpu_image,
        # Function environment overrides image defaults without adding a build
        # layer after its add_local_* source mounts (Modal forbids that order).
        env={
            "DEPLOYMENT_ENV": "development",
            "JAX_PLATFORMS": "cuda,cpu",
            # Reserve a contiguous pool before backend initialization so large
            # compiled sweep workspaces do not depend on allocator growth.
            "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": FIT_DEVICE_MEMORY_FRACTION,
            FIT_CACHE_VOLUME_ENV: cache_volume,
            "JAX_COMPILATION_CACHE_DIR": "/tmp/grassmoth-jax",
            "JAX_ENABLE_COMPILATION_CACHE": "true",
            "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": str(FIT_CACHE_MIN_COMPILE_SECONDS),
            "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "0",
            HOST_MEMORY_ALLOCATION_ENV: str(FIT_HOST_MEMORY_MIB * 1024**2),
            # Keep small host-side NumPy/SciPy operations within one BLAS thread.
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
        gpu=hardware.value,
        cpu=(_FIT_CPU_CORES, _FIT_CPU_CORES),
        memory=(FIT_HOST_MEMORY_MIB, FIT_HOST_MEMORY_MIB),
        timeout=payload.sampler.time_limit_seconds,
        startup_timeout=payload.sampler.time_limit_seconds,
        scaledown_window=2,
        # Modal generators do not support a retries option. Crash redelivery is
        # independently fenced by the per-dispatch claim before computation.
        volumes={
            "/cache": modal.Volume.from_name(cache_volume, create_if_missing=True),
            str(ANALYSIS_CACHE_ROOT): modal.Volume.from_name(
                ANALYSIS_CACHE_VOLUME, create_if_missing=True
            ),
        },
    )(_fit_gpu)
    with modal.Dict.ephemeral() as attempt, app.run():
        frame = io.BytesIO()
        for chunk in remote_fit.remote_gen(payload, attempt):
            frame.write(chunk)
        return decode_fit_result(frame.getvalue())


def pack_observations(observations: ObservationDataset) -> bytes:
    """Pack selected readings without retaining their parent simulation buffers."""
    series = {}
    for identity, selected in observations.series.items():
        history = selected.history
        readings = history.readings()
        series[identity] = {
            "observation": selected.observation.model_dump(mode="python"),
            "history": {
                **history.model_dump(
                    mode="python", exclude={"values", "support_start", "support_end"}
                ),
                "values": tuple(row[1] for row in readings),
                "support_start": tuple(row[2] for row in readings),
                "support_end": tuple(row[3] for row in readings),
            },
        }
    return pack_payload({"series": series, "time_origin": observations.time_origin.isoformat()})


def fit_on_modal(
    inputs: FitInputs,
    *,
    analysis_call: AnalysisCall,
    profile: bool = False,
    previous_cache: AnalysisCacheEntry | None = None,
) -> RemoteFitResult:
    """Compute once on Modal; return reports only after retaining the complete result."""
    result = _dispatch_fit(
        FitComputeInput(
            dynamical_model_spec_bytes=pack_result(inputs.selection.dynamical_model_spec),
            observations=pack_observations(inputs.data_for_model),
            sampler=inputs.sampler,
            compute_loo_diagnostics=inputs.compute_loo_diagnostics,
            analysis_call=analysis_call,
            profile=profile,
            previous=previous_cache
            if previous_cache is not None
            else None
            if inputs.previous is None
            else pack_result(inputs.previous),
        ),
        hardware=inputs.sampler.hardware,
    )
    completion = RemoteFitResult.model_validate(unpack_result(result.output))
    if completion.full_fit.entry.call != analysis_call or completion.transfer.call != analysis_call:
        raise ValueError("Remote fit completion belongs to a different call")
    return completion
