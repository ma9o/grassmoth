"""GPU compute for local fits; the caller still owns the study and publication."""

from __future__ import annotations

import hashlib
import io
import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from grassmoth.public import (
    DynamicalModelSpec,
    FitOutput,
    ObservationDataset,
    SamplerCheckpoint,
)
from grassmoth_server.compute.analysis_cache import (
    ANALYSIS_CACHE_ROOT,
    ANALYSIS_CACHE_VOLUME,
    AnalysisCall,
    cache_fit_result,
    fit_profile_directory,
)
from grassmoth_server.compute.resources import HOST_MEMORY_ALLOCATION_ENV
from grassmoth_server.study.persistence.codec import pack_result, unpack_result

if TYPE_CHECKING:
    from collections.abc import Iterator
    from datetime import datetime

    import modal

    from grassmoth.public import (
        InferenceHardware,
        QuestionSpec,
        ResolvedObservationSpec,
        SamplerSpec,
        StructuralSelection,
    )

_FIT_CPU_CORES = 8
_FIT_HOST_MEMORY_MIB = 32768
FIT_CACHE_VOLUME_ENV = "GRASSMOTH_FIT_CACHE_VOLUME"
_RESULT_HEADER_BYTES = 8 + hashlib.sha256().digest_size
_RESULT_CHUNK_BYTES = 8 * 1024**2


@dataclass(frozen=True)
class FitComputeInput:
    """Pinned, self-contained inputs; no remote workspace or storage credentials."""

    dynamical_model_spec_bytes: bytes
    question: QuestionSpec
    panel_parquet: bytes
    variables: tuple[ResolvedObservationSpec, ...]
    time_origin: datetime
    sampler: SamplerSpec
    compute_loo_diagnostics: bool
    analysis_call: AnalysisCall
    profile: bool = False
    continuation: bytes | None = None
    previous: bytes | None = None


@dataclass(frozen=True)
class FitComputeResult:
    """Complete fitted output to validate and persist through the caller's store."""

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


def execute_fit_compute(payload: FitComputeInput) -> FitComputeResult:
    """Use the ordinary fit/report/conditioning code with an in-memory array store."""
    from grassmoth_server.bootstrap import initialize_numerical_runtime

    initialize_numerical_runtime()
    import polars as pl

    from grassmoth.public import StructuralSelection
    from grassmoth_server.actions.fit.action import run_fit

    dynamical_model_spec = DynamicalModelSpec.model_validate(
        unpack_result(payload.dynamical_model_spec_bytes)
    ).materialized()
    result = run_fit(
        selection=StructuralSelection.for_question(dynamical_model_spec, payload.question),
        question=payload.question,
        data_for_model=ObservationDataset.from_frame(
            pl.read_parquet(io.BytesIO(payload.panel_parquet)),
            payload.variables,
            time_origin=payload.time_origin,
        ),
        time_origin=payload.time_origin,
        sampler=payload.sampler,
        compute_loo_diagnostics=payload.compute_loo_diagnostics,
        profile=payload.profile,
        profile_dir=fit_profile_directory(ANALYSIS_CACHE_ROOT, payload.analysis_call)
        if payload.profile
        else None,
        continuation=SamplerCheckpoint.model_validate(unpack_result(payload.continuation))
        if payload.continuation is not None
        else None,
        previous=FitOutput.model_validate(unpack_result(payload.previous))
        if payload.previous is not None
        else None,
    )
    return FitComputeResult(output=pack_result(result))


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
    logging.getLogger(__name__).info("Fit compute devices: %s", jax.devices())
    volume = modal.Volume.from_name(os.environ[FIT_CACHE_VOLUME_ENV])
    with local_compilation_cache(Path("/cache/jax-bundles"), commit=volume.commit):
        result = execute_fit_compute(payload)
        cache_fit_result(ANALYSIS_CACHE_ROOT, payload.analysis_call, result.output)
        modal.Volume.from_name(ANALYSIS_CACHE_VOLUME).commit()
        frame = encode_fit_result(result)
        del result
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
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.90",
            FIT_CACHE_VOLUME_ENV: cache_volume,
            "JAX_COMPILATION_CACHE_DIR": "/tmp/grassmoth-jax",
            "JAX_ENABLE_COMPILATION_CACHE": "true",
            "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": str(FIT_CACHE_MIN_COMPILE_SECONDS),
            "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "0",
            HOST_MEMORY_ALLOCATION_ENV: str(_FIT_HOST_MEMORY_MIB * 1024**2),
            # Keep small host-side NumPy/SciPy operations within one BLAS thread.
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
        gpu=hardware.value,
        cpu=(_FIT_CPU_CORES, _FIT_CPU_CORES),
        memory=(_FIT_HOST_MEMORY_MIB, _FIT_HOST_MEMORY_MIB),
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


def fit_on_modal(
    hardware: InferenceHardware,
    *,
    selection: StructuralSelection,
    data_for_model: ObservationDataset,
    time_origin: datetime,
    sampler: SamplerSpec,
    question: QuestionSpec,
    compute_loo_diagnostics: bool,
    analysis_call: AnalysisCall,
    profile: bool = False,
    continuation: SamplerCheckpoint | None = None,
    previous: FitOutput | None = None,
) -> FitOutput:
    """Transfer pinned values, compute once on Modal, then retain returned arrays locally."""
    # Prior phase traces are publication history, never transition inputs. Keep
    # them at the caller instead of uploading and downloading them unchanged.
    compute_previous = (
        None
        if previous is None
        else previous.revised(
            inference=previous.inference.revised(
                evidence=previous.inference.evidence.revised(phase_extra_fields={})
            )
        )
    )
    panel = io.BytesIO()
    data_for_model.recorded.frame.write_parquet(panel)
    result = _dispatch_fit(
        FitComputeInput(
            dynamical_model_spec_bytes=pack_result(selection.dynamical_model_spec),
            question=question,
            panel_parquet=panel.getvalue(),
            variables=data_for_model.variables,
            time_origin=time_origin,
            sampler=sampler,
            compute_loo_diagnostics=compute_loo_diagnostics,
            analysis_call=analysis_call,
            profile=profile,
            continuation=None if continuation is None else pack_result(continuation),
            previous=None if compute_previous is None else pack_result(compute_previous),
        ),
        hardware=hardware,
    )
    # Validate the complete response before the first local write. Exceptions
    # propagate to the action's normal error path; there is no local retry.
    output = FitOutput.model_validate(unpack_result(result.output))
    if previous is not None:
        output = output.revised(
            inference=output.inference.revised(
                evidence=output.inference.evidence.with_previous_phases(previous.inference.evidence)
            )
        )
    return output
