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

from nof1_causal_lab.actions.contracts.bodies import FitOutput
from nof1_causal_lab.artifacts.continuation import SamplerCheckpoint
from nof1_causal_lab.artifacts.model.dynamical_model_spec import DynamicalModelSpec
from nof1_causal_lab.artifacts.observations.dataset import ObservationDataset
from nof1_causal_lab.artifacts.reports.inference import InferenceEvidence, InferenceMetadata
from nof1_causal_lab.compute.resources import HOST_MEMORY_ALLOCATION_ENV
from nof1_causal_lab.storage.arrays import decode_array, encode_array
from nof1_causal_lab.study.persistence.codec import pack_result, unpack_result

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime

    import numpy as np

    from nof1_causal_lab.actions.fit.action import FitResult
    from nof1_causal_lab.artifacts.identity import ConstructId
    from nof1_causal_lab.artifacts.observations.spec import ResolvedObservationSpec
    from nof1_causal_lab.artifacts.probability.serialization import ArrayLoader
    from nof1_causal_lab.artifacts.sampling import InferenceHardware, SamplerSpec
    from nof1_causal_lab.models.structure import StructuralSelection

_FIT_CPU_CORES = 8
_FIT_HOST_MEMORY_MIB = 32768
FIT_CACHE_VOLUME_ENV = "NOF1_FIT_CACHE_VOLUME"
FIT_RESULT_MAX_BYTES = 256 * 1024**2
_RESULT_HEADER_BYTES = 8 + hashlib.sha256().digest_size


@dataclass(frozen=True)
class FitComputeInput:
    """Pinned, self-contained inputs; no remote workspace or storage credentials."""

    dynamical_model_spec_bytes: bytes
    outcome: ConstructId | None
    panel_parquet: bytes
    variables: tuple[ResolvedObservationSpec, ...]
    time_origin: datetime
    sampler: SamplerSpec
    compute_loo_diagnostics: bool
    continuation: bytes | None = None
    previous: bytes | None = None


@dataclass(frozen=True)
class FitComputeResult:
    """Complete fitted output to validate and persist through the caller's store."""

    dynamical_model_spec_bytes: bytes | None
    evidence: bytes
    metadata_json: str


def encode_fit_result(result: FitComputeResult) -> bytes:
    """Bound and losslessly compress the complete function return for transport."""
    import pyarrow as pa

    raw = pickle.dumps(result, protocol=5)
    if len(raw) > FIT_RESULT_MAX_BYTES:
        raise ValueError("Fit result exceeds the transport size bound")
    compressed = bytes(pa.Codec("zstd", compression_level=1).compress(raw))
    frame = len(raw).to_bytes(8, "big") + hashlib.sha256(raw).digest() + compressed
    if len(frame) > FIT_RESULT_MAX_BYTES:
        raise ValueError("Compressed fit result exceeds the transport size bound")
    return frame


def decode_fit_result(frame: bytes) -> FitComputeResult:
    """Decode a bounded worker return and verify its exact bytes before unpickling."""
    import pyarrow as pa

    if not _RESULT_HEADER_BYTES < len(frame) <= FIT_RESULT_MAX_BYTES:
        raise ValueError("Invalid fit result transport size")
    size = int.from_bytes(frame[:8], "big")
    if not 0 < size <= FIT_RESULT_MAX_BYTES:
        raise ValueError("Fit result exceeds the decoded size bound")
    raw = bytes(pa.decompress(frame[_RESULT_HEADER_BYTES:], decompressed_size=size, codec="zstd"))
    if len(raw) != size or hashlib.sha256(raw).digest() != frame[8:_RESULT_HEADER_BYTES]:
        raise ValueError("Fit result transport checksum does not match")
    result = pickle.loads(raw)
    if not isinstance(result, FitComputeResult):
        raise ValueError("Unexpected fit result transport value")
    return result


def execute_fit_compute(payload: FitComputeInput) -> FitComputeResult:
    """Use the ordinary fit/report/conditioning code with an in-memory array store."""
    from nof1_causal_lab.bootstrap import initialize_numerical_runtime

    initialize_numerical_runtime()
    import polars as pl

    from nof1_causal_lab.actions.fit.action import fit

    arrays: dict[str, np.ndarray] = {}

    def write_array(values: np.ndarray) -> str:
        identity, data = encode_array(values)
        arrays[identity] = decode_array(identity, data)
        return identity

    from nof1_causal_lab.models.structure import StructuralSelection

    dynamical_model_spec = DynamicalModelSpec.model_validate(
        unpack_result(payload.dynamical_model_spec_bytes)
    ).materialized()
    result = fit(
        selection=StructuralSelection(dynamical_model_spec, payload.outcome),
        data_for_model=ObservationDataset.from_frame(
            pl.read_parquet(io.BytesIO(payload.panel_parquet)),
            payload.variables,
            time_origin=payload.time_origin,
        ),
        time_origin=payload.time_origin,
        sampler=payload.sampler,
        array_writer=write_array,
        array_loader=arrays.__getitem__,
        compute_loo_diagnostics=payload.compute_loo_diagnostics,
        continuation=SamplerCheckpoint.model_validate(unpack_result(payload.continuation))
        if payload.continuation is not None
        else None,
        previous=FitOutput.model_validate(unpack_result(payload.previous))
        if payload.previous is not None
        else None,
    )
    conditioned = result["_dynamical_model_spec"]
    evidence = result["evidence"]
    return FitComputeResult(
        dynamical_model_spec_bytes=None if conditioned is None else pack_result(conditioned),
        evidence=pack_result(evidence),
        metadata_json=result["metadata"].model_dump_json(round_trip=True),
    )


def _fit_gpu(payload: FitComputeInput) -> bytes:
    import jax
    import modal

    from nof1_causal_lab.bootstrap import initialize_numerical_runtime
    from nof1_causal_lab.compute.compilation_cache import local_compilation_cache

    logging.basicConfig(level=logging.INFO)
    initialize_numerical_runtime()
    if jax.default_backend() != "gpu":
        raise RuntimeError("Modal fit requires a GPU backend")
    logging.getLogger(__name__).info("Fit compute devices: %s", jax.devices())
    volume = modal.Volume.from_name(os.environ[FIT_CACHE_VOLUME_ENV])
    with local_compilation_cache(Path("/cache/jax-bundles"), commit=volume.commit):
        return encode_fit_result(execute_fit_compute(payload))


def _dispatch_fit(payload: FitComputeInput, *, hardware: InferenceHardware) -> FitComputeResult:
    import modal

    from nof1_causal_lab.compute.compilation_cache import FIT_CACHE_MIN_COMPILE_SECONDS
    from nof1_causal_lab.compute.modal_images import gpu_image

    # A separate ephemeral app per call supports concurrent local study workers
    # and ships their current source; no deployed production function is replaced.
    app = modal.App("nof1-causal-lab-pipeline")
    cache_volume = os.environ.get(FIT_CACHE_VOLUME_ENV, "nof1-cached-fit-cache")
    remote_fit = app.function(
        image=gpu_image,
        # Function environment overrides image defaults without adding a build
        # layer after its add_local_* source mounts (Modal forbids that order).
        env={
            "PYTHONHASHSEED": "0",
            "DEPLOYMENT_ENV": "development",
            "JAX_PLATFORMS": "cuda,cpu",
            # Reserve a contiguous pool before backend initialization so large
            # compiled sweep workspaces do not depend on allocator growth.
            "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.90",
            FIT_CACHE_VOLUME_ENV: cache_volume,
            "JAX_COMPILATION_CACHE_DIR": "/tmp/nof1-jax",
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
        retries=0,
        volumes={"/cache": modal.Volume.from_name(cache_volume, create_if_missing=True)},
    )(_fit_gpu)
    with app.run():
        return decode_fit_result(remote_fit.remote(payload))


def fit_on_modal(
    hardware: InferenceHardware,
    *,
    selection: StructuralSelection,
    data_for_model: ObservationDataset,
    time_origin: datetime,
    sampler: SamplerSpec,
    array_writer: Callable[[np.ndarray], str],
    array_loader: ArrayLoader,
    compute_loo_diagnostics: bool,
    continuation: SamplerCheckpoint | None = None,
    previous: FitOutput | None = None,
) -> FitResult:
    """Transfer pinned values, compute once on Modal, then retain returned arrays locally."""
    from nof1_causal_lab.actions.fit.action import append_phase_evidence

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
            dynamical_model_spec_bytes=pack_result(
                selection.dynamical_model_spec, array_loader=array_loader
            ),
            outcome=selection.outcome,
            panel_parquet=panel.getvalue(),
            variables=data_for_model.variables,
            time_origin=time_origin,
            sampler=sampler,
            compute_loo_diagnostics=compute_loo_diagnostics,
            continuation=None
            if continuation is None
            else pack_result(continuation, array_loader=array_loader),
            previous=None
            if compute_previous is None
            else pack_result(compute_previous, array_loader=array_loader),
        ),
        hardware=hardware,
    )
    # Validate the complete response before the first local write. Exceptions
    # propagate to the action's normal error path; there is no local retry.
    from nof1_causal_lab.study.persistence.arrays import owned_arrays

    evidence = InferenceEvidence.model_validate(unpack_result(result.evidence))
    if previous is not None:
        evidence = append_phase_evidence(previous.inference.evidence, evidence)
    metadata = InferenceMetadata.model_validate_json(result.metadata_json)
    if result.dynamical_model_spec_bytes is None:
        return {"evidence": evidence, "metadata": metadata, "_dynamical_model_spec": None}
    payload = unpack_result(result.dynamical_model_spec_bytes)
    conditioned = DynamicalModelSpec.model_validate(payload).materialized()
    for identity, value in owned_arrays(payload).items():
        if array_writer(value.values) != identity:
            raise ValueError("Fit output array does not match its content identity")
    return {"evidence": evidence, "metadata": metadata, "_dynamical_model_spec": conditioned}
