"""Capture the actual compute environment at the fitting boundary."""

import hashlib
import importlib.metadata
import os
import platform
from importlib.resources import files
from pathlib import Path

import jax

from grassmoth.public import FitRuntime, InferenceHardware, SamplerSpec


def numerical_source_digest() -> str:
    """Record shipped source provenance without participating in compilation caching."""
    roots = (Path(str(files("grassmoth"))), Path(__file__).resolve().parents[2])
    digest = hashlib.sha256()
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            digest.update(path.relative_to(root.parent).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def fit_runtime(sampler: SamplerSpec) -> FitRuntime:
    """Hash shipped numerical source, including uncommitted edits, without requiring Git."""
    return FitRuntime(
        time_limit_seconds=None
        if sampler.hardware == InferenceHardware.LOCAL
        else sampler.time_limit_seconds,
        source_digest=numerical_source_digest(),
        versions={
            "python": platform.python_version(),
            **{
                name: importlib.metadata.version(name)
                for name in ("jax", "jaxlib", "numpyro", "dynestyx")
            },
        },
        compiler={
            "XLA_FLAGS": os.environ.get("XLA_FLAGS", ""),
            **{name: str(value) for name, value in jax.config.values.items()},
        },
        devices=tuple(
            f"{device.platform}:{device.device_kind}:{device.id}" for device in jax.devices()
        ),
    )
