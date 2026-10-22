"""Capture the actual compute environment at the fitting boundary."""

import importlib.metadata
import os
import platform

import jax

from nof1_causal_lab.public import FitRuntime, InferenceHardware, SamplerSpec
from nof1_causal_service.compute.compilation_cache import numerical_source_digest


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
