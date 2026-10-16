"""Capture the actual compute environment at the fitting boundary."""

import hashlib
import importlib.metadata
import os
import platform
from pathlib import Path

import jax

from nof1_causal_lab.artifacts.reports.inference import FitRuntime


def fit_runtime() -> FitRuntime:
    """Hash shipped numerical source, including uncommitted edits, without requiring Git."""
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return FitRuntime(
        source_digest=digest.hexdigest(),
        versions={
            "python": platform.python_version(),
            **{name: importlib.metadata.version(name) for name in ("jax", "jaxlib", "numpyro", "dynestyx")},
        },
        compiler={
            "XLA_FLAGS": os.environ.get("XLA_FLAGS", ""),
            **{name: str(value) for name, value in jax.config.values.items()},
        },
        devices=tuple(f"{device.platform}:{device.device_kind}:{device.id}" for device in jax.devices()),
    )
