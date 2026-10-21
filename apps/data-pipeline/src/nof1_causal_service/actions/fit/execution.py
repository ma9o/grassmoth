"""posterior: Bayesian inference and diagnostics."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

import jax

from nof1_causal_lab.artifacts.sampling import SamplerSpec

if TYPE_CHECKING:
    from jax.stages import Compiled

    from nof1_causal_lab.artifacts.sampling import FitSettingsSpec


def resolve_sampler_spec(
    settings: FitSettingsSpec, previous: SamplerSpec | None = None
) -> SamplerSpec:
    """Resolve public overrides once, before numerical preparation or execution."""
    from nof1_causal_service.config import get_config

    configured = get_config().inference.sampler if previous is None else previous
    return SamplerSpec(
        num_warmup=configured.num_warmup if settings.num_warmup is None else settings.num_warmup,
        num_samples_per_chain=configured.num_samples_per_chain
        if settings.num_samples_per_chain is None
        else settings.num_samples_per_chain,
        num_chains=configured.num_chains if settings.num_chains is None else settings.num_chains,
        seed=configured.seed if settings.seed is None else settings.seed,
        num_particles=configured.num_particles
        if settings.num_particles is None
        else settings.num_particles,
        hardware=configured.hardware if settings.hardware is None else settings.hardware,
        time_limit_seconds=configured.time_limit_seconds
        if settings.time_limit_seconds is None
        else settings.time_limit_seconds,
        retain_latent_paths=True,
        algorithm=configured.algorithm if settings.algorithm is None else settings.algorithm,
    )


_PROFILE_DIR_ENV = "NOF1_PROFILE_DIR"

# HLO ops whose counts characterize the device access pattern:
#   data movement      — transpose / copy / bitcast / reshape / broadcast /
#                         concatenate / slice
#   indexed access     — gather / scatter / dynamic-slice / dynamic-update-slice
#   tiny linear algebra — dot / triangular-solve / cholesky (the D=2-3 offenders)
#   fused compute      — fusion
#   serialization      — while (the lax.scan forward/backward filters)
_ACCESS_PATTERN_OPS = (
    "fusion",
    "dot",
    "triangular-solve",
    "cholesky",
    "transpose",
    "copy",
    "bitcast",
    "reshape",
    "broadcast",
    "concatenate",
    "slice",
    "gather",
    "scatter",
    "dynamic-slice",
    "dynamic-update-slice",
    "reduce",
    "while",
    "conditional",
    "sort",
    "custom-call",
)


def resolve_profile_dir(profile_dir: str | os.PathLike[str] | None) -> Path | None:
    """Resolve the effective profile directory, or ``None`` when profiling is off.

    ``profile_dir`` takes precedence; otherwise ``NOF1_PROFILE_DIR`` is consulted.
    The directory is created if it does not exist.
    """
    raw = profile_dir if profile_dir is not None else os.environ.get(_PROFILE_DIR_ENV)
    if raw is None or str(raw) == "":
        return None
    path = Path(raw)
    path.mkdir(parents=True, exist_ok=True)
    return path


def start_trace(profile_dir: Path | None, *, label: str) -> None:
    """Begin a ``jax.profiler`` trace into ``profile_dir / label`` (no-op if None)."""
    if profile_dir is None:
        return
    options = jax.profiler.ProfileOptions()
    options.host_tracer_level = 0
    options.python_tracer_level = 0
    options.include_dataset_ops = False
    options.enable_hlo_proto = False
    jax.profiler.start_trace(str(profile_dir / label), profiler_options=options)


def stop_trace(profile_dir: Path | None) -> None:
    """Stop the active ``jax.profiler`` trace (no-op if None)."""
    if profile_dir is None:
        return
    jax.profiler.stop_trace()


def _summarize_access_patterns(hlo_text: str) -> dict[str, int]:
    """Count producer instructions per op in the optimized HLO.

    Matches ``<op>(`` at a call site (operand references like ``%transpose.5``
    carry no paren, so they are not counted). A heuristic histogram, not a parse.
    """
    return {op: len(re.findall(rf"\b{re.escape(op)}\(", hlo_text)) for op in _ACCESS_PATTERN_OPS}


def dump_compiled_analysis(
    compiled: Compiled,
    profile_dir: Path | None,
    label: str,
) -> None:
    """Persist native producer compilation telemetry at the profiling shell boundary."""
    if profile_dir is None:
        return
    hlo_text = compiled.as_text()
    if hlo_text is None:
        raise RuntimeError("The JAX backend did not expose compiled HLO text")
    (profile_dir / f"{label}.hlo.txt").write_text(hlo_text)
    (profile_dir / f"{label}.cost.json").write_text(
        json.dumps(compiled.cost_analysis(), indent=2, sort_keys=True)
    )
    (profile_dir / f"{label}.access_patterns.json").write_text(
        json.dumps(_summarize_access_patterns(hlo_text), indent=2, sort_keys=True)
    )
