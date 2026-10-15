"""Parameter-recovery benchmark for the production particle-MCMC sampler.

The benchmark compares the two supported dSMC leaf proposals on the synthetic
nonlinear fixture. Both proposals target the exact Euler-Maruyama transition;
LAPS supplies paired starts and fixed proposal moments for both methods.

This is an evaluation entry point, not a test. Running it performs inference
and can be expensive:

    uv run python evaluation/benchmarks/benchmark.py
    uv run python evaluation/benchmarks/benchmark.py --proposals paid_mix --T 24
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from nof1_causal_lab.compute.resources import read_compute_budget
from nof1_causal_lab.models.ssm.preparation.preflight import ObservationPreflightFailure

if TYPE_CHECKING:
    from nof1_causal_lab.models.ssm.inference.types import MCMCPosterior

LeafProposal = Literal["amala_exact", "paid_mix"]
LEAF_PROPOSALS: tuple[LeafProposal, ...] = ("amala_exact", "paid_mix")
type BenchmarkRecord = dict[str, Any]

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "apps/data-pipeline/pyproject.toml").exists():
            return candidate
    raise RuntimeError("Could not locate repository root")


REPO_ROOT = _repo_root()
DEFAULT_OUTPUT = REPO_ROOT / "scratchpad" / "mpg_synthetic_nonlinear_parameter_recovery.json"


def _configure_jax_cache() -> None:
    cache_dir = REPO_ROOT / "scratchpad" / ".jax_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["JAX_COMPILATION_CACHE_DIR"] = str(cache_dir)
    os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "true"
    os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"
    os.environ["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] = "0"


def _parse_proposals(raw: str) -> tuple[LeafProposal, ...]:
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    unknown = sorted(set(values) - set(LEAF_PROPOSALS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown proposals: {unknown}; allowed: {list(LEAF_PROPOSALS)}"
        )
    if not values:
        raise argparse.ArgumentTypeError("at least one proposal is required")
    return cast("tuple[LeafProposal, ...]", values)


def _parse_metrics(raw: str) -> tuple[str, ...]:
    from nof1_causal_lab.models.ssm.inference.marginal_particle_gibbs.diagnostics import (
        MPGIBBS_DIAGNOSTIC_METRIC_VALUES,
    )

    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    unknown = sorted(set(values) - set(MPGIBBS_DIAGNOSTIC_METRIC_VALUES))
    if unknown:
        raise argparse.ArgumentTypeError(
            "unknown diagnostic metrics: "
            f"{unknown}; allowed: {list(MPGIBBS_DIAGNOSTIC_METRIC_VALUES)}"
        )
    return values


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    try:
        import jax
        import numpy as np

        array = np.asarray(jax.device_get(value))
    except (TypeError, ValueError):
        return str(value)
    if array.ndim == 0:
        return array.item()
    return array.tolist()


def _diagnostic_summary(result: MCMCPosterior) -> BenchmarkRecord:
    from nof1_causal_lab.artifacts.reports.posterior_diagnostics import ParticleSamplerDiagnostics
    from nof1_causal_lab.artifacts.sampling import MarginalParticleGibbsSpec

    diagnostics = result.diagnostics.sampler_diagnostics
    assert isinstance(diagnostics, ParticleSamplerDiagnostics)
    assert isinstance(diagnostics.settings.algorithm, MarginalParticleGibbsSpec)
    return {
        "dsmc_leaf_proposal": _json_ready(diagnostics.dsmc_leaf_proposal),
        "parameter_kernel": _json_ready(diagnostics.parameter_kernel),
        "parameter_preconditioned": _json_ready(diagnostics.parameter_preconditioned),
        "parameter_accept_rate": _json_ready(diagnostics.parameter_accept_rate),
        "latent_update_fraction": _json_ready(diagnostics.latent_update_fraction),
        "latent_frozen_fraction": _json_ready(diagnostics.latent_frozen_fraction),
        "latent_block_coords": _json_ready(diagnostics.latent_block_coords),
        "mcmc_phase_seconds": _json_ready(diagnostics.mcmc_phase_seconds),
        "diagnostic_metrics": _json_ready(diagnostics.diagnostic_metrics),
        "diagnostic_summary_phase": _json_ready(diagnostics.diagnostic_summary_phase),
        "final_param_step_size": _json_ready(diagnostics.final_param_step_size),
        "latent_delta": _json_ready(diagnostics.settings.algorithm.latent_delta),
    }


def _run_one(
    *,
    proposal: LeafProposal,
    model: Any,
    panel: Any,
    args: argparse.Namespace,
) -> BenchmarkRecord:
    from evaluation.recovery.extraction import parameter_recovery, scalar_posterior_ess

    from nof1_causal_lab.artifacts.sampling import MarginalParticleGibbsSpec, SamplerSpec
    from nof1_causal_lab.models.ssm.inference.fit import fit
    from nof1_causal_lab.models.ssm.inference.warmup.outcomes import InitializationFailure

    logger.info("starting dSMC/%s", proposal)
    started = time.monotonic()
    result = fit(
        model.prior_runtime_bundle,
        panel,
        sampler=SamplerSpec(
            num_warmup=args.num_warmup,
            num_samples_per_chain=args.num_samples,
            num_chains=args.num_chains,
            seed=args.seed,
            num_particles=args.num_particles,
            retain_latent_paths=args.retain_latent_paths,
            algorithm=MarginalParticleGibbsSpec(
                n_parameter_particles=args.n_parameter_particles,
                dsmc_leaf_proposal=proposal,
                latent_block_coords=args.latent_block_coords,
                diagnostic_metrics=args.diagnostic_metrics,
                param_step_size=args.param_step_size,
                param_step_size_min=args.param_step_size_min,
                param_step_size_max=args.param_step_size_max,
                param_target_accept=args.param_target_accept,
                adaptation_rate=args.adaptation_rate,
                amala_delta_init=args.amala_delta_init,
                amala_delta_min=args.amala_delta_min,
                amala_delta_max=args.amala_delta_max,
                amala_target_accept=args.amala_target_accept,
                paid_mix_z_weight=args.paid_mix_z_weight,
                paid_mix_pilot_weight=args.paid_mix_pilot_weight,
                paid_mix_pilot_var_scale=args.paid_mix_pilot_var_scale,
                paid_mix_wide_mult=args.paid_mix_wide_mult,
                compute_latent_posterior_summary=not args.skip_latent_posterior_summary,
            ),
        ),
        clock=time.monotonic,
        compute_budget=read_compute_budget(),
    )
    if isinstance(result, (ObservationPreflightFailure, InitializationFailure)):
        raise ValueError(result.message)
    elapsed_seconds = time.monotonic() - started
    logger.info("finished dSMC/%s in %.1fs", proposal, elapsed_seconds)
    return {
        "elapsed_seconds": elapsed_seconds,
        "diagnostics": _diagnostic_summary(result),
        "parameter_recovery": parameter_recovery(
            result,
            elapsed_seconds=elapsed_seconds,
        ),
        "posterior_ess": scalar_posterior_ess(
            result,
            max_sites=args.max_ess_sites,
            elapsed_seconds=elapsed_seconds,
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", dest="t_steps", type=int, default=32)
    parser.add_argument("--data-seed", type=int, default=71)
    parser.add_argument("--diffusion-scale", type=float, default=1.0)
    parser.add_argument("--proposals", type=_parse_proposals, default=LEAF_PROPOSALS)
    parser.add_argument("--num-warmup", type=int, default=500)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--num-chains", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1009)
    parser.add_argument("--n-particles", type=int, default=64)
    parser.add_argument("--n-parameter-particles", type=int, default=2)
    parser.add_argument("--latent-block-coords", type=int)
    parser.add_argument(
        "--diagnostic-metrics",
        type=_parse_metrics,
        default=("particle_identity", "parameter_movement"),
    )
    parser.add_argument("--param-step-size", type=float, default=0.02)
    parser.add_argument("--param-step-size-min", type=float, default=1e-6)
    parser.add_argument("--param-step-size-max", type=float, default=1e3)
    parser.add_argument("--param-target-accept", type=float)
    parser.add_argument("--adaptation-rate", type=float, default=0.05)
    parser.add_argument("--amala-delta-init", type=float, default=0.2)
    parser.add_argument("--amala-delta-min", type=float, default=1e-4)
    parser.add_argument("--amala-delta-max", type=float, default=1e2)
    parser.add_argument("--amala-target-accept", type=float, default=0.574)
    parser.add_argument("--paid-mix-z-weight", type=float, default=0.85)
    parser.add_argument("--paid-mix-pilot-weight", type=float, default=0.10)
    parser.add_argument("--paid-mix-pilot-var-scale", type=float, default=0.25)
    parser.add_argument("--paid-mix-wide-mult", type=float, default=4.0)
    parser.add_argument("--retain-latent-paths", action="store_true")
    parser.add_argument("--skip-latent-posterior-summary", action="store_true")
    parser.add_argument("--max-ess-sites", type=int, default=24)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    from nof1_causal_lab.bootstrap import initialize_numerical_runtime

    initialize_numerical_runtime()
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.n_parameter_particles < 2:
        raise ValueError("--n-parameter-particles must be at least 2")

    _configure_jax_cache()
    from evaluation.fixtures.synthetic_nonlinear import (
        bind_synthetic_nonlinear_panel,
        load_synthetic_nonlinear_model,
        simulate_synthetic_nonlinear_data,
    )

    data = simulate_synthetic_nonlinear_data(
        T=args.t_steps,
        seed=args.data_seed,
        diffusion_scale=args.diffusion_scale,
    )
    model = load_synthetic_nonlinear_model(T=args.t_steps, diffusion_scale=args.diffusion_scale)
    panel = bind_synthetic_nonlinear_panel(model, data)
    runs = {
        f"point:{proposal}": _run_one(
            proposal=proposal,
            model=model,
            panel=panel,
            args=args,
        )
        for proposal in args.proposals
    }
    artifact = {
        "artifact": "mpg_synthetic_nonlinear_parameter_recovery",
        "generated_at": datetime.now(UTC).isoformat(),
        "fixture": "synthetic_nonlinear",
        "config": {
            "T": args.t_steps,
            "data_seed": args.data_seed,
            "diffusion_scale": args.diffusion_scale,
            "proposals": args.proposals,
            "num_warmup": args.num_warmup,
            "num_samples": args.num_samples,
            "num_chains": args.num_chains,
            "seed": args.seed,
            "num_particles": args.num_particles,
            "n_parameter_particles": args.n_parameter_particles,
            "latent_block_coords": args.latent_block_coords,
            "diagnostic_metrics": args.diagnostic_metrics,
        },
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_ready(artifact), indent=2, sort_keys=True) + "\n")
    logger.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
