"""Reproduce the CPU SV teaching collection from apps/grassmoth-server.

Run ``python -m notebooks.support.sv_runs --reuse-tuning-from <directory>`` to
reuse an archived, independently selected pilot grid, or omit that option to
repeat the grid. The default stages prepare, run, time, assess and publish.
Individual stages and --only allow resumable work; only finalize writes latest.
RUN_LENGTHS is configuration, fingerprinted separately from implementation.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import tempfile
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from arviz_stats.base import array_stats

from . import sv_kernels as kernels
from . import sv_model as model

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "artifacts/sv_methods"

# This assignment alone is schedule configuration in the implementation digest.
RUN_LENGTHS = {
    "joint_path": 64000,
    "joint_shocks": 8000,
    "joint_chart": 8000,
    "nuts_ak_whole_path": 200000,
    "mpgibbs_ak_whole_path": 200000,
    "nuts_ak_blocks_path": 200000,
    "mpgibbs_ak_blocks_path": 200000,
    "nuts_mgrad_path": 32000,
    "mpgibbs_mgrad_path": 64000,
    "nuts_bootstrap_path": 32000,
    "mpgibbs_bootstrap_path": 64000,
    "nuts_ak_whole_chart": 80000,
    "mpgibbs_ak_whole_chart": 80000,
    "nuts_ak_blocks_chart": 80000,
    "mpgibbs_ak_blocks_chart": 80000,
    "nuts_mgrad_chart": 8000,
    "mpgibbs_mgrad_chart": 32000,
    "nuts_bootstrap_chart": 8000,
    "mpgibbs_bootstrap_chart": 16000,
    "reference": 64000,
}


def _json(path: Path, value: Any) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=path.name, delete=False
    ) as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _sources() -> tuple[str, dict[str, str], dict[str, str]]:
    sources = {
        path.name: path.read_text()
        for path in (Path(model.__file__), Path(kernels.__file__), Path(__file__))
    }
    normalized = sources.copy()
    runner = sources[Path(__file__).name]
    assignment = next(
        node
        for node in ast.parse(runner).body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "RUN_LENGTHS" for t in node.targets)
    )
    lines = runner.splitlines(keepends=True)
    normalized[Path(__file__).name] = (
        "".join(lines[: assignment.lineno - 1])
        + "RUN_LENGTHS = <schedule>\n"
        + "".join(lines[assignment.end_lineno :])
    )
    full_hashes = {
        name: hashlib.sha256(text.encode()).hexdigest() for name, text in sources.items()
    }
    return _fingerprint(normalized), full_hashes, sources


def _diagnostics(summary: np.ndarray) -> dict[str, Any]:
    return {
        "mean": summary.mean(axis=(0, 1)).tolist(),
        "sd": summary.reshape(-1, summary.shape[-1]).std(axis=0).tolist(),
        "mcse": np.asarray(array_stats.mcse(summary, chain_axis=0, draw_axis=1)).tolist(),
        "rhat": np.asarray(array_stats.rhat(summary, chain_axis=0, draw_axis=1)).tolist(),
        "ess_bulk": np.asarray(array_stats.ess(summary, chain_axis=0, draw_axis=1)).tolist(),
        "ess_mean": np.asarray(
            array_stats.ess(summary, method="mean", chain_axis=0, draw_axis=1)
        ).tolist(),
    }


def _run(name, spec, readings, keys, eta_start, path_start, warmup, draws):
    print(f"START {name}: compile", flush=True)
    function = jax.jit(
        jax.vmap(
            partial(kernels.run_method, spec, warmup=warmup, draws=draws), in_axes=(None, 0, 0, 0)
        )
    )
    start = time.perf_counter()
    compiled = function.lower(readings, keys, eta_start, path_start).compile()
    compilation_seconds = time.perf_counter() - start
    start = time.perf_counter()
    result = compiled(readings, keys, eta_start, path_start)
    jax.block_until_ready(result)
    sampling_seconds = time.perf_counter() - start
    arrays = jax.tree.map(np.asarray, result)
    assert np.all(np.isfinite(arrays.parameters)), name
    assert np.all(np.isfinite(arrays.paths)), name
    assert np.all(np.isfinite(arrays.substep_summaries)), name
    summaries = arrays.substep_summaries[:, :, -1]
    diagnostics = _diagnostics(summaries)
    assert all(np.all(np.isfinite(value)) for value in diagnostics.values()), name
    assert min(diagnostics["ess_mean"]) > 0, name
    assert min(diagnostics["ess_bulk"]) > 0, name
    density = jax.jit(jax.vmap(model.log_joint, in_axes=(0, 0, None)))
    for chain in range(arrays.parameters.shape[0]):
        for start in range(0, draws, 1000):
            values = density(
                arrays.parameters[chain, start : start + 1000],
                arrays.paths[chain, start : start + 1000],
                readings,
            )
            assert np.all(np.isfinite(values)), (name, "nonfinite retained target", chain, start)
    print(
        f"DONE {name}: compile={compilation_seconds:.2f}s run={sampling_seconds:.2f}s "
        f"max_rhat={max(diagnostics['rhat']):.3f} min_ess={min(diagnostics['ess_bulk']):.1f}",
        flush=True,
    )
    return arrays, {
        "compile_seconds": compilation_seconds,
        "sampling_seconds": sampling_seconds,
        "diagnostics": diagnostics,
    }


def _spec_record(spec):
    if isinstance(spec, kernels.JointNUTSSpec):
        return {"family": "joint_nuts", "nuts": asdict(spec.settings)}
    if isinstance(spec, kernels.NUTSSmootherSpec):
        return {
            "family": "nuts_smoother",
            "nuts": asdict(spec.settings),
            "smoother": type(spec.smoother).__name__,
            "smoother_settings": asdict(spec.smoother),
        }
    return {
        "family": "mpgibbs",
        "parameter_cholesky": np.asarray(spec.parameter_cholesky).tolist(),
        "smoother": type(spec.smoother).__name__,
        "smoother_settings": asdict(spec.smoother),
    }


def _save(destination: Path, run, history, chart, metadata: dict[str, Any]):
    draws = run.parameters.shape[1]
    selected = np.unique(np.r_[np.linspace(0, draws - 1, min(60, draws), dtype=int), draws - 1])
    arrays = {
        "parameters": run.parameters,
        "summaries": run.substep_summaries[:, :, -1],
        "substep_summaries": run.substep_summaries[:, : min(60, draws)],
        "selected_path_indices": selected,
        "selected_paths": run.paths[:, selected],
        "last_eta": run.parameters[:, -1],
        "last_path": run.paths[:, -1],
        "mean_path": run.paths.mean(axis=(0, 1)),
        "true_path": np.asarray(history.h),
        "returns": np.asarray(history.r),
        "squared_returns": np.asarray(history.r2),
        "chart_center": np.asarray(chart.center),
        "chart_slopes": np.asarray(chart.slopes),
        "chart_exponents": np.asarray(chart.exponents),
        "chart_reference": np.asarray(chart.reference),
    }
    if isinstance(run.statistics, kernels.NUTSStatistics):
        arrays.update(
            nuts_acceptance=run.statistics.acceptance_probabilities,
            leapfrog_steps=run.statistics.leapfrog_steps,
            divergences=run.statistics.divergences,
        )
    else:
        arrays.update(
            candidate_probabilities=run.statistics.candidate_probabilities,
            selected_labels=run.statistics.selected_labels,
        )
    if isinstance(run.tuning, kernels.NUTSTuning):
        arrays.update(
            nuts_step_size=run.tuning.step_size, nuts_inverse_mass=run.tuning.inverse_mass
        )
    else:
        arrays["parameter_cholesky"] = run.tuning.parameter_cholesky
    np.savez_compressed(destination, allow_pickle=False, **arrays)
    _json(destination.with_suffix(".json"), metadata)


def _read_chart(arrays) -> model.Chart:
    return model.Chart(*(jnp.asarray(arrays[f"chart_{field}"]) for field in model.Chart._fields))


def _smoother(name, delta, particles, block_length):
    match name:
        case "ak_whole":
            return kernels.AuxKalmanSpec(delta)
        case "ak_blocks":
            return kernels.BlockedKalmanSpec(delta, block_length)
        case "mgrad":
            return kernels.MGradSpec(particles, delta)
        case "bootstrap":
            return kernels.BootstrapSpec(particles)
        case _:
            raise ValueError(name)


def _tune(
    readings,
    chart,
    pilot,
    *,
    particles,
    block_length,
    replicates,
    screen_warmup,
    screen_draws,
    confirm_warmup,
    confirm_draws,
    checkpoint,
):
    """Freeze delta by path ESJD and parameter scale by actual multi-chain mean ESS."""
    eta = chart.reference
    target = kernels.sv_target(eta[None], readings)
    warmup = 256
    function = partial(
        kernels.run_path_chain,
        kernels.BootstrapSpec(particles),
        sweeps=warmup + 2 * replicates,
    )
    start = time.perf_counter()
    references = jax.jit(function)(
        jax.random.key(220), target, jnp.asarray(pilot["last_path"][0])
    ).stored_paths[warmup::2]
    jax.block_until_ready(references)
    preparation_seconds = time.perf_counter() - start
    keys = jax.random.split(jax.random.key(221), replicates)
    delta_scans = {}
    for name, grid in (
        ("ak_whole", (0.01, 0.03, 0.07, 0.1, 0.2, 0.3, 0.6)),
        ("ak_blocks", (0.03, 0.1, 0.3, 0.6, 1.0, 3.0)),
        ("mgrad", (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)),
    ):

        def _evaluate(delta, method_name=name):
            spec = _smoother(method_name, delta, particles, block_length)

            def _step(inputs):
                key, reference = inputs
                trace = kernels.smoother_trace(spec, key, target, reference)
                assert isinstance(
                    trace, (kernels.BankTrace, kernels.KalmanTrace, kernels.BlockedKalmanTrace)
                )
                difference = trace.path - reference
                return jnp.array([jnp.mean(difference**2), jnp.mean(difference) ** 2])

            return jax.lax.map(_step, (keys, references), batch_size=8)

        records = []
        start = time.perf_counter()
        compiled = jax.jit(_evaluate).lower(jnp.asarray(grid[0])).compile()
        compile_seconds = time.perf_counter() - start
        for delta in grid:
            start = time.perf_counter()
            values = np.asarray(compiled(jnp.asarray(delta)))
            assert np.all(np.isfinite(values)), (name, delta)
            record = {
                "delta": delta,
                "esjd_per_day": float(values[:, 0].mean()),
                "path_mean_esjd": float(values[:, 1].mean()),
                "mcse_esjd": float(values[:, 0].std(ddof=1) / np.sqrt(replicates)),
                "seconds": time.perf_counter() - start,
            }
            records.append(record)
            print(f"TUNE {name} delta={delta}: ESJD/day={record['esjd_per_day']:.4g}", flush=True)
        selected = max(records, key=lambda row: row["esjd_per_day"])["delta"]
        delta_scans[name] = {
            "selected": selected,
            "scan": records,
            "compile_seconds": compile_seconds,
        }
    delta_scans["bootstrap"] = {"selected": None, "scan": []}

    covariance = np.cov(pilot["parameters"].reshape(-1, 3), rowvar=False)
    cholesky = jnp.asarray(np.linalg.cholesky(covariance))
    chains = pilot["last_eta"].shape[0]
    # Independent paths for tuning starts. Neither these nor the scale pilots are
    # retained comparison draws. The original pilot alone defines chart/covariance.
    initial_etas = jnp.asarray(pilot["last_eta"])
    initial_paths = jnp.asarray(pilot["last_path"])
    independent = jax.jit(
        jax.vmap(
            partial(
                kernels.run_method, kernels.JointNUTSSpec(model.INNOVATIONS), warmup=750, draws=1500
            ),
            in_axes=(None, 0, 0, 0),
        )
    )(readings, jax.random.split(jax.random.key(330), chains), initial_etas, initial_paths)
    jax.block_until_ready(independent)
    starts_eta = independent.parameters[:, -1]
    starts_path = independent.paths[:, -1]
    independent_summaries = np.asarray(independent.substep_summaries[:, :, -1])
    del independent
    criterion = "minimum worst sweeps per mean effective sample over level, log vol, mean h; independent confirmation of two screening winners"
    result = {
        "delta_criterion": "mean squared physical path jump per day at fixed pilot-mean eta",
        "parameter_criterion": criterion,
        "reference_preparation_seconds": preparation_seconds,
        "conditional_reference_warmup": warmup,
        "conditional_reference_stride": 2,
        "replicates": replicates,
        "delta_seed": 221,
        "reference_seed": 220,
        "parameter_start_seed": 330,
        "parameter_start_warmup": 750,
        "parameter_start_draws": 1500,
        "parameter_start_rhat": np.asarray(
            array_stats.rhat(independent_summaries, chain_axis=0, draw_axis=1)
        ).tolist(),
        "screen_seed": 340,
        "confirmation_seed": 341,
        "functional_indices": [0, 2, 3],
        "functional_names": ["level", "log vol", "mean h"],
        "chains": chains,
        "fixed_eta": np.asarray(eta).tolist(),
        "posterior_cholesky": np.asarray(cholesky).tolist(),
        "delta": delta_scans,
        "parameter_scale": {},
    }
    checkpoint.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for coordinate_name, coordinate in (("path", model.PHYSICAL), ("chart", chart)):
        for name in delta_scans:
            spec = _smoother(name, delta_scans[name]["selected"], particles, block_length)
            arm_name = f"mpgibbs_{name}_{coordinate_name}"
            arm_index = len(result["parameter_scale"])
            phases = {}
            grid = (0.1, 0.2, 0.4, 0.6, 0.8, 1.6)
            for phase, warm, count, seed in (
                ("screen", screen_warmup, screen_draws, 340),
                ("confirmation", confirm_warmup, confirm_draws, 341),
            ):
                keys = jax.random.split(jax.random.fold_in(jax.random.key(seed), arm_index), chains)

                def _evaluate(
                    scale,
                    coords=coordinate,
                    path_spec=spec,
                    warmup=warm,
                    draws=count,
                    phase_keys=keys,
                ):
                    method = kernels.MPGibbsSpec(coords, path_spec, scale * cholesky)
                    run = jax.vmap(
                        partial(kernels.run_method, method, warmup=warmup, draws=draws),
                        in_axes=(None, 0, 0, 0),
                    )(readings, phase_keys, starts_eta, starts_path)
                    assert isinstance(run.statistics, kernels.EnsembleStatistics)
                    # The unreturned trajectories are pruned from the compiled scan output.
                    return run.substep_summaries[:, :, -1, :], run.statistics.selected_labels

                start = time.perf_counter()
                compiled = jax.jit(_evaluate).lower(jnp.asarray(grid[0])).compile()
                compile_seconds = time.perf_counter() - start
                records = []
                for scale in grid:
                    start = time.perf_counter()
                    summary, labels = jax.tree.map(np.asarray, compiled(jnp.asarray(scale)))
                    seconds = time.perf_counter() - start
                    assert np.all(np.isfinite(summary)), (arm_name, scale)
                    essentials = summary[:, :, [0, 2, 3]]
                    ess = np.asarray(
                        array_stats.ess(essentials, method="mean", chain_axis=0, draw_axis=1)
                    )
                    rhats = np.asarray(array_stats.rhat(essentials, chain_axis=0, draw_axis=1))
                    valid = bool(
                        np.all(np.isfinite(ess)) and np.all(ess > 0) and np.all(np.isfinite(rhats))
                    )
                    record = {
                        "scale": scale,
                        "ess_mean": [float(x) if np.isfinite(x) else None for x in ess],
                        "rhat": [float(x) if np.isfinite(x) else None for x in rhats],
                        "worst_sweeps_per_ess": float(chains * count / ess.min())
                        if valid
                        else None,
                        "label_change_rate": float(labels.mean()),
                        "seconds": seconds,
                        "valid": valid,
                    }
                    records.append(record)
                    print(
                        f"TUNE {arm_name} {phase} scale={scale}: worst sweeps/ESS={record['worst_sweeps_per_ess']} flip={record['label_change_rate']:.3f}",
                        flush=True,
                    )
                ranked = sorted(
                    (row for row in records if row["valid"]),
                    key=lambda row: row["worst_sweeps_per_ess"],
                )
                if not ranked:
                    raise RuntimeError(f"No finite ESS estimate for {arm_name} {phase}")
                phases[phase] = {
                    "warmup": warm,
                    "draws": count,
                    "seed": seed,
                    "fold_in": arm_index,
                    "compile_seconds": compile_seconds,
                    "scan": records,
                }
                grid = tuple(row["scale"] for row in ranked[:2])
                del compiled
            result["parameter_scale"][arm_name] = {
                "selected": ranked[0]["scale"],
                "criterion": criterion,
                **phases,
            }
            checkpoint.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            print(f"SELECT {arm_name} scale={ranked[0]['scale']}", flush=True)
    return result


def _arm_specs(chart, tuning, particles, block_length):
    arms = [
        ("joint_path", kernels.JointNUTSSpec(model.PHYSICAL)),
        ("joint_shocks", kernels.JointNUTSSpec(model.INNOVATIONS)),
        ("joint_chart", kernels.JointNUTSSpec(chart)),
    ]
    cholesky = jnp.asarray(tuning["posterior_cholesky"])
    for coordinate_name, coordinate in (("path", model.PHYSICAL), ("chart", chart)):
        for name in ("ak_whole", "ak_blocks", "mgrad", "bootstrap"):
            smoother = _smoother(name, tuning["delta"][name]["selected"], particles, block_length)
            ensemble = f"mpgibbs_{name}_{coordinate_name}"
            scale = tuning["parameter_scale"][ensemble]["selected"]
            arms.extend(
                (
                    (
                        f"nuts_{name}_{coordinate_name}",
                        kernels.NUTSSmootherSpec(coordinate, smoother),
                    ),
                    (ensemble, kernels.MPGibbsSpec(coordinate, smoother, scale * cholesky)),
                )
            )
    return arms


def _prepare(destination, history, config, source_hash, reuse):
    pilot_file = destination / "pilot.npz"
    if not pilot_file.exists():
        eta = model.to_eta(0.0, 0.05, 0.3)
        path = model.refine_path(eta, jnp.zeros(config["days"]), history.r2)
        run, metadata = _run(
            "pilot",
            kernels.JointNUTSSpec(model.INNOVATIONS),
            history.r2,
            jax.random.split(jax.random.key(101), config["chains"]),
            jnp.broadcast_to(eta, (config["chains"], 3)),
            jnp.broadcast_to(path, (config["chains"], config["days"])),
            config["pilot_warmup"],
            config["pilot_draws"],
        )
        center = jnp.asarray(run.parameters.mean(axis=(0, 1)))
        chart = jax.jit(model.fit_chart)(
            center, jnp.asarray(run.paths.mean(axis=(0, 1))), history.r2
        )
        _save(
            pilot_file,
            run,
            history,
            chart,
            {
                **metadata,
                "name": "pilot",
                "config": config,
                "source_hash": source_hash,
                "random_seed": 101,
            },
        )
    with np.load(pilot_file) as pilot:
        if (destination / "tuning.json").exists():
            return
        if reuse is not None:
            old_metadata = json.loads((reuse / "pilot.json").read_text())
            assert old_metadata["config"]["seed"] == config["seed"]
            assert old_metadata["config"]["days"] == config["days"]
            with np.load(reuse / "pilot.npz") as previous:
                names = (
                    "parameters",
                    "last_eta",
                    "last_path",
                    "chart_center",
                    "chart_slopes",
                    "chart_exponents",
                    "chart_reference",
                    "squared_returns",
                )
                for name in names:
                    assert np.array_equal(pilot[name], previous[name]), ("pilot changed", name)
            evidence = json.loads((reuse / "tuning.json").read_text())
            archive = destination / "tuning-evidence"
            archive.mkdir(exist_ok=True)
            shutil.copyfile(reuse / "tuning.json", archive / "tuning.json")
            shutil.copyfile(reuse / "pilot.json", archive / "pilot.json")
            source_archive = reuse / "sources"
            if not source_archive.is_dir():
                raise FileNotFoundError(
                    f"Archive the tuning implementation in {source_archive} first"
                )
            shutil.copytree(source_archive, archive / "sources", dirs_exist_ok=True)
            evidence["reuse_provenance"] = {
                "source_hash": old_metadata["source_hash"],
                "source_directory": str(reuse.resolve()),
                "evidence_sha256": hashlib.sha256((reuse / "tuning.json").read_bytes()).hexdigest(),
                "pilot_bit_identical_fields": names,
                "reason": "Previously independent tuning evidence; final pilot regenerated and identical. See verification.md for valid-path regression checks.",
            }
        else:
            start = time.perf_counter()
            evidence = _tune(
                history.r2,
                _read_chart(pilot),
                pilot,
                particles=config["particles"],
                block_length=config["block_length"],
                replicates=config["tuning_replicates"],
                screen_warmup=config["scale_screen_warmup"],
                screen_draws=config["scale_screen_draws"],
                confirm_warmup=config["scale_confirm_warmup"],
                confirm_draws=config["scale_confirm_draws"],
                checkpoint=destination / "tuning-progress.json",
            )
            evidence["total_seconds"] = time.perf_counter() - start
        evidence["selection_applied_to_source_hash"] = source_hash
        _json(destination / "tuning.json", evidence)


def _run_arms(destination, history, chart, tuning, config, source_hash, schedule_hash, only):
    arms = _arm_specs(chart, tuning, config["particles"], config["block_length"])
    arms.append(("reference", kernels.JointNUTSSpec(model.INNOVATIONS)))
    with np.load(destination / "pilot.npz") as pilot:
        starts_eta, starts_path = jnp.asarray(pilot["last_eta"]), jnp.asarray(pilot["last_path"])
    for index, (name, spec) in enumerate(arms):
        if only and name not in only:
            continue
        result_file = destination / f"{name}.npz"
        draws = RUN_LENGTHS[name]
        if result_file.exists():
            existing = json.loads(result_file.with_suffix(".json").read_text())
            if existing["source_hash"] != source_hash:
                raise ValueError(f"Source mismatch for {name}")
            if existing["config"]["draws"] == draws:
                print(f"CACHED {name}: {draws} draws", flush=True)
                continue
            archive = destination / f"retained-{existing['config']['draws']}"
            archive.mkdir(exist_ok=True)
            for source in (result_file, result_file.with_suffix(".json")):
                source.replace(archive / source.name)
        seed = 139 if name == "reference" else 102
        key = (
            jax.random.key(seed)
            if name == "reference"
            else jax.random.fold_in(jax.random.key(seed), index)
        )
        run, metadata = _run(
            name,
            spec,
            history.r2,
            jax.random.split(key, config["chains"]),
            starts_eta,
            starts_path,
            config["warmup"],
            draws,
        )
        _save(
            result_file,
            run,
            history,
            chart,
            {
                **metadata,
                "name": name,
                "spec": _spec_record(spec),
                "delta": getattr(getattr(spec, "smoother", None), "delta", None),
                "parameter_scale": tuning["parameter_scale"][name]["selected"]
                if isinstance(spec, kernels.MPGibbsSpec)
                else None,
                "parameter_tuning_criterion": tuning["parameter_criterion"]
                if isinstance(spec, kernels.MPGibbsSpec)
                else None,
                "config": {**config, "draws": draws},
                "source_hash": source_hash,
                "schedule_hash": schedule_hash,
                "random_seed": seed,
                "fold_in": None if name == "reference" else index,
            },
        )
        del run
        jax.clear_caches()


def _time_arms(destination, history, chart, tuning, config, source_hash):
    """Sequential, isolated throughput: compilation excluded; warmup included."""
    arms = _arm_specs(chart, tuning, config["particles"], config["block_length"])
    with np.load(destination / "pilot.npz") as pilot:
        starts_eta, starts_path = jnp.asarray(pilot["last_eta"]), jnp.asarray(pilot["last_path"])
    costs, records = {}, {}
    for index, (name, spec) in enumerate(arms):
        run, record = _run(
            "timing_" + name,
            spec,
            history.r2,
            jax.random.split(jax.random.fold_in(jax.random.key(140), index), config["chains"]),
            starts_eta,
            starts_path,
            500,
            500,
        )
        costs[name] = 1000 * record["sampling_seconds"] / (config["chains"] * 1000)
        records[name] = record
        del run
        jax.clear_caches()
    _json(destination / "timing.json", costs)
    _json(
        destination / "timing-metadata.json",
        {
            "source_hash": source_hash,
            "warmup": 500,
            "draws": 500,
            "chains": config["chains"],
            "seed": 140,
            "fold_in": "arm index in manifest order",
            "arms": records,
            "definition": "Sequential CPU post-compilation seconds / [chains * (warmup + draws)] * 1000. Short adaptation schedule, amortized throughput; not total cost to precision.",
            "jax_version": jax.__version__,
            "devices": [str(device) for device in jax.devices()],
        },
    )


def _finalize(destination, config, source_hash, schedule_hash):
    """Publish only a complete, finite, sufficiently explored collection."""
    arm_names = [name for name in RUN_LENGTHS if name != "reference"]
    records = {name: json.loads((destination / f"{name}.json").read_text()) for name in RUN_LENGTHS}
    reference = records["reference"]["diagnostics"]
    rows, flags = [], []
    functionals = ("level", "log decay", "log vol", "mean h", "RMS residual", "middle h")
    for name, record in records.items():
        assert record["source_hash"] == source_hash, name
        assert record["config"]["draws"] == RUN_LENGTHS[name], name
        assert (destination / f"{name}.npz").exists(), name
        d = record["diagnostics"]
        assert all(np.all(np.isfinite(value)) for value in d.values()), name
        z = (np.asarray(d["mean"]) - reference["mean"]) / np.hypot(d["mcse"], reference["mcse"])
        rhat = max(d["rhat"])
        if rhat > 1.01:
            flags.append(
                f"{name}: Rhat {rhat:.5f} > 1.01; increase its RUN_LENGTHS entry and rerun."
            )
        z_flags = ", ".join(
            f"{functionals[i]} ({value:+.2f})" for i, value in enumerate(z) if abs(value) > 3
        )
        rows.append(
            f"| {name} | {RUN_LENGTHS[name]:,} | {rhat:.5f} | {min(d['ess_mean']):.1f} | {max(abs(z)):.2f} | {z_flags or '—'} |"
        )
    timing = json.loads((destination / "timing.json").read_text())
    assert set(timing) == set(arm_names)
    assert all(np.isfinite(value) and value > 0 for value in timing.values())
    assert (
        json.loads((destination / "timing-metadata.json").read_text())["source_hash"] == source_hash
    )
    verification = (
        "# Collection verification\n\n"
        f"Implementation `{source_hash}`; schedule `{schedule_hash}`.\n\n"
        "The runner checked every retained parameter, physical path, substep summary and "
        "diagnostic for finiteness. Every retained state's exact nonlinear joint log density "
        "was evaluated separately and required to be finite. Mean and bulk ESS were positive. "
        "This publication pass checked all 19 arms plus the independent reference against "
        "the implementation hash and current length table, and checked the complete isolated "
        "timing pass against that implementation. Rhat must be at most 1.01 for publication.\n\n"
        "These automated artifact checks do not replace kernel invariance or density oracles. "
        "Independent scientific verification is recorded separately for this implementation.\n"
    )
    (destination / "verification.md").write_text(verification)
    report = (
        "# Final SV collection assessment\n\n"
        + f"Implementation `{source_hash}`; schedule `{schedule_hash}`.\n\n"
    )
    report += "Each row compares six posterior means with an independently keyed joint-shocks reference (4 × 64,000 draws). z uses the square root of the sum of squared MCSEs. These are Monte Carlo agreement checks, not posterior calibration tests. Multiple comparisons can produce isolated |z| > 3. Finite pilot selection uncertainty remains.\n\n"
    report += (
        "| Arm | Draws/chain | Worst Rhat | Min mean ESS | Max abs z | abs z > 3 |\n|---|---:|---:|---:|---:|---|\n"
        + "\n".join(rows)
        + "\n\n"
    )
    report += "\n".join(flags) if flags else "All reported Rhat values are at most 1.01.\n"
    (destination / "assessment.md").write_text(report)
    if flags:
        raise RuntimeError("\n".join(flags))
    manifest = {
        "config": config,
        "source_hash": source_hash,
        "schedule_hash": schedule_hash,
        "directory": str(destination.resolve()),
        "arms": {name: f"{name}.npz" for name in arm_names},
        "pilot": "pilot.npz",
        "tuning": "tuning.json",
        "reference": "reference.npz",
        "timing": "timing.json",
        "verification": "verification.md",
        "assessment": "assessment.md",
        "timing_definition": "See timing-metadata.json: isolated CPU, 500 warmup + 500 retained updates per chain; amortized post-compilation throughput.",
    }
    _json(destination / "manifest.json", manifest)
    _json(destination.parent / "latest.json", manifest)
    print(f"FINAL {destination}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (
        ("days", 1000),
        ("seed", 721),
        ("chains", 4),
        ("warmup", 1000),
        ("pilot-warmup", 750),
        ("pilot-draws", 1500),
        ("particles", 32),
        ("tuning-replicates", 128),
        ("block-length", 128),
        ("scale-screen-warmup", 500),
        ("scale-screen-draws", 1500),
        ("scale-confirm-warmup", 1000),
        ("scale-confirm-draws", 4000),
    ):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument(
        "--stage", choices=("all", "prepare", "runs", "timing", "finalize"), default="all"
    )
    parser.add_argument("--only", nargs="+", choices=tuple(RUN_LENGTHS))
    parser.add_argument("--reuse-tuning-from", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    jax.config.update("jax_platforms", "cpu")
    jax.config.update("jax_enable_x64", True)
    assert jax.default_backend() == "cpu"
    config = {
        name: value
        for name, value in vars(args).items()
        if name not in ("stage", "only", "reuse_tuning_from", "output")
    }
    evidence_hash = (
        None
        if args.reuse_tuning_from is None
        else hashlib.sha256((args.reuse_tuning_from / "tuning.json").read_bytes()).hexdigest()
    )
    config["reused_tuning_evidence_sha256"] = evidence_hash
    source_hash, full_hashes, sources = _sources()
    schedule_hash = _fingerprint(RUN_LENGTHS)
    cache_id = _fingerprint({"config": config, "source_hash": source_hash})[:16]
    destination = args.output / f"T{args.days}-{cache_id}"
    destination.mkdir(parents=True, exist_ok=True)
    source_dir = destination / "sources" / schedule_hash[:16]
    source_dir.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        # Provenance bytes are data, not duplicate executable modules for lint.
        (source_dir / (name + ".txt")).write_text(source)
    _json(
        source_dir / "provenance.json",
        {
            "source_hash": source_hash,
            "full_file_sha256": full_hashes,
            "schedule_hash": schedule_hash,
            "run_lengths": RUN_LENGTHS,
        },
    )
    print(f"COLLECTION {destination}", flush=True)
    history = model.simulate(args.days, args.seed)
    if args.stage in ("all", "prepare"):
        _prepare(destination, history, config, source_hash, args.reuse_tuning_from)
    if args.stage in ("all", "runs", "timing"):
        with np.load(destination / "pilot.npz") as pilot:
            chart = _read_chart(pilot)
        tuning = json.loads((destination / "tuning.json").read_text())
        if args.stage in ("all", "runs"):
            _run_arms(
                destination, history, chart, tuning, config, source_hash, schedule_hash, args.only
            )
        if args.stage in ("all", "timing"):
            _time_arms(destination, history, chart, tuning, config, source_hash)
    if args.stage in ("all", "finalize"):
        _finalize(destination, config, source_hash, schedule_hash)


if __name__ == "__main__":
    main()
