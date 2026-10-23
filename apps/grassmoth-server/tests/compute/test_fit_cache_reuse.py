"""Whole library fits reuse JAX's cache in fresh and previously traced processes."""

import json
import os
import subprocess
import sys
from pathlib import Path
from time import monotonic

import pytest

pytestmark = [pytest.mark.inference(concern="sampling"), pytest.mark.inference(concern="warmup")]


def test_whole_fits_reuse_jax_cache_across_processes(tmp_path):
    root = Path(__file__).resolve().parents[4]
    results = []
    started = monotonic()
    for phase in ("populate", "reuse"):
        output = tmp_path / f"{phase}.json"
        process = subprocess.run(
            [sys.executable, "-m", "grassmoth_server_test_support.fit_cache", phase, str(output)],
            cwd=root,
            env={
                **os.environ,
                "PYTHONPATH": str(root / "apps/grassmoth-server/tests"),
                "PYTHONHASHSEED": "0",
                "JAX_PLATFORMS": "cpu",
                "JAX_COMPILATION_CACHE_DIR": str(tmp_path / "cache"),
                "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
                "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "-1",
                "GRASSMOTH_DISABLE_JAX_PERSISTENT_CACHE": "0",
            },
            capture_output=True,
            text=True,
            timeout=240,
            check=False,
        )
        assert process.returncode == 0, process.stdout + process.stderr
        results.append(json.loads(output.read_text()))
    print(json.dumps({"seconds": monotonic() - started, "phases": results}, indent=2))
    # The short C0 fixture exercises every scientific program in the F03 inventory;
    # eager primitives are counted too, not restricted to the main sweep.
    assert {
        "project_response_trajectory",
        "draw_sites",
        "affine_chart",
        "draw",
        "project",
        "block",
        "scan",
        "<lambda>",
    } <= set(results[0][0]["traced"])
    # Chain initialization is an anonymous vmapped function. Together with site
    # discovery, the chart, initial draws, projection and sweep it gives six
    # prepared programs before direct compilation paths are routed through it.
    assert sum(item["name"] == "jit(invoke)" for item in results[0][0]["programs"]) >= 6
    assert all(item["compilations"] > 0 for item in results[0])
    assert all(item["compilations"] == 0 for item in results[1]), results[1]
