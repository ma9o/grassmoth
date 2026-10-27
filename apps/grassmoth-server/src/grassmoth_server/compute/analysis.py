"""Run an analysis script beside a verified cached fit on Modal CPU."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from grassmoth.public import Value
from grassmoth_server.actions.contracts.identity import CallId
from grassmoth_server.compute.analysis_cache import (
    ANALYSIS_CACHE_ROOT,
    ANALYSIS_CACHE_VOLUME,
    AnalysisCall,
    cached_fit_result,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


class AnalysisOutput(Value):
    """A script's exit status and combined standard output from its CPU runner."""

    exit_code: int
    log: str


def _analyze_cpu(
    call: AnalysisCall, script: bytes, arguments: tuple[str, ...]
) -> Iterator[AnalysisOutput]:
    """Pass a verified MessagePack body as argv[1]; results never enter an image."""
    result = cached_fit_result(ANALYSIS_CACHE_ROOT, call)
    with TemporaryDirectory() as directory:
        path = Path(directory) / "analysis.py"
        path.write_bytes(script)
        completed = subprocess.run(
            (sys.executable, str(path), str(result), *arguments),
            cwd=directory,
            env={**os.environ, "GRASSMOTH_RESULT_PATH": str(result)},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    yield AnalysisOutput(exit_code=completed.returncode, log=completed.stdout.decode())


def analyze_on_modal(
    call: AnalysisCall, script: bytes, arguments: tuple[str, ...]
) -> AnalysisOutput:
    """Send only script bytes and lookup metadata to a bounded CPU-only runner."""
    import modal

    from grassmoth_server.compute.modal_images import cpu_image

    app = modal.App("grassmoth-saved-fit-analysis")
    analyze = app.function(
        image=cpu_image,
        cpu=(2, 2),
        memory=(32768, 32768),
        timeout=1800,
        scaledown_window=2,
        volumes={str(ANALYSIS_CACHE_ROOT): modal.Volume.from_name(ANALYSIS_CACHE_VOLUME)},
        env={
            "JAX_PLATFORMS": "cpu",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
    )(_analyze_cpu)
    with app.run():
        (output,) = analyze.remote_gen(call, script, arguments)
        return AnalysisOutput.model_validate(output)


def main() -> None:
    """Resolve a published fit using journal metadata, then analyze its cached output."""
    from grassmoth_server.study.persistence.history import StudyRepository
    from grassmoth_server.study.records import Applied

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    analyze = commands.add_parser("analyze", help="Analyze a cached fit on Modal CPU.")
    analyze.add_argument("--workspace", required=True)
    analyze.add_argument("--call", required=True, type=CallId)
    analyze.add_argument("script", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    script = args.script[1:] if args.script[:1] == ["--"] else args.script
    if not script:
        parser.error("supply -- script.py [arguments]")
    call = AnalysisCall(workspace_id=args.workspace, call_id=args.call)
    revision = StudyRepository(call.workspace_id).call(call.call_id)
    if (
        revision is None
        or revision.record.attempt.action != "fit"
        or not isinstance(revision.record.attempt.outcome, Applied)
    ):
        parser.error("--call must select a published successful fit in --workspace")
    output = analyze_on_modal(call, Path(script[0]).read_bytes(), tuple(script[1:]))
    print(output.log, end="")
    raise SystemExit(output.exit_code)
