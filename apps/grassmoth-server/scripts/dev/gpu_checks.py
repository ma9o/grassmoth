"""Run repository checks in an ephemeral L40S container, retaining logs and edits.

Only Git-visible source files are uploaded. Ignored results, credentials, caches
and local dependencies stay on the workstation. The local entry point performs
no checks; the requested command and dependency installation run remotely.
"""

from __future__ import annotations

import io
import os
import shlex
import subprocess
import urllib.request
import zipfile
from pathlib import Path

import modal

from grassmoth_server.compute.modal_images import ROOT, gpu_image

app = modal.App("grassmoth-gpu-checks")
results = modal.Volume.from_name("grassmoth-gpu-verification", create_if_missing=True)
image = gpu_image
if modal.is_local():
    source_files = (
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT
        )
        .decode()
        .split("\0")
    )
    for source_file in sorted(set(source_files) - {""}):
        source = ROOT / source_file
        if source.is_file():
            image = image.add_local_file(source, "/workspace/" + source_file)


@app.function(
    image=image,
    gpu="L40S",
    cpu=8,
    memory=32768,
    timeout=600,
    startup_timeout=300,
    scaledown_window=2,
    retries=0,
    volumes={"/results": results},
    env={"JAX_PLATFORMS": "cuda", "JAX_ENABLE_X64": "false"},
)
def run_checks(job: str, command: str, support: dict[str, str]) -> int:
    """Install tooling, run the explicit command and persist any generated edits."""
    root = Path("/workspace")
    (root / ".venv").symlink_to("/root/.venv", target_is_directory=True)
    output = Path("/results") / job
    output.mkdir()
    for filename, content in support.items():
        destination = root / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content)
    environment = {
        **os.environ,
        "PATH": "/root/bun-linux-x64:/root/.venv/bin:" + os.environ["PATH"],
        "PYTHONPATH": str(root / "packages/grassmoth/src")
        + ":"
        + str(root / "apps/grassmoth-server/src")
        + ":"
        + str(root / "packages/checks/src"),
        "UV_PROJECT_ENVIRONMENT": "/root/.venv",
        "UV_NO_SYNC": "1",
        "DEPLOYMENT_ENV": "development",
        "GRASSMOTH_DATA_ROOT": "/tmp/grassmoth-checks-data",
        "TURBO_TELEMETRY_DISABLED": "1",
    }

    def run(arguments: list[str]) -> int:
        process = subprocess.Popen(
            arguments,
            cwd=root,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert process.stdout is not None
        with (output / "output.log").open("a") as log:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
        return process.wait()

    try:
        with (
            urllib.request.urlopen(
                "https://github.com/oven-sh/bun/releases/download/bun-v1.3.14/bun-linux-x64.zip"
            ) as response,
            zipfile.ZipFile(io.BytesIO(response.read())) as archive,
        ):
            archive.extractall("/root")
        Path("/root/bun-linux-x64/bun").chmod(0o755)
        Path("/root/bun-linux-x64/bunx").symlink_to("bun")
        for arguments in (
            ["git", "init"],
            ["git", "add", "--all"],
            [
                "uv",
                "sync",
                "--frozen",
                "--all-packages",
                "--group",
                "dev",
                "--group",
                "cloud",
                "--group",
                "gpu",
            ],
            ["bun", "install", "--frozen-lockfile"],
        ):
            code = run(arguments)
            if code:
                return code
        return run(["bash", "-c", command])
    finally:
        patch = subprocess.run(
            ["git", "diff", "--binary"], cwd=root, capture_output=True, check=False
        )
        (output / "changes.patch").write_bytes(patch.stdout)
        results.commit()


@app.local_entrypoint()
def main(job: str, command: str, support: str = "") -> None:
    """Submit an explicit check command without executing it on the workstation."""
    files = {filename: (ROOT / filename).read_text() for filename in shlex.split(support)}
    raise SystemExit(run_checks.remote(job, command, files))
