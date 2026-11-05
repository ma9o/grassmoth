"""Run an explicitly selected numerical test subset on one ephemeral Modal L40S.

Example: uv run modal run apps/grassmoth-server/scripts/dev/gpu_tests.py \
    --files packages/grassmoth/tests/models/ssm/inference/test_nuts_da.py
"""

import shlex
import subprocess
import sys

import modal

from grassmoth_server.compute.modal_images import LIBRARY, ROOT, SERVER, gpu_image

app = modal.App("grassmoth-gpu-tests")
image = (
    gpu_image.add_local_dir(LIBRARY / "tests", "/root/packages/grassmoth/tests")
    .add_local_dir(SERVER / "tests", "/root/apps/grassmoth-server/tests")
    .add_local_dir(ROOT / "packages/checks/src", "/root/packages/checks/src")
)


@app.function(
    image=image,
    gpu="L40S",
    cpu=8,
    memory=32768,
    timeout=300,
    startup_timeout=300,
    scaledown_window=2,
    retries=0,
    env={"JAX_PLATFORMS": "cuda", "JAX_ENABLE_X64": "false"},
)
def run_tests(arguments: tuple[str, ...]) -> tuple[int, str]:
    """Execute pytest with the installed locked dependencies and the mounted source."""
    import jax

    if jax.default_backend() != "gpu":
        raise RuntimeError("Numerical tests require the requested GPU backend")
    process = subprocess.Popen(
        [sys.executable, "-m", "pytest", *arguments],
        cwd="/root",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert process.stdout is not None
    lines: list[str] = []
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    return process.wait(), "".join(lines)


@app.local_entrypoint()
def main(
    files: str,
    select: str = "inference",
    expression: str = "",
    config: str = "packages/grassmoth/pyproject.toml",
) -> None:
    """Require explicit files; never default to the full numerical suite."""
    arguments = (
        "-c",
        config,
        *shlex.split(files),
        "-v",
        "-m",
        select,
        "-n",
        "0",
        "-k",
        expression,
    )
    code, output = run_tests.remote(arguments)
    print(output)
    if code:
        raise SystemExit(code)
