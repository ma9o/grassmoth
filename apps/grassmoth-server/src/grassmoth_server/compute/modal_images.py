"""Reusable Modal images for hosted reads and numerical compute."""

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[5]
SERVER = ROOT / "apps/grassmoth-server"
LIBRARY = ROOT / "packages/grassmoth"


def _image(*, gpu: bool) -> modal.Image:
    """Install locked workspace dependencies, then mount both source packages."""
    groups = "--group dev --group cloud" + (" --group gpu" if gpu else "")
    return (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git")
        .pip_install("uv")
        .add_local_file(ROOT / "pyproject.toml", "/root/pyproject.toml", copy=True)
        .add_local_file(ROOT / "uv.lock", "/root/uv.lock", copy=True)
        .add_local_file(
            LIBRARY / "pyproject.toml", "/root/packages/grassmoth/pyproject.toml", copy=True
        )
        .add_local_file(
            SERVER / "pyproject.toml", "/root/apps/grassmoth-server/pyproject.toml", copy=True
        )
        .run_commands(
            "uv sync --project /root --frozen --all-packages --no-install-workspace " + groups
        )
        .env(
            {
                "PATH": "/root/.venv/bin:$PATH",
                "PYTHONPATH": "/root/packages/grassmoth/src:/root/apps/grassmoth-server/src",
                "DEPLOYMENT_ENV": "production",
            }
        )
        .add_local_file(SERVER / "config.yaml", "/root/apps/grassmoth-server/config.yaml")
        .add_local_dir(LIBRARY / "src", "/root/packages/grassmoth/src")
        .add_local_dir(SERVER / "src", "/root/apps/grassmoth-server/src")
    )


cpu_image = _image(gpu=False)
gpu_image = _image(gpu=True)
