"""Terminate only registered benchmark allocations at their declared wall limits."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from nof1_causal_lab.compute.modal_budget import ModalAllocation


def expired(allocation: ModalAllocation, now: float) -> bool:
    """Completed calls and expired reservations must not retain live paid containers."""
    return (
        allocation.finished_at is not None
        or now
        >= allocation.created_at
        + allocation.policy.startup_seconds
        + allocation.policy.timeout_seconds
    )


def watch(root: Path) -> None:
    """Maintain admission heartbeats and stop registered apps, including stalled builds."""
    root.mkdir(parents=True, exist_ok=True)
    while not (root / "STOP").exists():
        directories = tuple(path for path in root.iterdir() if path.is_dir())
        for path in directories:
            (path / "watchdog.heartbeat").touch()
        allocations = tuple(
            ModalAllocation.model_validate_json(path.read_text())
            for directory in directories
            for path in directory.glob("*.json")
        )
        due = {item.app_name: item for item in allocations if expired(item, time.time())}
        if due:
            result = subprocess.run(
                ["modal", "app", "list", "--json"],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            apps = json.loads(result.stdout)
            (root / "apps.json").write_text(result.stdout)
            for app in apps:
                if app["description"] in due and app["state"] not in {"stopped", "stopping..."}:
                    allocation = due[app["description"]]
                    subprocess.run(
                        ["modal", "app", "stop", app["app_id"], "--yes"], check=True, timeout=30
                    )
                    with (root / "stopped.jsonl").open("a") as stream:
                        stream.write(
                            json.dumps(
                                {
                                    "allocation": allocation.identity,
                                    "app_id": app["app_id"],
                                    "time": time.time(),
                                }
                            )
                            + "\n"
                        )
        time.sleep(2)


def main() -> None:
    """Watch the exclusively owned allocation directories of one authorized campaign."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry", type=Path)
    watch(parser.parse_args().registry)


if __name__ == "__main__":
    main()
