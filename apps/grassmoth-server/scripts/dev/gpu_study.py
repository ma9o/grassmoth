"""Run an isolated public-HTTP study entirely inside one ephemeral L40S container.

The local entry point only reads request files and sends them to Modal. The
server, worker, simulation, fit and diagnostics all execute remotely. Study
history, request envelopes, profiles and logs remain on the result Volume.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import modal

from grassmoth_server.compute.modal_images import ROOT, SERVER, gpu_image

if TYPE_CHECKING:
    from grassmoth.public import JsonObject
    from grassmoth_server.actions.contracts.polling import ActionSuccess

app = modal.App("grassmoth-gpu-study")
results = modal.Volume.from_name("grassmoth-gpu-verification", create_if_missing=True)
image = gpu_image.add_local_file(
    SERVER / "scripts/dev/temporal_dev_server.py", "/root/temporal_dev_server.py"
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
    volumes={"/results": results},
    env={
        "JAX_PLATFORMS": "cuda",
        "JAX_ENABLE_X64": "false",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    },
)
def run_study(
    job: str, truth_json: str, model_json: str, settings_json: str, source_patch: str
) -> str:
    """Create fixtures and execute one fit through the ordinary HTTP workflow."""
    from pydantic import TypeAdapter

    from grassmoth_server.actions.contracts.polling import ActionPoll

    poll_adapter = TypeAdapter(ActionPoll)
    root = Path("/results") / job
    root.mkdir()
    if source_patch:
        (root / "source.patch").write_text(source_patch)
        subprocess.run(
            ["git", "apply", "--whitespace=nowarn", "-"],
            input=source_patch,
            text=True,
            check=True,
            cwd="/root",
        )
    environment = {
        **os.environ,
        "DEPLOYMENT_ENV": "development",
        "READ_ONLY_FACADE": "0",
        "TEMPORAL_ADDRESS": "127.0.0.1:17233",
        "TEMPORAL_TASK_QUEUE": "gpu-verification",
        "GRASSMOTH_DATA_ROOT": str(root / "data"),
    }
    processes: list[subprocess.Popen[bytes]] = []
    logs = []

    def start(name: str, arguments: list[str]) -> None:
        log = (root / f"{name}.log").open("w")
        logs.append(log)
        processes.append(
            subprocess.Popen(arguments, env=environment, stdout=log, stderr=subprocess.STDOUT)
        )

    try:
        start("temporal", [sys.executable, "/root/temporal_dev_server.py", "--port", "17233"])
        for _ in range(120):
            try:
                with socket.create_connection(("127.0.0.1", 17233), timeout=0.5):
                    break
            except OSError:
                time.sleep(0.5)
        else:
            raise RuntimeError("Remote Temporal did not start")
        start("worker", [sys.executable, "-m", "grassmoth_server.actions.temporal.worker"])
        start(
            "api",
            [sys.executable, "-m", "uvicorn", "grassmoth_server.api.app:app", "--port", "18100"],
        )
        with httpx.Client(
            base_url="http://127.0.0.1:18100", headers={"Accept": "application/json"}, timeout=120
        ) as client:
            for _ in range(120):
                try:
                    client.get("/api/workspaces").raise_for_status()
                    break
                except httpx.TransportError:
                    time.sleep(0.5)
            else:
                raise RuntimeError("Remote HTTP server did not start")

            def run(action: str, payload: JsonObject) -> ActionSuccess:
                request = {"action": action, "input": payload}
                (root / f"{action}-request.json").write_text(json.dumps(request))
                url = f"/api/studies/GPUNUTS/{action}"
                response = client.post(url, json=request)
                response.raise_for_status()
                envelope = response.json()
                print(f"{job}: {action} submitted {envelope['call_id']}", flush=True)
                while envelope["status"] not in ("success", "failed"):
                    time.sleep(1)
                    response = client.get(f"{url}/{envelope['call_id']}")
                    response.raise_for_status()
                    envelope = response.json()
                (root / f"{action}-result.json").write_text(json.dumps(envelope))
                print(f"{job}: {action} {envelope['status']}", flush=True)
                if envelope["status"] != "success":
                    raise RuntimeError(str(envelope["messages"][-3:]))
                decoded = poll_adapter.validate_python(envelope)
                assert decoded.status == "success"
                return decoded

            def produced(commit: str, artifact: str) -> str:
                timeline = client.get("/api/studies/GPUNUTS/timeline").json()
                return next(
                    item["revision"]
                    for attempt in timeline["attempts"]
                    if attempt["commit_id"] == commit
                    for item in attempt["record"]["attempt"]["outcome"]["effects"]["produced"]
                    if item["artifact_id"] == artifact
                )

            question = run(
                "edit_question",
                {
                    "question": {
                        "text": "How persistent is daily volatility?",
                        "outcome": {
                            "id": "construct:h",
                            "name": "Daily volatility",
                            "reasoning": {"text": "The modeled course of daily volatility."},
                            "role": "endogenous",
                            "temporal_status": "time_varying",
                        },
                        "queries": {},
                    }
                },
            )
            truth = run(
                "edit_model",
                {
                    "parent_ref": produced(question.commit_id, "model"),
                    "dynamical_model_spec": json.loads(truth_json),
                },
            )
            truth_ref = produced(truth.commit_id, "model")
            simulation = run(
                "simulate",
                {
                    "dynamical_model_spec_ref": truth_ref,
                    "simulation": {"start": "2020-01-01", "horizon": "2999d"},
                    "seed": 721,
                    "num_replicates": 1,
                },
            )
            model = run(
                "edit_model",
                {"parent_ref": truth_ref, "dynamical_model_spec": json.loads(model_json)},
            )
            settings = json.loads(settings_json)
            # "local" means this L40S container; no second GPU allocation is needed.
            settings["hardware"] = "local"
            result = run(
                "fit",
                {
                    "target": {
                        "kind": "new",
                        "dynamical_model_spec_ref": produced(model.commit_id, "model"),
                        "data_ref": {"revision": simulation.commit_id, "replicate_index": 0},
                    },
                    "settings": settings,
                },
            )
            body = json.loads(result.model_dump_json())["body"]
            summary = json.dumps(
                {
                    "profile": body["profile"],
                    "diagnostics": body["inference"]["core"]["inference_metadata"][
                        "sampler_diagnostics"
                    ],
                }
            )
            (root / "summary.json").write_text(summary)
            print(summary, flush=True)
            return summary
    finally:
        for process in reversed(processes):
            process.terminate()
        for process in reversed(processes):
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log in logs:
            log.close()
        print(
            (root / "worker.log").read_text()[-12000:]
            if (root / "worker.log").exists()
            else "Worker did not start",
            flush=True,
        )
        results.commit()


@app.local_entrypoint()
def main(job: str, settings: str, source_patch: str = "") -> None:
    """Ship fixture documents and a fit-settings JSON file; execute nothing locally."""
    case = ROOT / "benchmarks/cases/suite/SV-cubic"
    run_study.remote(
        job,
        (case / "truth.json").read_text(),
        (case / "fit.json").read_text(),
        Path(settings).read_text(),
        Path(source_patch).read_text() if source_patch else "",
    )
