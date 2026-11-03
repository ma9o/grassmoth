"""Fresh-process runner for the whole-fit cache regression, using real C0 fits."""

import json
import logging
import sys
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import monotonic

import jax
import polars as pl

from grassmoth.actions.contracts.io import FitInputs, FitOutput
from grassmoth.actions.fit.action import fit
from grassmoth.artifacts.model.dynamical_model_spec import DynamicalModelSpec
from grassmoth.artifacts.model.question import QuestionSpec
from grassmoth.artifacts.reports.inference import FitRuntime
from grassmoth.artifacts.sampling import (
    GaussianInitializationSpec,
    GuidedPathSpec,
    SamplerSpec,
)
from grassmoth.compute.budget import ComputeBudget, SharedMemoryBudget
from grassmoth.compute.context import ModelExecutionContext
from grassmoth.models.ssm.compile.inputs import compile_executable_model
from grassmoth.models.structure import StructuralSelection
from grassmoth_server.compute.compilation_cache import compile_program, prepare_program
from grassmoth_test_support.helpers import observation_dataset

ROOT = Path(__file__).resolve().parents[4]
ORIGIN = datetime(2020, 1, 1, tzinfo=UTC)


def fixture(scale=1.0, offset=0.0):
    model = json.loads((ROOT / "benchmarks/cases/suite/C0/fit.json").read_text())
    model["distributions"]["distribution:tiny-beta"]["params"]["scale"] = scale
    question = QuestionSpec.model_validate(
        {
            "text": "Identify the input effect on y.",
            "outcome": "construct:y",
            "queries": {
                "input contrast": {
                    "start": "2020-01-01",
                    "horizon": "16d",
                    "interventions": [{"target": "construct:u", "value": 1.0}],
                }
            },
        }
    )
    selection = StructuralSelection.for_question(DynamicalModelSpec.model_validate(model), question)
    compiled = compile_executable_model(selection)
    variables = tuple(item.observation for item in compiled.observations)
    rows = [
        {
            "indicator_id": str(variable.id),
            "value": 1.0 if str(variable.id) == "indicator:u" else 0.1 * index + offset,
            "anchor_time": ORIGIN + timedelta(seconds=675 * index),
            "support_start": ORIGIN + timedelta(seconds=675 * index),
            "support_end": ORIGIN + timedelta(seconds=675 * index),
            "support_kind": variable.support_kind.value,
            "summary_operator": variable.summary_operator.value,
            "anchor_policy": variable.anchor_policy.value,
            "observation_window": "675s",
        }
        for variable in variables
        for index in range(3)
    ]
    data = observation_dataset(pl.DataFrame(rows), variables, time_origin=ORIGIN)
    return question, selection, data


def run(phase, output):
    events = Counter()
    traces = set()
    compiled = []
    last_hit = False

    def on_event(event: str, **_metadata: str | int) -> None:
        nonlocal last_hit
        events[event] += 1
        if event == "/jax/compilation_cache/compile_requests_use_cache":
            last_hit = False
        elif event == "/jax/compilation_cache/cache_hits":
            last_hit = True

    def on_duration(event: str, duration_secs: float, **metadata: str | int) -> None:
        if event == "/jax/core/compile/jaxpr_trace_duration":
            traces.add(metadata["fun_name"])
        elif event == "/jax/core/compile/backend_compile_duration":
            compiled.append(
                {"name": metadata["fun_name"], "seconds": duration_secs, "hit": last_hit}
            )

    jax.monitoring.register_event_listener(on_event)
    jax.monitoring.register_event_duration_secs_listener(on_duration)
    results = []
    budget = ComputeBudget(memory=SharedMemoryBudget(available_bytes=256 * 1024**2))
    runtime = FitRuntime(
        source_digest="cache-regression", versions={}, compiler={}, devices=("cpu",)
    )

    def execute(label, template, *, prior=1.0, offset=0.0, seed=19, previous=None):
        question, selection, data = fixture(prior, offset)
        sampler = template.revised(seed=seed)
        events.clear()
        traces.clear()
        compiled.clear()
        started = monotonic()
        result = fit(
            FitInputs(
                selection=selection,
                question=question,
                data_for_model=data,
                sampler=sampler,
                compute_loo_diagnostics=False,
                previous=previous,
            ),
            execution_context=ModelExecutionContext(
                compiler=compile_program,
                prepare_program=prepare_program,
                budget=budget,
                clock=monotonic,
                wall_clock=lambda: datetime.now(UTC),
                logger=logging.getLogger(__name__),
                profile=False,
            ),
            runtime=runtime,
        )
        assert isinstance(result, FitOutput), result
        requests = events["/jax/compilation_cache/compile_requests_use_cache"]
        hits = events["/jax/compilation_cache/cache_hits"]
        assert len(compiled) == requests, (label, compiled, events)
        record = {
            "case": label,
            "compilations": requests - hits,
            "hits": hits,
            "compile_seconds": sum(item["seconds"] for item in compiled if not item["hit"]),
            "seconds": monotonic() - started,
            "programs": list(compiled),
            "traced": sorted(traces),
        }
        results.append(record)
        output.write_text(json.dumps(results, indent=2))
        print(label, record["compilations"], f"{record['seconds']:.3f}s", flush=True)
        return result

    samplers = {
        name: SamplerSpec(
            hardware="local",
            num_chains=2,
            num_warmup=2,
            num_samples_per_chain=16,
            path_update=GuidedPathSpec(method=method, num_particles=4),
            initialization=GaussianInitializationSpec(),
            diagnostic_metrics=("particle_identity", "parameter_movement", "particle_weights"),
        )
        for name, method in (("mgrad", "particle_mgrad"), ("twisted", "twisted_particle_agrad"))
    }
    first = {name: execute(f"{name}/A", sampler) for name, sampler in samplers.items()}
    if phase == "reuse":
        for name, sampler in samplers.items():
            execute(f"{name}/same", sampler)
            if name == "mgrad":
                execute("mgrad/data", sampler, offset=0.05)
                execute("mgrad/seed", sampler, seed=20)
                execute("mgrad/prior", sampler, prior=1.2)
            execute(f"{name}/continue", sampler, previous=first[name])


if __name__ == "__main__":
    run(sys.argv[1], Path(sys.argv[2]))
