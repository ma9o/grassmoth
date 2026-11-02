# Experiment and benchmark guidance

- Read the [scientific library architecture](../packages/grassmoth/AGENTS.md) before experiments, comparisons or scientific notebook work.

- Experiments and comparisons use the server's public study actions over HTTP, following the [benchmark workflow](README.md#running) and [study API reference](../docs/reference/study-api.md). Develop missing capabilities as the smallest coherent patch to their framework owners, applied in an isolated checkout with its own server and study; retain the patch and experiment declaration in `scratchpad/`. Exercise the patched framework through the same actions. Do not recreate model preparation, initialization, sampling orchestration, diagnostics or convergence gates in experiment harnesses.

- External callers use the [public study library](../docs/guides/public_study_library.md). Check each scratchpad experiment or notebook with `bun run --cwd apps/grassmoth-server lint:public <caller-path>`; paths are relative to `apps/grassmoth-server`.

- Budget GPU benchmarks carefully: a B200 on Modal costs $6/hour.
- Analyze saved fit results through `grassmoth-server analyze --workspace <study> --call <id> -- script.py`; keep results on its Modal Volume and never put results into images.
- Run independent GPU benchmark scenarios concurrently on separate GPUs. Do not serialize them on one L40S; isolate each scenario's GPU so timing comparisons remain meaningful, and preserve shared initialization dependencies.
