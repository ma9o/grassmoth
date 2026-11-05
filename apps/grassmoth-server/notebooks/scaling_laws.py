import marimo

__generated_with = "0.25.0"
app = marimo.App(
    width="full",
    app_title="Scaling laws",
    css_file="support/scaling_laws.css",
)


@app.cell(hide_code=True)
def imports_marimo():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def imports_support():
    from notebooks.support import scaling_laws as sl
    from notebooks.support import scaling_laws_drawing as drawing
    from notebooks.support import scaling_laws_figures as figs

    return drawing, figs, sl


@app.cell(hide_code=True)
def evidence_and_laws(sl):
    evidence = sl.load_evidence()
    laws = sl.fit_laws(evidence)
    return evidence, laws


@app.cell(hide_code=True)
def controls(drawing, evidence, mo, sl):
    def _scrubber(steps, value):
        return mo.ui.slider(steps=steps, value=value, include_input=True, full_width=True)

    gpu_control = mo.ui.radio(options=list(evidence.offered_gpus), value="L40S", inline=True)
    chains_control = _scrubber(sl.scrub_steps(1, 10000), 80)
    draws_control = _scrubber(sl.scrub_steps(1, 1000000), 1000)
    warmup_control = _scrubber(sl.scrub_steps(0, 1000000, 0), 2000)
    time_control = _scrubber(sl.scrub_steps(2, 1000000, 1644, 2049), 3000)
    states_control = _scrubber(sl.scrub_steps(1, 200), 1)
    labels_control = _scrubber(sl.scrub_steps(2, 64), 2)
    particles_control = _scrubber(sl.scrub_steps(2, 8192), 128)
    block_control = _scrubber(sl.scrub_steps(2, 10000), 300)
    windowed_control = mo.ui.checkbox(value=False, label="Window memory")

    def _group(name, *methods):
        dots = "".join(
            f'<span class="sl-side-dot" style="background:{drawing.HUES[method]}"></span>'
            for method in methods
        )
        return f'<div class="sl-side-group">{dots}{name}</div>'

    def _control(name, element):
        return f'<div><div class="sl-side-label">{name}</div>{element}</div>'

    _items = [
        _group("GPU"),
        f"<div>{gpu_control}</div>",
        _group("All methods", *sl.METHODS),
        _control("Chains", chains_control),
        _control("Retained draws", draws_control),
        _control("Warmup sweeps", warmup_control),
        _control("Time points", time_control),
        _control("States", states_control),
        _control("MPGibbs particles", labels_control),
        _group("mGRAD", "mgrad"),
        _control("mGRAD particles", particles_control),
        _group("Blocked Kalman", "kalman_blocks"),
        _control("Block size", block_control),
        f"<div>{windowed_control}</div>",
    ]
    # Displayed here, in the cell that creates them, so moving one never rebuilds the sidebar.
    mo.sidebar(mo.Html(f'<div class="sl-side">{"".join(_items)}</div>'), width="270px")
    return (
        block_control,
        chains_control,
        draws_control,
        gpu_control,
        labels_control,
        particles_control,
        states_control,
        time_control,
        warmup_control,
        windowed_control,
    )


@app.cell(hide_code=True)
def scenario_from_controls(
    block_control,
    chains_control,
    draws_control,
    gpu_control,
    labels_control,
    particles_control,
    sl,
    states_control,
    time_control,
    warmup_control,
    windowed_control,
):
    # One set of settings; each lane swaps in its own method.
    scenario = sl.Scenario(
        gpu=gpu_control.value,
        method=sl.METHODS[0],
        chains=chains_control.value,
        particles=particles_control.value,
        labels=labels_control.value,
        block=block_control.value,
        windowed=windowed_control.value,
        time_points=time_control.value,
        states=states_control.value,
        warmup=warmup_control.value,
        draws=draws_control.value,
    )
    return (scenario,)


@app.cell(hide_code=True)
def first_screen(drawing, evidence, laws, mo, scenario):
    mo.Html(drawing.first_screen(scenario, laws, evidence))
    return


@app.cell(hide_code=True)
def law_grid(drawing, laws, mo, scenario):
    mo.Html(drawing.law_grid(scenario, laws))
    return


@app.cell(hide_code=True)
def law_pictures(drawing, evidence, figs, laws, mo, sl):
    def _head(title, caption):
        return mo.Html(drawing.heading(title, caption))

    mo.ui.tabs(
        {
            "Chains": mo.vstack(
                [
                    _head(
                        "Sweep time is latency plus per-chain work",
                        "Open marks ran outside the fit: other builds, or the same source on other "
                        "allocations.",
                    ),
                    figs.sweep_time(evidence, laws),
                    _head(
                        "Processing fill is the distance to the throughput limit",
                        "Lines are the law, marks are measured. <code>nvidia-smi</code> busy stays "
                        "flat while throughput climbs, so it can't measure this.",
                    ),
                    figs.fill_against_busy(evidence, laws),
                ]
            ),
            "mGRAD memory": mo.vstack(
                [
                    _head(
                        "mGRAD's memory is its particle-pair table",
                        "The only fitted constant covers the particle banks, a few percent of the "
                        "total.",
                    ),
                    figs.pair_table(evidence, laws),
                ]
            ),
            "Kalman memory": mo.vstack(
                [
                    _head(
                        "Kalman barely uses GPU memory until sampling ends",
                        "Long fits peak after sampling, at about three copies of the retained "
                        "paths. Circle L40S, square H200, diamond L4.",
                    ),
                    figs.kalman_memory(evidence, laws),
                ]
            ),
            "mGRAD particles": mo.vstack(
                [
                    _head(
                        "Particles cost memory first and time second",
                        f"Lines are the law. The time law runs "
                        f"{sl.particle_time_error_range(laws, evidence)} against these runs.",
                    ),
                    figs.particles(evidence, laws),
                ]
            ),
            "Block size": mo.vstack(
                [
                    _head(
                        "Block size has a floor, and the whole path costs about 30% more",
                        "Only France has small blocks measured. Its window memory forces extra "
                        "block groups.",
                    ),
                    figs.block_size(evidence),
                ]
            ),
            "States": mo.vstack(
                [
                    _head(
                        "State dimension rests on one pair of models",
                        "SV to Lorenz-96 across builds. TAIL and France show how far the model "
                        "alone moves it.",
                    ),
                    figs.states(evidence, laws),
                ]
            ),
            "GPUs": mo.vstack(
                [
                    _head(
                        "The GPU changes speed, and spec sheets get most of the way",
                        "Fitted after seeing these runs, on three GPUs. A10G and B200 have no runs "
                        "yet.",
                    ),
                    figs.cross_gpu(evidence, laws),
                ]
            ),
        }
    )
    return


@app.cell(hide_code=True)
def reference(drawing, evidence, laws, mo, sl):
    def _table(rows):
        return mo.ui.table(
            rows,
            selection=None,
            pagination=False,
            show_column_summaries=False,
            show_data_types=False,
        )

    mo.vstack(
        [
            mo.Html(drawing.heading("Reference", "Constants, checks and the raw records.")),
            mo.accordion(
                {
                    "Laws as formulas, with fitted constants": _table(sl.formula_rows(laws)),
                    "Every law against every run": _table(sl.check_rows(laws, evidence)),
                    "GPUs": _table(sl.gpu_rows(evidence)),
                    "Runs": _table(sl.run_rows(evidence)),
                    "Runs that would close the gaps": mo.md(r"""
    A few dollars in short fits, roughly, with host RSS added to the telemetry.

    | Gap | Runs |
    |:---|:---|
    | A10G and B200 have no runs | Aux Kalman and mGRAD on each, at two chain counts at least 4× apart |
    | Latency against per-chain cost off the L40S | Aux Kalman on H200, at 20 and 320 chains |
    | mGRAD's curve off the H200 | mGRAD on L40S, at 10 and 20 chains |
    | State dimension | Lorenz-96 at D=10 and D=20, each at two chain counts |
    | Time points alone | One model at a second T |
    | MPGibbs particles | Any model at K=4 |
    | Host RAM and CPU while compiling | Any of the above, once the telemetry records them |
    """),
                    "Adding a run": mo.md(r"""
    Append a row to `notebooks/support/scaling_laws_runs.json`, then rerun the notebook.
    A `ladder` named `sv-<method>-<gpu>`, such as `sv-kalman-whole-h200`, becomes that GPU's measured
    curve in the calculator. Rows without a ladder still show up in the checks.

    `scratchpad/gpu-fill-20261104/extract_runs.py` built this table and overwrites it when rerun, so
    extend that script or edit the file by hand, not both.
    """),
                },
                multiple=True,
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
