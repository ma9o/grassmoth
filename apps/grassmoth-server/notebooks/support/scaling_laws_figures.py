"""Plotly figures for the notebook's law pictures: saved runs as marks, laws as lines.

Method is the only categorical colour: mGRAD blue, whole-path Kalman orange, blocked Kalman aqua.
Everything else is ink or grey, told apart by symbol, line style and direct labels.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from notebooks.support import scaling_laws as sl

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from notebooks.support.scaling_laws import Curve, Evidence, Laws, Run

C_MGRAD = "#2a78d6"
C_WHOLE = "#eb6834"
C_BLOCKS = "#1baf7a"
C_INK = "#17212b"
C_INK_2 = "#5a6673"
C_MUTED = "#8b96a1"
C_GRID = "#e4e8ec"
FONT = "Archivo, system-ui, sans-serif"

METHOD_COLOURS = {"mgrad": C_MGRAD, "kalman_whole": C_WHOLE, "kalman_blocks": C_BLOCKS}
GPU_SYMBOLS = {"L40S": "circle", "H200": "square", "L4": "diamond"}


def _tick_text(tick: float) -> str:
    """Short labels: 300, 1,000, 100k, 10M."""
    if tick >= 1e6:
        return f"{tick / 1e6:g}M"
    if tick >= 1e5:
        return f"{tick / 1e3:g}k"
    return f"{tick:,g}"


def _log_axis(ticks: Iterable[float], title: str = "") -> dict[str, object]:
    """A log axis labelled at exactly these values."""
    values = [float(tick) for tick in ticks]
    return {
        "type": "log",
        "tickmode": "array",
        "tickvals": values,
        "ticktext": [_tick_text(tick) for tick in values],
        "title": {"text": title},
    }


def _finish(fig: go.Figure, *, height: int = 420) -> go.Figure:
    """No panel behind the plot, recessive axes, and the legend above it."""
    fig.update_layout(
        height=height,
        margin={"l": 64, "r": 24, "t": 64, "b": 54},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": FONT, "color": C_INK_2, "size": 12.5},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.04,
            "xanchor": "left",
            "x": 0,
            "font": {"family": FONT, "size": 12.5, "color": C_INK_2},
        },
        hoverlabel={"bgcolor": "#ffffff", "font": {"family": FONT, "color": C_INK, "size": 12.5}},
    )
    fig.update_xaxes(gridcolor=C_GRID, zeroline=False, linecolor=C_GRID, ticks="", automargin=True)
    fig.update_yaxes(gridcolor=C_GRID, zeroline=False, linecolor=C_GRID, ticks="", automargin=True)
    return fig


def _finish_panels(
    fig: go.Figure, *, height: int, legend_y: float = 1.16, top: int = 104
) -> go.Figure:
    """Small multiples: subplot titles under a legend that sits clear of them."""
    fig.update_annotations(font={"family": FONT, "size": 13, "color": C_INK})
    fig = _finish(fig, height=height)
    fig.update_layout(margin={"t": top}, legend={"y": legend_y})
    return fig


def _marker(colour: str, symbol: str = "circle", *, hollow: bool = False) -> dict[str, object]:
    """A 10 px mark with a thin ring, or an open one for runs outside the fit."""
    return {
        "size": 10,
        "symbol": f"{symbol}-open" if hollow else symbol,
        "color": colour,
        "line": {"color": colour if hollow else "#ffffff", "width": 2 if hollow else 1},
    }


def _line(colour: str, *, dash: str = "solid", width: float = 2) -> dict[str, object]:
    return {"color": colour, "width": width, "dash": dash}


def _hover(run: Run, *extra: str) -> str:
    """Name a run fully: its setting, its evidence and where it came from."""
    parts = [
        f"<b>{run.id}</b>",
        f"{run.model}, T={run.T:,}, D={run.D}, {run.gpu}, {run.C} chains"
        + (f", N={run.N}" if run.N else "")
        + (f", L={run.block}" if run.block else ""),
        *extra,
    ]
    if run.note:
        parts.append(run.note)
    return "<br>".join(parts) + "<extra></extra>"


def pair_table(evidence: Evidence, laws: Laws) -> go.Figure:
    """Predicted against measured mGRAD memory: every run should sit on the diagonal."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[1, 1000],
            y=[1, 1000],
            mode="lines",
            line=_line(C_MUTED, dash="dot", width=1.5),
            name="Law = measured",
            hoverinfo="skip",
        )
    )

    def points(runs: Sequence[Run]) -> tuple[list[float], list[float], list[str]]:
        predicted, observed, hovers = [], [], []
        for run in runs:
            scenario = sl.Scenario(run.gpu, "mgrad", run.C, particles=run.N or sl.REFERENCE_N)
            value = sl.sweep_bytes_per_chain(scenario, laws) * run.C / sl.GIB
            measured = run.sweep_memory_gib or 0.0
            predicted.append(value)
            observed.append(measured)
            hovers.append(
                _hover(
                    run,
                    f"Law {value:,.2f} GiB, measured {measured:,.2f} GiB, "
                    f"{100 * (value / measured - 1):+.1f}%",
                )
            )
        return predicted, observed, hovers

    runs = [
        run
        for run in evidence.runs
        if run.model == "SV" and run.variant == "mgrad" and run.sweep_memory_gib is not None
    ]
    chain_runs = [run for run in runs if run.N == sl.REFERENCE_N]
    particle_runs = [run for run in runs if run.N != sl.REFERENCE_N]
    x, y, hover = points(chain_runs)
    labelled = {
        4: ("4 chains", "bottom right"),
        40: ("40 chains", "top left"),
        80: ("80 chains", "top left"),
    }
    seen: set[int] = set()
    text, positions = [], []
    for run in chain_runs:
        label, position = labelled.get(run.C, ("", "top left"))
        text.append("" if run.C in seen else label)
        positions.append(position)
        seen.add(run.C)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers+text",
            text=text,
            textposition=positions,
            textfont={"color": C_INK_2},
            marker=_marker(C_MGRAD),
            name="Chains vary, N=128",
            hovertemplate=hover,
        )
    )
    x, y, hover = points(particle_runs)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers+text",
            text=[f"N={run.N}, {run.C} chains" for run in particle_runs],
            textposition="bottom right",
            textfont={"color": C_INK_2},
            marker=_marker(C_MGRAD, "diamond"),
            name="Particles vary",
            hovertemplate=hover,
        )
    )
    failed = next(run for run in evidence.runs if run.failed_alloc_gib is not None)
    predicted = sl.pair_bytes(failed.C, failed.K or 2, failed.T, failed.N or 0, failed.D) / sl.GIB
    fig.add_trace(
        go.Scatter(
            x=[predicted],
            y=[failed.failed_alloc_gib],
            mode="markers+text",
            text=["Lorenz-96 at 40 states: the allocation that failed"],
            textposition="middle left",
            textfont={"color": C_INK_2},
            marker=_marker(C_MGRAD, "star", hollow=True),
            name="T×D check",
            hovertemplate=_hover(
                failed,
                f"Law {predicted:,.2f} GiB, allocation {failed.failed_alloc_gib:,.2f} GiB",
            ),
        )
    )
    ticks = (1, 3, 10, 30, 100, 300, 1000)
    fig.update_xaxes(_log_axis(ticks, "Law: C·K·T·N²·D·4 B plus particle banks, GiB"))
    fig.update_yaxes(_log_axis(ticks, "Measured, GiB"))
    return _finish(fig, height=440)


def kalman_memory(evidence: Evidence, laws: Laws) -> go.Figure:
    """Left, Kalman while sampling; right, what every fit holds on the GPU after sampling."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("While sampling: MiB per chain", "After sampling: GiB on the GPU"),
        horizontal_spacing=0.1,
    )
    runs = [
        run
        for run in evidence.runs
        if run.model == "SV" and run.method == "kalman" and run.mem_sampling_gib is not None
    ]
    names = {"kalman_whole": "Whole path", "kalman_blocks": "Blocks"}
    for variant, name in names.items():
        for gpu, symbol in GPU_SYMBOLS.items():
            members = sorted(
                (run for run in runs if run.variant == variant and run.gpu == gpu),
                key=lambda run: run.C,
            )
            if not members:
                continue
            fig.add_trace(
                go.Scatter(
                    x=[run.C for run in members],
                    y=[(run.mem_sampling_gib or 0.0) * 1024 / run.C for run in members],
                    mode="markers",
                    marker=_marker(METHOD_COLOURS[variant], symbol),
                    name=name,
                    showlegend=gpu == "L40S",
                    hovertemplate=[
                        _hover(
                            run,
                            f"{(run.mem_sampling_gib or 0.0) * 1024 / run.C:.1f} MiB per chain",
                        )
                        for run in members
                    ],
                ),
                row=1,
                col=1,
            )
    law = laws.kalman_bytes * sl.REFERENCE_T / sl.MIB
    fig.add_trace(
        go.Scatter(
            x=[30, 400],
            y=[law, law],
            mode="lines",
            line=_line(C_MUTED, dash="dot", width=1.5),
            name=f"Law, {law:.1f} MiB",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.update_xaxes(_log_axis((40, 80, 160, 320), "Chains"), row=1, col=1)
    fig.update_yaxes(range=[0, 13], row=1, col=1)

    post = [
        run for run in evidence.runs if run.mem_conditioning_gib is not None and run.S is not None
    ]
    coordinates = [run.C * (run.S or 0) * run.T / 1e9 for run in post]
    fig.add_trace(
        go.Scatter(
            x=coordinates,
            y=[run.mem_conditioning_gib for run in post],
            mode="markers",
            marker=_marker(C_INK, "circle"),
            name="Conditioning",
            hovertemplate=[
                _hover(run, f"Conditioning {run.mem_conditioning_gib:.2f} GiB") for run in post
            ],
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=coordinates,
            y=[run.mem_diagnostics_gib for run in post],
            mode="markers",
            marker=_marker(C_MUTED, "square"),
            name="Diagnostics",
            hovertemplate=[
                _hover(run, f"Diagnostics {run.mem_diagnostics_gib:.2f} GiB") for run in post
            ],
        ),
        row=1,
        col=2,
    )
    grid = [0.1 * 1.25**step for step in range(21)]
    fig.add_trace(
        go.Scatter(
            x=grid,
            y=[laws.conditioning_units * value * 1e9 * sl.FLOAT32_BYTES / sl.GIB for value in grid],
            mode="lines",
            line=_line(C_INK),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=grid,
            y=[
                laws.diagnostics_offset_gib
                + laws.diagnostics_units * value * 1e9 * sl.FLOAT32_BYTES / sl.GIB
                for value in grid
            ],
            mode="lines",
            line=_line(C_MUTED, dash="dash"),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )
    pool = evidence.gpu("L40S").pool_gib
    fig.add_trace(
        go.Scatter(
            x=[grid[0], grid[-1]],
            y=[pool, pool],
            mode="lines+text",
            line=_line(C_MUTED, dash="dot", width=1.5),
            text=["L40S pool", ""],
            textposition="top right",
            textfont={"color": C_INK_2},
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )
    full = evidence.run("profile/kalman-blocked-l40s-full")
    fig.add_annotation(
        x=math.log10(full.C * (full.S or 0) * full.T / 1e9),
        y=math.log10(full.mem_conditioning_gib or 1.0),
        text="80 chains × 10,000 draws",
        showarrow=True,
        arrowcolor=C_MUTED,
        ax=-92,
        ay=-8,
        font={"family": FONT, "color": C_INK_2},
        row=1,
        col=2,
    )
    fig.update_xaxes(
        _log_axis((0.1, 0.3, 1, 3, 10), "Chains × draws × time points, billions"), row=1, col=2
    )
    fig.update_yaxes(_log_axis((1, 3, 10, 30, 100)), row=1, col=2)
    return _finish_panels(fig, height=450, legend_y=1.14)


def _subtitle(*lines: str) -> str:
    """Formula lines under a panel title, at a size that can be read."""
    return "".join(
        f'<br><span style="font-size:12px;color:{C_INK_2}">{line}</span>' for line in lines
    )


def _fit_label(curve: Curve) -> str:
    return _subtitle(
        f"a = {curve.a_ms:.1f} ms, b = {curve.b_ms:.3f} ms per chain",
        f"50% at {curve.half_chains:.0f} chains",
    )


def sweep_time(evidence: Evidence, laws: Laws) -> go.Figure:
    """Sweep time against chains for each measured ladder, with runs outside the fit left open."""
    panels = (
        ("kalman_whole", "sv-kalman-whole-l40s", "L40S", "Aux Kalman, whole path, L40S", 340),
        ("kalman_blocks", "sv-kalman-blocks-l40s", "L40S", "Aux Kalman, blocks, L40S", 340),
        ("mgrad", "sv-mgrad-h200", "H200", "mGRAD, N=128, H200", 90),
    )
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            f"{title}{_fit_label(laws.ladders[ladder])}" for _, ladder, _, title, _ in panels
        ],
        horizontal_spacing=0.07,
    )
    for name, hollow in (("In the fit", False), ("Outside the fit", True)):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=_marker(C_INK_2, hollow=hollow),
                name=name,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
    for column, (variant, ladder, gpu, _, top) in enumerate(panels, start=1):
        curve = laws.ladders[ladder]
        colour = METHOD_COLOURS[variant]
        fig.add_trace(
            go.Scatter(
                x=[0, top],
                y=[curve.time_ms(0), curve.time_ms(top)],
                mode="lines",
                line=_line(colour),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=column,
        )
        members = [
            run
            for run in evidence.runs
            if run.model == "SV"
            and run.variant == variant
            and run.gpu == gpu
            and run.sampling_ms is not None
            and run.N in (None, sl.REFERENCE_N)
        ]
        for hollow in (False, True):
            subset = [run for run in members if (run.ladder is None) == hollow]
            fig.add_trace(
                go.Scatter(
                    x=[run.C for run in subset],
                    y=[run.sampling_ms for run in subset],
                    mode="markers",
                    marker=_marker(colour, hollow=hollow),
                    showlegend=False,
                    hovertemplate=[
                        _hover(
                            run,
                            f"Law {curve.time_ms(run.C):.1f} ms, measured {run.sampling_ms:.1f} ms",
                        )
                        for run in subset
                    ],
                ),
                row=1,
                col=column,
            )
        fig.update_xaxes(title_text="Chains", range=[0, top], row=1, col=column)
        fig.update_yaxes(rangemode="tozero", row=1, col=column)
    fig.update_yaxes(title_text="Milliseconds per sweep", row=1, col=1)
    return _finish_panels(fig, height=450, legend_y=1.26, top=128)


def fill_against_busy(evidence: Evidence, laws: Laws) -> go.Figure:
    """The throughput-limit definition of fill beside the utilization nvidia-smi reports."""
    fig = go.Figure()
    grid = [10 ** (step / 20) for step in range(61)]
    series = (
        ("sv-kalman-whole-l40s", "kalman_whole", "L40S", "Aux Kalman on L40S"),
        ("sv-mgrad-h200", "mgrad", "H200", "mGRAD on H200"),
    )
    for ladder, variant, gpu, name in series:
        curve = laws.ladders[ladder]
        colour = METHOD_COLOURS[variant]
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=[100 * curve.fill(value) for value in grid],
                mode="lines",
                line=_line(colour),
                name=name,
                legendgroup=name,
                hoverinfo="skip",
            )
        )
        members = sorted(
            (
                run
                for run in evidence.runs
                if run.model == "SV"
                and run.variant == variant
                and run.gpu == gpu
                and run.sampling_ms is not None
                and run.N in (None, sl.REFERENCE_N)
            ),
            key=lambda run: run.C,
        )
        fig.add_trace(
            go.Scatter(
                x=[run.C for run in members],
                y=[100 * curve.b_ms * run.C / (run.sampling_ms or 1.0) for run in members],
                mode="markers",
                marker=_marker(colour),
                name=name,
                legendgroup=name,
                showlegend=False,
                hovertemplate=[
                    _hover(
                        run,
                        f"{1000 * run.C / (run.sampling_ms or 1.0):,.0f} chain-sweeps per second, "
                        f"limit {curve.ceiling_per_second:,.0f}",
                    )
                    for run in members
                ],
            )
        )
    busy = sorted(
        (
            run
            for run in evidence.runs
            if run.ladder == "sv-kalman-whole-l40s" and run.busy_pct is not None
        ),
        key=lambda run: run.C,
    )
    fig.add_trace(
        go.Scatter(
            x=[run.C for run in busy],
            y=[run.busy_pct for run in busy],
            mode="markers+lines",
            line=_line(C_MUTED, dash="dot", width=1.5),
            marker=_marker(C_MUTED, "square"),
            name="nvidia-smi busy, aux Kalman on L40S",
            hovertemplate=[_hover(run, f"Busy {run.busy_pct:.0f}%") for run in busy],
        )
    )
    fig.update_xaxes(_log_axis((1, 3, 10, 30, 100, 300, 1000), "Chains"))
    fig.update_yaxes(title_text="Percent", range=[0, 105])
    return _finish(fig, height=400)


def particles(evidence: Evidence, laws: Laws) -> go.Figure:
    """mGRAD particles: memory follows N², time barely moves until chains multiply it."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Memory: GiB", "Time: milliseconds per sweep"),
        horizontal_spacing=0.1,
    )
    curve = laws.ladders["sv-mgrad-h200"]
    runs = [
        run
        for run in evidence.runs
        if run.model == "SV" and run.variant == "mgrad" and run.gpu == "H200" and run.N is not None
    ]
    grid = [64 * 2 ** (step / 4) for step in range(17)]
    for chains, symbol, dash in ((4, "circle", "dot"), (80, "square", "solid")):
        members = sorted((run for run in runs if chains == run.C), key=lambda run: run.N or 0)
        memory = [run for run in members if run.sweep_memory_gib is not None]
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=[
                    sl.sweep_bytes_per_chain(
                        sl.Scenario("H200", "mgrad", chains, particles=round(value)), laws
                    )
                    * chains
                    / sl.GIB
                    for value in grid
                ],
                mode="lines",
                line=_line(C_MGRAD, dash=dash),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[run.N for run in memory],
                y=[run.sweep_memory_gib for run in memory],
                mode="markers",
                marker=_marker(C_MGRAD, symbol),
                name=f"{chains} chains",
                hovertemplate=[_hover(run, f"{run.sweep_memory_gib:.1f} GiB") for run in memory],
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=[
                    curve.a_ms + chains * curve.b_ms * (value / sl.REFERENCE_N) ** 2
                    for value in grid
                ],
                mode="lines",
                line=_line(C_MGRAD, dash=dash),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=2,
        )
        timed = [run for run in members if run.sampling_ms is not None]
        fig.add_trace(
            go.Scatter(
                x=[run.N for run in timed],
                y=[run.sampling_ms for run in timed],
                mode="markers",
                marker=_marker(C_MGRAD, symbol),
                showlegend=False,
                hovertemplate=[
                    _hover(
                        run,
                        "Law "
                        f"{curve.a_ms + run.C * curve.b_ms * ((run.N or 0) / sl.REFERENCE_N) ** 2:.0f}"
                        f" ms, measured {run.sampling_ms:.0f} ms",
                    )
                    for run in timed
                ],
            ),
            row=1,
            col=2,
        )
    ticks = (64, 128, 256, 512, 1024)
    fig.update_xaxes(_log_axis(ticks, "mGRAD particles N"), row=1, col=1)
    fig.update_xaxes(_log_axis(ticks, "mGRAD particles N"), row=1, col=2)
    fig.update_yaxes(_log_axis((0.3, 1, 3, 10, 30, 100, 300)), row=1, col=1)
    fig.update_yaxes(range=[0, 240], row=1, col=2)
    return _finish_panels(fig, height=430, legend_y=1.14)


def block_size(evidence: Evidence) -> go.Figure:
    """Sweep time against block size: France with window memory, SV without."""
    categories = ["10", "25", "50", "100", "200", "300", "whole"]
    fig = go.Figure()
    france = {
        (str(run.block) if run.block else "whole"): run
        for run in evidence.runs
        if run.id.startswith("france/block-")
    }
    ordered = [france[name] for name in categories]
    fig.add_trace(
        go.Scatter(
            x=categories,
            y=[run.sampling_ms for run in ordered],
            mode="lines+markers",
            line=_line(C_BLOCKS),
            marker=_marker(C_BLOCKS),
            name="France, window memory, 40 chains",
            hovertemplate=[
                _hover(run, f"{run.sampling_ms:.1f} ms per sweep, compile {run.compile_s:.0f} s")
                for run in ordered
            ],
        )
    )
    sv = (evidence.run("profile/kalman-blocked-l40s"), evidence.run("profile/kalman-whole-l40s"))
    fig.add_trace(
        go.Scatter(
            x=["300", "whole"],
            y=[run.sampling_ms for run in sv],
            mode="lines+markers",
            line=_line(C_BLOCKS, dash="dot"),
            marker=_marker(C_BLOCKS, "square"),
            name="SV, Markov, 80 chains",
            hovertemplate=[
                _hover(
                    run,
                    f"{run.sampling_ms:.1f} ms per sweep, {run.mem_sampling_gib:.2f} GiB, "
                    f"compile {run.compile_s:.0f} s",
                )
                for run in sv
            ],
        )
    )
    fig.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=categories,
        title_text="Block size L, in time points",
    )
    fig.update_yaxes(title_text="Milliseconds per sweep", rangemode="tozero")
    return _finish(fig, height=400)


def states(evidence: Evidence, laws: Laws) -> go.Figure:
    """What's known about state dimension: mostly one pair of models, drawn as exactly that."""
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "Kalman flops per chain and time point"
            + _subtitle(f"{laws.flops_constant:,.0f} + {laws.flops_cubic:.0f}·D³, two points"),
            "Kalman per-chain µs per time point"
            + _subtitle(f"whole path about D^{laws.kalman_d_exponent:.1f}, SV to Lorenz-96"),
            "mGRAD per-chain ns per particle pair"
            + _subtitle(f"about D^{laws.mgrad_d_exponent:.1f}, SV to Lorenz-96"),
        ),
        horizontal_spacing=0.09,
    )
    grid = [10 ** (step * math.log10(60) / 30) for step in range(31)]
    whole = (evidence.run("profile/kalman-whole"), evidence.run("lorenz96/kalman-whole"))
    fig.add_trace(
        go.Scatter(
            x=grid,
            y=[laws.flops_constant + laws.flops_cubic * value**3 for value in grid],
            mode="lines",
            line=_line(C_WHOLE, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[run.D for run in whole],
            y=[sl.flops_per_chain_time(run) for run in whole],
            mode="markers+text",
            text=[run.model for run in whole],
            textposition=["top right", "bottom left"],
            textfont={"color": C_INK_2},
            marker=_marker(C_WHOLE),
            name="Whole path",
            hovertemplate=[
                _hover(run, f"{sl.flops_per_chain_time(run):,.0f} flops per chain and time point")
                for run in whole
            ],
        ),
        row=1,
        col=1,
    )
    points = sl.kalman_points(evidence, laws)
    for variant, colour, symbol, name in (
        ("kalman_whole", C_WHOLE, "circle", "Whole path"),
        ("kalman_blocks", C_BLOCKS, "square", "Blocks"),
    ):
        members = [(run, value) for run, value in points if run.variant == variant]
        fig.add_trace(
            go.Scatter(
                x=[run.D for run, _ in members],
                y=[value for _, value in members],
                mode="markers+text",
                text=[run.model if variant == "kalman_blocks" else "" for run, _ in members],
                textposition=[
                    {"France": "top right", "TAIL": "bottom right", "Lorenz-96": "bottom left"}.get(
                        run.model, "middle right"
                    )
                    for run, _ in members
                ],
                textfont={"color": C_INK_2},
                marker=_marker(colour, symbol),
                name=name,
                showlegend=variant == "kalman_blocks",
                hovertemplate=[_hover(run, f"{value:.3f} µs") for run, value in members],
            ),
            row=1,
            col=2,
        )
    first = next(value for run, value in points if run.id == "profile/kalman-whole")
    last = next(value for run, value in points if run.id == "lorenz96/kalman-whole")
    fig.add_trace(
        go.Scatter(
            x=[1, 40],
            y=[first, last],
            mode="lines",
            line=_line(C_WHOLE, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=1,
        col=2,
    )
    mgrad = sl.mgrad_points(evidence, laws)
    fig.add_trace(
        go.Scatter(
            x=[dimension for _, dimension, _ in mgrad],
            y=[value for _, _, value in mgrad],
            mode="lines+markers+text",
            text=[name for name, _, _ in mgrad],
            textposition=["top right", "bottom left"],
            textfont={"color": C_INK_2},
            line=_line(C_MGRAD, dash="dot"),
            marker=_marker(C_MGRAD),
            name="mGRAD",
            hovertemplate=[
                f"<b>{name}</b><br>D={dimension:.0f}<br>{value:.5f} ns per pair<extra></extra>"
                for name, dimension, value in mgrad
            ],
        ),
        row=1,
        col=3,
    )
    ticks = (1, 2, 5, 10, 20, 40)
    for column in (1, 2, 3):
        fig.update_xaxes(_log_axis(ticks, "States D"), row=1, col=column)
    fig.update_yaxes(_log_axis((1e3, 1e4, 1e5, 1e6, 1e7)), row=1, col=1)
    fig.update_yaxes(_log_axis((0.01, 0.1, 1, 10, 100)), row=1, col=2)
    fig.update_yaxes(_log_axis((1e-3, 1e-2, 1e-1, 1)), row=1, col=3)
    return _finish_panels(fig, height=440, legend_y=1.2, top=112)


def cross_gpu(evidence: Evidence, laws: Laws) -> go.Figure:
    """Predicting another GPU, with and without its spec sheet, as a ratio to what it measured."""
    checks = sl.cross_gpu_checks(laws, evidence)[::-1]
    labels = [check.label for check in checks]
    fig = go.Figure()
    fig.add_shape(
        type="line",
        x0=1,
        x1=1,
        y0=0,
        y1=1,
        yref="paper",
        line=_line(C_MUTED, dash="dot", width=1.5),
    )
    for check in checks:
        fig.add_trace(
            go.Scatter(
                x=[check.blind_ms / check.observed_ms, check.spec_ms / check.observed_ms],
                y=[check.label, check.label],
                mode="lines",
                line=_line(C_GRID, width=2),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[check.blind_ms / check.observed_ms for check in checks],
            y=labels,
            mode="markers+text",
            text=[f"{100 * (check.blind_ms / check.observed_ms - 1):+.0f}%" for check in checks],
            textposition="top center",
            textfont={"color": C_INK_2},
            marker=_marker(C_MUTED, "circle", hollow=True),
            name="Ignoring the GPU",
            hovertemplate=[
                _hover(
                    check.run,
                    f"{check.source_gpu}'s curve unchanged: {check.blind_ms:.1f} ms, "
                    f"measured {check.observed_ms:.1f} ms",
                )
                for check in checks
            ],
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[check.spec_ms / check.observed_ms for check in checks],
            y=labels,
            mode="markers+text",
            text=[f"{100 * (check.spec_ms / check.observed_ms - 1):+.0f}%" for check in checks],
            textposition="top right",
            textfont={"color": C_INK},
            marker=_marker(C_INK, "diamond"),
            name="Spec-sheet scaling",
            hovertemplate=[
                _hover(
                    check.run,
                    f"Scaled from {check.source_gpu}: {check.spec_ms:.1f} ms, "
                    f"measured {check.observed_ms:.1f} ms",
                )
                for check in checks
            ],
        )
    )
    fig.update_xaxes(title_text="Predicted ÷ measured sweep time; 1 is exact", range=[0.35, 1.45])
    fig.update_yaxes(categoryorder="array", categoryarray=labels, range=[-0.5, len(labels) - 0.3])
    return _finish(fig, height=360)
