"""Plotly figures for the stochastic volatility methods walkthrough.

Every function takes plain arrays and small records and returns a figure, so the notebook
decides what to compute. Colour says which idea a mark belongs to, and it stays the same
across the notebook: particles blue, Gaussian moves violet, physics orange, conditionals aqua.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

# ── Palette: validated categorical slots (blue, orange, aqua, violet) on a light surface ──
C_PARTICLE = "#2a78d6"
C_PHYSICS = "#eb6834"
C_CONDITIONAL = "#1baf7a"
C_KALMAN = "#6250d6"
C_INK = "#0b0b0b"
C_INK_2 = "#52514e"
C_MUTED = "#9aa0a6"
C_GRID = "#e6e5e1"
C_LINES = "#d3d2cc"
C_SURFACE = "#fcfcfb"
C_MARGINAL = "rgba(154, 160, 166, 0.45)"
C_BAND = "rgba(154, 160, 166, 0.28)"


def _log_ticks(values: Iterable[float]) -> dict[str, Any]:
    """A log axis labelled at exactly these values: Plotly's default minor labels read as bare digits."""
    ticks = [float(value) for value in values]
    return {
        "type": "log",
        "tickmode": "array",
        "tickvals": ticks,
        "ticktext": [f"{tick:g}" for tick in ticks],
    }


def _layout(
    fig: go.Figure, *, title: str, height: int = 420, legend: bool = True, legend_y: float = -0.16
) -> go.Figure:
    fig.update_layout(
        title={"text": title, "font": {"size": 15, "color": C_INK}, "x": 0.0, "xanchor": "left"},
        height=height,
        paper_bgcolor=C_SURFACE,
        plot_bgcolor=C_SURFACE,
        font={"color": C_INK_2, "size": 12},
        margin={"l": 70, "r": 30, "t": 70, "b": 60},
        legend={"orientation": "h", "y": legend_y, "x": 0.0, "font": {"size": 11}},
        showlegend=legend,
        hovermode="closest",
    )
    fig.update_annotations(font={"size": 12, "color": C_INK_2})
    fig.update_xaxes(
        showgrid=True, gridcolor=C_GRID, gridwidth=1, zeroline=False, linecolor=C_LINES
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=C_GRID, gridwidth=1, zeroline=False, linecolor=C_LINES
    )
    return fig


def _line(
    x: np.ndarray,
    y: np.ndarray,
    *,
    name: str,
    color: str = C_INK_2,
    width: float = 1.5,
    dash: str = "solid",
    legend: bool = True,
    opacity: float = 1.0,
) -> go.Scatter:
    return go.Scatter(
        x=x,
        y=y,
        mode="lines",
        name=name,
        line={"color": color, "width": width, "dash": dash},
        showlegend=legend,
        opacity=opacity,
        hovertemplate=f"{name}<br>%{{x:.3~f}}, %{{y:.3~f}}<extra></extra>",
    )


def _band(x: np.ndarray, low: np.ndarray, high: np.ndarray, *, name: str, color: str) -> go.Scatter:
    return go.Scatter(
        x=np.concatenate((x, x[::-1])),
        y=np.concatenate((high, low[::-1])),
        fill="toself",
        fillcolor=color,
        line={"width": 0},
        name=name,
        hoverinfo="skip",
    )


def _marginal_bars(values: np.ndarray, *, name: str, legend: bool = True, bins: int = 40) -> go.Bar:
    density, edges = np.histogram(values, bins=bins, density=True)
    return go.Bar(
        x=(edges[:-1] + edges[1:]) / 2,
        y=density,
        width=np.diff(edges),
        marker={"color": C_MARGINAL, "line": {"width": 0}},
        name=name,
        showlegend=legend,
        hovertemplate=f"{name}<br>%{{x:.3~f}}: %{{y:.3~f}}<extra></extra>",
    )


def fig_history(days: np.ndarray, returns: np.ndarray, log_variance: np.ndarray) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "What we observe: daily returns",
            "What we want: the hidden log-variance, level + path",
        ),
    )
    fig.add_trace(_line(days, returns, name="Return", width=0.8, legend=False), row=1, col=1)
    fig.add_trace(
        _line(days, log_variance, name="True level + path", color=C_INK, width=1.5, legend=False),
        row=2,
        col=1,
    )
    fig.update_xaxes(title_text="Day", row=2, col=1)
    fig.update_yaxes(title_text="Return", row=1, col=1)
    fig.update_yaxes(title_text="Log-variance", row=2, col=1)
    return _layout(fig, title="One simulated history", height=430, legend=False)


def fig_posterior_paths(
    days: np.ndarray, truth: np.ndarray, draws: np.ndarray, *, examples: int = 3
) -> go.Figure:
    """Draws has shape (D, T): posterior draws of the same physical quantity as truth."""
    low, high = np.quantile(draws, [0.05, 0.95], axis=0)
    fig = go.Figure()
    fig.add_trace(_band(days, low, high, name="90% posterior band", color=C_BAND))
    picks = np.linspace(0, draws.shape[0] - 1, examples).astype(int)
    for rank, pick in enumerate(picks):
        fig.add_trace(
            _line(
                days,
                draws[pick],
                name="Posterior draws of the whole path",
                color=C_PARTICLE,
                width=0.8,
                legend=rank == 0,
            )
        )
    fig.add_trace(_line(days, truth, name="Truth", color=C_INK, width=1.5))
    fig.update_xaxes(title_text="Day")
    fig.update_yaxes(title_text="Path h")
    return _layout(fig, title="A posterior draw includes a whole path", height=380, legend_y=-0.26)


def fig_ties(
    level: np.ndarray,
    mean_path: np.ndarray,
    log_vol: np.ndarray,
    rms_residual: np.ndarray,
    *,
    level_grid: np.ndarray,
    level_conditional: np.ndarray,
    log_vol_grid: np.ndarray,
    log_vol_conditional: np.ndarray,
    chosen: int,
) -> go.Figure:
    """Top: posterior draws, showing each tie. Bottom: marginal and one conditional."""
    fig = make_subplots(
        rows=2,
        cols=2,
        shared_xaxes="columns",
        row_heights=[0.62, 0.38],
        vertical_spacing=0.06,
        horizontal_spacing=0.1,
        subplot_titles=(
            "Level and the path's average",
            "Vol-of-vol and the path's daily residuals",
        ),
    )

    def _cloud(x: np.ndarray, y: np.ndarray, col: int, *, legend: bool) -> None:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker={"color": C_MUTED, "size": 4, "opacity": 0.45},
                name="Posterior draws",
                showlegend=legend,
                hovertemplate="%{x:.3~f}, %{y:.3~f}<extra></extra>",
            ),
            row=1,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=[x[chosen]],
                y=[y[chosen]],
                mode="markers",
                marker={
                    "color": C_CONDITIONAL,
                    "size": 11,
                    "line": {"color": C_SURFACE, "width": 2},
                },
                name="One chosen draw",
                showlegend=legend,
                hovertemplate="Chosen draw<br>%{x:.3~f}, %{y:.3~f}<extra></extra>",
            ),
            row=1,
            col=col,
        )

    _cloud(level, mean_path, 1, legend=True)
    _cloud(log_vol, rms_residual, 2, legend=False)
    fig.add_trace(_marginal_bars(level, name="Marginal: readings only"), row=2, col=1)
    fig.add_trace(
        _line(
            level_grid,
            level_conditional,
            name="Conditional: given that draw's whole path",
            color=C_CONDITIONAL,
            width=2,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        _marginal_bars(log_vol, name="Marginal: readings only", legend=False), row=2, col=2
    )
    fig.add_trace(
        _line(
            log_vol_grid,
            log_vol_conditional,
            name="Conditional",
            color=C_CONDITIONAL,
            width=2,
            legend=False,
        ),
        row=2,
        col=2,
    )
    fig.update_yaxes(title_text="Mean of h", row=1, col=1)
    fig.update_yaxes(title_text="RMS transition residual", row=1, col=2)
    fig.update_yaxes(title_text="Density", row=2, col=1)
    fig.update_yaxes(title_text="Density", row=2, col=2)
    fig.update_xaxes(title_text="Level μ", row=2, col=1)
    fig.update_xaxes(title_text="log vol-of-vol, log σ", row=2, col=2)
    fig.update_layout(barmode="overlay", bargap=0)
    return _layout(fig, title="Parameters are tied to the path", height=560)


def fig_reading_factor(
    grid: np.ndarray,
    factors: Sequence[tuple[str, np.ndarray, np.ndarray]],
) -> go.Figure:
    """Each entry is (label, exact log factor, Gaussian log factor), both peak-normalized."""
    fig = make_subplots(
        rows=1,
        cols=len(factors),
        subplot_titles=tuple(label for label, _, _ in factors),
        horizontal_spacing=0.08,
    )
    for column, (_, exact, gaussian) in enumerate(factors, start=1):
        fig.add_trace(
            _line(
                grid,
                np.exp(exact),
                name="Exact reading factor",
                color=C_INK,
                width=2,
                legend=column == 1,
            ),
            row=1,
            col=column,
        )
        fig.add_trace(
            _line(
                grid,
                np.exp(gaussian),
                name="Gaussian fit at its peak",
                color=C_KALMAN,
                width=2,
                dash="dash",
                legend=column == 1,
            ),
            row=1,
            col=column,
        )
        fig.update_xaxes(title_text="Level + path on that day", row=1, col=column)
    fig.update_yaxes(title_text="Factor, peak = 1", row=1, col=1)
    return _layout(fig, title="One day's reading is not Gaussian", height=340, legend_y=-0.32)


def fig_lattice(
    particles: np.ndarray,
    ancestors: np.ndarray,
    probabilities: np.ndarray,
    selected: np.ndarray,
    path: np.ndarray,
    *,
    backward_probabilities: np.ndarray,
    stage: int,
    title: str,
    auxiliary: np.ndarray | None = None,
) -> go.Figure:
    """One conditional particle sweep, revealed step by step.

    Stages below T show the forward pass through that day, with candidates sized by weight.
    Later stages draw the new path backward from the final day. On the day being chosen,
    candidates are sized by their backward probability, which also looks at the day chosen after.
    Candidate 0 is the current path at every day.
    """
    days, count = particles.shape
    forward_end = min(stage, days - 1)
    active = days - 1 - (stage - days) if stage >= days else None
    fig = go.Figure()
    link_x: list[float | None] = []
    link_y: list[float | None] = []
    for day in range(1, forward_end + 1):
        for particle in range(1, count):
            link_x += [day - 1, day, None]
            link_y += [particles[day - 1, ancestors[day, particle]], particles[day, particle], None]
    fig.add_trace(
        go.Scatter(
            x=link_x,
            y=link_y,
            mode="lines",
            line={"color": C_LINES, "width": 1},
            name="Each new candidate's parent",
            hoverinfo="skip",
        )
    )
    shown = np.arange(forward_end + 1)
    if auxiliary is not None:
        fig.add_trace(
            go.Scatter(
                x=shown,
                y=auxiliary[shown],
                mode="markers",
                marker={
                    "color": C_PHYSICS,
                    "size": 9,
                    "symbol": "diamond-open",
                    "line": {"width": 2},
                },
                name="Noisy copy u of the old path",
                hovertemplate="Noisy copy on day %{x}: %{y:.3f}<extra></extra>",
            )
        )
    sizes = probabilities[shown].copy()
    if active is not None:
        sizes[active] = backward_probabilities[active]
    fig.add_trace(
        go.Scatter(
            x=np.repeat(shown, count),
            y=particles[shown].ravel(),
            mode="markers",
            marker={
                "color": C_PARTICLE,
                "size": 5 + 20 * np.sqrt(sizes.ravel()),
                "opacity": 0.75,
                "line": {"color": C_SURFACE, "width": 1},
            },
            name="Candidates, sized by weight",
            customdata=sizes.ravel(),
            hovertemplate="Day %{x}<br>Value %{y:.3f}<br>Size %{customdata:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        _line(
            shown,
            particles[shown, 0],
            name="Current path: candidate 0, keeps its own parent",
            color=C_INK,
            width=1.5,
            dash="dot",
        )
    )
    if active is not None:
        kept = np.arange(active, days)
        fig.add_trace(
            go.Scatter(
                x=kept,
                y=path[kept],
                mode="lines+markers",
                line={"color": C_PHYSICS, "width": 3},
                marker={"color": C_PHYSICS, "size": 9, "line": {"color": C_SURFACE, "width": 2}},
                name="New path, drawn backward",
                customdata=selected[kept],
                hovertemplate="Day %{x}: candidate %{customdata}, value %{y:.3f}<extra></extra>",
            )
        )
        fig.add_annotation(
            x=active,
            y=1.0,
            yref="paper",
            yshift=14,
            showarrow=False,
            text="Being chosen: sized by backward probability",
            xanchor="left" if active < days / 2 else "right",
        )
    fig.update_xaxes(title_text="Day", range=[-0.5, days - 0.5], dtick=1)
    fig.update_yaxes(title_text="Stored path value")
    return _layout(fig, title=title, height=430, legend_y=-0.2)


def fig_one_day(
    grid: np.ndarray,
    *,
    transition: np.ndarray,
    reading: np.ndarray,
    target: np.ndarray,
    bump: np.ndarray,
    proposal: np.ndarray,
    reference: float,
    auxiliary: float,
    title: str,
) -> go.Figure:
    """Densities over one day's value, each normalized on the grid."""
    fig = go.Figure()
    fig.add_trace(
        _line(
            grid,
            reading,
            name="Reading likelihood, rescaled for display",
            color=C_MUTED,
            width=1.5,
            dash="dot",
        )
    )
    fig.add_trace(
        _line(grid, transition, name="Dynamics from the parent", color=C_CONDITIONAL, width=2)
    )
    fig.add_trace(_line(grid, bump, name="Bump around the noisy copy", color=C_PHYSICS, width=2))
    fig.add_trace(
        _line(grid, proposal, name="Proposal: dynamics × bump", color=C_PARTICLE, width=3)
    )
    fig.add_trace(
        _line(
            grid,
            target,
            name="True one-day target: dynamics × reading",
            color=C_INK,
            width=2,
            dash="dash",
        )
    )
    left, right = sorted(((reference, "old value"), (auxiliary, "noisy copy")))
    for (value, label), position in ((left, "top left"), (right, "top right")):
        fig.add_vline(
            x=value,
            line={"color": C_LINES, "width": 1},
            annotation_text=label,
            annotation_position=position,
        )
    fig.update_xaxes(title_text="This day's value of the path")
    tallest = max(float(np.max(curve)) for curve in (transition, target, bump, proposal))
    fig.update_yaxes(title_text="Density", range=[0.0, 1.2 * tallest])
    return _layout(fig, title=title, height=380, legend_y=-0.24)


def fig_kalman_move(
    days: np.ndarray,
    *,
    reference: np.ndarray,
    auxiliary: np.ndarray,
    proposal_mean: np.ndarray,
    proposal_sd: np.ndarray,
    proposed: np.ndarray,
    target_change: float,
    auxiliary_change: float,
    proposal_change: float,
    acceptance: float,
    title: str,
) -> go.Figure:
    """Left: the Gaussian proposal around the noisy copy. Right: the exact ledger."""
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.72, 0.28],
        horizontal_spacing=0.09,
        subplot_titles=("One whole-path proposal", "The exact correction"),
    )
    fig.add_trace(
        _band(
            days,
            proposal_mean - 2 * proposal_sd,
            proposal_mean + 2 * proposal_sd,
            name="Gaussian proposal, ± 2 sd",
            color="rgba(98, 80, 214, 0.18)",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=days,
            y=auxiliary,
            mode="markers",
            marker={"color": C_PHYSICS, "size": 6, "symbol": "diamond-open"},
            name="Noisy copy u",
            hovertemplate="Day %{x}: %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(_line(days, reference, name="Current path", color=C_INK, width=1.5), row=1, col=1)
    fig.add_trace(
        _line(days, proposed, name="Proposed path", color=C_KALMAN, width=2), row=1, col=1
    )
    labels = ["Exact target", "Noisy copy", "Proposal, reversed", "Sum: log ratio"]
    values = [
        target_change,
        auxiliary_change,
        proposal_change,
        target_change + auxiliary_change + proposal_change,
    ]
    colors = [C_INK_2, C_PHYSICS, C_KALMAN, C_INK]
    fig.add_trace(
        go.Bar(
            x=labels,
            y=values,
            marker={"color": colors, "line": {"width": 0}},
            showlegend=False,
            hovertemplate="%{x}: %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="Day", row=1, col=1)
    fig.update_yaxes(title_text="Stored path value", row=1, col=1)
    fig.update_yaxes(title_text="Change in log density", row=1, col=2)
    return _layout(
        fig,
        title=f"{title} (acceptance probability {acceptance:.2f})",
        height=400,
        legend_y=-0.2,
    )


def fig_bank_weights(
    target_increments: np.ndarray,
    bank_densities: np.ndarray,
    probabilities: np.ndarray,
    *,
    title: str,
) -> go.Figure:
    """A candidate's log weight is the sum of these two parts, each shown relative to its mean."""
    candidates = np.arange(probabilities.size)
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.12,
        subplot_titles=("The two parts of the log weight", "The resulting probability"),
    )
    fig.add_trace(
        go.Bar(
            x=candidates,
            y=target_increments - target_increments.mean(),
            name="True one-day factors",
            marker={"color": C_INK_2, "line": {"width": 0}},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=candidates,
            y=bank_densities - bank_densities.mean(),
            name="Density of the other candidates, if this one were current",
            marker={"color": C_PARTICLE, "line": {"width": 0}},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=candidates,
            y=probabilities,
            name="Probability",
            marker={"color": C_MUTED, "line": {"width": 0}},
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="Candidate (0 is the current path)", dtick=1)
    fig.update_yaxes(title_text="Relative log weight", row=1, col=1)
    fig.update_yaxes(title_text="Probability", row=1, col=2)
    fig.update_layout(barmode="group", bargap=0.25)
    return _layout(fig, title=title, height=340, legend_y=-0.3)


def fig_leapfrog(
    days: np.ndarray,
    positions: np.ndarray,
    energies: np.ndarray,
    *,
    title: str,
    frames: int = 5,
) -> go.Figure:
    """Left: the path at several points along one trajectory. Right: the energy error."""
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.72, 0.28],
        horizontal_spacing=0.09,
        subplot_titles=("The whole path moves at every step", "Energy error along the way"),
    )
    steps = positions.shape[0] - 1
    picks = np.linspace(0, steps, frames).astype(int)
    for rank, pick in enumerate(picks):
        share = rank / max(frames - 1, 1)
        fig.add_trace(
            _line(
                days,
                positions[pick],
                name="Start" if pick == 0 else f"After {pick} steps",
                color=C_INK if pick == 0 else C_PHYSICS,
                width=1.5 if pick == 0 else 1.0 + share,
                opacity=1.0 if pick == 0 else 0.35 + 0.65 * share,
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        _line(
            np.arange(steps + 1),
            energies - energies[0],
            name="Energy − starting energy",
            color=C_PHYSICS,
            width=2,
            legend=False,
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="Day", row=1, col=1)
    fig.update_xaxes(title_text="Leapfrog step", row=1, col=2)
    fig.update_yaxes(title_text="Path value", row=1, col=1)
    fig.update_yaxes(title_text="Energy error", row=1, col=2)
    return _layout(fig, title=title, height=380, legend_y=-0.22)


def fig_kalman_scan(
    days: np.ndarray,
    deltas: np.ndarray,
    acceptance: np.ndarray,
    *,
    blocks: tuple[int, np.ndarray] | None = None,
) -> go.Figure:
    """Whole-path acceptance against series length, one line per δ; shape (deltas, days)."""
    fig = go.Figure()
    for rank, delta in enumerate(deltas):
        share = (rank + 1) / deltas.size
        fig.add_trace(
            go.Scatter(
                x=days,
                y=acceptance[rank],
                mode="lines+markers",
                name=f"δ = {delta:g}",
                line={"color": C_KALMAN, "width": 2},
                marker={"color": C_KALMAN, "size": 8},
                opacity=0.3 + 0.7 * share,
            )
        )
    if blocks is not None:
        fig.add_trace(
            go.Scatter(
                x=days,
                y=blocks[1],
                mode="lines+markers",
                name=f"{blocks[0]}-day blocks, per block, largest δ",
                line={"color": C_INK, "width": 2, "dash": "dash"},
                marker={"color": C_INK, "size": 8},
            )
        )
    fig.update_xaxes(title_text="Days in the series", **_log_ticks(days))
    fig.update_yaxes(title_text="Acceptance probability", range=[0, 1.02])
    return _layout(fig, title="One verdict for many days", height=360, legend_y=-0.24)


def fig_later_reading(
    days: np.ndarray,
    results: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    late_day: int,
) -> go.Figure:
    """Mean change by day in what is proposed and in what is returned."""
    colors = {"Auxiliary Kalman": C_KALMAN, "Particle-mGRAD": C_PARTICLE}
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.09,
        subplot_titles=("What is proposed", "What is returned"),
    )
    for name, (proposed, returned) in results.items():
        for column, values in ((1, proposed), (2, returned)):
            fig.add_trace(
                _line(
                    days,
                    values,
                    name=name,
                    color=colors[name],
                    width=2,
                    legend=column == 1,
                ),
                row=1,
                col=column,
            )
    for column in (1, 2):
        fig.add_vline(
            x=late_day, line={"color": C_MUTED, "width": 1, "dash": "dot"}, row=1, col=column
        )
        fig.update_xaxes(
            title_text="Day (dotted line: the changed reading)",
            range=[late_day - 15, late_day + 15],
            row=1,
            col=column,
        )
    fig.update_yaxes(title_text="Mean change in path value", row=1, col=1)
    return _layout(
        fig,
        title=f"Same random numbers, one much larger return on day {late_day}",
        height=380,
        legend_y=-0.24,
    )


def fig_kernel_race(
    traces: dict[str, np.ndarray],
    *,
    conditional_mean: np.ndarray,
    conditional_sd: np.ndarray,
    marginal_mean: np.ndarray,
    marginal_sd: np.ndarray,
    columns: Sequence[tuple[int, str]],
    title: str,
) -> go.Figure:
    """Each kernel's parameter trace against the bands set by the conditional and the marginal."""
    colors = {"Random walk": C_PARTICLE, "MALA": C_KALMAN, "NUTS": C_PHYSICS}
    fig = make_subplots(
        rows=1,
        cols=len(columns),
        horizontal_spacing=0.09,
        subplot_titles=tuple(label for _, label in columns),
    )
    steps = next(iter(traces.values())).shape[0]
    x = np.arange(steps)
    for position, (column, label) in enumerate(columns, start=1):
        marginal_low = np.full(steps, marginal_mean[column] - 2 * marginal_sd[column])
        marginal_high = np.full(steps, marginal_mean[column] + 2 * marginal_sd[column])
        conditional_low = np.full(steps, conditional_mean[column] - 2 * conditional_sd[column])
        conditional_high = np.full(steps, conditional_mean[column] + 2 * conditional_sd[column])
        marginal_band = _band(x, marginal_low, marginal_high, name="Marginal ± 2 sd", color=C_BAND)
        conditional_band = _band(
            x,
            conditional_low,
            conditional_high,
            name="Conditional at the held path ± 2 sd",
            color="rgba(27, 175, 122, 0.3)",
        )
        marginal_band.showlegend = position == 1
        conditional_band.showlegend = position == 1
        fig.add_trace(marginal_band, row=1, col=position)
        fig.add_trace(conditional_band, row=1, col=position)
        for name, values in traces.items():
            fig.add_trace(
                _line(
                    x,
                    values[:, column],
                    name=name,
                    color=colors[name],
                    width=1.5,
                    legend=position == 1,
                ),
                row=1,
                col=position,
            )
        fig.update_xaxes(title_text="Kernel steps, path held", row=1, col=position)
        fig.update_yaxes(title_text=label, row=1, col=position)
    return _layout(fig, title=title, height=380, legend_y=-0.24)


def fig_decode(
    days: np.ndarray,
    responses: dict[str, np.ndarray],
    ledgers: dict[str, np.ndarray],
    *,
    title: str,
) -> go.Figure:
    """Left: how the physical path responds. Right: what the move costs in log density.

    A ledger is (prior, initial law, transitions, readings, Jacobian). The bars combine
    transitions with the Jacobian, because holding shocks or the chart makes them cancel.
    """
    colors = {
        "Hold the path": C_INK_2,
        "Hold the daily shocks": C_PHYSICS,
        "Hold the chart": C_PARTICLE,
    }
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.6, 0.4],
        horizontal_spacing=0.09,
        subplot_titles=("Change in the physical path", "Change in log density"),
    )
    groups = ("Prior and initial law", "Transitions + Jacobian", "Readings", "Total")
    for name, change in responses.items():
        fig.add_trace(_line(days, change, name=name, color=colors[name], width=1.5), row=1, col=1)
        terms = ledgers[name]
        values = [terms[0] + terms[1], terms[2] + terms[4], terms[3], terms.sum()]
        fig.add_trace(
            go.Bar(
                x=groups,
                y=values,
                name=name,
                marker={"color": colors[name], "line": {"width": 0}},
                showlegend=False,
                hovertemplate=f"{name}<br>%{{x}}: %{{y:.1f}}<extra></extra>",
            ),
            row=1,
            col=2,
        )
    fig.update_xaxes(title_text="Day", row=1, col=1)
    fig.update_yaxes(title_text="New h − old h", row=1, col=1)
    fig.update_yaxes(title_text="Δ log density", row=1, col=2)
    fig.update_layout(barmode="group", bargap=0.2)
    return _layout(fig, title=title, height=400, legend_y=-0.24)


def fig_widths(
    names: Sequence[str],
    parameters: Sequence[str],
    mean: np.ndarray,
    spread: np.ndarray,
    *,
    title: str,
) -> go.Figure:
    """Dot plot of conditional sd over marginal sd; mean has shape (names, parameters)."""
    colors = (C_PARTICLE, C_PHYSICS, C_KALMAN)
    fig = go.Figure()
    for column, parameter in enumerate(parameters):
        fig.add_trace(
            go.Scatter(
                x=mean[:, column],
                y=list(names),
                mode="markers",
                marker={
                    "color": colors[column],
                    "size": 11,
                    "line": {"color": C_SURFACE, "width": 2},
                },
                error_x={
                    "type": "data",
                    "array": spread[:, column],
                    "color": colors[column],
                    "thickness": 1,
                },
                name=parameter,
                hovertemplate=f"{parameter}<br>%{{y}}: %{{x:.2f}}<extra></extra>",
            )
        )
    fig.add_vline(
        x=1.0, line={"color": C_LINES, "width": 1}, annotation_text="As wide as<br>the marginal"
    )
    fig.update_xaxes(title_text="Conditional sd ÷ marginal sd", range=[0, 1.15])
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, title=title, height=340, legend_y=-0.28)


def fig_pmmh_noise(days: np.ndarray, particles: np.ndarray, spread: np.ndarray) -> go.Figure:
    fig = go.Figure()
    for row, length in enumerate(days):
        share = (row + 1) / days.size
        fig.add_trace(
            go.Scatter(
                x=particles,
                y=spread[row],
                mode="lines+markers",
                name=f"{length} days",
                line={"color": C_PARTICLE, "width": 2},
                marker={"color": C_PARTICLE, "size": 8},
                opacity=0.3 + 0.7 * share,
            )
        )
    fig.update_xaxes(title_text="Particles", **_log_ticks(particles))
    fig.update_yaxes(
        title_text="Sd of the log-likelihood", **_log_ticks((0.1, 0.2, 0.5, 1, 2, 5, 10))
    )
    return _layout(
        fig,
        title="How noisy this particle filter's likelihood estimate is",
        height=340,
        legend_y=-0.28,
    )


def fig_mpgibbs_stages(
    days: np.ndarray,
    *,
    paths_before: np.ndarray,
    paths_after: np.ndarray,
    probability_before: np.ndarray,
    probability_after: np.ndarray,
    final_probability: float,
    selected: int,
    title: str,
) -> go.Figure:
    """Path per candidate, and the new candidate's running probability along the days."""
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.55, 0.45],
        horizontal_spacing=0.09,
        subplot_titles=(
            "Each candidate's physical path",
            "Running probability of the new candidate",
        ),
    )
    fig.add_trace(
        _line(
            days,
            paths_before[0],
            name="Current candidate, before",
            color=C_INK,
            width=1,
            dash="dot",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        _line(
            days,
            paths_before[1],
            name="New candidate, before",
            color=C_PARTICLE,
            width=1,
            dash="dot",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        _line(
            days,
            paths_after[0],
            name="Current candidate, after the path move",
            color=C_INK,
            width=2,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        _line(
            days,
            paths_after[1],
            name="New candidate, after the path move",
            color=C_PARTICLE,
            width=2,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        _line(days, probability_before, name="At the old stored path", color=C_MUTED, width=2),
        row=1,
        col=2,
    )
    fig.add_trace(
        _line(days, probability_after, name="At the new stored path", color=C_PHYSICS, width=2),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="Day")
    fig.update_yaxes(title_text="Path h", row=1, col=1)
    fig.update_yaxes(title_text="Probability", range=[0, 1.02], row=1, col=2)
    choice = "the new candidate" if selected == 1 else "the current candidate"
    return _layout(
        fig,
        title=f"{title}: final probability of the new candidate {final_probability:.2f}, flip chose {choice}",
        height=420,
        legend_y=-0.2,
    )


def fig_slope_check(
    from_chart: np.ndarray,
    from_covariance: np.ndarray,
    from_cholesky: np.ndarray,
    raw_block: np.ndarray,
) -> go.Figure:
    days = np.arange(from_chart.size)
    fig = go.Figure()
    fig.add_trace(
        _line(
            days,
            from_covariance,
            name="Regression slope, from the exact joint covariance",
            color=C_INK,
            width=5,
        )
    )
    fig.add_trace(
        _line(
            days,
            from_cholesky,
            name="From the Cholesky factor of that covariance",
            color=C_CONDITIONAL,
            width=2.5,
        )
    )
    fig.add_trace(
        _line(
            days,
            from_chart,
            name="Translation, from differentiating the mean path",
            color=C_PARTICLE,
            width=1.5,
            dash="dash",
        )
    )
    fig.add_trace(
        _line(days, raw_block, name="Raw covariance block, Cov(h, level)", color=C_MUTED, width=1.5)
    )
    fig.update_xaxes(title_text="Day")
    fig.update_yaxes(title_text="Value")
    return _layout(
        fig,
        title="On a jointly Gaussian twin, the translation is the regression slope",
        height=360,
        legend_y=-0.3,
    )


def fig_barker_reference(
    particles: np.ndarray,
    *,
    mpgibbs: tuple[np.ndarray, np.ndarray],
    pmmh: tuple[np.ndarray, np.ndarray],
    ideal: tuple[float, float],
    ideal_metropolis: tuple[float, float],
    held: tuple[float, float],
) -> go.Figure:
    """Each value is (mean, standard error across independent chains)."""
    fig = go.Figure()
    for values, name, color in (
        (mpgibbs, "MPGibbs: new candidate wins", C_PARTICLE),
        (pmmh, "PMMH: candidate accepted", C_PHYSICS),
    ):
        fig.add_trace(
            go.Scatter(
                x=particles,
                y=values[0],
                error_y={"type": "data", "array": 2 * values[1], "color": color, "thickness": 1},
                mode="lines+markers",
                name=name,
                line={"color": color, "width": 2},
                marker={"color": color, "size": 8},
            )
        )
    for (value, _), name, color in (
        (ideal, "Flip after a perfect mixture path draw", C_INK),
        (ideal_metropolis, "Metropolis with the exact marginal likelihood", C_PHYSICS),
        (held, "Flip at a held path: taking turns", C_CONDITIONAL),
    ):
        fig.add_trace(
            go.Scatter(
                x=[particles[0], particles[-1]],
                y=[value, value],
                mode="lines",
                name=name,
                line={"color": color, "width": 1.5, "dash": "dash"},
            )
        )
    fig.update_xaxes(title_text="Particles", **_log_ticks(particles))
    fig.update_yaxes(title_text="How often the candidate wins", rangemode="tozero")
    return _layout(
        fig,
        title="MPGibbs against its references, on a Gaussian oracle",
        height=420,
        legend_y=-0.24,
    )


PLANES = {
    "scale": (2, 4, "log vol-of-vol, log σ", "RMS transition residual"),
    "level": (0, 3, "Level μ", "Mean of h"),
}


def fig_substeps(
    substeps: np.ndarray,
    *,
    cloud: np.ndarray,
    plane: str,
    parameters_first: bool,
    title: str,
) -> go.Figure:
    """substeps has shape (sweeps, 3, 6): one sequence of states from one chain."""
    x_column, y_column, x_title, y_title = PLANES[plane]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cloud[:, x_column],
            y=cloud[:, y_column],
            mode="markers",
            marker={"color": C_MUTED, "size": 4, "opacity": 0.3},
            name="Reference posterior draws",
            hoverinfo="skip",
        )
    )
    sweeps = substeps.shape[0]

    def _segments(start: int, stop: int) -> tuple[list[float | None], list[float | None]]:
        xs: list[float | None] = []
        ys: list[float | None] = []
        for sweep in range(sweeps):
            xs += [substeps[sweep, start, x_column], substeps[sweep, stop, x_column], None]
            ys += [substeps[sweep, start, y_column], substeps[sweep, stop, y_column], None]
        return xs, ys

    first, second = _segments(0, 1), _segments(1, 2)
    parameter, path = (first, second) if parameters_first else (second, first)
    fig.add_trace(
        go.Scatter(
            x=path[0],
            y=path[1],
            mode="lines",
            line={"color": C_PARTICLE, "width": 1.5},
            name="Path move" if parameters_first else "Path move, under both candidates",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=parameter[0],
            y=parameter[1],
            mode="lines",
            line={"color": C_PHYSICS, "width": 1.5},
            name="Parameter move" if parameters_first else "The flip",
            hoverinfo="skip",
        )
    )
    fig.update_xaxes(title_text=x_title)
    fig.update_yaxes(title_text=y_title)
    return _layout(fig, title=title, height=440, legend_y=-0.18)


def fig_agreement(
    names: Sequence[str],
    means: np.ndarray,
    errors: np.ndarray,
    reference: np.ndarray,
    reference_errors: np.ndarray,
    *,
    columns: Sequence[tuple[int, str]],
) -> go.Figure:
    """Each run's posterior mean with ± 2 Monte Carlo standard errors, against the reference."""
    fig = make_subplots(
        rows=1,
        cols=len(columns),
        shared_yaxes=True,
        horizontal_spacing=0.025,
        subplot_titles=tuple(label for _, label in columns),
    )
    for position, (column, _) in enumerate(columns, start=1):
        fig.add_trace(
            go.Scatter(
                x=means[:, column],
                y=list(names),
                mode="markers",
                marker={"color": C_PARTICLE, "size": 7},
                error_x={
                    "type": "data",
                    "array": 2 * errors[:, column],
                    "color": C_PARTICLE,
                    "thickness": 1,
                },
                showlegend=False,
                hovertemplate="%{y}: %{x:.3f}<extra></extra>",
            ),
            row=1,
            col=position,
        )
        fig.add_vrect(
            x0=reference[column] - 2 * reference_errors[column],
            x1=reference[column] + 2 * reference_errors[column],
            fillcolor=C_BAND,
            line_width=0,
            row=1,
            col=position,
        )
        fig.add_vline(
            x=reference[column],
            line={"color": C_INK, "width": 1, "dash": "dash"},
            row=1,
            col=position,
        )
    fig.update_yaxes(autorange="reversed", tickfont={"size": 10})
    fig.update_xaxes(nticks=4)
    return _layout(
        fig,
        title="Same posterior: means with ± 2 Monte Carlo errors, against the reference",
        height=max(380, 30 * len(names) + 120),
        legend=False,
    )


def fig_regimes(
    regimes: dict[str, tuple[np.ndarray, dict[str, np.ndarray], float]],
) -> go.Figure:
    """Each regime: deltas, sweeps per effective sample by smoother, and the bootstrap level."""
    colors = {
        "Particle-mGRAD": C_PARTICLE,
        "Auxiliary Kalman, whole path": C_KALMAN,
        "Auxiliary Kalman, blocks": "#a79be8",
    }
    fig = make_subplots(
        rows=1,
        cols=len(regimes),
        horizontal_spacing=0.09,
        subplot_titles=tuple(regimes),
    )
    for position, (_, (deltas, lines, bootstrap)) in enumerate(regimes.items(), start=1):
        for name, values in lines.items():
            fig.add_trace(
                go.Scatter(
                    x=deltas,
                    y=np.where(np.isfinite(values), values, np.nan),
                    mode="lines+markers",
                    name=name,
                    line={"color": colors[name], "width": 2},
                    marker={"color": colors[name], "size": 7},
                    showlegend=position == 1,
                ),
                row=1,
                col=position,
            )
        fig.add_trace(
            go.Scatter(
                x=[deltas[0], deltas[-1]],
                y=[bootstrap, bootstrap],
                mode="lines",
                name="Bootstrap conditional SMC",
                line={"color": C_INK, "width": 1.5, "dash": "dash"},
                showlegend=position == 1,
            ),
            row=1,
            col=position,
        )
        fig.update_xaxes(
            title_text="δ", type="log", dtick=1, exponentformat="power", row=1, col=position
        )
        fig.update_yaxes(
            title_text="Sweeps per effective sample",
            **_log_ticks((1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)),
            row=1,
            col=position,
        )
    return _layout(
        fig, title="The path's mean, with the parameters known", height=400, legend_y=-0.24
    )


def fig_joint_moves(moves: dict[str, np.ndarray], *, cloud: np.ndarray) -> go.Figure:
    """Consecutive retained draws of joint NUTS on one plane, one panel per coordinate system."""
    x_column, y_column, x_title, y_title = PLANES["scale"]
    fig = make_subplots(
        rows=1,
        cols=len(moves),
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.03,
        subplot_titles=tuple(moves),
    )
    for position, values in enumerate(moves.values(), start=1):
        fig.add_trace(
            go.Scatter(
                x=cloud[:, x_column],
                y=cloud[:, y_column],
                mode="markers",
                marker={"color": C_MUTED, "size": 3, "opacity": 0.3},
                name="Reference posterior draws",
                showlegend=position == 1,
                hoverinfo="skip",
            ),
            row=1,
            col=position,
        )
        fig.add_trace(
            go.Scatter(
                x=values[:, 0, x_column],
                y=values[:, 0, y_column],
                mode="lines+markers",
                line={"color": C_PHYSICS, "width": 1.2},
                marker={"color": C_PHYSICS, "size": 4},
                name="Forty consecutive draws",
                showlegend=position == 1,
            ),
            row=1,
            col=position,
        )
        fig.update_xaxes(title_text=x_title, row=1, col=position)
    fig.update_yaxes(title_text=y_title, row=1, col=1)
    fig.update_annotations(font={"size": 11})
    return _layout(fig, title="Joint NUTS, in three coordinate systems", height=380, legend_y=-0.24)


def fig_joint_trajectories(trajectories: dict[str, np.ndarray], *, cloud: np.ndarray) -> go.Figure:
    """One leapfrog trajectory per coordinate system, on the plane where vol-of-vol is tied."""
    x_column, y_column, x_title, y_title = PLANES["scale"]
    fig = make_subplots(
        rows=1,
        cols=len(trajectories),
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.03,
        subplot_titles=tuple(trajectories),
    )
    for position, values in enumerate(trajectories.values(), start=1):
        fig.add_trace(
            go.Scatter(
                x=cloud[:, x_column],
                y=cloud[:, y_column],
                mode="markers",
                marker={"color": C_MUTED, "size": 3, "opacity": 0.3},
                name="Reference posterior draws",
                showlegend=position == 1,
                hoverinfo="skip",
            ),
            row=1,
            col=position,
        )
        fig.add_trace(
            go.Scatter(
                x=values[:, x_column],
                y=values[:, y_column],
                mode="lines",
                line={"color": C_PHYSICS, "width": 2},
                name="One trajectory",
                showlegend=position == 1,
            ),
            row=1,
            col=position,
        )
        fig.add_trace(
            go.Scatter(
                x=values[:1, x_column],
                y=values[:1, y_column],
                mode="markers",
                marker={"color": C_INK, "size": 9, "line": {"color": C_SURFACE, "width": 2}},
                name="Where it starts",
                showlegend=position == 1,
            ),
            row=1,
            col=position,
        )
        fig.update_xaxes(title_text=x_title, row=1, col=position)
    fig.update_yaxes(title_text=y_title, row=1, col=1)
    fig.update_annotations(font={"size": 11})
    return _layout(
        fig,
        title="What one joint trajectory does (fixed-length HMC with NUTS-tuned settings)",
        height=380,
        legend_y=-0.24,
    )


def fig_speed(
    rows: Sequence[tuple[str, np.ndarray, np.ndarray]],
    *,
    columns: Sequence[tuple[int, str]],
    shocks: np.ndarray,
) -> go.Figure:
    """Sweeps per effective sample, chart off to chart on, per family and quantity.

    Each row is (label, sweeps with the chart off, sweeps with it on). Lower is better.
    """
    fig = make_subplots(
        rows=1,
        cols=len(columns),
        shared_yaxes=True,
        horizontal_spacing=0.03,
        subplot_titles=tuple(label for _, label in columns),
    )
    names = [label for label, _, _ in rows]
    for position, (column, _) in enumerate(columns, start=1):
        first = position == 1
        link_x: list[float | None] = []
        link_y: list[str | None] = []
        for label, off, on in rows:
            link_x += [float(off[column]), float(on[column]), None]
            link_y += [label, label, None]
        fig.add_trace(
            go.Scatter(
                x=link_x,
                y=link_y,
                mode="lines",
                line={"color": C_LINES, "width": 3},
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=position,
        )
        fig.add_trace(
            go.Scatter(
                x=[off[column] for _, off, _ in rows],
                y=names,
                mode="markers",
                marker={"color": C_INK_2, "size": 11, "line": {"color": C_SURFACE, "width": 2}},
                name="Chart off",
                showlegend=first,
                hovertemplate="%{y}, chart off: %{x:.1f}<extra></extra>",
            ),
            row=1,
            col=position,
        )
        fig.add_trace(
            go.Scatter(
                x=[on[column] for _, _, on in rows],
                y=names,
                mode="markers",
                marker={"color": C_PARTICLE, "size": 11, "line": {"color": C_SURFACE, "width": 2}},
                name="Chart on",
                showlegend=first,
                hovertemplate="%{y}, chart on: %{x:.1f}<extra></extra>",
            ),
            row=1,
            col=position,
        )
        fig.add_trace(
            go.Scatter(
                x=[shocks[column]],
                y=[names[0]],
                mode="markers",
                marker={
                    "color": C_PHYSICS,
                    "size": 12,
                    "symbol": "diamond",
                    "line": {"color": C_SURFACE, "width": 2},
                },
                name="Joint NUTS in daily shocks",
                showlegend=first,
                hovertemplate="Joint NUTS, daily shocks: %{x:.1f}<extra></extra>",
            ),
            row=1,
            col=position,
        )
        fig.update_xaxes(
            title_text="Sweeps per effective sample", type="log", dtick=1, row=1, col=position
        )
    fig.update_yaxes(autorange="reversed", tickfont={"size": 11})
    return _layout(
        fig,
        title="What each choice buys: lower is better",
        height=max(380, 36 * len(names) + 150),
        legend_y=-0.16,
    )


def fig_flip_rates(
    ideal: tuple[np.ndarray, np.ndarray],
    panels: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
) -> go.Figure:
    """How often MPGibbs's new candidate wins, by candidate scale, against the ideal flip."""
    colors = {
        "Bootstrap conditional SMC": C_INK,
        "Particle-mGRAD": C_PARTICLE,
        "Auxiliary Kalman, whole path": C_KALMAN,
        "Auxiliary Kalman, blocks": "#a79be8",
    }
    fig = make_subplots(
        rows=1,
        cols=len(panels),
        shared_yaxes=True,
        horizontal_spacing=0.04,
        subplot_titles=tuple(panels),
    )
    for position, lines in enumerate(panels.values(), start=1):
        first = position == 1
        fig.add_trace(
            go.Scatter(
                x=ideal[0],
                y=ideal[1],
                mode="lines",
                name="Ideal flip: perfect path draw, Gaussian marginal",
                line={"color": C_MUTED, "width": 5},
                showlegend=first,
            ),
            row=1,
            col=position,
        )
        for name, (scales, rates) in lines.items():
            fig.add_trace(
                go.Scatter(
                    x=scales,
                    y=rates,
                    mode="lines+markers",
                    name=name,
                    line={"color": colors[name], "width": 2},
                    marker={"color": colors[name], "size": 7},
                    showlegend=first,
                ),
                row=1,
                col=position,
            )
        fig.update_xaxes(
            title_text="Candidate scale: proposal sd per whitened coordinate", row=1, col=position
        )
    fig.update_yaxes(
        title_text="How often the new candidate wins", rangemode="tozero", row=1, col=1
    )
    return _layout(
        fig, title="The chart lets candidates sit further apart", height=420, legend_y=-0.24
    )
