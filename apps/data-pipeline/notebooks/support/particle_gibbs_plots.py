"""Figures for the particle-sampler walkthrough.

Every function draws records produced by ``notebooks.support.particle_gibbs_surface``
(``Surface``, ``Coupling``, ``Chain``, ``Sweep``, ``LapsRun``) and never moves a
sampler. Grid quantities (slices, slice centres, shadows) are read off ``Surface``.

Colours follow the entity across the notebook: exact Gibbs, conditional DSMC,
mPGibbs, LAPS and joint NUTS keep their categorical slot in every figure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from plotly.subplots import make_subplots
from scipy.special import logsumexp

from .particle_gibbs_surface import autocorrelation

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .particle_gibbs_surface import (
        Chain,
        CouplingRow,
        LapsRun,
        Surface,
        Sweep,
        World,
    )

FloatArray = NDArray[np.float64]

_FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

# ── Theme ─────────────────────────────────────────────────────────────────────

_BLUE_RAMP = (
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
)


@dataclass(frozen=True)
class Theme:
    """Chart chrome and entity colours for one display mode."""

    surface: str
    ink: str
    ink2: str
    muted: str
    grid: str
    axis: str
    gibbs: str
    dsmc: str
    mpgibbs: str
    laps: str
    nuts: str
    backdrop_scale: tuple[tuple[float, str], ...]
    relief_scale: tuple[tuple[float, str], ...]


def _scale(colors: Sequence[str]) -> tuple[tuple[float, str], ...]:
    return tuple((index / (len(colors) - 1), color) for index, color in enumerate(colors))


LIGHT = Theme(
    surface="#fcfcfb",
    ink="#0b0b0b",
    ink2="#52514e",
    muted="#898781",
    grid="#e1e0d9",
    axis="#c3c2b7",
    gibbs="#2a78d6",
    dsmc="#eb6834",
    mpgibbs="#1baf7a",
    laps="#eda100",
    nuts="#e87ba4",
    backdrop_scale=_scale(("#fcfcfb", "#ecebe7", "#d9d7d0", "#bdbbb3", "#9a988f")),
    relief_scale=_scale(_BLUE_RAMP),
)

DARK = Theme(
    surface="#1a1a19",
    ink="#f0efec",
    ink2="#c3c2b7",
    muted="#898781",
    grid="#2c2c2a",
    axis="#383835",
    gibbs="#3987e5",
    dsmc="#d95926",
    mpgibbs="#199e70",
    laps="#c98500",
    nuts="#d55181",
    backdrop_scale=_scale(("#1a1a19", "#262625", "#323230", "#44443f", "#62615b")),
    relief_scale=_scale(tuple(reversed(_BLUE_RAMP))),
)


def theme_for(mode: str) -> Theme:
    """``"dark"`` selects the dark steps; anything else the light ones."""
    return DARK if mode == "dark" else LIGHT


def _log_ticks(values: FloatArray) -> dict[str, list[float] | list[str]]:
    """1-2-5 tick values spanning ``values``, labelled in full."""
    positive = values[np.isfinite(values) & (values > 0)]
    low, high = np.floor(np.log10(positive.min())), np.ceil(np.log10(positive.max()))
    ticks = [
        float(mantissa * 10.0**exponent)
        for exponent in np.arange(low, high + 1)
        for mantissa in (1, 2, 5)
        if positive.min() / 1.5 <= mantissa * 10.0**exponent <= positive.max() * 1.5
    ]
    return {"tickvals": ticks, "ticktext": [f"{tick:g}" for tick in ticks]}


def _layout(
    fig: go.Figure,
    theme: Theme,
    *,
    height: int,
    title: str | None = None,
    legend_below: bool = False,
) -> go.Figure:
    """Shared chrome. Multi-panel figures put the legend below, clear of panel titles."""
    legend = (
        {"yref": "container", "y": 0.0, "yanchor": "bottom"}
        if legend_below
        else {"yanchor": "bottom", "y": 1.02}
    )
    fig.update_layout(
        height=height,
        title=None if title is None else {"text": title, "x": 0.0, "xanchor": "left"},
        font={"family": _FONT, "size": 13, "color": theme.ink2},
        paper_bgcolor=theme.surface,
        plot_bgcolor=theme.surface,
        margin={
            "l": 60,
            "r": 24,
            "t": 56 if title or legend_below else 28,
            "b": 110 if legend_below else 52,
        },
        legend={
            **legend,
            "orientation": "h",
            "xanchor": "left",
            "x": 0.0,
            "font": {"color": theme.ink2},
        },
        hoverlabel={"font": {"family": _FONT}},
    )
    fig.update_xaxes(
        gridcolor=theme.grid,
        linecolor=theme.axis,
        zerolinecolor=theme.grid,
        tickfont={"color": theme.muted},
        title_font={"color": theme.ink2},
    )
    fig.update_yaxes(
        gridcolor=theme.grid,
        linecolor=theme.axis,
        zerolinecolor=theme.grid,
        tickfont={"color": theme.muted},
        title_font={"color": theme.ink2},
    )
    return fig


# ── Grid quantities read off the surface ──────────────────────────────────────


def _slice_centres(surface: Surface) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """``E[x | theta, y]`` per theta column and ``E[theta | x, y]`` per x row.

    Columns or rows carrying a negligible share of the mass are dropped, since their
    centres are numerically meaningless.
    """
    density = surface.density
    column_mass = np.trapezoid(density, surface.x, axis=0)
    row_mass = np.trapezoid(density, surface.theta, axis=1)
    columns = column_mass > 1e-4 * column_mass.max()
    rows = row_mass > 1e-4 * row_mass.max()
    x_centres = (
        np.trapezoid(surface.x[:, None] * density, surface.x, axis=0)[columns]
        / column_mass[columns]
    )
    theta_centres = (
        np.trapezoid(density * surface.theta, surface.theta, axis=1)[rows] / row_mass[rows]
    )
    return surface.theta[columns], x_centres, theta_centres, surface.x[rows]


def _interpolated_column(surface: Surface, theta: float) -> FloatArray:
    """Joint density along x at a given theta, normalized to unit area."""
    index = int(np.clip(np.searchsorted(surface.theta, theta) - 1, 0, surface.theta.size - 2))
    left, right = surface.theta[index], surface.theta[index + 1]
    weight = float(np.clip((theta - left) / (right - left), 0.0, 1.0))
    column = (1.0 - weight) * surface.density[:, index] + weight * surface.density[:, index + 1]
    return column / np.trapezoid(column, surface.x)


def _interpolated_row(surface: Surface, x: float) -> FloatArray:
    """Joint density along theta at a given x, normalized to unit area."""
    index = int(np.clip(np.searchsorted(surface.x, x) - 1, 0, surface.x.size - 2))
    low, high = surface.x[index], surface.x[index + 1]
    weight = float(np.clip((x - low) / (high - low), 0.0, 1.0))
    row = (1.0 - weight) * surface.density[index, :] + weight * surface.density[index + 1, :]
    return row / np.trapezoid(row, surface.theta)


def _latent_marginal(surface: Surface) -> FloatArray:
    marginal = np.trapezoid(surface.density, surface.theta, axis=1)
    return marginal / np.trapezoid(marginal, surface.x)


def _backdrop(surface: Surface, theme: Theme, *, showlegend: bool = False) -> go.Contour:
    return go.Contour(
        x=surface.theta,
        y=surface.x,
        z=surface.density,
        colorscale=[list(stop) for stop in theme.backdrop_scale],
        showscale=False,
        contours={"coloring": "fill", "showlines": True},
        line={"color": theme.axis, "width": 0.6},
        ncontours=12,
        hoverinfo="skip",
        name="p(θ, x | y)",
        showlegend=showlegend,
    )


def _regression_lines(surface: Surface, theme: Theme, *, row: int, col: int, fig: go.Figure):
    theta_grid, x_centres, theta_centres, x_grid = _slice_centres(surface)
    fig.add_trace(
        go.Scatter(
            x=theta_grid,
            y=x_centres,
            mode="lines",
            line={"color": theme.dsmc, "width": 2, "dash": "dash"},
            name="E[x | θ, y]: smoothing-slice centres",
            hovertemplate="θ %{x:.2f}<br>E[x | θ, y] %{y:.2f}<extra></extra>",
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=theta_centres,
            y=x_grid,
            mode="lines",
            line={"color": theme.gibbs, "width": 2, "dash": "dash"},
            name="E[θ | x, y]: parameter-slice centres",
            hovertemplate="x %{y:.2f}<br>E[θ | x, y] %{x:.2f}<extra></extra>",
        ),
        row=row,
        col=col,
    )


def _frame_axes(fig: go.Figure, surface: Surface, *, row: int, col: int) -> None:
    theta = surface.theta[_central(surface.theta, surface.theta_marginal)]
    x = surface.x[_central(surface.x, _latent_marginal(surface))]
    fig.update_xaxes(range=[theta[0], theta[-1]], title_text="θ (parameter)", row=row, col=col)
    fig.update_yaxes(range=[x[0], x[-1]], title_text="x (latent)", row=row, col=col)


def start_point(surface: Surface) -> tuple[float, float]:
    """A plausible but overdispersed start: the 2% quantile of ``p(theta | y)``, with
    the latent at that theta's smoothing-slice centre."""
    theta = float(np.interp(0.02, _grid_cdf(surface.theta, surface.theta_marginal), surface.theta))
    column = _interpolated_column(surface, theta)
    return theta, float(np.trapezoid(surface.x * column, surface.x))


def smoother_echo(slice_var: float, delta: float, *, kappa: float) -> float:
    """Lag-1 autocorrelation of the latent under an exact draw from ``pi(x | z)``.

    On a Gaussian slice the auxiliary mean has slope
    ``a = 1 - kappa * delta / (2 * slice_var)``. Conditioning on that auxiliary
    gives the infinite-particle regression slope ``a² v / (a² v + delta/2)``.
    """
    variance = 0.5 * delta
    slope = 1 - kappa * variance / slice_var
    explained = slope**2 * slice_var
    return explained / (explained + variance)


def proposal_sd(step: float, scale: float = 1.0) -> float:
    """Standard deviation of theta' - theta for mPGibbs: two halves of ``2 step scale²``."""
    return 2.0 * float(np.sqrt(step)) * scale


# ── 1. The surface ────────────────────────────────────────────────────────────


def _grid_cdf(values: FloatArray, density: FloatArray) -> FloatArray:
    """Trapezoidal probability masses, including nonuniform display grids."""
    cdf = np.concatenate(([0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(values))))
    return cdf / cdf[-1]


def _central(values: FloatArray, weights: FloatArray, mass: float = 0.996) -> NDArray[np.bool_]:
    """Grid points inside the central ``mass`` of ``weights``, padded by 12% each side."""
    cdf = _grid_cdf(values, weights)
    low, high = np.interp([(1.0 - mass) / 2.0, (1.0 + mass) / 2.0], cdf, values)
    pad = 0.12 * (high - low)
    return (values >= low - pad) & (values <= high + pad)


def surface_figure(
    surface: Surface,
    theme: Theme,
    *,
    theta_cut: float | None = None,
    x_cut: float | None = None,
) -> go.Figure:
    """The joint posterior as a 3D relief with its two shadows and optional cuts.

    The view is cropped to the central 99.6% of each shadow and looks across the
    ridge, so its length is visible.
    """
    keep_theta = _central(surface.theta, surface.theta_marginal)
    latent = _latent_marginal(surface)
    keep_x = _central(surface.x, latent)
    theta, x = surface.theta[keep_theta], surface.x[keep_x]
    density = surface.density[np.ix_(keep_x, keep_theta)]
    top = float(density.max())
    theta_shadow = surface.theta_marginal[keep_theta]
    latent_shadow = latent[keep_x]
    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=theta,
            y=x,
            z=density,
            colorscale=[list(stop) for stop in theme.relief_scale],
            showscale=False,
            contours={
                "z": {
                    "show": True,
                    "color": theme.ink2,
                    "project": {"z": True},
                    "width": 1,
                    "start": 0.1 * top,
                    "end": 0.9 * top,
                    "size": 0.2 * top,
                }
            },
            lighting={"ambient": 0.85, "diffuse": 0.4, "specular": 0.05, "roughness": 0.9},
            hovertemplate="θ %{x:.2f}<br>x %{y:.2f}<br>density %{z:.3f}<extra></extra>",
            name="p(θ, x | y)",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=theta,
            y=np.full(theta.size, x[-1]),
            z=theta_shadow / theta_shadow.max() * 0.9 * top,
            mode="lines",
            line={"color": theme.ink, "width": 6},
            name="p(θ | y): the parameter's shadow",
            hovertemplate="θ %{x:.2f}<extra>p(θ | y)</extra>",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=np.full(x.size, theta[0]),
            y=x,
            z=latent_shadow / latent_shadow.max() * 0.9 * top,
            mode="lines",
            line={"color": theme.muted, "width": 4},
            name="p(x | y): the latent's shadow",
            hovertemplate="x %{y:.2f}<extra>p(x | y)</extra>",
        )
    )
    if theta_cut is not None:
        index = int(np.argmin(np.abs(theta - theta_cut)))
        column = _interpolated_column(surface, theta_cut)[keep_x]
        fig.add_trace(
            go.Scatter3d(
                x=np.full(x.size, theta_cut),
                y=x,
                z=column / column.max() * float(density[:, index].max()) * 1.01,
                mode="lines",
                line={"color": theme.dsmc, "width": 8},
                name=f"cut at θ = {theta_cut:.2f}: the smoothing slice",
                hoverinfo="skip",
            )
        )
    if x_cut is not None:
        index = int(np.argmin(np.abs(x - x_cut)))
        row = _interpolated_row(surface, x_cut)[keep_theta]
        fig.add_trace(
            go.Scatter3d(
                x=theta,
                y=np.full(theta.size, x_cut),
                z=row / row.max() * float(density[index, :].max()) * 1.01,
                mode="lines",
                line={"color": theme.gibbs, "width": 8},
                name=f"cut at x = {x_cut:.2f}: the parameter slice",
                hoverinfo="skip",
            )
        )
    axis = {
        "backgroundcolor": theme.surface,
        "gridcolor": theme.grid,
        "zerolinecolor": theme.axis,
        "linecolor": theme.axis,
        "tickfont": {"color": theme.muted},
        "showbackground": True,
    }
    fig.update_layout(
        scene={
            "xaxis": {**axis, "title": {"text": "θ (parameter)", "font": {"color": theme.ink2}}},
            "yaxis": {**axis, "title": {"text": "x (latent)", "font": {"color": theme.ink2}}},
            "zaxis": {
                **axis,
                "title": {"text": "density", "font": {"color": theme.ink2}},
                "showticklabels": False,
            },
            "camera": {
                "eye": {"x": 1.2, "y": -1.4, "z": 1.0},
                "center": {"x": 0, "y": 0, "z": -0.15},
            },
            "aspectmode": "manual",
            "aspectratio": {"x": 1.0, "y": 1.0, "z": 0.5},
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        }
    )
    _layout(fig, theme, height=600)
    fig.update_layout(margin={"l": 0, "r": 0, "t": 36, "b": 0})
    return fig


# ── 2. Cuts and shadows ───────────────────────────────────────────────────────


def cuts_figure(surface: Surface, theme: Theme, *, theta_cut: float, x_cut: float) -> go.Figure:
    """Top view with both cuts, then each cut against the shadow it should match."""
    fig = make_subplots(
        rows=1,
        cols=3,
        column_widths=[0.4, 0.3, 0.3],
        horizontal_spacing=0.08,
        subplot_titles=(
            "Top view",
            f"Parameter slice at x = {x_cut:.2f} vs shadow",
            f"Smoothing slice at θ = {theta_cut:.2f} vs shadow",
        ),
    )
    fig.add_trace(_backdrop(surface, theme), row=1, col=1)
    _regression_lines(surface, theme, row=1, col=1, fig=fig)
    fig.add_vline(x=theta_cut, line={"color": theme.dsmc, "width": 1.5}, row=1, col=1)
    fig.add_hline(y=x_cut, line={"color": theme.gibbs, "width": 1.5}, row=1, col=1)
    _frame_axes(fig, surface, row=1, col=1)

    row = _interpolated_row(surface, x_cut)
    fig.add_trace(
        go.Scatter(
            x=surface.theta,
            y=row,
            mode="lines",
            fill="tozeroy",
            line={"color": theme.gibbs, "width": 2},
            name="p(θ | x, y): parameter slice",
            showlegend=False,
            hovertemplate="θ %{x:.2f}<br>%{y:.3f}<extra>p(θ | x, y)</extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=surface.theta,
            y=surface.theta_marginal,
            mode="lines",
            line={"color": theme.ink, "width": 2},
            name="shadows: p(θ | y), p(x | y)",
            hovertemplate="θ %{x:.2f}<br>%{y:.3f}<extra>p(θ | y)</extra>",
        ),
        row=1,
        col=2,
    )
    column = _interpolated_column(surface, theta_cut)
    fig.add_trace(
        go.Scatter(
            x=surface.x,
            y=column,
            mode="lines",
            fill="tozeroy",
            line={"color": theme.dsmc, "width": 2},
            name="p(x | θ, y): smoothing slice",
            showlegend=False,
            hovertemplate="x %{x:.2f}<br>%{y:.3f}<extra>p(x | θ, y)</extra>",
        ),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Scatter(
            x=surface.x,
            y=_latent_marginal(surface),
            mode="lines",
            line={"color": theme.ink, "width": 2},
            showlegend=False,
            hovertemplate="x %{x:.2f}<br>%{y:.3f}<extra>p(x | y)</extra>",
        ),
        row=1,
        col=3,
    )
    fig.update_xaxes(title_text="θ (parameter)", row=1, col=2)
    fig.update_xaxes(title_text="x (latent)", row=1, col=3)
    fig.update_yaxes(showticklabels=False, row=1, col=2)
    fig.update_yaxes(showticklabels=False, row=1, col=3)
    fig.update_annotations(font={"color": theme.ink2, "size": 13})
    return _layout(fig, theme, height=430, legend_below=True)


# ── 3. Chains on the surface ──────────────────────────────────────────────────


@dataclass(frozen=True)
class Series:
    """One chain to draw, with its display name and entity colour."""

    name: str
    chain: Chain
    color: str


def _polyline(chain: Chain, sweeps: int) -> tuple[FloatArray, FloatArray]:
    """The first ``sweeps`` retained states; blocked chains as a staircase.

    The path starts at the first retained draw: discarded warmup transitions are not
    drawn, so no segment spans them.
    """
    thetas, xs = chain.theta[:sweeps], chain.x[:sweeps]
    if not chain.blocked:
        return thetas, xs
    # theta moves first (horizontal), then x (vertical).
    path_theta = np.empty(2 * thetas.size - 1)
    path_x = np.empty(2 * thetas.size - 1)
    path_theta[0::2], path_x[0::2] = thetas, xs
    path_theta[1::2], path_x[1::2] = thetas[1:], xs[:-1]
    return path_theta, path_x


def paths_figure(
    surface: Surface,
    theme: Theme,
    series: Sequence[Series],
    *,
    sweeps: int,
) -> go.Figure:
    """The first ``sweeps`` retained moves of each chain over the top view."""
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(_backdrop(surface, theme), row=1, col=1)
    _regression_lines(surface, theme, row=1, col=1, fig=fig)
    for item in series:
        path_theta, path_x = _polyline(item.chain, sweeps)
        fig.add_trace(
            go.Scatter(
                x=path_theta,
                y=path_x,
                mode="lines",
                line={"color": item.color, "width": 2},
                name=item.name,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=item.chain.theta[:sweeps],
                y=item.chain.x[:sweeps],
                mode="markers",
                marker={
                    "color": item.color,
                    "size": 7,
                    "line": {"color": theme.surface, "width": 2},
                },
                showlegend=False,
                customdata=np.arange(1, sweeps + 1),
                hovertemplate="retained draw %{customdata}<br>θ %{x:.2f}<br>x %{y:.2f}<extra>"
                + item.name
                + "</extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=item.chain.theta[:1],
                y=item.chain.x[:1],
                mode="markers",
                marker={
                    "color": "rgba(0,0,0,0)",
                    "size": 15,
                    "line": {"color": item.color, "width": 2.5},
                },
                showlegend=False,
                hovertemplate="first retained draw<extra>" + item.name + "</extra>",
            ),
            row=1,
            col=1,
        )
    _frame_axes(fig, surface, row=1, col=1)
    return _layout(fig, theme, height=520)


def echo_figure(theme: Theme, chain: Chain, *, mean: float, share: float, name: str) -> go.Figure:
    """Consecutive parameter draws: the regression slope is the coupling share."""
    current, following = chain.theta[:-1], chain.theta[1:]
    slope = float(np.polyfit(current - mean, following - mean, 1)[0])
    span = np.array([current.min(), current.max()])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=current,
            y=following,
            mode="markers",
            marker={"color": theme.gibbs, "size": 4, "opacity": 0.35},
            name=f"{name}: (θₖ, θₖ₊₁), fitted slope {slope:.2f}",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=span,
            y=mean + share * (span - mean),
            mode="lines",
            line={"color": theme.ink, "width": 2},
            name=f"stationary linear predictor, slope s = {share:.2f}",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=span,
            y=span,
            mode="lines",
            line={"color": theme.muted, "width": 1, "dash": "dot"},
            name="no movement (slope 1)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=span,
            y=np.full(2, mean),
            mode="lines",
            line={"color": theme.muted, "width": 1},
            name="independent draws (slope 0)",
        )
    )
    fig.update_xaxes(title_text="θ at sweep k")
    fig.update_yaxes(title_text="θ at sweep k + 1")
    return _layout(fig, theme, height=430)


def mixing_figure(
    theme: Theme,
    series: Sequence[Series],
    *,
    share: float | None,
    max_lag: int = 40,
    window: int = 400,
) -> go.Figure:
    """Parameter traces and autocorrelations side by side."""
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.58, 0.42],
        horizontal_spacing=0.09,
        subplot_titles=(f"θ over the first {window} sweeps", "Autocorrelation of θ"),
    )
    lags = np.arange(max_lag + 1)
    for item in series:
        fig.add_trace(
            go.Scatter(
                x=np.arange(1, window + 1),
                y=item.chain.theta[:window],
                mode="lines",
                line={"color": item.color, "width": 1.5},
                name=item.name,
                hovertemplate="sweep %{x}<br>θ %{y:.3f}<extra>" + item.name + "</extra>",
            ),
            row=1,
            col=1,
        )
        acf = autocorrelation(item.chain.theta, max_lag)
        fig.add_trace(
            go.Scatter(
                x=lags,
                y=acf,
                mode="lines",
                line={"color": item.color, "width": 2},
                showlegend=False,
                hovertemplate="lag %{x}<br>%{y:.2f}<extra>" + item.name + "</extra>",
            ),
            row=1,
            col=2,
        )
        fig.add_annotation(
            x=min(max_lag, 6 + 4 * len(fig.layout.annotations)),
            y=float(acf[min(max_lag, 6 + 4 * len(fig.layout.annotations))]),
            text=item.name,
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font={"color": theme.ink2, "size": 11},
            row=1,
            col=2,
        )
    if share is not None:
        fig.add_trace(
            go.Scatter(
                x=lags,
                y=share**lags,
                mode="lines",
                line={"color": theme.ink, "width": 1.5, "dash": "dash"},
                name="Gaussian exact Gibbs: sᵏ",
            ),
            row=1,
            col=2,
        )
    fig.update_xaxes(title_text="sweep", row=1, col=1)
    fig.update_xaxes(title_text="lag (sweeps)", row=1, col=2)
    fig.update_yaxes(title_text="θ", row=1, col=1)
    fig.update_yaxes(range=[-0.15, 1.02], row=1, col=2)
    fig.update_annotations(font={"color": theme.ink2})
    return _layout(fig, theme, height=420, legend_below=True)


# ── 4. One particle sweep ─────────────────────────────────────────────────────


def sweep_figure(
    world: World,
    sweep: Sweep,
    theme: Theme,
    *,
    delta: float,
    color: str,
    x_range: tuple[float, float],
) -> go.Figure:
    """Anatomy of one conditional DSMC sweep at a single time step.

    Top: the target the sweep samples (the smoothing slice, or the mixture over both
    parameter labels), and what an exact draw given the auxiliary point would look
    like: ``pi(x | z) ∝ target(x) N(z; x + kappa delta/2 grad log target(x), delta/2)``.
    Bottom: the particles with
    their selection probabilities; particle 0 is the reference.
    """
    labels = sweep.thetas.size
    grid = np.linspace(*x_range, 600)
    component_log_density = np.array(
        [
            weight + world.log_latent(grid, float(theta)) + world.log_obs(grid)
            for weight, theta in zip(sweep.label_log_prior, sweep.thetas, strict=True)
        ]
    )
    target_log_density = logsumexp(component_log_density, axis=0)
    components = np.exp(component_log_density - target_log_density.max())
    target = components.sum(axis=0)
    area = np.trapezoid(target, grid)
    var = 0.5 * delta
    component_scores = np.array(
        [
            (world.x_given_theta(float(theta))[0] - grid) / world.x_given_theta(float(theta))[1]
            for theta in sweep.thetas
        ]
    )
    gradient = (np.exp(component_log_density - target_log_density) * component_scores).sum(axis=0)
    auxiliary_mean = grid + sweep.kappa * var * gradient
    reachable_log_density = target_log_density - 0.5 * np.square(auxiliary_mean - sweep.aux) / var
    reachable = np.exp(reachable_log_density - reachable_log_density.max())
    reachable /= np.trapezoid(reachable, grid)
    proposal = np.exp(-0.5 * np.square(grid - sweep.center) / var) / np.sqrt(2 * np.pi * var)

    fig = make_subplots(
        rows=2,
        cols=2 if labels > 1 else 1,
        shared_xaxes=True,
        row_heights=[0.62, 0.38],
        column_widths=[0.8, 0.2] if labels > 1 else [1.0],
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
        specs=[[{}, {"rowspan": 2}], [{}, None]] if labels > 1 else [[{}], [{}]],
        subplot_titles=(
            (
                "What the sweep targets",
                "Label given the new x",
                "Particles and selection probability",
            )
            if labels > 1
            else ("What the sweep targets", "Particles and selection probability")
        ),
    )
    if labels > 1:
        names = ("label θ (current)", "label θ′ (proposed)")
        colors = (theme.ink2, color)
        for component, name, line_color in zip(components, names, colors, strict=True):
            fig.add_trace(
                go.Scatter(
                    x=grid,
                    y=component / area,
                    mode="lines",
                    line={"color": line_color, "width": 1.5, "dash": "dot"},
                    name=f"{name} share of the target",
                    hoverinfo="skip",
                ),
                row=1,
                col=1,
            )
    fig.add_trace(
        go.Scatter(
            x=grid,
            y=target / area,
            mode="lines",
            fill="tozeroy",
            line={"color": theme.muted, "width": 1.5},
            name="target: mixture over labels" if labels > 1 else "target: p(x | θ, y)",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=grid,
            y=reachable,
            mode="lines",
            line={"color": color, "width": 3},
            name="π(x | z): exact draw given z (N → ∞)",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=grid,
            y=proposal,
            mode="lines",
            line={"color": theme.ink2, "width": 1, "dash": "dash"},
            name="particle proposal N(c, δ/2)",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    reference = float(sweep.particles[0])
    chosen = float(sweep.particles[sweep.chosen])
    for value, symbol, label in (
        (reference, "circle-open", "reference x"),
        (sweep.aux, "diamond", "auxiliary z"),
        (sweep.center, "triangle-up", "centre c"),
    ):
        fig.add_trace(
            go.Scatter(
                x=[value],
                y=[0.0],
                mode="markers+text",
                marker={"color": theme.ink, "size": 11, "symbol": symbol, "line": {"width": 2}},
                text=[label],
                textposition="top center",
                textfont={"color": theme.ink2, "size": 11},
                showlegend=False,
                hovertemplate=f"{label} %{{x:.3f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    stems_x = np.repeat(sweep.particles, 3)
    stems_y = np.column_stack(
        [np.zeros_like(sweep.weights), sweep.weights, np.full_like(sweep.weights, np.nan)]
    ).ravel()
    fig.add_trace(
        go.Scatter(
            x=stems_x,
            y=stems_y,
            mode="lines",
            line={"color": color, "width": 1.5},
            showlegend=False,
            hoverinfo="skip",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=sweep.particles,
            y=sweep.weights,
            mode="markers",
            marker={"color": color, "size": 7, "line": {"color": theme.surface, "width": 2}},
            name="particles",
            customdata=np.arange(sweep.particles.size),
            hovertemplate="particle %{customdata}<br>x %{x:.3f}<br>P(select) %{y:.3f}<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[reference, chosen],
            y=[float(sweep.weights[0]), float(sweep.weights[sweep.chosen])],
            mode="markers+text",
            marker={
                "color": "rgba(0,0,0,0)",
                "size": 16,
                "symbol": ["circle-open", "circle-open"],
                "line": {"color": [theme.ink2, theme.ink], "width": [2, 3]},
            },
            text=["reference", "chosen"] if sweep.chosen else ["", "chosen = reference"],
            textposition="top center",
            textfont={"color": theme.ink2, "size": 11},
            showlegend=False,
            hoverinfo="skip",
        ),
        row=2,
        col=1,
    )
    if labels > 1:
        fig.add_trace(
            go.Bar(
                x=["θ stays", "θ′ moves"],
                y=sweep.label_probs,
                marker={"color": [theme.ink2, color], "cornerradius": 4},
                text=[f"{value:.2f}" for value in sweep.label_probs],
                textposition="outside",
                textfont={"color": theme.ink2},
                showlegend=False,
                hovertemplate="%{x}: %{y:.3f}<extra></extra>",
            ),
            row=1,
            col=2,
        )
        fig.update_yaxes(range=[0, 1.15], showticklabels=False, row=1, col=2)
    fig.update_xaxes(range=list(x_range), row=1, col=1)
    fig.update_xaxes(range=list(x_range), title_text="x (latent)", row=2, col=1)
    fig.update_yaxes(showticklabels=False, title_text="density", row=1, col=1)
    fig.update_yaxes(title_text="P(select)", rangemode="tozero", row=2, col=1)
    fig.update_annotations(font={"color": theme.ink2, "size": 13})
    return _layout(fig, theme, height=560, legend_below=True)


def mpgibbs_move_figure(surface: Surface, sweep: Sweep, theme: Theme) -> go.Figure:
    """The two parameter labels as vertical lines, and where the sweep went."""
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(_backdrop(surface, theme), row=1, col=1)
    current, proposed = float(sweep.thetas[0]), float(sweep.thetas[1])
    for value, line_color, label in (
        (current, theme.ink2, "θ (current)"),
        (proposed, theme.mpgibbs, "θ′ (proposed)"),
    ):
        fig.add_trace(
            go.Scatter(
                x=[value, value],
                y=[surface.x[0], surface.x[-1]],
                mode="lines",
                line={"color": line_color, "width": 2},
                name=label,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
    start = (current, float(sweep.particles[0]))
    end = (float(sweep.thetas[sweep.label]), float(sweep.particles[sweep.chosen]))
    fig.add_trace(
        go.Scatter(
            x=[sweep.parameter_aux],
            y=[start[1]],
            mode="markers",
            marker={"color": theme.mpgibbs, "size": 11, "symbol": "diamond-open"},
            name="u: parameter proposal auxiliary (not a chain state)",
            hovertemplate="u %{x:.3f}<extra>actual parameter auxiliary</extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_annotation(
        x=end[0],
        y=end[1],
        ax=start[0],
        ay=start[1],
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=3,
        arrowwidth=2,
        arrowcolor=theme.ink,
    )
    fig.add_trace(
        go.Scatter(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            mode="markers+text",
            marker={
                "color": [theme.ink2, theme.mpgibbs],
                "size": 11,
                "line": {"width": 2, "color": theme.surface},
            },
            text=["before", "after"],
            textposition=["bottom left", "top right"],
            textfont={"color": theme.ink2},
            showlegend=False,
            hovertemplate="θ %{x:.3f}<br>x %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    _frame_axes(fig, surface, row=1, col=1)
    return _layout(fig, theme, height=440)


# ── 5. LAPS ───────────────────────────────────────────────────────────────────


def laps_figure(surface: Surface, run: LapsRun, theme: Theme, *, step: int) -> go.Figure:
    """The ensemble at one executed step, the tuning it learned, and the handoff.

    ``run.positions`` holds executed steps only: burn-in until the early stop
    (``run.boundary`` steps), then the adjusted steps.
    """
    total = run.positions.shape[0]
    step = int(np.clip(step, 0, total - 1))
    phase = "unadjusted burn-in" if step < run.boundary else "adjusted (MAMS)"
    fig = make_subplots(
        rows=2,
        cols=2,
        column_widths=[0.56, 0.44],
        row_heights=[0.6, 0.4],
        specs=[[{"rowspan": 2}, {}], [None, {}]],
        horizontal_spacing=0.1,
        vertical_spacing=0.16,
        subplot_titles=(
            f"Ensemble at step {step + 1} of {total} ({phase})",
            "Step size (log scale)",
            "Adjusted-phase acceptance",
        ),
    )
    fig.add_trace(_backdrop(surface, theme), row=1, col=1)
    trail = run.positions[max(0, step - 12) : step + 1, :12, :]
    for member in range(trail.shape[1]):
        fig.add_trace(
            go.Scatter(
                x=trail[:, member, 0],
                y=trail[:, member, 1],
                mode="lines",
                line={"color": theme.laps, "width": 1},
                opacity=0.6,
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
    members = run.positions[step]
    fig.add_trace(
        go.Scatter(
            x=members[:, 0],
            y=members[:, 1],
            mode="markers",
            marker={"color": theme.laps, "size": 7, "line": {"color": theme.surface, "width": 1.5}},
            name=f"{members.shape[0]} ensemble members",
            hovertemplate="θ %{x:.2f}<br>x %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    if step == total - 1:
        fig.add_trace(
            go.Scatter(
                x=run.starts[:, 0],
                y=run.starts[:, 1],
                mode="markers",
                marker={
                    "color": "rgba(0,0,0,0)",
                    "size": 15,
                    "line": {"color": theme.ink, "width": 2},
                },
                name="starts handed to the particle chains",
                hovertemplate="start θ %{x:.2f}<br>x %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        centre = float(np.mean(run.starts[:, 0]))
        low = float(surface.x[0]) + 0.06 * float(surface.x[-1] - surface.x[0])
        fig.add_trace(
            go.Scatter(
                x=[centre - run.parameter_chol, centre + run.parameter_chol],
                y=[low, low],
                mode="lines+markers",
                line={"color": theme.ink, "width": 3},
                marker={"symbol": "line-ns", "size": 12, "line": {"width": 3, "color": theme.ink}},
                name="± parameter_chol: the proposal scale handed to mPGibbs",
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
    _frame_axes(fig, surface, row=1, col=1)

    executed = np.arange(1, total + 1)
    fig.add_trace(
        go.Scatter(
            x=executed,
            y=run.step_size,
            mode="lines",
            line={"color": theme.laps, "width": 2},
            showlegend=False,
            hovertemplate="step %{x}<br>step size %{y:.3g}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    for row in (1, 2):
        fig.add_vline(x=run.boundary + 0.5, line={"color": theme.ink2, "width": 1}, row=row, col=2)
        fig.add_vline(x=step + 1, line={"color": theme.ink, "width": 1.5}, row=row, col=2)
    fig.add_annotation(
        x=run.boundary + 0.5,
        y=1.0,
        xref="x2",
        yref="y2 domain",
        text="burn-in → adjusted ",
        showarrow=False,
        xanchor="right",
        yanchor="top",
        font={"color": theme.ink2, "size": 11},
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(run.boundary + 1, run.boundary + run.acceptance.size + 1),
            y=run.acceptance,
            mode="lines+markers",
            line={"color": theme.laps, "width": 2},
            marker={"size": 5},
            showlegend=False,
            hovertemplate="step %{x}<br>acceptance %{y:.2f}<extra></extra>",
        ),
        row=2,
        col=2,
    )
    fig.add_hline(y=0.7, line={"color": theme.ink2, "width": 1, "dash": "dot"}, row=2, col=2)
    fig.update_yaxes(type="log", **_log_ticks(run.step_size), row=1, col=2)
    fig.update_yaxes(range=[0, 1.05], row=2, col=2)
    fig.update_xaxes(range=[0, total + 1], row=1, col=2)
    fig.update_xaxes(range=[0, total + 1], title_text="executed step", row=2, col=2)
    fig.update_annotations(font={"color": theme.ink2, "size": 13})
    return _layout(fig, theme, height=540, legend_below=True)


# ── 6. Comparisons ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Score:
    """Mixing of one sampler at the current setting."""

    name: str
    color: str
    theta_ess_per_sweep: float
    x_ess_per_sweep: float
    note: str


def scoreboard_figure(
    theme: Theme, scores: Sequence[Score], *, gibbs_theory: float | None
) -> go.Figure:
    """ESS per sweep for the parameter and for the latent, one bar per sampler."""
    names = [score.name for score in scores]
    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.04,
        subplot_titles=("θ: ESS per sweep", "x: ESS per sweep"),
    )
    for col, values in (
        (1, [score.theta_ess_per_sweep for score in scores]),
        (2, [score.x_ess_per_sweep for score in scores]),
    ):
        fig.add_trace(
            go.Bar(
                y=names,
                x=values,
                orientation="h",
                marker={"color": [score.color for score in scores], "cornerradius": 4},
                text=[f"{value:.3f}" for value in values],
                textposition="outside",
                textfont={"color": theme.ink2},
                customdata=[score.note for score in scores],
                hovertemplate="%{y}<br>%{x:.3f} per sweep<br>%{customdata}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=col,
        )
        fig.update_xaxes(range=[0, max(1.05, 1.18 * max(values))], row=1, col=col)
    if gibbs_theory is not None:
        fig.add_vline(
            x=gibbs_theory,
            line={"color": theme.ink, "width": 1.5, "dash": "dash"},
            annotation={
                "text": "exact Gibbs, Gaussian theory",
                "font": {"color": theme.ink2, "size": 11},
            },
            row=1,
            col=1,
        )
    fig.update_yaxes(autorange="reversed", row=1, col=1)
    fig.update_annotations(font={"color": theme.ink2, "size": 13})
    return _layout(fig, theme, height=90 + 46 * len(scores))


@dataclass(frozen=True)
class CouplingCurve:
    """ESS per sweep of one sampler across coupling shares."""

    name: str
    color: str
    share: FloatArray
    ess_per_sweep: FloatArray


def coupling_curves(rows: Sequence[CouplingRow], theme: Theme) -> tuple[CouplingCurve, ...]:
    """Group sweep rows by sampler, in entity colours."""
    names = {
        "gibbs": ("exact Gibbs (reference)", theme.gibbs),
        "dsmc": ("exact θ + DSMC (arm G)", theme.dsmc),
        "mpgibbs": ("mPGibbs + DSMC", theme.mpgibbs),
    }
    curves = []
    for key, (name, color) in names.items():
        selected = sorted((row for row in rows if row.sampler == key), key=lambda row: row.share)
        if selected:
            curves.append(
                CouplingCurve(
                    name,
                    color,
                    np.array([row.share for row in selected]),
                    np.array([row.theta_ess_per_sweep for row in selected]),
                )
            )
    return tuple(curves)


def coupling_figure(theme: Theme, curves: Sequence[CouplingCurve]) -> go.Figure:
    """How each sampler's parameter mixing responds to the coupling share."""
    fig = go.Figure()
    grid = np.linspace(0.0, 0.99, 200)
    fig.add_trace(
        go.Scatter(
            x=grid,
            y=(1.0 - grid) / (1.0 + grid),
            mode="lines",
            line={"color": theme.ink, "width": 1.5, "dash": "dash"},
            name="exact Gibbs, Gaussian theory: (1 − s)/(1 + s)",
        )
    )
    for curve in curves:
        fig.add_trace(
            go.Scatter(
                x=curve.share,
                y=curve.ess_per_sweep,
                mode="lines+markers",
                line={"color": curve.color, "width": 2},
                marker={"size": 8, "line": {"color": theme.surface, "width": 2}},
                name=curve.name,
                hovertemplate="s %{x:.2f}<br>%{y:.3f} per sweep<extra>" + curve.name + "</extra>",
            )
        )
        fig.add_annotation(
            x=float(curve.share[-1]),
            y=float(np.log10(max(curve.ess_per_sweep[-1], 1e-4))),
            text=curve.name,
            showarrow=False,
            xanchor="left",
            font={"color": theme.ink2, "size": 11},
        )
    fig.update_xaxes(title_text="coupling share s", range=[0, 1.12])
    fig.update_yaxes(title_text="θ ESS per sweep", type="log")
    return _layout(fig, theme, height=480)


# ── 7. The benchmark itself ───────────────────────────────────────────────────


@dataclass(frozen=True)
class Arm:
    """One arm of the 2026-10-14 T=512 benchmark (report.html, 'Where mixing is limited')."""

    label: str
    family: str
    parameter_ess: float
    state_ess: float
    draws: int


BENCHMARK_ARMS = (
    Arm("A · joint LAPS init → mPGibbs+DSMC", "together: mPGibbs", 9.5, 173.9, 2000),
    Arm("B · cold mPGibbs+DSMC", "together: mPGibbs", 4.3, 32.5, 2000),
    Arm("C · joint LAPS → mPGibbs warmup", "together: mPGibbs", 36.4, 262.9, 2000),
    Arm("E · blocked LAPS init → mPGibbs", "together: mPGibbs", 5.7, 244.3, 2000),
    Arm("D · blocked LAPS + DSMC", "in turn: θ | path, path | θ", 263.2, 290.9, 2000),
    Arm("F · NUTS on θ + DSMC", "in turn: θ | path, path | θ", 1089.4, 275.4, 2000),
    Arm("G · exact θ draw + DSMC", "in turn: θ | path, path | θ", 992.6, 302.1, 2000),
)


def benchmark_figure(theme: Theme) -> go.Figure:
    """Minimum ESS per retained draw for parameters (filled) and latent states (open)."""
    colors = {
        "together: mPGibbs": theme.mpgibbs,
        "in turn: θ | path, path | θ": theme.dsmc,
    }
    fig = go.Figure()
    for family, color in colors.items():
        arms = [arm for arm in BENCHMARK_ARMS if arm.family == family]
        fig.add_trace(
            go.Scatter(
                x=[arm.parameter_ess / arm.draws for arm in arms],
                y=[arm.label for arm in arms],
                mode="markers",
                marker={"color": color, "size": 12, "line": {"color": theme.surface, "width": 2}},
                name=f"{family}: parameters",
                customdata=[arm.parameter_ess for arm in arms],
                hovertemplate="%{y}<br>parameter min ESS %{customdata:.1f}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[arm.state_ess / arm.draws for arm in arms],
                y=[arm.label for arm in arms],
                mode="markers",
                marker={"color": color, "size": 12, "symbol": "circle-open", "line": {"width": 2}},
                name=f"{family}: latent states",
                customdata=[arm.state_ess for arm in arms],
                hovertemplate="%{y}<br>state min ESS %{customdata:.1f}<extra></extra>",
            )
        )
    fig.add_vline(
        x=4672.5 / 4000,
        line={"color": theme.nuts, "width": 2},
        annotation={
            "text": "golden joint NUTS (all 1,026 coordinates)",
            "font": {"color": theme.ink2, "size": 11},
            "xanchor": "right",
        },
    )
    fig.update_xaxes(type="log", title_text="minimum bulk ESS per retained draw")
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, theme, height=420)
