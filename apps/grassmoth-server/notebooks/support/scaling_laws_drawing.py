"""The notebook's hand-drawn pictures: one lane per method, chain limits per GPU, and the law grid.

Everything here is a function of the laws and the evidence, so it refits with them. Readings sit in
fixed rows outside the marks, so the numbers stay put as settings change. Headings are ARIA headings
rather than h1 and h2, because marimo would otherwise float an outline beside so short a page.
"""

from __future__ import annotations

import html
import math
from dataclasses import replace
from typing import TYPE_CHECKING

from notebooks.support import scaling_laws as sl

if TYPE_CHECKING:
    from notebooks.support.scaling_laws import Evidence, FitState, Laws, Method, Scenario

HUES: dict[Method, str] = {
    "mgrad": "#2a78d6",
    "kalman_whole": "#eb6834",
    "kalman_blocks": "#1baf7a",
}
STATE_WORDS: dict[FitState, str] = {
    "fits": "Fits",
    "rejected": "Planner rejects",
    "after_sampling": "Fails after sampling",
    "over_pool": "Doesn't fit",
}

WIDTH = 888
GUTTER = 148
RIGHT_MARGIN = 4
COLUMNS = 4

HEADER = 24
READING = 26
BAND = 46
SPLIT = 4
LOWER = 24
LANE_GAP = 16
CAP = 8

PLOT_RIGHT = 636
AXIS_DECADES = 4
ROW_HEIGHT = 20
GROUP_GAP = 8
VALUE_COLUMNS = (("50%", 690), ("90%", 748), ("Planner", 816), ("Pool", 884))


def _column_x(column: float) -> float:
    """Left edge of a stage column; fractions land inside it."""
    return GUTTER + (WIDTH - GUTTER - RIGHT_MARGIN) / COLUMNS * column


def _hatch(pattern_id: str, hue: str) -> str:
    """Light diagonal strokes: evidence shouldn't shout over the quantity."""
    return (
        f'<pattern id="{pattern_id}" width="10" height="10" patternUnits="userSpaceOnUse" '
        'patternTransform="rotate(45)">'
        f'<rect width="10" height="10" fill="{hue}" fill-opacity="0.08"/>'
        f'<rect width="1.4" height="10" fill="{hue}" fill-opacity="0.75"/></pattern>'
    )


def _text(x: float, y: float, content: str, classes: str, anchor: str = "middle") -> str:
    escaped = html.escape(content, quote=False)
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" class="{classes}">{escaped}</text>'
    )


def _line(x1: float, y1: float, x2: float, y2: float, style: str) -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" style="{style}"/>'


def _reading(center: float, y: float, big: str, small: str, big_class: str = "big") -> str:
    """A number and its unit on one line, centred in its column."""
    first = f'<tspan class="{big_class}">{html.escape(big, quote=False)}</tspan>' if big else ""
    gap = ' dx="6"' if big else ""
    return (
        f'<text x="{center:.1f}" y="{y:.1f}" text-anchor="middle">{first}'
        f'<tspan class="small"{gap}>{html.escape(small, quote=False)}</tspan></text>'
    )


def _overflow_cap(column: int, pool_y: float) -> str:
    """A plain red cap with a torn top: the real height doesn't fit the picture."""
    left, right = _column_x(column), _column_x(column + 1)
    top = pool_y - CAP
    points = [f"{left:.1f},{pool_y:.1f}"]
    teeth = max(2, round((right - left) / 12))
    for index in range(teeth + 1):
        x = left + (right - left) * index / teeth
        points.append(f"{x:.1f},{top + (4 if index % 2 else 0):.1f}")
    points.append(f"{right:.1f},{pool_y:.1f}")
    return f'<polygon points="{" ".join(points)}" style="fill:var(--sl-critical)"/>'


def _lane(scenario: Scenario, laws: Laws, evidence: Evidence, top: float) -> str:
    """One method: memory rises towards the pool, processing hangs towards its limit."""
    rows = sl.estimate(scenario, laws, evidence)
    speed = sl.scenario_speed(scenario, laws, evidence)
    limits = sl.scenario_limits(scenario, laws, evidence)
    state = sl.fit_state(scenario, limits)
    trace = sl.matching_trace(scenario, evidence)
    pool_gib = evidence.gpu(scenario.gpu).pool_gib
    method = scenario.method
    hue = HUES[method]

    pool_y = top + READING
    base = pool_y + BAND
    drop_top = base + SPLIT
    limit_y = drop_top + BAND
    left, right = _column_x(0), _column_x(COLUMNS)
    budget_y = base - sl.PLANNING_FRACTION * BAND
    solid = f"fill:{hue};fill-opacity:0.22"
    hatched = f"fill:url(#sl-hatch-{method})"
    edge = f"stroke:{hue};stroke-width:2"
    rule = "stroke:var(--sl-ink);stroke-width:1.2"
    faint = "stroke:var(--sl-line);stroke-width:1"
    bad = state in {"after_sampling", "over_pool"}

    parts = [
        f'<rect x="0" y="{pool_y:.1f}" width="3" height="{limit_y - pool_y:.1f}" style="fill:{hue}"/>',
        _text(11, pool_y + 13, sl.LANE_LABELS[method], "lane", "start"),
        _text(11, pool_y + 32, STATE_WORDS[state], "state bad" if bad else "state", "start"),
        _text(11, pool_y + 49, f"Planner limit {limits.planner:,}", "small", "start"),
        _text(left + 6, pool_y + 14, "Memory", "band", "start"),
        _text(left + 6, drop_top + 15, "Processing", "band", "start"),
    ]
    parts += [
        _line(_column_x(column), pool_y, _column_x(column), limit_y, faint)
        for column in range(1, COLUMNS)
    ]

    outline = [f"M{left:.1f},{base:.1f}"]
    for column, row in enumerate(rows):
        lift = max(1.5, min(1.0, row.pool_share) * BAND)
        outline += [
            f"L{_column_x(column):.1f},{base - lift:.1f}",
            f"L{_column_x(column + 1):.1f},{base - lift:.1f}",
        ]
    outline.append(f"L{right:.1f},{base:.1f}Z")
    fill = solid if sl.memory_is_measured(scenario) else hatched
    parts.append(f'<path d="{" ".join(outline)}" style="{fill};{edge}"/>')

    if trace is not None:
        for column, points in enumerate(trace.stages):
            if len(points) < 2:
                continue
            drawn = " ".join(
                f"{_column_x(column + share):.1f},{base - min(1.0, gib / pool_gib) * BAND:.1f}"
                for share, gib in points
            )
            parts.append(
                f'<polyline points="{drawn}" '
                'style="fill:none;stroke:var(--sl-ink);stroke-width:1;opacity:0.85"/>'
            )

    parts += [
        _line(left, pool_y, right, pool_y, rule),
        _line(
            left,
            budget_y,
            right,
            budget_y,
            "stroke:var(--sl-secondary);stroke-width:0.8;stroke-dasharray:3 4",
        ),
        _line(left, limit_y, right, limit_y, rule),
    ]

    for column, row in enumerate(rows):
        center = _column_x(column + 0.5)
        if row.pool_share > 1:
            parts.append(_overflow_cap(column, pool_y))
            parts.append(
                _reading(
                    center,
                    pool_y - 11,
                    sl.format_share(row.pool_share),
                    f"{row.memory_gib:,.1f} GiB, over pool",
                    "big over",
                )
            )
        else:
            parts.append(
                _reading(
                    center,
                    pool_y - 7,
                    sl.format_share(row.pool_share),
                    f"{row.memory_gib:,.2f} GiB",
                )
            )
        if row.fill is None:
            parts.append(_reading(center, limit_y + 18, "", f"{row.idle}, {row.time}"))
            continue
        drop = row.fill * BAND
        fill = solid if speed.measured else hatched
        x, width = _column_x(column), _column_x(1) - _column_x(0)
        parts.append(
            f'<rect x="{x:.1f}" y="{drop_top:.1f}" width="{width:.1f}" height="{drop:.1f}" style="{fill}"/>'
        )
        parts.append(_line(x, drop_top + drop, x + width, drop_top + drop, edge))
        parts.append(_reading(center, limit_y + 18, f"{100 * row.fill:.0f}%", row.time))
    return "".join(parts)


def method_lanes(scenario: Scenario, laws: Laws, evidence: Evidence) -> str:
    """All three methods at these settings, under one set of stage headings."""
    variants = sl.each_method(scenario)
    lane = READING + BAND + SPLIT + BAND + LOWER
    height = HEADER + lane * len(variants) + LANE_GAP * (len(variants) - 1)
    stages = [row.stage for row in sl.estimate(scenario, laws, evidence)]
    parts = [
        f'<svg class="sl-drawing" viewBox="0 0 {WIDTH} {height}" role="img" '
        'aria-label="GPU memory and processing fill at each stage, for each method">',
        "<defs>",
        *(_hatch(f"sl-hatch-{method}", hue) for method, hue in HUES.items()),
        "</defs>",
    ]
    parts += [
        _text(_column_x(column + 0.5), 15, stage, "stage") for column, stage in enumerate(stages)
    ]
    for index, variant in enumerate(variants):
        parts.append(_lane(variant, laws, evidence, HEADER + index * (lane + LANE_GAP)))
    parts.append("</svg>")
    return "".join(parts)


def _ruler_x(chains: float) -> float:
    """Chains on the shared log axis, clamped to it."""
    share = math.log10(max(1.0, min(10.0**AXIS_DECADES, chains))) / AXIS_DECADES
    return GUTTER + share * (PLOT_RIGHT - GUTTER)


def _count(chains: float) -> str:
    return "<1" if chains < 1 else f"{chains:,.0f}"


def gpu_ruler(scenario: Scenario, laws: Laws, evidence: Evidence) -> str:
    """Per method and GPU: how many chains fill the processing, and where memory stops them."""
    names = evidence.offered_gpus
    top = 26
    variants = sl.each_method(scenario)
    bottom = top + ROW_HEIGHT * len(names) * len(variants) + GROUP_GAP * (len(variants) - 1)
    height = bottom + 22
    faint = "stroke:var(--sl-line);stroke-width:1"
    parts = [
        f'<svg class="sl-drawing" viewBox="0 0 {WIDTH} {height}" role="img" '
        'aria-label="Where processing and memory limits fall by chain count, per method and GPU">',
        "<defs>",
        *(_hatch(f"sl-ruler-hatch-{method}", hue) for method, hue in HUES.items()),
        "</defs>",
    ]
    for decade in range(AXIS_DECADES + 1):
        tick = 10**decade
        x = _ruler_x(tick)
        anchor = "end" if decade == AXIS_DECADES else "start" if decade == 0 else "middle"
        parts.append(_line(x, top, x, bottom, faint))
        parts.append(_text(x, bottom + 16, f"{tick:,}", "small", anchor))
    parts.append(_text(GUTTER - 12, bottom + 16, "Chains", "small", "end"))
    for label, x in VALUE_COLUMNS:
        parts.append(_text(x, 14, label, "heading", "end"))

    y_top = float(top)
    for variant in variants:
        method = variant.method
        hue = HUES[method]
        group_height = ROW_HEIGHT * len(names)
        parts.append(
            f'<rect x="0" y="{y_top + 2:.1f}" width="3" height="{group_height - 4}" style="fill:{hue}"/>'
        )
        parts.append(_text(11, y_top + ROW_HEIGHT * 0.7, sl.LANE_LABELS[method], "lane", "start"))
        for index, name in enumerate(names):
            at_gpu = replace(variant, gpu=name)
            limits = sl.scenario_limits(at_gpu, laws, evidence)
            speed = sl.scenario_speed(at_gpu, laws, evidence)
            y = y_top + ROW_HEIGHT * index + ROW_HEIGHT / 2
            strong = " strong" if name == scenario.gpu else ""
            low, high = _ruler_x(limits.half_fill), _ruler_x(limits.high_fill)
            planner, pool = _ruler_x(limits.planner), _ruler_x(limits.pool)
            parts.append(_text(GUTTER - 12, y + 4, name, f"gpu{strong}", "end"))
            parts.append(_line(GUTTER, y, PLOT_RIGHT, y, faint))
            reachable = min(high, planner)
            fill = f"fill:{hue}" if speed.measured else f"fill:url(#sl-ruler-hatch-{method})"
            if low < reachable:
                parts.append(
                    f'<rect x="{low:.1f}" y="{y - 4.5:.1f}" width="{reachable - low:.1f}" height="9" style="{fill}"/>'
                )
            if high > reachable:
                start = max(low, reachable)
                parts.append(
                    f'<rect x="{start:.1f}" y="{y - 4.5:.1f}" width="{high - start:.1f}" height="9" '
                    f'style="fill:none;stroke:{hue};stroke-width:1;stroke-dasharray:3 3"/>'
                )
            parts.append(
                _line(planner, y - 7.5, planner, y + 7.5, "stroke:var(--sl-ink);stroke-width:2.5")
            )
            parts.append(_line(pool, y - 7.5, pool, y + 7.5, "stroke:var(--sl-ink);stroke-width:1"))
            values = (limits.half_fill, limits.high_fill, limits.planner, limits.pool)
            for (_, x), value in zip(VALUE_COLUMNS, values, strict=True):
                parts.append(_text(x, y + 4, _count(value), f"value{strong}", "end"))
        y_top += group_height + GROUP_GAP

    chosen = _ruler_x(scenario.chains)
    parts.append(
        _line(
            chosen,
            top - 6,
            chosen,
            bottom,
            "stroke:var(--sl-ink);stroke-width:1.5;stroke-dasharray:1 3",
        )
    )
    parts.append(_text(chosen, 14, f"{scenario.chains:,} chains", "small strong"))
    parts.append("</svg>")
    return "".join(parts)


def heading(title: str, caption: str) -> str:
    """A one-sentence heading and a short caption. The caption may carry inline markup."""
    return (
        '<div class="sl"><div class="sl-section" role="heading" aria-level="2">'
        f'{html.escape(title, quote=False)}</div><p class="sl-caption">{caption}</p></div>'
    )


def first_screen(scenario: Scenario, laws: Laws, evidence: Evidence) -> str:
    """The answer, one lane per method, and how many chains each GPU could take."""
    gpu = evidence.gpu(scenario.gpu)
    head = html.escape(sl.headline(scenario, laws, evidence), quote=False)
    below = html.escape(sl.detail(scenario), quote=False)
    budget = sl.PLANNING_FRACTION * gpu.pool_gib
    bases = "".join(
        f'<li><b style="color:{HUES[variant.method]}">{sl.LANE_LABELS[variant.method]}</b> '
        f"{html.escape(sl.lane_basis(variant, laws, evidence), quote=False)}</li>"
        for variant in sl.each_method(scenario)
    )
    return f"""<div class="sl">
    <p class="sl-title">Scaling laws for GPU fits, from {len(evidence.runs)} saved runs</p>
    <div class="sl-headline" role="heading" aria-level="1">{head}</div>
    <p class="sl-detail">{below}</p>
    {method_lanes(scenario, laws, evidence)}
    <p class="sl-basis">Top line: the {gpu.pool_gib:.0f} GiB pool, with the {budget:.0f} GiB planner budget dashed.
    Bottom line: the throughput limit. Solid is measured; hatched is scaled from another GPU, another model or the code.
    A thin line is a saved run's real memory trace.</p>
    <details class="sl-details"><summary>What backs each lane</summary><ul>{bases}</ul></details>
    <div class="sl-section" role="heading" aria-level="2">How many chains each GPU can take</div>
    <p class="sl-caption">Bar: 50% to 90% of the throughput limit, solid until the planner stops it.
    Heavy stop: planner. Light stop: GPU pool.</p>
    {gpu_ruler(scenario, laws, evidence)}
    </div>"""


def law_grid(scenario: Scenario, laws: Laws) -> str:
    """How each setting scales each quantity, for all three methods side by side."""
    top = "".join(
        f'<th class="gap"></th><th class="group" colspan="{len(sl.SCALING_COLUMNS)}" '
        f'style="--sl-hue:{HUES[method]}">{sl.LANE_LABELS[method]}</th>'
        for method in sl.METHODS
    )
    columns = "".join(f"<th>{name}</th>" for name in sl.SCALING_COLUMNS)
    below = "".join(f'<th class="gap"></th>{columns}' for _ in sl.METHODS)
    rows = []
    for row in sl.scaling_rows(scenario, laws):
        cells = []
        for method in sl.METHODS:
            cells.append('<td class="gap"></td>')
            if method not in row.cells:
                cells.append(
                    f'<td class="unused" colspan="{len(sl.SCALING_COLUMNS)}">not used</td>'
                )
                continue
            cells.extend(
                f'<td class="{cell.backing}" style="--sl-hue:{HUES[method]}">{cell.text}</td>'
                if cell.text
                else '<td class="flat">–</td>'
                for cell in row.cells[method]
            )
        rows.append(f"<tr><th>{row.setting}</th>{''.join(cells)}</tr>")
    return f"""<div class="sl" style="--sl-hue:#5a6673">
    <div class="sl-section" role="heading" aria-level="2">What each setting scales</div>
    <p class="sl-caption">Mem is memory while sampling, a is latency, b is per-chain cost, After is after sampling, Comp is compile.
    A dash is flat. Spec means cores × clock, plus DRAM for mGRAD. The tabs below show the runs behind each law.</p>
    <table class="sl-grid"><thead><tr><th></th>{top}</tr><tr><th></th>{below}</tr></thead>
    <tbody>{"".join(rows)}</tbody></table>
    <p class="sl-key"><span class="sl-swatch measured"></span>Measured<span class="sl-swatch pair"></span>One pair of models<span class="sl-swatch code"></span>Code only</p>
    </div>"""
