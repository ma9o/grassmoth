import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def imports_marimo():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def intro(mo):
    mo.md(r"""
    # Why our path update can't move smooth shapes, and what the Kalman move changes

    On 2026-10-23 we refit C0 three ways; only the latent path update differed:

    | Arm | Production gate | Worst x ESS (bulk / tail) | Worst y ESS (bulk / tail) |
    | :--- | :--- | ---: | ---: |
    | Today's sampler (A0) | fails | 434 / 333 | 194 / 268 |
    | A0 with smoother marginals | fails | 501 / 369 | 1,246 / 998 |
    | A0 + Kalman path move | **passes** | 12,105 / 11,775 | 12,210 / 11,984 |

    This notebook shows why, on a toy you can change: one path shaped like C0's `x`
    (decay 0.6 per day, an input that flips every half day with effect 0.5, diffusion
    0.45, readings with noise 0.2), over two days. Three exact updates run on it:

    - **Separable local**, today's dSMC proposal: candidates drawn near the current
      path, independently at each step.
    - **Separable smoother marginals**, the dSMC paper's §4.1 proposal: candidates drawn
      from a Gaussian smoother's estimate at each step, still independently.
    - **Kalman path move**: one whole path drawn from a Gaussian model of the dynamics
      and readings, then accepted or rejected against the exact model.

    Both separable updates stitch their candidates with forward filtering and backward
    sampling: the exact distribution over every combination of candidates. No stitcher
    does better, dSMC's tree included.
    """)
    return


@app.cell
def imports_lab():
    import numpy as np

    import notebooks.support.latent_path_proposals as lab

    return lab, np


@app.cell
def controls(mo):
    grid = mo.ui.dropdown(
        options={"16": 16, "32": 32, "64": 64, "128 (C0)": 128, "256": 256},
        value="128 (C0)",
        label="grid intervals per day",
    )
    every = mo.ui.dropdown(
        options={"every step (C0)": 1, "every 4th step": 4, "every 16th step": 16},
        value="every step (C0)",
        label="readings",
    )
    particles = mo.ui.dropdown(
        options={"8": 8, "16": 16, "32": 32, "64": 64},
        value="32",
        label="candidates per step",
    )
    delta = mo.ui.dropdown(
        options={"0.01": 0.01, "0.03": 0.03, "0.1": 0.1, "0.3": 0.3, "1": 1.0},
        value="0.1",
        label="δ, separable local",
    )
    mo.hstack([grid, every, particles, delta], justify="start", gap=1.5)
    return delta, every, grid, particles


@app.cell
def toy_posterior(every, grid, lab):
    toy = lab.make_toy(intervals_per_day=grid.value, every=every.value)
    model = lab.reference_model(toy)
    start = lab.posterior_draws(model, 1, seed=1)[0]
    return model, start, toy


@app.cell(hide_code=True)
def shapes_md(mo):
    mo.md(r"""
    ## 1. The posterior and its two kinds of shape

    Any path is a sum of cosine shapes, from one slow swing across the window to an
    up-down alternation every step. Below, three posterior draws are split at a period of
    half a day. At any one time both parts are about the same size, but they behave
    differently. The **smooth part** drifts in long, coherent swings and is pinned only
    by the noisy readings. The **wiggly part** is rough and short-lived, and is pinned by
    the dynamics. Per shape (bottom right), the smoothest shape varies about a hundred
    times more than the wiggliest on C0's grid. A finer grid or sparser readings stretch
    that further.
    """)
    return


@app.cell
def shapes_view(lab, model, toy):
    lab.fig_posterior_shapes(toy, model, lab.posterior_draws(model, 3, seed=2))
    return


@app.cell
def shapes_numbers(mo, model, np, toy):
    _sd = np.sqrt(np.diag(model.covariance))
    _covariance = model.covariance
    _correlation = float(
        np.median(
            _covariance[np.arange(toy.steps - 1), np.arange(1, toy.steps)] / (_sd[:-1] * _sd[1:])
        )
    )
    mo.md(
        f"{toy.steps} grid steps. Each state is known to about ±{np.median(_sd):.3f} "
        f"(one posterior SD), and neighbouring states are {_correlation:.0%} correlated."
    )
    return


@app.cell(hide_code=True)
def updates_md(mo):
    mo.md(r"""
    ## 2. Each update: locally, globally and by shape

    One row per update. Read across each row:

    - **Locally:** what the update proposes in a 30-step stretch, all rows on the same
      scale. The separable updates propose a cloud of values at each step (grey dots),
      independently of the neighbouring steps, and stitch one per step into the new
      path. The Kalman move proposes whole paths (thin lines) already shaped by the
      dynamics.
    - **Globally:** the smooth part of the path, minus the posterior mean, over 40
      consecutive sweeps, one row per sweep. Vertical streaks are a smooth shape the
      update can't shift. A pattern that changes every row means a fresh smooth shape
      every sweep.
    - **By shape:** the fraction of each shape's posterior variance refreshed per sweep,
      about 1 − lag-1 autocorrelation. 1 means a fresh draw every sweep.
    """)
    return


@app.cell
def update_runs(delta, lab, model, particles, start, toy):
    views = lab.update_views(toy, model, start, particles=particles.value, delta=delta.value)
    local_range = lab.local_range(toy, start, views)
    strip_scale = lab.smooth_scale(toy, model)
    return local_range, strip_scale, views


@app.cell
def local_row(lab, local_range, model, start, strip_scale, toy, views):
    lab.fig_update_row(
        toy, model, start, views["local"], local_range=local_range, zmax=strip_scale, first=True
    )
    return


@app.cell
def marginal_row(lab, local_range, model, start, strip_scale, toy, views):
    lab.fig_update_row(
        toy, model, start, views["marginal"], local_range=local_range, zmax=strip_scale, first=False
    )
    return


@app.cell
def kalman_row(lab, local_range, model, start, strip_scale, toy, views):
    lab.fig_update_row(
        toy, model, start, views["kalman"], local_range=local_range, zmax=strip_scale, first=False
    )
    return


@app.cell
def update_table(lab, mo, model, np, toy, views):
    _sd = np.sqrt(np.diag(model.covariance))
    _rows = []
    for _key, _view in views.items():
        _smooth, _wiggly = lab.shape_summary(lab.refresh(_view.chain, model), toy)
        _error = np.max(np.abs(_view.chain.mean(axis=0) - model.mean) / _sd)
        _acceptance = "" if _view.acceptance is None else f"{_view.acceptance:.2f}"
        _rows.append(
            f"| {lab.NAMES[_key]} | {_smooth:.2f} | {_wiggly:.2f} | {_acceptance} | {_error:.2f} |"
        )
    mo.md(
        "\n".join(
            [
                "| Update | Smooth refresh | Wiggle refresh | Acceptance | Largest mean error (SDs) |",
                "| :--- | ---: | ---: | ---: | ---: |",
                *_rows,
                "",
                "All three updates agree with the exact posterior; the last column is Monte "
                "Carlo error only. So dSMC isn't implemented wrong, it is slow on smooth "
                "shapes. A refresh of r per sweep means roughly (2 − r)/r sweeps per "
                "independent draw of that shape.",
            ]
        )
    )
    return


@app.cell(hide_code=True)
def knobs_md(mo):
    mo.md(r"""
    ## 3. What changes the picture

    - **δ (left).** Small δ keeps candidates next to the current path, so moves are tiny.
      Large δ scatters them, so few line up and stitching keeps the current path. No δ
      comes close to the Kalman move. Our sampler tunes δ to a 50% update rate, which
      can't see this; in C0 it pushed δ to its cap at 80–92% of steps.
    - **Grid density and readings (middle).** Tighter links between neighbouring states
      starve separable proposals, while the Kalman move stays near 1. This is the
      prediction for C2 (sparse readings), C4 (twice the grid) and France (a daily grid
      with monthly readings). The panel shows a stored scan; the button recomputes it in
      about a minute.
    - **A frozen Kalman model (right).** The move accepts or rejects the whole path, so
      it is only as good as its Gaussian model. Frozen at the wrong input effect, its
      acceptance (dotted) and smooth refresh fall, though here they stay above the
      separable proposal. Rebuilding the model every sweep at the current parameters
      avoids this; that is the open decision. Readings far from Gaussian, or a few far
      more informative than the rest, hurt it the same way (Corenflos and Särkkä, §3 and
      §4.4). That is why the C0 run kept dSMC's local updates and added the Kalman move
      after them.
    """)
    return


@app.cell
def grid_button(mo):
    recompute = mo.ui.run_button(label="Recompute the grid scan")
    mo.vstack([recompute])
    return (recompute,)


@app.cell
def knobs_view(lab, model, particles, recompute, start, toy, views):
    _deltas = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
    _delta_smooth = [
        lab.shape_summary(
            lab.refresh(
                lab.run_separable(toy, start, sweeps=350, particles=particles.value, delta=_value)[
                    50:
                ],
                model,
            ),
            toy,
        )[0]
        for _value in _deltas
    ]
    _current = {
        _key: lab.shape_summary(lab.refresh(_view.chain, model), toy)[0]
        for _key, _view in views.items()
    }
    _scan = lab.GRID_SCAN
    if recompute.value:
        _scan = {
            "intervals_per_day": lab.GRID_SCAN["intervals_per_day"],
            "dense": lab.scan_grid(1),
            "sparse": lab.scan_grid(16),
        }
    _effects = [0.5, 0.4, 0.3, 0.1, -0.1]
    _acceptance, _smooth = [], []
    for _effect in _effects:
        _paths, _rate = lab.run_kalman(toy, start, lab.reference_model(toy, _effect), sweeps=350)
        _acceptance.append(_rate)
        _smooth.append(lab.shape_summary(lab.refresh(_paths[50:], model), toy)[0])
    lab.fig_knobs(_deltas, _delta_smooth, _current, _scan, _effects, _acceptance, _smooth)
    return


@app.cell(hide_code=True)
def c0_md(mo):
    mo.md(r"""
    ## 4. C0, and how we got here

    The same per-shape measurement on the real C0 fits: 2,049 steps per path, the README
    schedule (128 particles, 4 chains, 2,000 + 4,000 draws, seed 19), one A10 each, on
    byte-identical data. Today's sampler redraws one of the two paths per sweep, so it
    tops out near 0.5 even for wiggles, and its tree stitches less well than the toy's
    exact stitcher. The Kalman arm adds a joint move of both paths after every sweep. It
    accepted 92% of its proposals at δ = 10⁴ (its cap) for about 7% more sampling time.
    The three fits cost $0.54 together; their records are in
    `scratchpad/latent-proposals-20261023/`.
    """)
    return


@app.cell
def c0_view(lab):
    lab.fig_c0_bands()
    return


@app.cell(hide_code=True)
def history_md(mo):
    mo.md(r"""
    **How we ended up without a dynamics-aware update.** The repo had one in April 2026 and
    traded it for parallel depth while consolidating:

    | Date | Commit | Change |
    | :--- | :--- | :--- |
    | Apr 17–24 | `15c232c2`, `7d6bf481`, `a2a7887d` | Auxiliary Kalman samplers, a parallel Kalman filter, and Particle-mGRAD with sequential conditional SMC, whose proposals used the dynamics |
    | May 12 | `019ec808` | Particle-mGRAD's path update moved to parallel dSMC for O(log T) depth; dSMC needs proposals independent across time, so the dynamics moved out of the proposal and into the stitching weights |
    | May 21 | `3e8687f0` | The composite aux-Kalman experiment was deleted; the commit records no reason |
    | May 29 – Jun 3 | `5ff65e5b`, `7b9e778b` | Inference consolidated onto MPGibbs + dSMC |
    | Jun 22, Oct 14 | `5393cd42`, `5b0d7f88` | Guard against linearized likelihood backends; Laplace/IEKS initialization removed |

    The trade wasn't necessary. The auxiliary Kalman move draws its whole path with a
    parallel Kalman filter (an associative scan, the same O(log T) depth as dSMC) and
    corrects it exactly.

    **Open decisions:**
    - Whether the move's Gaussian model may be rebuilt every sweep at the current
      parameters. The rule today allows linearization only at initialization, so the C0
      run froze it after warmup.
    - Hardening the move before it joins the sampler: tests, lint and the fit chart, plus
      support for interval readings, replayed accumulators, exact readings on free
      coordinates and tempering, all of which France needs.

    References: dSMC, [arXiv 2202.02264](https://arxiv.org/abs/2202.02264) §4; auxiliary
    Kalman samplers, [arXiv 2303.00301](https://arxiv.org/abs/2303.00301) §2–4;
    Particle-MALA and Particle-mGRAD, [arXiv 2401.14868](https://arxiv.org/abs/2401.14868) §4.
    """)
    return


if __name__ == "__main__":
    app.run()
