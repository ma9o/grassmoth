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
    # How path proposals work, by type of latent

    On 2026-10-23 we refit C0 three ways; only the update of the latent paths differed:

    | Arm | Production gate | Worst x ESS (bulk / tail) | Worst y ESS (bulk / tail) |
    | :--- | :--- | ---: | ---: |
    | Today's sampler (A0) | fails | 434 / 333 | 194 / 268 |
    | A0 with smoother marginals | fails | 501 / 369 | 1,246 / 998 |
    | A0 + Kalman path move | **passes** | 12,105 / 11,775 | 12,210 / 11,984 |

    This notebook explains that result from the ground up, by looking straight at the
    posterior of the latent path. It has five parts:

    1. **Two steps.** Cut the path down to two neighbouring steps. The posterior is then
       a picture in a plane, so every idea can be drawn.
    2. **Every proposal on every type of latent.** Seven ways to update the path, each
       tried on four kinds of latent: wiggly, smooth, very smooth, and smooth with binary
       readings.
    3. **A whole path.** The same seven updates on 256 steps, where "smooth" and "wiggly"
       become shapes over time.
    4. **What limits each family.**
    5. **C0, and how we got here.**

    The toy is C0's `x` path: decay 0.6 per day, an input that flips every half day with
    effect 0.5, diffusion 0.45 and readings with noise 0.2, on C0's grid of 128 steps per
    day. Every update here is *exact*: run long enough, each one samples the true
    posterior. They differ in how fast they explore it, and the pictures show why.
    """)
    return


@app.cell
def imports_lab():
    import numpy as np

    import notebooks.support.latent_path_proposals as lab

    return lab, np


@app.cell(hide_code=True)
def model_md(mo):
    mo.md(r"""
    ## Part 1 · Two steps: a posterior you can see

    ### 1. Where the posterior comes from

    Take two neighbouring steps of the path, x₁ and x₂, one grid interval apart (11¼
    minutes on C0's grid). Two things inform them:

    - **The dynamics.** x₂ is x₁ plus a small drift plus noise. On C0's grid that noise
      has SD 0.45 · √(1/128) ≈ 0.04, so the dynamics say *x₂ ≈ x₁, to within about
      ±0.04*. They say little about the level itself: before any reading, x₁ is only
      known to ±0.4.
    - **The readings.** Each step has a noisy reading, y₁ and y₂, with noise SD 0.2.

    Below, every point (x₁, x₂) of the plane is a whole two-step path, and the diagonal
    x₁ = x₂ holds the flat paths. The posterior is the product of the two ingredients:
    the dynamics squeeze it onto the diagonal, and the readings cut a stretch out of it.
    """)
    return


@app.cell
def latent_types(lab):
    planes = (
        lab.make_plane("Wiggly: ½ day per step", intervals_per_day=2),
        lab.make_plane("Smooth: C0's grid", intervals_per_day=128),
        lab.make_plane("Very smooth: 16× finer", intervals_per_day=2048),
        lab.make_plane("Smooth, binary readings", intervals_per_day=128, binary=True),
    )
    c0 = planes[1]
    return c0, planes


@app.cell
def model_view(c0, lab):
    lab.fig_two_step_model(c0)
    return


@app.cell(hide_code=True)
def model_numbers(c0, lab, mo, np):
    _variances, _ = lab.axes(c0.model)
    _covariance = c0.model.covariance
    _correlation = _covariance[0, 1] / np.sqrt(_covariance[0, 0] * _covariance[1, 1])
    mo.md(
        f"""
    The posterior's long axis has SD {np.sqrt(_variances[1]):.2f} and its short axis
    {np.sqrt(_variances[0]):.3f}: the ellipse is {np.sqrt(_variances[1] / _variances[0]):.1f}
    times longer than wide, and the two steps are {_correlation:.0%} correlated.

    **Two directions, two kinds of shape.** The long axis moves both steps together: it
    shifts the level of the path. Call it the **smooth** direction. The short axis moves
    the steps apart: a **wiggle**. The readings pin the smooth direction only loosely,
    because each one is noisy. The dynamics pin the wiggle tightly, because neighbouring
    steps can't differ by much. The right-hand panel draws each axis as a two-step path.
    """
    )
    return


@app.cell(hide_code=True)
def types_md(mo):
    mo.md(r"""
    ### 2. Four types of latent

    How long and thin the ellipse gets depends on how tightly the dynamics tie
    neighbouring steps together, compared with how much the readings say. That is what
    *type of latent* means below:

    - **Wiggly**: steps half a day apart. The dynamics barely link them, so the posterior
      is nearly round, and each step is pinned mostly by its own reading.
    - **Smooth**: C0's grid, the ellipse from section 1.
    - **Very smooth**: a grid 16 times finer. Neighbours are tied 4 times tighter, so the
      ellipse is about 4 times thinner. Sparser readings or a smaller diffusion do the
      same.
    - **Smooth, binary readings**: C0's grid, but each reading only says whether the
      state is above zero, through a soft threshold. Both readings say "above", so the
      posterior is cut off just below zero and trails off up the diagonal: it isn't
      Gaussian. France has readings like these.
    """)
    return


@app.cell
def types_view(lab, planes):
    lab.fig_latents(planes)
    return


@app.cell(hide_code=True)
def auxiliary_md(mo):
    mo.md(r"""
    ### 3. How an exact update moves: jitter, then redraw

    An MCMC update takes the current path and returns a new one. Repeated over and over,
    the paths it visits become draws from the posterior. Six of the seven updates in this
    notebook work in two stages:

    1. **Jitter.** Draw an *auxiliary point* u: the current path plus independent noise
       of variance δ/2 at each step. δ is the update's step size.
    2. **Redraw.** Draw the new path from the posterior *times* a bump centred on u (the
       dashed ellipses below). The new path lands where the bump and the posterior
       overlap.

    Stage 2 is where the updates differ. Stage 1 already limits all of them: the bump
    pulls the new path towards u, and u remembers where the current path was. Even a
    perfect stage 2 can renew, along a direction with posterior variance v, at most

    $$\text{refresh} \le \frac{\delta}{\delta + 2v}.$$

    **Refresh** is the fraction of a direction renewed per update: 1 means every update
    is a fresh, independent posterior draw along it, and 0 means it never moves. It is
    the mean squared jump over twice the variance, about 1 minus the lag-1
    autocorrelation; a refresh of r takes roughly (2 − r)/r updates per independent
    draw.
    """)
    return


@app.cell
def auxiliary_view(c0, lab):
    lab.fig_auxiliary(c0, (0.01, 0.1, 1.0))
    return


@app.cell(hide_code=True)
def auxiliary_numbers(c0, lab, mo):
    _variances, _ = lab.axes(c0.model)
    mo.md(
        f"""
    The wiggly direction has a tiny variance, so even δ = 0.01 renews most of it. The
    smooth direction has {_variances[1] / _variances[0]:.0f} times more, so it needs δ
    near 1. **For every update below, the question is how large a δ its stage 2 can
    handle.** (When u lands off the chart, an arrow points to it: a large δ throws it far
    away, and the bump becomes so wide that stage 2 draws from almost the whole posterior.)
    """
    )
    return


@app.cell(hide_code=True)
def closeup_md(mo):
    mo.md(r"""
    ### 4. Today's update, up close

    Today's dSMC update (Particle-RWM in the papers, `drift: none` in our sampler) does
    stage 2 with **separable** candidates:

    1. At each step, draw N candidate values around u, *ignoring the other step*. The
       current value is always one of them.
    2. Any combination of one candidate per step is a possible new path. In the plane
       these are the crossings of a grid: vertical lines at the step-1 candidates,
       horizontal lines at the step-2 candidates.
    3. A *stitcher* picks one crossing at random, in proportion to how well it fits the
       posterior times the bump. Dot size shows each crossing's chance. The stitcher
       here is exact, the best possible; dSMC's tree is a fast approximation of it.

    The controls set the number of candidates per step and which random draw you see.
    They drive sections 4 and 5 and all of Part 2.
    """)
    return


@app.cell(hide_code=True)
def plane_controls(mo):
    candidates = mo.ui.dropdown(
        options={"4": 4, "8": 8, "16": 16, "32": 32}, value="8", label="candidates per step"
    )
    draw = mo.ui.slider(start=0, stop=9, value=0, label="random draw", show_value=True)
    mo.hstack([candidates, draw], justify="start", gap=1.5)
    return candidates, draw


@app.cell
def plane_studies(candidates, draw, lab, planes):
    studies = {
        kind: tuple(
            lab.plane_study(kind, plane, particles=candidates.value, seed=draw.value)
            for plane in planes
        )
        for kind in lab.UPDATES
    }
    return (studies,)


@app.cell(hide_code=True)
def closeup_runs(c0, candidates, draw, lab, studies):
    closeup = (
        lab.study(
            "local",
            c0.toy,
            c0.start,
            c0.model,
            particles=candidates.value,
            delta=0.01,
            sweeps=3000,
            burn=0,
            seed=draw.value,
        ),
        studies["local"][1],
        lab.study(
            "local",
            c0.toy,
            c0.start,
            c0.model,
            particles=candidates.value,
            delta=3.0,
            sweeps=3000,
            burn=0,
            seed=draw.value,
        ),
    )
    return (closeup,)


@app.cell(hide_code=True)
def closeup_view(c0, closeup, lab):
    _small, _best, _large = closeup
    lab.fig_plane_panels(
        c0,
        [
            (f"δ = {_small.delta:g}: one update", _small, False),
            (f"best δ = {_best.delta:g}: one update", _best, False),
            (f"δ = {_large.delta:g}: one update", _large, False),
            (f"60 updates at δ = {_best.delta:g}", _best, True),
        ],
        title="Today's update on the smooth latent: a grid of crossings, and a thin diagonal ridge",
    )
    return


@app.cell(hide_code=True)
def closeup_numbers(c0, closeup, lab, mo):
    _variances, _ = lab.axes(c0.model)
    _small, _best, _large = closeup
    _refresh = [lab.axis_refresh(result.chain, c0.model)[0] for result in closeup]
    mo.md(
        f"""
    Left to right:

    - **Small δ ({_small.delta:g}).** All candidates sit near the current path, so every
      crossing is close by. The update moves, but only a little: a smooth refresh of
      {_refresh[0]:.2f}.
    - **The best δ ({_best.delta:g}).** A few crossings land on the ridge further away. This
      is as good as it gets: {_refresh[1]:.2f}, against a ceiling of
      {lab.ceiling(_best.delta, _variances[1]):.2f} at that δ.
    - **Large δ ({_large.delta:g}).** Candidates scatter so widely that almost no crossing
      lands inside the thin ellipse. The current path is always a crossing, and it fits,
      so the stitcher mostly keeps it: {_refresh[2]:.2f}.
    - **60 updates at the best δ.** The chain creeps along the ridge.

    The grid is aligned with the axes, while the ridge runs diagonally. To land on the
    ridge away from the current path, both steps' candidates have to line up by chance.
    """
    )
    return


@app.cell(hide_code=True)
def ridge_md(mo):
    mo.md(r"""
    ### 5. Two ways to follow the ridge

    The fix is to put the dynamics into stage 2. There are two ways to do it:

    - **Follow the parent** (guided proposals: Particle-aGRAD with κ = 0). Draw the
      step-1 candidates first. Then draw each step-2 candidate *from* one of them, its
      parent, through the dynamics, pulled towards u. Each step-2 candidate lands next to
      its parent's diagonal (hollow circles), so the crossings that fit are spread along
      the ridge. The price: step 2 waits for step 1, so the update runs **sequentially**
      in time.
    - **Propose the whole path at once** (the auxiliary Kalman path move). Build a
      Gaussian model of the dynamics and readings, draw a whole path from it, and accept
      or reject that path against the exact posterior. Its proposals (violet) already
      have the ellipse's shape. On this latent the Gaussian model is exact, so every
      proposal is accepted, δ can be infinite (no jitter, so no ceiling), and every
      update is an independent draw.
    """)
    return


@app.cell(hide_code=True)
def ridge_view(c0, lab, studies):
    _guided, _kalman = studies["guided"][1], studies["kalman"][1]
    lab.fig_plane_panels(
        c0,
        [
            (f"Guided, δ = {_guided.delta:g}: one update", _guided, False),
            ("Guided: 60 updates", _guided, True),
            ("Kalman move: one update", _kalman, False),
            ("Kalman move: 60 updates", _kalman, True),
        ],
        title="Two updates that follow the ridge",
    )
    return


@app.cell(hide_code=True)
def ridge_numbers(c0, lab, mo, studies):
    _refresh = {
        kind: lab.axis_refresh(studies[kind][1].chain, c0.model)[0]
        for kind in ("local", "guided", "kalman")
    }
    mo.md(
        f"""
    Smooth refresh per update on this latent: today's update {_refresh["local"]:.2f},
    guided {_refresh["guided"]:.2f}, Kalman move {_refresh["kalman"]:.2f}.
    """
    )
    return


@app.cell(hide_code=True)
def family_md(mo):
    mo.md(r"""
    ## Part 2 · Every proposal on every type of latent

    Seven updates, sorted by two questions: does a candidate at one step look at the
    other steps, and what else does it use?

    | What else shapes a candidate | **Separable**: ignores the other steps | **Sequential**: extends a candidate at the step before |
    | :--- | :--- | :--- |
    | Only u | Particle-RWM: today's dSMC (`drift: none`) | Guided: Particle-aGRAD with κ = 0 |
    | u shifted by a gradient | Particle-aMALA (`drift: amala`): the gradient of the step's whole log density, its dynamics included | Particle-aGRAD with κ = 1: the gradient of the reading only |
    | A look ahead | Smoother marginals (dSMC §4.1): no u; candidates from a Gaussian smoother's estimate of each step | Twisted Particle-aGRAD, κ = 0: each proposal also sees u at every later step |

    The seventh, the **Kalman path move**, proposes the whole path at once.

    Separable candidates can all be drawn at the same time and stitched in parallel,
    which is what dSMC does. Sequential ones can't: step t waits for step t − 1. The
    Kalman move is parallel too, through a different trick (an associative scan).

    Not drawn: Particle-MALA and Particle-mGRAD are aMALA and aGRAD with u integrated out
    of the weights, so they propose the same way. `drift: amala_plus` (Particle-aMALA+)
    adds the next step's dynamics to aMALA's gradient; it is separable too.

    **How to read the figures.** One figure per update, one column per type of latent.
    The top row is one update from the current path (black dot), drawn as in sections 4
    and 5. The bottom row is its first 60 updates, titled with the refresh of the smooth
    and wiggly directions measured over 3,000 updates. Each update runs at the δ that
    refreshes the smooth direction most on that latent. Colour says what a candidate
    uses: blue only u, orange a gradient, green a look ahead, violet the whole path.
    """)
    return


@app.cell(hide_code=True)
def local_md(mo):
    mo.md(r"""
    #### Particle-RWM: today's dSMC

    Candidates scatter around u at each step, ignoring the other step. On the wiggly
    latent the posterior is round, so the grid covers it well and the update moves
    freely. As the latent gets smoother, the ridge gets thinner than the grid's spacing
    and the update creeps. With binary readings it creeps the same way: the trouble is
    the ridge, not the readings.
    """)
    return


@app.cell
def local_view(lab, planes, studies):
    lab.fig_proposal("local", planes, studies["local"])
    return


@app.cell(hide_code=True)
def amala_md(mo):
    mo.md(r"""
    #### Particle-aMALA

    The same grid, but u is first pushed along the gradient of each step's log density:
    the pull of its reading and of its link to the other step. On a fine grid that link
    is very stiff (its curvature is 1/0.04² ≈ 630 on C0's grid), so a push of any useful
    size overshoots the ridge, and only a tiny δ works. It stalls on every smooth latent,
    worst on the smoothest.
    """)
    return


@app.cell
def amala_view(lab, planes, studies):
    lab.fig_proposal("amala", planes, studies["amala"])
    return


@app.cell(hide_code=True)
def marginal_md(mo):
    mo.md(r"""
    #### Smoother marginals

    No jitter and no u: each step's candidates come from a Gaussian smoother's estimate
    of where that step is, so they spread over the whole range the posterior allows for
    that step. There is no δ, so no ceiling. But the grid is still aligned with the axes:
    on smooth latents only a few crossings fall on the ridge, and the update often keeps
    the current path.
    """)
    return


@app.cell
def marginal_view(lab, planes, studies):
    lab.fig_proposal("marginal", planes, studies["marginal"])
    return


@app.cell(hide_code=True)
def guided_md(mo):
    mo.md(r"""
    #### Guided: Particle-aGRAD with κ = 0

    Each step-2 candidate is drawn from a step-1 candidate, its parent, through the
    dynamics, and pulled towards u. The hollow circles mark these pairs: they sit on the
    ridge however thin it gets, so smooth latents slow it down far less than they slow
    the separable updates. Its best δ is
    large, where the pull towards u fades and the proposal is the dynamics alone; with a
    single state per step, that works well.
    """)
    return


@app.cell
def guided_view(lab, planes, studies):
    lab.fig_proposal("guided", planes, studies["guided"])
    return


@app.cell(hide_code=True)
def agrad_md(mo):
    mo.md(r"""
    #### Particle-aGRAD with κ = 1

    Guided, with u first pushed along the reading's gradient. That gradient is gentle
    (curvature 1/0.2² = 25 for noisy-value readings), but it still overshoots once δ
    passes four times the reading's variance, 0.16, so this update has to stay local. In
    this toy the cap costs more than the gradient gives. The papers find the gradient
    pays when there are many states per step, where every update has to stay local
    anyway.
    """)
    return


@app.cell
def agrad_view(lab, planes, studies):
    lab.fig_proposal("agrad", planes, studies["agrad"])
    return


@app.cell(hide_code=True)
def twisted_md(mo):
    mo.md(r"""
    #### Twisted Particle-aGRAD with κ = 0

    Guided, plus a look ahead: before proposing step 1, a backward Kalman pass works out
    what u at step 2 implies for step 1, so even the step-1 candidates spread along the
    ridge. With one state per step and a large δ, the plain guided update already does
    well, so the twist adds little here. It matters when δ has to stay small.
    """)
    return


@app.cell
def twisted_view(lab, planes, studies):
    lab.fig_proposal("twisted", planes, studies["twisted"])
    return


@app.cell(hide_code=True)
def kalman_md(mo):
    mo.md(r"""
    #### The Kalman path move

    One whole-path proposal from a Gaussian with the posterior's mean and covariance,
    then an exact accept-or-reject. It handles every Gaussian latent perfectly, however
    thin: every proposal is accepted and every update is a fresh draw. With binary
    readings the Gaussian no longer matches: it puts mass below the cut-off, where the
    posterior has almost none, so those proposals are rejected and the chain waits. This
    Gaussian is the best one possible; a model linearized around one path, as a real
    sampler builds it, can fit worse.
    """)
    return


@app.cell
def kalman_view(lab, planes, studies):
    lab.fig_proposal("kalman", planes, studies["kalman"])
    return


@app.cell(hide_code=True)
def path_md(mo):
    mo.md(r"""
    ## Part 3 · A whole path: smooth and wiggly shapes

    With 256 steps the plane becomes a 256-dimensional space, and the ellipse an
    ellipsoid with 256 axes. Its axes are cosine shapes, from one slow swing across the
    whole window to an up-down alternation at every step. The two-step smooth and wiggly
    directions are the first and the last of them. Below, three posterior draws are split
    at a period of half a day into a smooth part and a wiggly part.
    """)
    return


@app.cell(hide_code=True)
def path_controls(mo):
    grid = mo.ui.dropdown(
        options={"16": 16, "32": 32, "64": 64, "128 (C0)": 128, "256": 256},
        value="128 (C0)",
        label="grid steps per day",
    )
    every = mo.ui.dropdown(
        options={"every step (C0)": 1, "every 4th step": 4, "every 16th step": 16},
        value="every step (C0)",
        label="readings",
    )
    particles = mo.ui.dropdown(
        options={"8": 8, "16": 16, "32": 32, "64": 64}, value="32", label="candidates per step"
    )
    mo.hstack([grid, every, particles], justify="start", gap=1.5)
    return every, grid, particles


@app.cell
def path_toy(every, grid, lab):
    toy = lab.make_toy(intervals_per_day=grid.value, every=every.value)
    model = lab.reference_model(toy)
    start = lab.posterior_draws(model, 1, seed=1)[0]
    return model, start, toy


@app.cell
def shapes_view(lab, model, toy):
    lab.fig_posterior_shapes(toy, model, lab.posterior_draws(model, 3, seed=2))
    return


@app.cell(hide_code=True)
def shapes_numbers(mo, model, np, toy):
    _sd = np.sqrt(np.diag(model.covariance))
    _covariance = model.covariance
    _correlation = float(
        np.median(
            _covariance[np.arange(toy.steps - 1), np.arange(1, toy.steps)] / (_sd[:-1] * _sd[1:])
        )
    )
    mo.md(
        f"""
    {toy.steps} grid steps. Each state is known to about ±{np.median(_sd):.3f} (one
    posterior SD), and neighbouring states are {_correlation:.0%} correlated. The smooth
    part drifts in long swings pinned only by the noisy readings; the wiggly part is rough
    and pinned by the dynamics.
    """
    )
    return


@app.cell
def delta_scan(lab, model, particles, start, toy):
    scan = lab.scan_delta(toy, model, start, particles=particles.value)
    path_deltas = lab.best_deltas(scan)
    return path_deltas, scan


@app.cell
def path_studies(lab, model, particles, path_deltas, start, toy):
    paths = {
        kind: lab.study(
            kind,
            toy,
            start,
            model,
            particles=particles.value,
            delta=path_deltas[kind],
            sweeps=450,
            burn=50,
        )
        for kind in lab.UPDATES
    }
    local = lab.local_range(toy, start, [result.picture for result in paths.values()])
    strip = lab.smooth_scale(toy, model)
    return local, paths, strip


@app.cell(hide_code=True)
def rows_md(mo):
    mo.md(r"""
    ### Seven updates along the path

    One row per update, each at the δ that gives it the best smooth refresh on this path
    (Part 4 shows the whole scan):

    - **One update, in a 30-step stretch.** Separable updates propose a cloud of values
      at each step (grey dots) and stitch one per step. Sequential updates draw each
      candidate from a parent at the step before (grey lines). The Kalman move proposes
      whole paths (thin lines).
    - **Smooth part of the path over 40 sweeps.** The path's smooth part minus the
      posterior mean, one row per sweep: red above the mean, blue below. Vertical streaks
      are a smooth shape the update can't shift. A pattern that changes from row to row
      means a fresh smooth shape every sweep.
    - **Refreshed per sweep, by shape**, from the smoothest shape (left) to the wiggliest
      (right).
    """)
    return


@app.cell
def local_row(lab, local, model, paths, start, strip, toy):
    lab.fig_path_row(toy, model, start, paths["local"], local=local, zmax=strip, first=True)
    return


@app.cell
def amala_row(lab, local, model, paths, start, strip, toy):
    lab.fig_path_row(toy, model, start, paths["amala"], local=local, zmax=strip, first=False)
    return


@app.cell
def marginal_row(lab, local, model, paths, start, strip, toy):
    lab.fig_path_row(toy, model, start, paths["marginal"], local=local, zmax=strip, first=False)
    return


@app.cell
def guided_row(lab, local, model, paths, start, strip, toy):
    lab.fig_path_row(toy, model, start, paths["guided"], local=local, zmax=strip, first=False)
    return


@app.cell
def agrad_row(lab, local, model, paths, start, strip, toy):
    lab.fig_path_row(toy, model, start, paths["agrad"], local=local, zmax=strip, first=False)
    return


@app.cell
def twisted_row(lab, local, model, paths, start, strip, toy):
    lab.fig_path_row(toy, model, start, paths["twisted"], local=local, zmax=strip, first=False)
    return


@app.cell
def kalman_row(lab, local, model, paths, start, strip, toy):
    lab.fig_path_row(toy, model, start, paths["kalman"], local=local, zmax=strip, first=False)
    return


@app.cell(hide_code=True)
def path_table(lab, mo, model, np, paths, toy):
    _sd = np.sqrt(np.diag(model.covariance))
    _rows = []
    for _kind, _result in paths.items():
        _smooth, _wiggly = lab.shape_summary(lab.refresh(_result.chain, model), toy)
        _error = np.max(np.abs(_result.chain.mean(axis=0) - model.mean) / _sd)
        _family = (
            "separable"
            if _kind in lab.SEPARABLE
            else "sequential"
            if _kind in lab.SEQUENTIAL
            else "whole path"
        )
        _delta = "∞" if not np.isfinite(_result.delta) else f"{_result.delta:g}"
        _acceptance = "" if _result.acceptance is None else f"{_result.acceptance:.0%}"
        _rows.append(
            f"| {lab.NAMES[_kind]} | {_family} | {_delta} | {_smooth:.2f} | {_wiggly:.2f} | {_acceptance} | {_error:.2f} |"
        )
    mo.md(
        "\n".join(
            [
                "| Update | Family | δ | Smooth refresh | Wiggle refresh | Accepted | Largest mean error (SDs) |",
                "| :--- | :--- | ---: | ---: | ---: | ---: | ---: |",
                *_rows,
                "",
                "All seven agree with the exact posterior: the last column is Monte Carlo error from "
                "400 sweeps. Today's dSMC isn't wrong; it is slow on smooth shapes. The order matches "
                "the smooth latent in Part 2.",
            ]
        )
    )
    return


@app.cell(hide_code=True)
def limits_md(mo):
    mo.md(r"""
    ## Part 4 · What limits each family

    Two scans, each update at its best δ unless δ is the axis:

    - **δ (left two panels).** The dotted line is the ceiling from section 3, averaged
      over the path's smooth shapes. An update that drew stage 2 perfectly would sit on
      it.
    - **Path length (right).** Today's update, smoother marginals, the twisted update and
      the Kalman move on paths of 2 to 256 steps of C0's grid. The two-step point is
      Part 1's setting; the 256-step point is Part 3's path.

    The path-length scan takes about a minute.
    """)
    return


@app.cell
def length_scan(lab, particles):
    lengths = lab.scan_length(particles=particles.value)
    return (lengths,)


@app.cell(hide_code=True)
def limits_view(lab, lengths, model, paths, scan, toy):
    _ceilings = [lab.smooth_ceiling(delta, toy, model) for delta in lab.SCAN_DELTAS]
    _kalman = lab.shape_summary(lab.refresh(paths["kalman"].chain, model), toy)[0]
    lab.fig_limits(scan, _ceilings, _kalman, lengths)
    return


@app.cell(hide_code=True)
def limits_after_md(mo):
    mo.md(r"""
    - **Separable updates** fall away from the ceiling once δ is large enough to matter:
      their grid can't follow the ridge. aMALA collapses almost at once, because its
      gradient overshoots.
    - **Sequential updates** without a gradient track the ceiling at small δ and level
      off where their particles run out. aGRAD even rises above the dotted line around
      δ = 0.1: the ceiling assumes a plain jitter, and aGRAD's push towards the readings
      makes u forget the current path faster. Past δ ≈ 0.16 the push overshoots and it
      collapses.
    - **The Kalman move** sits above everything at δ = ∞.
    - **Path length.** Going from 2 steps to 8 costs today's update most of its smooth
      refresh, and after that it stays low. Smoother marginals and the twisted update
      lose less, and the Kalman move nothing. On paths shorter than half a day "smooth"
      is just the level of the whole window, so those points are noisier.
    """)
    return


@app.cell(hide_code=True)
def c0_md(mo):
    mo.md(r"""
    ## Part 5 · C0, and how we got here

    The same per-shape measurement on the real C0 fits: 2,049 steps per path, the README
    schedule (128 particles, 4 chains, 2,000 + 4,000 draws, seed 19), one A10 each, on
    byte-identical data. Today's sampler redraws one of the two paths per sweep, so it
    tops out near 0.5 even for wiggles, and its tree stitches less well than the toy's
    exact stitcher. The Kalman arm adds a joint move of both paths after every sweep; it
    accepted 92% of its proposals at δ = 10⁴ (its cap). The three fits' records are in
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
    **How we ended up without a dynamics-aware update.** The repo had one in April 2026
    and traded it for parallel depth while consolidating:

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
      run froze it after warmup. Part 2's binary latent shows what a Gaussian that
      doesn't fit costs.
    - Hardening the move before it joins the sampler: tests, lint and the fit chart, plus
      support for interval readings, replayed accumulators, exact readings on free
      coordinates and tempering, all of which France needs.

    References: dSMC, [arXiv 2202.02264](https://arxiv.org/abs/2202.02264) §4; auxiliary
    Kalman samplers and guided proposals, [arXiv 2303.00301](https://arxiv.org/abs/2303.00301)
    §2–4; Particle-aMALA, Particle-aGRAD and the twisted variant,
    [arXiv 2401.14868](https://arxiv.org/abs/2401.14868) §3–4.
    """)
    return


if __name__ == "__main__":
    app.run()
