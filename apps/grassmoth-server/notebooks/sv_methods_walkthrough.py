import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def imports_marimo():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def whole_story(mo):
    mo.md(r"""
    # Sampling stochastic volatility, one idea at a time

    Daily returns tell you how volatile each day was, but only noisily. Behind them sits a
    hidden path, one log-variance per day, and three unknown numbers that govern it. We want
    the posterior over the numbers and the whole path together. With 1,000 days of data, that
    is one point in 1,003 dimensions.

    Every method below targets that same posterior exactly. What differs is how they move around it.

    **A method is four separate choices.** Most of the confusion about these samplers comes
    from treating each one as a single thing. Taken apart:

    | Choice | The question | Options you'll meet | Part |
    |:---|:---|:---|:---|
    | **Path move** | With the parameters fixed, how do we get a different path? | Candidates day by day (particles), one Gaussian draw (auxiliary Kalman), a physics trajectory (HMC) | I |
    | **Parameter move** | With the path fixed, how do we get different parameters? | A random walk, MALA or NUTS | II |
    | **Stored path** | Which version of the path does the sampler keep, and so hold while the parameters move? | The path itself, its daily shocks, or chart coordinates | III |
    | **Organization** | How do the parameter and path moves fit together? | Taking turns, one joint trajectory, or updating the path while undecided between two parameter candidates | II, IV |

    The families you asked about are settings of those four choices. "With or without the chart" is the last column:

    | Family | Organization | Parameter move | Path move | Stored path |
    |:---|:---|:---|:---|:---|
    | **NUTS + smoother** | Taking turns | NUTS | Auxiliary Kalman, Particle-mGRAD or plain bootstrap conditional SMC | $h$ or the chart |
    | **MPGibbs + smoother** | Move the path while undecided | Propose one new candidate, and decide after the path has moved | The same path moves, now aimed at a mixture of both candidates | $h$ or the chart |
    | **Joint NUTS** | One trajectory | The same trajectory | The same trajectory | $h$, daily shocks or the chart |

    **Exactness composes, speed doesn't.** Each piece keeps the posterior in place, so a valid combination gives the right
    answer without a new proof, and Part VI checks that they all agree. Speed is different: what one choice buys depends on
    the others. So Part V names the places where the choices touch, and Part VI reads the speeds one choice at a time, then
    together.

    **How to read this.** Top to bottom: each part adds one choice on the same model and uses only what came before, with
    real computation you can poke at. Parts I to IV each end with a table that sums them up, so to find a topic, skim the
    tables. The closed drawers hold derivations and side experiments, and the main line works without them. For the
    conclusions first, skip to the last section, "Where they agree, where they differ". The sliders recompute live on this
    machine's CPU. The long comparisons are read from saved runs, and the last drawer says how to regenerate them.
    """)
    return


@app.cell(hide_code=True)
def vocabulary(mo):
    mo.md(r"""
    | Word | What it means here |
    |:---|:---|
    | **Path** | $h$: the hidden log-variance, one value per day |
    | **Stored path** | The path as the sampler keeps it and holds it. For now that is $h$ itself; Part III changes this |
    | **Reading** | One day's squared return |
    | **Parameters** | θ = (level μ, decay κ, vol-of-vol σ). Samplers move them on the unconstrained scale η = (μ, log κ, log σ) |
    | **Sweep** | One full round of a sampler's moves, giving one new θ and path |
    | **Smoother** | A path move that uses all the readings |
    | **Conditional** (a "slice" in the pictures) | A parameter's distribution given a whole stored path and the readings |
    | **Marginal** (a "shadow") | A parameter's distribution given the readings only, with the path integrated out |
    | **Auxiliary** | An extra random variable added to make a move, then thrown away: momentum, or a noisy copy |
    | **Candidate** | A possible next value, among those a move compares |
    | **Exact correction** | The step that uses true densities, so that approximations cost speed but never the answer |
    """)
    return


@app.cell(hide_code=True)
def imports_lab():
    import os

    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_ENABLE_X64"] = "true"

    import jax
    import jax.numpy as jnp
    import numpy as np

    from notebooks.support import sv_demos as demos
    from notebooks.support import sv_figures as figs
    from notebooks.support import sv_kernels as k
    from notebooks.support import sv_model as sv

    return demos, figs, jax, jnp, k, np, sv


@app.cell(hide_code=True)
def model_md(mo):
    mo.md(r"""
    ## The model and the data

    Each day has a hidden log-variance $h_t$. It drifts back towards zero and takes a random
    shock:

    $$h_{t+1} = h_t - \kappa h_t - 0.03\,h_t^3 + \sigma\,\varepsilon_t,
    \qquad \varepsilon_t \sim N(0, 1), \qquad h_0 \sim N(0, 1).$$

    That day's return is $r_t \sim N(0, e^{\mu + h_t})$. A return's sign says nothing about
    variance, so its square, the **reading**, carries all the information it has about $h_t$.

    Three numbers are unknown:

    | Parameter | Meaning | Prior | Truth here |
    |:---|:---|:---|:---|
    | Level μ | Overall log-variance | Normal, mean 0, sd 2 | −0.2 |
    | Decay κ | How fast the path returns to zero | Half-normal, scale 0.1 | 0.02 |
    | Vol-of-vol σ | Size of the daily shocks | Half-normal, scale 0.5 | 0.2 |

    This is the daily model of the `SV-cubic` benchmark case, whose notes call decay θ; here θ
    means all three parameters. The case has 3,000 days. We use a fresh simulation of 1,000 from
    the same model, because every idea shows at that length and the runs stay quick on a laptop.
    """)
    return


@app.cell(hide_code=True)
def data_setup(demos, sv):
    history = sv.simulate(demos.DAYS, demos.SEED)
    truth = sv.to_eta(*sv.TRUTH)
    return history, truth


@app.cell(hide_code=True)
def runs_load(demos):
    runs = demos.load_runs()
    return (runs,)


@app.cell(hide_code=True)
def method_labels():
    method_labels = {
        "joint_path": "Joint NUTS, h",
        "joint_shocks": "Joint NUTS, daily shocks",
        "joint_chart": "Joint NUTS, chart",
        "nuts_bootstrap_path": "NUTS + bootstrap, chart off",
        "nuts_bootstrap_chart": "NUTS + bootstrap, chart on",
        "nuts_mgrad_path": "NUTS + mGRAD, chart off",
        "nuts_mgrad_chart": "NUTS + mGRAD, chart on",
        "nuts_ak_blocks_path": "NUTS + Kalman blocks, chart off",
        "nuts_ak_blocks_chart": "NUTS + Kalman blocks, chart on",
        "nuts_ak_whole_path": "NUTS + Kalman whole path, chart off",
        "nuts_ak_whole_chart": "NUTS + Kalman whole path, chart on",
        "mpgibbs_bootstrap_path": "MPGibbs + bootstrap, chart off",
        "mpgibbs_bootstrap_chart": "MPGibbs + bootstrap, chart on",
        "mpgibbs_mgrad_path": "MPGibbs + mGRAD, chart off",
        "mpgibbs_mgrad_chart": "MPGibbs + mGRAD, chart on",
        "mpgibbs_ak_blocks_path": "MPGibbs + Kalman blocks, chart off",
        "mpgibbs_ak_blocks_chart": "MPGibbs + Kalman blocks, chart on",
        "mpgibbs_ak_whole_path": "MPGibbs + Kalman whole path, chart off",
        "mpgibbs_ak_whole_chart": "MPGibbs + Kalman whole path, chart on",
    }
    smoother_labels = {
        "bootstrap": "Bootstrap conditional SMC",
        "mgrad": "Particle-mGRAD",
        "ak_blocks": "Auxiliary Kalman, blocks",
        "ak_whole": "Auxiliary Kalman, whole path",
    }
    return method_labels, smoother_labels


@app.cell(hide_code=True)
def data_view(figs, history, np, truth):
    figs.fig_history(
        np.arange(history.h.shape[0]),
        np.asarray(history.r),
        np.asarray(truth[0] + history.h),
    )
    return


@app.cell(hide_code=True)
def posterior_md(mo):
    mo.md(r"""
    **A posterior draw is three parameters plus a whole path.** Below is the posterior of the
    path from a long reference run of one of the samplers this notebook builds: 4 chains of 3,000
    retained sweeps. The grey band shows how much of the path the readings leave open. Each of the three draws (blue)
    is much rougher than the band's centre, because each draw has the path's day-to-day shocks in
    it, and the average of many draws doesn't.
    """)
    return


@app.cell(hide_code=True)
def reference_compute(demos, history, np):
    reference = demos.reference_posterior(history.r2)
    reference_etas = np.asarray(reference["etas"]).reshape(-1, 3)
    reference_paths = np.asarray(reference["paths"]).reshape(-1, history.h.shape[0])
    return reference_etas, reference_paths


@app.cell(hide_code=True)
def posterior_view(figs, history, np, reference_paths):
    figs.fig_posterior_paths(np.arange(history.h.shape[0]), np.asarray(history.h), reference_paths)
    return


@app.cell(hide_code=True)
def hard_md(mo):
    mo.md(r"""
    ## Why this is hard

    Two facts make this posterior hard. The rest of the notebook is about getting around them.

    **The path's days are tied to each other.** The dynamics say tomorrow is close to today, with
    a precision of about $1/\sigma^2 = 25$. One day's reading is weak by comparison: its expected
    Fisher information about that day is ½, though a rare, huge return says more about its own day.
    So one reading barely moves one day, and only long stretches of readings shape the path. A good path move has to change many days
    together, and proposals that ignore the neighbours get rejected.

    **The parameters are tied to the path, in two ways.**

    - *Level and path.* Readings depend only on level + path. Raise the level by 1 and lower
      every day by 1, and the readings' likelihood doesn't change. Only the dynamics' pull towards
      zero and the priors break this symmetry, so the posterior has a long, approximately linear
      ridge (left).
    - *Vol-of-vol and path.* The path's daily transition residuals are draws with standard
      deviation σ, and a whole path holds 999 of them. Given a long path and the other parameters, σ
      is pinned down within about $1/\sqrt{2 \cdot 999} \approx 2\%$. The readings alone say much less
      (right).

    In the figure, each dot is one posterior draw. Underneath, grey is a parameter's **marginal**,
    the cloud's shadow: what the readings say with the path integrated out. Green is one
    **conditional**, a slice through the cloud: what that parameter can do given one draw's
    whole path and the other two parameters.
    """)
    return


@app.cell(hide_code=True)
def ties_compute(demos, history, jax, jnp, np, reference_etas, reference_paths, sv):
    # The draw nearest the posterior's centre, so that the slices show a typical case.
    _standard = (reference_etas - reference_etas.mean(axis=0)) / reference_etas.std(axis=0)
    chosen_draw = int(np.argmin(np.sum(_standard**2, axis=1)))
    _eta = jnp.asarray(reference_etas[chosen_draw])
    _held = jnp.asarray(reference_paths[chosen_draw])

    def _grid(column):
        low, high = reference_etas[:, column].min(), reference_etas[:, column].max()
        return np.linspace(low - 0.1, high + 0.1, 500)

    level_grid, log_vol_grid = _grid(0), _grid(2)
    level_conditional = demos.conditional_density(
        sv.PHYSICAL, _eta, _held, history.r2, 0, level_grid
    )
    log_vol_conditional = demos.conditional_density(
        sv.PHYSICAL, _eta, _held, history.r2, 2, log_vol_grid
    )
    rms_residuals = np.sqrt(
        np.mean(
            np.asarray(
                jax.vmap(sv.residuals)(jnp.asarray(reference_etas), jnp.asarray(reference_paths))
            )
            ** 2,
            axis=1,
        )
    )
    ratio_level = reference_etas[:, 0].std() / demos.conditional_sd(level_conditional, level_grid)
    ratio_log_vol = reference_etas[:, 2].std() / demos.conditional_sd(
        log_vol_conditional, log_vol_grid
    )
    return (
        chosen_draw,
        level_conditional,
        level_grid,
        log_vol_conditional,
        log_vol_grid,
        ratio_level,
        ratio_log_vol,
        rms_residuals,
    )


@app.cell(hide_code=True)
def ties_view(
    chosen_draw,
    figs,
    level_conditional,
    level_grid,
    log_vol_conditional,
    log_vol_grid,
    reference_etas,
    reference_paths,
    rms_residuals,
):
    figs.fig_ties(
        reference_etas[:, 0],
        reference_paths.mean(axis=1),
        reference_etas[:, 2],
        rms_residuals,
        level_grid=level_grid,
        level_conditional=level_conditional,
        log_vol_grid=log_vol_grid,
        log_vol_conditional=log_vol_conditional,
        chosen=chosen_draw,
    )
    return


@app.cell(hide_code=True)
def narrow_md(mo, ratio_level, ratio_log_vol):
    mo.md(f"""
    **Why a narrow conditional hurts.** Suppose a sampler takes turns: parameters with the path
    held, then path with the parameters held. Each parameter step can only move within the green
    curve. Here the marginal is {ratio_level:.1f}× wider than the conditional for level, and
    {ratio_log_vol:.1f}× wider for log vol-of-vol.

    For a two-dimensional Gaussian with perfect draws from each conditional, a ratio r means about
    2r² − 1 **sweeps per effective sample**, meaning the sweeps it takes to learn as much about a
    quantity as one independent draw would: about {2 * ratio_level**2 - 1:.0f} for level and
    {2 * ratio_log_vol**2 - 1:.0f} for vol-of-vol. Real posteriors aren't Gaussian, these slices
    fix the other two parameters, and real path moves aren't perfect, so read these numbers as the
    scale of the problem. Part VI measures the real ones.

    Everything from here on is one of two things: making the path move well (Part I), or getting
    the parameters past this limit (Parts II to IV).
    """)
    return


@app.cell(hide_code=True)
def part_one_md(mo):
    mo.md(r"""
    ## Part I · Moving the path, with the parameters known

    Fix the parameters at the truth, so the only unknown is the path. The target is then a chain:
    one dynamics factor and one reading factor per day. A move that updates the path given all the
    readings is often called a **smoother**, though one move rarely gives an independent draw.

    If both factors were Gaussian, a Kalman smoother would draw an exact path in one pass. We can
    use that as an answer key for testing, but it isn't available here. The cubic drift is nonlinear,
    and a day's reading is clearly not Gaussian, as the figure shows. A drawn path from
    a Gaussian model is therefore not a draw from the real posterior.

    Three ideas get exact moves anyway, each using the model's structure differently:

    1. **Many candidates per day**, recombined across days (particle methods, I.1 to I.3).
    2. **One Gaussian draw of the whole path**, then an exact correction (auxiliary Kalman, I.4).
    3. **Follow the gradient** with physics (HMC, I.5).
    """)
    return


@app.cell(hide_code=True)
def reading_view(demos, figs, np):
    _grid = np.linspace(-6.0, 3.0, 400)
    figs.fig_reading_factor(
        _grid,
        [
            ("A very small return, r² = 0.01", *demos.reading_factor(_grid, 0.01)),
            ("A typical return, r² = 1", *demos.reading_factor(_grid, 1.0)),
        ],
    )
    return


@app.cell(hide_code=True)
def particles_md(mo):
    mo.md(r"""
    ### I.1 Many candidates per day: conditional SMC

    A particle filter keeps N candidate values for each day. To reach tomorrow, each candidate
    picks a parent among today's, in proportion to the parents' weights, then steps forward through
    the dynamics. Its weight says how well it explains tomorrow's reading.

    Two additions turn this filter into an exact move for the path (Andrieu, Doucet & Holenstein
    2010; Whiteley 2010):

    - **The current path stays in the running.** Candidate 0 is the current path on every day. That
      is what makes this an MCMC move that leaves the posterior exactly in place, for any N of at
      least 2. More candidates improve movement, not exactness.
    - **The new path is drawn backward.** Pick a final-day candidate by weight. Then walk back: on
      each earlier day, pick a candidate by its weight times how well it leads to the day already
      chosen. This isn't tied to any one particle's ancestry, so the new path can stitch together
      pieces of different lineages. N candidates over T days offer about Nᵀ possible paths for
      N × T work.

    Step through one sweep below, on the first 12 days with 8 candidates, starting from the true
    path. Because the data were simulated from these parameters, the true path is itself an exact
    posterior draw, so we start in the right place.
    """)
    return


@app.cell(hide_code=True)
def tiny_setup(history, jax, k, truth):
    tiny_days = 12
    tiny_count = 8
    tiny_path = history.h[:tiny_days]
    tiny_target = k.sv_target(truth[None], history.r2[:tiny_days])
    bootstrap_move = jax.jit(
        lambda key: k.csmc_trace(key, tiny_target, tiny_path, particles=tiny_count)
    )
    mgrad_move = jax.jit(
        lambda key, delta: k.mgrad_trace(
            key, tiny_target, tiny_path, particles=tiny_count, delta=delta
        )
    )
    return bootstrap_move, mgrad_move, tiny_days


@app.cell(hide_code=True)
def lattice_controls(mo, tiny_days):
    lattice_stage = mo.ui.slider(
        start=0, stop=2 * tiny_days - 1, value=2 * tiny_days - 1, label="Step through the sweep"
    )
    lattice_seed = mo.ui.slider(start=0, stop=20, value=3, label="Random numbers")
    mo.hstack([lattice_stage, lattice_seed], justify="start", gap=2)
    return lattice_seed, lattice_stage


@app.cell(hide_code=True)
def lattice_view(bootstrap_move, figs, jax, lattice_seed, lattice_stage, np):
    _trace = bootstrap_move(jax.random.key(lattice_seed.value))
    figs.fig_lattice(
        np.asarray(_trace.particles),
        np.asarray(_trace.ancestors),
        np.asarray(_trace.probabilities),
        np.asarray(_trace.selected_indices),
        np.asarray(_trace.path),
        backward_probabilities=np.asarray(_trace.backward_probabilities),
        stage=lattice_stage.value,
        title="One sweep of bootstrap conditional SMC",
    )
    return


@app.cell(hide_code=True)
def noisy_copy_md(mo):
    mo.md(r"""
    ### I.2 A shared trick: the noisy copy

    Bootstrap candidates come from the dynamics alone. They have no explicit pull towards the
    current path's value on that day, and they don't look at the reading they're about to meet. That
    works when one reading says little, as here. It degrades when readings are very sharp or each day
    has many coordinates, because random candidates then rarely land anywhere useful.

    The fix is to look near the current path. Draw a **noisy copy** $u_t = h_t + \text{noise}$,
    with noise variance δ/2, and then ask for a new path near $u$. Formally this adds $u$ to the
    target as $\pi(h)\,N(u; h, \delta/2)$. The marginal of $h$ is untouched, so any move that
    preserves $\pi(h \mid u)$, followed by forgetting $u$, preserves $\pi$
    (Titsias & Papaspiliopoulos 2018; Corenflos & Särkkä 2025).

    **δ is a dial.** With small δ, the copy pins the new path close to the old one: the problem
    becomes narrow, nearly Gaussian and easy, but moves are small. With large δ the copy says
    little: moves can be large, but the problem is as hard as it was.

    The two methods you asked about share this first step. They differ in what comes next.
    Particle-mGRAD makes many candidates, day by day. Auxiliary Kalman draws one whole path.

    The figure shows one day. The proposal multiplies the dynamics by a bump around the copy.
    Slide δ. When it's small the proposal hugs the copy. When it's large the proposal approaches the
    dynamics. The dashed line is what the proposal is trying to imitate. Here it's almost the same as
    the dynamics, because one reading is so weak next to them.

    **The nudge.** The copy can also lean towards the reading, by centring it slightly uphill of the current
    value: at $h_t + \tfrac{\delta}{2} g_t$, where $g_t$ is the gradient of that day's reading at $h_t$.
    Particle-mGRAD does this, and auxiliary Kalman doesn't. Flip the switch at a large δ on the largest return:
    the proposal now approaches the dynamics shifted by $\sigma^2 g_t$, rather than the dynamics themselves.
    """)
    return


@app.cell(hide_code=True)
def anatomy_controls(mo):
    anatomy_delta = mo.ui.slider(
        steps=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
        value=0.03,
        label="δ",
        show_value=True,
    )
    anatomy_gradient = mo.ui.switch(value=False, label="Nudge the copy by the reading's gradient")
    anatomy_day = mo.ui.dropdown(
        options={"An ordinary day": "ordinary", "The largest return so far": "largest"},
        value="An ordinary day",
        label="Which day",
    )
    mo.hstack([anatomy_delta, anatomy_gradient, anatomy_day], justify="start", gap=2)
    return anatomy_day, anatomy_delta, anatomy_gradient


@app.cell(hide_code=True)
def anatomy_view(
    anatomy_day,
    anatomy_delta,
    anatomy_gradient,
    figs,
    history,
    jax,
    jnp,
    k,
    np,
    truth,
):
    _day = (
        40 if anatomy_day.value == "ordinary" else 1 + int(np.argmax(np.asarray(history.r2[1:200])))
    )

    def _on_grid(grid):
        return k.one_day_anatomy(
            jax.random.key(7),
            truth,
            history.h[_day - 1],
            history.h[_day],
            history.r2[_day],
            grid,
            delta=anatomy_delta.value,
            gradient=anatomy_gradient.value,
        )

    # The grid has to reach the bump, which a large δ and a large reading can push far away.
    _old = float(history.h[_day])
    _copy = float(_on_grid(jnp.asarray([_old])).auxiliary)
    _reach = 3.0 * np.sqrt(anatomy_delta.value / 2.0)
    _grid = jnp.linspace(min(_old - 1.6, _copy - _reach), max(_old + 1.6, _copy + _reach), 1201)
    _anatomy = _on_grid(_grid)
    _reading = np.asarray(_anatomy.reading_density)
    figs.fig_one_day(
        np.asarray(_grid),
        transition=np.asarray(_anatomy.transition_density),
        reading=_reading / np.trapezoid(_reading, np.asarray(_grid)),
        target=np.asarray(_anatomy.target_density),
        bump=np.asarray(_anatomy.auxiliary_density),
        proposal=np.asarray(_anatomy.proposal_density),
        reference=float(history.h[_day]),
        auxiliary=float(_anatomy.auxiliary),
        title=f"One day's proposal, δ = {anatomy_delta.value}",
    )
    return


@app.cell(hide_code=True)
def mgrad_md(mo):
    mo.md(r"""
    ### I.3 Particle-mGRAD: candidates near the noisy copy

    Particle-mGRAD (Corenflos & Finke 2024) is conditional SMC with three changes.

    1. **Nudge the copy,** as in I.2. This is the only gradient it uses, because the Gaussian
       dynamics enter exactly in the next step.
    2. **Blend the dynamics and the copy.** Each candidate is drawn from the product of two
       Gaussians: the dynamics from its parent, and the bump around $u_t$. Its mean is
       $(1 - H) \times \text{dynamics mean} + H \times u_t$, with
       $H = \sigma^2 / (\sigma^2 + \delta/2)$. So small δ stays near the old path. As δ grows, the
       mean approaches the dynamics mean plus $\sigma^2 g$, where $g$ is the reading's gradient at the old
       path. Without the nudge that limit is bootstrap's proposal. With it, the shift is small on this
       model because $\sigma^2 g$ is small, but that isn't true in general.
    3. **Integrate the copy out of the weights.** A candidate's weight is its true one-day factors
       times the density of all the *other* candidates, supposing this one had been the current
       value, with $u$ integrated out. That is what the "m", for marginal, refers to: the copy
       in the weights. The path isn't integrated out, and the proposal still uses the sampled $u$.

    Everything else matches I.1: candidate 0 is the current path, parents are chosen by weight, and
    the new path is drawn backward using true densities. Slide δ to see the candidates tighten
    around the noisy copy (diamonds) or spread out along the dynamics.
    """)
    return


@app.cell(hide_code=True)
def mgrad_controls(mo, tiny_days):
    mgrad_delta = mo.ui.slider(
        steps=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
        value=0.03,
        label="δ",
        show_value=True,
    )
    mgrad_stage = mo.ui.slider(
        start=0, stop=2 * tiny_days - 1, value=2 * tiny_days - 1, label="Step through the sweep"
    )
    mgrad_seed = mo.ui.slider(start=0, stop=20, value=3, label="Random numbers")
    mo.hstack([mgrad_delta, mgrad_stage, mgrad_seed], justify="start", gap=2)
    return mgrad_delta, mgrad_seed, mgrad_stage


@app.cell(hide_code=True)
def mgrad_view(demos, figs, jax, mgrad_delta, mgrad_move, mgrad_seed, mgrad_stage, mo, np):
    _trace = mgrad_move(jax.random.key(mgrad_seed.value), mgrad_delta.value)
    _lattice = figs.fig_lattice(
        np.asarray(_trace.particles),
        np.asarray(_trace.ancestors),
        np.asarray(_trace.probabilities),
        np.asarray(_trace.selected_indices),
        np.asarray(_trace.path),
        backward_probabilities=np.asarray(_trace.backward_probabilities),
        stage=mgrad_stage.value,
        title=f"One sweep of Particle-mGRAD, δ = {mgrad_delta.value}",
        auxiliary=np.asarray(_trace.auxiliary.values),
    )
    _day = 6
    _weights = figs.fig_bank_weights(
        np.asarray(_trace.target_increments[_day]),
        np.asarray(_trace.proposal_corrections[_day]),
        np.asarray(_trace.probabilities[_day]),
        title=f"What the weights are made of, on day {_day}",
    )
    mo.vstack(
        [
            _lattice,
            mo.accordion(
                {
                    "Detail: what a candidate's weight is made of": _weights,
                    "The code for this move": mo.ui.code_editor(
                        demos.kernel_source("marginal_bank_log_densities", "_particle_trace"),
                        language="python",
                        disabled=True,
                    ),
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def kalman_md(mo):
    mo.md(r"""
    ### I.4 Auxiliary Kalman: one Gaussian draw, one exact verdict

    Auxiliary Kalman (Corenflos & Särkkä 2025) starts with the same noisy copy, then does something
    very different. Around the copy it builds a linear-Gaussian stand-in for the model: linearized
    dynamics, each reading replaced by a Gaussian with the same slope and curvature, and the copy
    itself as one more Gaussian reading per day. That stand-in is a state-space model a Kalman
    smoother can solve, so one pass draws a complete path, with every day's proposal shaped by
    readings on both sides of it.

    The stand-in only proposes. The true model decides, in one Metropolis–Hastings verdict with
    three terms, the bars on the right of the figure:

    $$\log \text{ratio} = \underbrace{\log\frac{\pi(h')}{\pi(h)}}_{\text{exact target}} + \underbrace{\log\frac{N(u; h', \delta/2)}{N(u; h, \delta/2)}}_{\text{noisy copy}} + \underbrace{\log\frac{q(h \mid u)}{q(h' \mid u)}}_{\text{proposal, reversed}}.$$

    The stand-in is built only from what the move holds fixed, the copy $u$ and the parameters, so
    the same Gaussian serves both directions. Building it at the current path instead is also valid,
    but then the reverse direction rebuilds it at the proposed path, and the ratio has to pay for
    that.

    Slide δ. With a small δ almost every proposal is accepted, and the move is small. With a large δ
    the moves get bigger and the verdict gets harsher.
    """)
    return


@app.cell(hide_code=True)
def kalman_setup(history, jax, k, truth):
    kalman_days = 48
    _target = k.sv_target(truth[None], history.r2[:kalman_days])
    _path = history.h[:kalman_days]
    kalman_move = jax.jit(lambda key, delta: k.aux_kalman_trace(key, _target, _path, delta=delta))
    return kalman_days, kalman_move


@app.cell(hide_code=True)
def kalman_controls(mo):
    kalman_delta = mo.ui.slider(
        steps=[0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0], value=0.1, label="δ", show_value=True
    )
    kalman_seed = mo.ui.slider(start=0, stop=20, value=1, label="Random numbers")
    mo.hstack([kalman_delta, kalman_seed], justify="start", gap=2)
    return kalman_delta, kalman_seed


@app.cell(hide_code=True)
def kalman_view(figs, jax, kalman_days, kalman_delta, kalman_move, kalman_seed, np):
    _trace = kalman_move(jax.random.key(kalman_seed.value), kalman_delta.value)
    _ledger = _trace.ledger
    figs.fig_kalman_move(
        np.arange(kalman_days),
        reference=np.asarray(_trace.reference),
        auxiliary=np.asarray(_trace.auxiliary),
        proposal_mean=np.asarray(_trace.proposal_mean),
        proposal_sd=np.asarray(_trace.proposal_sd),
        proposed=np.asarray(_trace.proposed),
        target_change=float(_ledger.proposed_target - _ledger.current_target),
        auxiliary_change=float(_ledger.proposed_auxiliary - _ledger.current_auxiliary),
        proposal_change=float(_ledger.current_proposal - _ledger.proposed_proposal),
        acceptance=float(_trace.acceptance_probability),
        title=f"One auxiliary Kalman move, δ = {kalman_delta.value}",
    )
    return


@app.cell(hide_code=True)
def kalman_scan_compute(demos, history, truth):
    kalman_scan = demos.kalman_scan(truth, history.r2, history.h)
    return (kalman_scan,)


@app.cell(hide_code=True)
def blocks_md(kalman_scan, mo):
    mo.md(rf"""
    **One verdict for many days, and why blocks exist.** Every day's Gaussian mismatch feeds into the same log
    ratio. At a fixed δ, a longer series means a harsher verdict. At δ = {kalman_scan.deltas[-1]:g}, acceptance
    falls from {kalman_scan.whole_path[-1, 0]:.2f} at {kalman_scan.days[0]} days to
    {kalman_scan.whole_path[-1, -1]:.2f} at {kalman_scan.days[-1]}.

    The way out is **blocks**. Update a stretch of about {kalman_scan.block_length} days at a time,
    holding its neighbours fixed, with its own verdict. One bad stretch no longer stops the rest,
    so δ can stay larger. The dashed line is acceptance per block at the largest δ. The cost is that
    each block only moves given its neighbours, so the boundaries move less freely. Shifting the
    block edges between sweeps helps with that. From here on, "whole path" and "blocks" name these two ways
    of running auxiliary Kalman.
    """)
    return


@app.cell(hide_code=True)
def blocks_view(figs, kalman_scan):
    figs.fig_kalman_scan(
        kalman_scan.days,
        kalman_scan.deltas,
        kalman_scan.whole_path,
        blocks=(kalman_scan.block_length, kalman_scan.blocks),
    )
    return


@app.cell(hide_code=True)
def kalman_code_drawer(demos, mo):
    mo.accordion(
        {
            "The code for this move": mo.ui.code_editor(
                demos.kernel_source("_aux_kalman_block"), language="python", disabled=True
            ),
        }
    )
    return


@app.cell(hide_code=True)
def hmc_md(mo):
    mo.md(r"""
    ### I.5 Physics: HMC on the path

    HMC uses a different auxiliary: **momentum**, one number per day. It then simulates a frictionless
    ball on the log-density surface, in small leapfrog steps. Every day moves a little at every step,
    guided by the gradient of the whole density, dynamics and readings together. At the end, one verdict
    corrects the simulation's numerical energy error. NUTS is the same physics, except that it chooses the
    trajectory's length as it goes and selects a point from the whole tree by a rule built to stay
    exact.

    Unlike the other two, HMC doesn't use the chain structure of the problem beyond what the gradient
    contains: nothing here knows it is a time series, and that is both its strength and its limit. It
    only needs a differentiable density. But the tight day-to-day dynamics make some directions very
    stiff, which forces small steps, while slow changes to the whole path need long trajectories to
    cross. So each move takes many gradient evaluations, each passing over all the days.

    The **mass matrix** is HMC's way of handling directions with different scales. It sets the momentum's
    covariance, which amounts to a linear rescaling of the coordinates. A diagonal one rescales each coordinate,
    and a dense one can also undo fixed correlations between them.

    One leapfrog step with fresh momentum is exactly MALA: a gradient step plus noise.
    """)
    return


@app.cell(hide_code=True)
def leapfrog_controls(mo):
    leapfrog_step = mo.ui.slider(
        steps=[0.01, 0.02, 0.05, 0.1, 0.15, 0.2], value=0.05, label="Step size", show_value=True
    )
    leapfrog_seed = mo.ui.slider(start=0, stop=20, value=2, label="Random numbers")
    mo.hstack([leapfrog_step, leapfrog_seed], justify="start", gap=2)
    return leapfrog_seed, leapfrog_step


@app.cell(hide_code=True)
def leapfrog_setup(history, jax, k, truth):
    leapfrog_days = 48
    _target = k.sv_target(truth[None], history.r2[:leapfrog_days])
    _path = history.h[:leapfrog_days]
    leapfrog_move = jax.jit(
        lambda key, step: k.leapfrog_trace(key, _target, _path, step_size=step, steps=30)
    )
    return leapfrog_days, leapfrog_move


@app.cell(hide_code=True)
def leapfrog_view(figs, jax, leapfrog_days, leapfrog_move, leapfrog_seed, leapfrog_step, np):
    _trace = leapfrog_move(jax.random.key(leapfrog_seed.value), leapfrog_step.value)
    figs.fig_leapfrog(
        np.arange(leapfrog_days),
        np.asarray(_trace.positions),
        np.asarray(_trace.energies),
        title=(
            f"One fixed-length HMC trajectory, step {leapfrog_step.value} "
            f"(acceptance probability {float(_trace.acceptance_probability):.2f})"
        ),
    )
    return


@app.cell(hide_code=True)
def later_md(mo):
    mo.md(r"""
    ### I.6 Where do later readings enter?

    It's easy to say that auxiliary Kalman "sees the future" and Particle-mGRAD doesn't. That's only
    half true, and this experiment shows which half. Run the same moves twice with the same random
    numbers, but make one day's return much larger the second time. Then measure how the average path
    changes, day by day.

    - **In what is proposed** (left), the difference is real. Auxiliary Kalman's Gaussian reacts on both
      sides of the changed day, because its smoother spreads information in both directions. Particle-mGRAD's
      proposals can't react before that day, since they run forward and only look at their own day.
    - **In what is returned** (right), the difference mostly disappears. Particle-mGRAD's backward draw
      brings the later reading into earlier days too.

    So both are smoothers. Auxiliary Kalman puts later readings into the proposal, and Particle-mGRAD puts
    them into the selection. The old path also carries earlier sweeps' knowledge of later readings, through
    the noisy copy. Twisted variants add explicit look-ahead to particle proposals.
    """)
    return


@app.cell(hide_code=True)
def later_compute(demos, history, truth):
    later_day = 40
    later_length = 64
    later = demos.later_reading_experiment(
        truth,
        history.r2[:later_length],
        history.h[:later_length],
        late_day=later_day,
        late_reading=40.0,
    )
    return later, later_day, later_length


@app.cell(hide_code=True)
def later_view(figs, later, later_day, later_length, np):
    figs.fig_later_reading(
        np.arange(later_length),
        {name: tuple(values) for name, values in later.items()},
        late_day=later_day,
    )
    return


@app.cell(hide_code=True)
def side_by_side_md(mo):
    mo.md(r"""
    ### I.7 Side by side, with the parameters known

    How do these compare in practice? Every chain below starts from an exact posterior path, with the
    parameters known. The left panel is our stochastic volatility model, on all 1,000 days. The right panel is a
    different model for contrast, a Gaussian twin whose readings are sharp, so that one reading says more about
    its day than the dynamics do. The vertical axis is the number of sweeps per effective sample of the path's
    mean, so lower is better.
    """)
    return


@app.cell(hide_code=True)
def regimes_compute(demos, history, truth):
    regimes = demos.fixed_parameter_regimes(truth, history.r2, history.h)
    return (regimes,)


@app.cell(hide_code=True)
def regimes_view(figs, regimes):
    figs.fig_regimes(
        {name: (regime.deltas, regime.lines, regime.bootstrap) for name, regime in regimes.items()}
    )
    return


@app.cell(hide_code=True)
def regimes_tables(demos, mo, regimes):
    mo.vstack(
        [
            item
            for name, regime in regimes.items()
            for item in (
                mo.md(f"**{name}**"),
                mo.md(demos.markdown_table(regime.rows)),
            )
        ]
    )
    return


@app.cell(hide_code=True)
def regimes_md(mo, regimes):
    _weak, _sharp = regimes.values()
    mo.md(f"""
    The curves are one short exploratory chain per setting; the tables use a separately seeded, longer
    chain at the setting each curve picks. A missing point means the chain never moved in that scan.

    **On this model, with the parameters known, moving the path isn't hard.** A day's reading is weak
    next to the dynamics, so candidates drawn from the dynamics land in sensible places, and plain
    bootstrap conditional SMC draws a nearly fresh path each sweep: {_weak.bootstrap:.1f} sweeps per
    effective sample of the path's mean. Particle-mGRAD with a large δ comes close, at {_weak.mgrad:.1f},
    because its proposal is then near the dynamics. A small δ only holds it back. Auxiliary Kalman pays for
    the readings' non-Gaussian shape, which limits δ, and it needs about {_weak.kalman:.0f} sweeps on the whole path.

    **Sweeps aren't the only cost.** One Kalman pass or one short trajectory is far cheaper here than N
    particles per day, so per millisecond the order changes: see the last column. These timings are for
    this teaching code, with one coordinate per day.

    **With very sharp readings the ranking flips.** Now a reading is far more precise than a day's shock,
    so candidates from the dynamics rarely land near it, and bootstrap needs {_sharp.bootstrap:.0f} sweeps. In
    one dimension it takes readings this sharp to hurt it. With many coordinates per day it fails much sooner.
    Particle-mGRAD copes at {_sharp.mgrad:.1f} once δ is matched to the reading's variance, and a δ too large
    makes the gradient nudge overshoot. Auxiliary Kalman's stand-in is now the true model at these known
    parameters, so its verdict accepts every time, at any δ. That alone doesn't make sweeps independent, since the new
    path still depends on the old one through the noisy copy, but at the large δ chosen here almost none of that
    dependence is left: {_sharp.kalman:.1f} sweeps per effective sample.

    So none of them is simply best. Which one wins depends on how informative the readings are, how Gaussian
    they look, how many coordinates each day has, and what you count as cost. And an easy path problem at known
    parameters doesn't mean the joint problem is easy. That is what Parts II to VI are about.
    """)
    return


@app.cell(hide_code=True)
def table_s(mo):
    mo.md(r"""
    ### Summary: moving the path

    | Question | Bootstrap conditional SMC | Particle-mGRAD | Auxiliary Kalman | Conditional path HMC |
    |:---|:---|:---|:---|:---|
    | What does it need from the model? | Transitions you can sample, factors you can evaluate | Gaussian dynamics with fixed variance, a differentiable reading | Densities you can evaluate, a Gaussian stand-in you can build | A differentiable path density |
    | What auxiliary does it add? | No Gaussian path auxiliary (its bank of particles is its auxiliary) | A noisy copy of the path, nudged by the reading's gradient | A noisy copy of the path | Momentum for every day |
    | Where do candidates come from? | N per day, each extending a parent through the dynamics | N per day, each blending its parent's dynamics with the copy | One whole path, or one block at a time, from a Gaussian stand-in built around the copy | One trajectory of leapfrog steps (NUTS: a tree of them) |
    | How does the current path enter? | Kept as a candidate | Kept as a candidate, and it defines the copy | It defines the copy, and a rejection keeps it | It's where the trajectory starts |
    | How do the dynamics shape proposals? | Exactly | Exact mean from the parent | Linearized around the copy | Through the gradient only |
    | How do readings shape proposals? | Not at all, only through weights | That day's gradient, through the nudge | A Gaussian fit to each reading, spread across days by the smoother | Through the gradient at every step |
    | How do later readings enter? | Later weights and the backward draw | Later weights and the backward draw, and via the old path in the copy | The proposal itself | The gradient as the trajectory runs |
    | How is the next path selected? | Backward draw, using weights and true densities | The same, with weights that integrate the copy out | One accept or reject per path or block | HMC: accept or reject the endpoint. NUTS: a rule over the tree |
    | What makes it exact? | The construction conditioned on the current path, with correct weights | The same, with the integrated density of the other candidates | True densities in a forward and reverse ratio | A reversible, volume-preserving integrator, with true densities in the correction |
    | What limits movement? | Candidates can miss when readings are very sharp or days have many coordinates | Small δ moves little, too large δ overshoots, and no explicit look-ahead | Gaussian mismatch adds up into one verdict; blocks limit how much | Stiff directions force small steps, and slow directions need long trajectories |
    | Work per sweep, one coordinate per day | N × T | N × T | About T per colour of blocks | Leapfrog steps × T |
    | Its knobs | N | N, δ | δ, block length | Step size, mass matrix, trajectory length or tree depth |

    All four target the same path posterior, and all four are exact. Particle-mGRAD and auxiliary Kalman
    share their first step and differ in everything after it.
    """)
    return


@app.cell(hide_code=True)
def map_drawer(mo):
    mo.accordion(
        {
            "Detail: a map of the path moves, and where they touch": mo.md(r"""
    Two independent choices generate most of the path moves in the literature. Each row adds knowledge to the
    proposal, and each column is a way to use candidates. Only the bold ones are taught above.

    | What the proposal knows | One whole path, one verdict | Many candidates per day |
    |:---|:---|:---|
    | Only the noisy copy | Random-walk Metropolis | Particle-RWM |
    | + the readings' gradient | MALA | Particle-MALA |
    | + the Gaussian dynamics, one day at a time | | **Particle-aGRAD / Particle-mGRAD** |
    | + all days' dynamics and readings, as one Gaussian | **Auxiliary Kalman** | Twisted Particle-aGRAD |

    Without the noisy copy, the same two columns hold independent Metropolis–Hastings, with the dynamics as the
    proposal, and **bootstrap conditional SMC**.

    **Where they touch,** with the conventions stated, since these are easy to garble:

    - One leapfrog step with fresh momentum and a matched metric proposes exactly what MALA proposes, with δ the
      square of the step size, and its energy correction equals MALA's ratio. This isn't true of NUTS in general.
    - With one day and two total candidates, Particle-RWM and Particle-MALA reduce to random-walk Metropolis and
      MALA if the final candidate is chosen by the paper's forced-move rule, which gives min(1, R). Our kernels
      use a plain draw instead, which gives R / (1 + R), Barker's rule. Both are valid.
    - Under the assumptions of Corenflos & Finke's propositions (§4.6), Particle-mGRAD approaches bootstrap conditional
      SMC when the dynamics are very tight, and Particle-MALA when they are very diffuse. Those limits vary the
      dynamics' covariance, which isn't the same as turning up δ, as the nudge shows in I.3.
    - On a linear-Gaussian model at known parameters, auxiliary Kalman's stand-in equals the true model, so its
      verdict accepts every time. That's because its pseudo-readings match the true Gaussian readings exactly, not
      only to first order.

    **What the noisy copy costs, in the simplest case.** Take a Gaussian direction with variance $v$, and an exact
    draw of the new value given the copy. Then the lag-one correlation is $2v / (\delta + 2v)$, so such a move renews a
    fraction $\delta / (\delta + 2v)$ of that direction. Small δ renews directions with large $v$ slowly, and these are
    typically the slow ones. This is a statement about that exact draw, not a limit on every move that keeps the
    conditional in place.
    """)
        }
    )
    return


@app.cell(hide_code=True)
def part_one_short(mo):
    mo.callout(
        mo.md(r"""
    **Part I in short.** With the parameters known, every path move is exact. They differ in what the proposal knows
    and how many candidates it uses. Particle-mGRAD and auxiliary Kalman start the same way, then diverge. On this model
    plain particles already move the path well per sweep, and sharper readings would change that.

    **What it hands on: a path move.** Give it a current path and a target density over it, and it returns a new path that
    leaves that target in place. It needn't know what else the sampler does.
    """),
        kind="neutral",
    )
    return


@app.cell(hide_code=True)
def part_two_md(mo):
    mo.md(r"""
    ## Part II · Moving the parameters, with the stored path held

    Now let the parameters move. The simplest plan holds the stored path still and updates θ from its
    conditional. That is an ordinary three-dimensional target, so any standard kernel works:

    - **Random-walk Metropolis:** add noise, then accept or reject.
    - **MALA:** add a gradient step as well as noise, and account for the asymmetry in the verdict.
    - **NUTS:** physics in three dimensions. Every leapfrog step evaluates the whole path's density
      at the moved parameters, so one NUTS move costs several passes over the days.

    The figure tunes each one on the conditional at one held path, then restarts all three from the same
    faraway θ. They all arrive in the same green band. A better kernel gets there in fewer steps, but
    none of them leaves it. The limit isn't the kernel. The limit is holding the path.
    """)
    return


@app.cell(hide_code=True)
def race_compute(demos, history, reference_etas, reference_paths, sv):
    race = demos.parameter_kernel_race(
        reference_etas, reference_paths, history.r2, sv.PHYSICAL, name="path"
    )
    return (race,)


@app.cell(hide_code=True)
def race_view(figs, race):
    figs.fig_kernel_race(
        race.traces,
        conditional_mean=race.conditional_mean,
        conditional_sd=race.conditional_sd,
        marginal_mean=race.marginal_mean,
        marginal_sd=race.marginal_sd,
        columns=((0, "Level μ"), (2, "log vol-of-vol, log σ")),
        title="Three kernels, one held path",
    )
    return


@app.cell(hide_code=True)
def table_p1(mo):
    mo.md(r"""
    **A mass matrix can't change this either.** On the parameters alone, a mass matrix is a linear
    rescaling inside the conditional. It helps a kernel cross the green band faster. It can't widen the band.

    **What NUTS buys here.** Roughly a fresh draw from the conditional at every sweep, at the price of
    several path densities per sweep. With three parameters that's a modest gain over a well-tuned random
    walk. With many parameters the gain grows, because a random walk's steps have to shrink as dimensions
    are added, and gradient-guided moves degrade more slowly.

    | Question | Random walk | MALA | NUTS |
    |:---|:---|:---|:---|
    | How is a candidate made? | Add noise | Gradient step plus noise | Simulate a trajectory, build a tree of candidates |
    | What derivative does it need? | None | Conditional log-density gradient | The same, at each leapfrog step |
    | What is held? | The stored path | The stored path | The stored path |
    | What selects the next value? | Accept or reject | Accept or reject, including the asymmetric proposal | A rule over the tree that keeps the target exact |
    | What can a better kernel improve? | Movement within the conditional | Movement within the conditional | Movement within the conditional |
    | What can a mass matrix do? | Shape the noise | Shape the noise and the drift | Shape the momentum |
    | What can none of them do? | Widen the conditional | Widen the conditional | Widen the conditional |
    """)
    return


@app.cell(hide_code=True)
def taking_turns_md(mo):
    mo.md(r"""
    ### Taking turns: the first whole sampler

    Part I's move needs the parameters fixed, and this part's move needs the path fixed. So alternate them: one sweep is a
    parameter move followed by a path move.

    **The two choices compose.** Each step keeps the posterior in place, so the whole sweep does too, whichever pair you
    choose, and neither needs to know what the other is. NUTS on θ with a smoother from Part I is the family called
    **NUTS + smoother**.

    **A running example.** NUTS + bootstrap is the simplest such pairing, and from here on we'll watch what each new choice
    does to it. Here it is on all 1,000 days, from the saved runs with 32 particles. Part VI gives their settings and checks
    that they agree.
    """)
    return


@app.cell(hide_code=True)
def taking_turns_table(demos, method_labels, mo, runs):
    mo.md(demos.markdown_table(runs.example_rows(method_labels, ("nuts_bootstrap_path",))))
    return


@app.cell(hide_code=True)
def taking_turns_reading(mo, regimes, runs):
    _weak = next(iter(regimes.values()))
    _sweeps = runs.sweeps_per_effective_sample("nuts_bootstrap_path")
    mo.md(rf"""
    Each move on its own comes close to a fresh draw: NUTS from θ's conditional, and bootstrap from the path's, at
    {_weak.bootstrap:.1f} sweeps per effective sample with the parameters known. Together they still need
    {_sweeps[0]:.0f} sweeps for level and {_sweeps[2]:.0f} for log σ. So neither move is at fault. The limit is the tie
    between them, the narrow conditional from the opening, and a sampler taking turns can only walk along that ridge in small
    steps.

    To get past it, either hold something else (Part III) or stop holding the path during the parameter move (Part IV).
    """)
    return


@app.cell(hide_code=True)
def part_two_short(mo):
    mo.callout(
        mo.md(r"""
    **Part II in short.** With the path held, a better kernel or a better mass matrix reaches the conditional faster, and
    nothing gets past it. Getting past it takes a change in what's held, or in how the moves are organized.

    **What it hands on: a parameter move,** which does for θ what a path move does for the path, and **taking turns**, the
    simplest way to combine the two.
    """),
        kind="neutral",
    )
    return


@app.cell(hide_code=True)
def part_three_md(mo):
    mo.md(r"""
    ## Part III · Changing what is held

    The stored path doesn't have to be $h$. A sampler may store any invertible function of $(\theta, h)$
    and hold that instead. When θ then moves with the stored values fixed, the physical path moves too. The
    posterior stays exactly the same, as long as the density includes the change of variables.

    | Stored path | How the physical path comes back |
    |:---|:---|
    | **The path itself** | $h_t = s_t$ |
    | **Daily shocks** (non-centred; Papaspiliopoulos, Roberts & Sköld 2007) | Rebuilt forward through the dynamics: $h_{t+1} = f_\theta(h_t) + \sigma s_{t+1}$ |
    | **The chart** | $h_t = c_t + B_t(\eta - \eta^*) + e^{A_t(\eta - \eta^*)}(z_t - c_t)$ |

    Here $\eta$ is θ on the sampler's unconstrained scale, $(\mu, \log\kappa, \log\sigma)$, and $\eta^*$ is the chart's
    reference point, the posterior mean from a pilot run. The chart's three pieces come from a Gaussian
    approximation of the path at $\eta^*$:

    - **Centre $c$:** the approximate posterior mean path at the reference.
    - **Translation $B$:** how that mean path responds when the parameters move, computed by differentiating
      the Gaussian approximation.
    - **Scale $A$:** a structural choice, not derived from the Gaussian: 1 for $\log\sigma$ and 0 otherwise, so
      deviations from the centre resize with σ. It applies on day 0 as well, whose physical law stays fixed, and
      the density terms pay for that.

    At $\eta = \eta^*$ the chart is the identity. All it adds is a response to the parameters. The Gaussian
    only shapes the coordinates: every density is still the exact nonlinear model, plus the chart's Jacobian.
    Its coefficients are frozen before the retained draws, but the map itself still depends on θ, so NUTS on θ
    differentiates through it.

    Move the sliders, which start from one posterior draw. **Raise log σ** with the path held, and the transitions pay
    heavily, because the old residuals no longer fit the new σ. With the shocks held, the standardized shocks don't
    change at all: the physical transition density falls by $(T-1)\,\Delta\log\sigma$ and the Jacobian adds exactly that
    back. Those two cancel in the table, and only a small change in the readings is left. The chart behaves much the same,
    by resizing deviations around its centre. **Raise the level** with the path held, and the readings pay. Holding
    the shocks doesn't help, because level doesn't enter the dynamics. The chart shifts the path down to compensate, so
    the readings barely notice, and the transitions pay a little for moving the path against the pull towards zero.
    """)
    return


@app.cell(hide_code=True)
def decode_controls(mo):
    decode_level = mo.ui.slider(
        start=-0.3, stop=0.3, step=0.05, value=0.0, label="Change in level", show_value=True
    )
    decode_log_vol = mo.ui.slider(
        start=-0.4, stop=0.4, step=0.05, value=0.2, label="Change in log σ", show_value=True
    )
    decode_parts = mo.ui.dropdown(
        options={
            "Translation and scale": "both",
            "Translation only": "translation",
            "Scale only": "scale",
        },
        value="Translation and scale",
        label="Chart parts",
    )
    mo.hstack([decode_level, decode_log_vol, decode_parts], justify="start", gap=2)
    return decode_level, decode_log_vol, decode_parts


@app.cell(hide_code=True)
def decode_view(
    chosen_draw,
    decode_level,
    decode_log_vol,
    decode_parts,
    demos,
    figs,
    history,
    jnp,
    mo,
    np,
    reference_etas,
    reference_paths,
    runs,
    sv,
):
    _chart = {
        "both": runs.chart,
        "translation": sv.chart_ablation(runs.chart, scale=False),
        "scale": sv.chart_ablation(runs.chart, translation=False),
    }[decode_parts.value]
    _eta = jnp.asarray(reference_etas[chosen_draw])
    _path = jnp.asarray(reference_paths[chosen_draw])
    _moved = _eta + jnp.asarray([decode_level.value, 0.0, decode_log_vol.value])
    _coordinates = {
        "Hold the path": sv.PHYSICAL,
        "Hold the daily shocks": sv.INNOVATIONS,
        "Hold the chart": _chart,
    }
    _ledgers = {
        name: demos.decode_ledger(coordinate, _eta, _path, _moved, history.r2)
        for name, coordinate in _coordinates.items()
    }
    _shown = 300
    mo.vstack(
        [
            figs.fig_decode(
                np.arange(_shown),
                {
                    name: ledger.path[:_shown] - np.asarray(_path[:_shown])
                    for name, ledger in _ledgers.items()
                },
                {name: ledger.terms for name, ledger in _ledgers.items()},
                title="What each choice does when the parameters change, starting from one posterior draw",
            ),
            mo.md(
                demos.markdown_table(
                    [
                        {
                            "Held": name,
                            "Prior": round(float(ledger.terms[0]), 1),
                            "Initial law": round(float(ledger.terms[1]), 1),
                            "Transitions": round(float(ledger.terms[2]), 1),
                            "Readings": round(float(ledger.terms[3]), 1),
                            "Jacobian": round(float(ledger.terms[4]), 1),
                            "Total": round(float(ledger.terms.sum()), 1),
                        }
                        for name, ledger in _ledgers.items()
                    ]
                )
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def widths_compute(demos, history, reference_etas, reference_paths, runs, sv):
    width_coordinates = {
        "Hold the path": sv.PHYSICAL,
        "Hold the daily shocks": sv.INNOVATIONS,
        "Chart, translation only": sv.chart_ablation(runs.chart, scale=False),
        "Chart, scale only": sv.chart_ablation(runs.chart, translation=False),
        "Chart, both": runs.chart,
    }
    coordinate_widths = demos.coordinatewise_widths_cached(
        reference_etas, reference_paths, history.r2, width_coordinates
    )
    block_widths = demos.block_conditional_widths(
        reference_etas,
        reference_paths,
        history.r2,
        {
            name: width_coordinates[name]
            for name in ("Hold the path", "Hold the daily shocks", "Chart, both")
        },
    )
    return block_widths, coordinate_widths


@app.cell(hide_code=True)
def widths_view(coordinate_widths, figs, mo, np):
    _names = tuple(coordinate_widths)
    _mean = np.stack([coordinate_widths[name].mean for name in _names])
    _spread = np.stack([coordinate_widths[name].spread for name in _names])
    mo.vstack(
        [
            mo.md(r"""
    **How much wider does each choice make the conditional?** For each held quantity, below is the sd of one
    parameter given that stored path and the other two parameters, as a fraction of the marginal sd, across
    60 posterior draws. The dots show the mean and the bars the spread across draws. A value near 1 means that
    holding this quantity leaves that parameter about as much room as its marginal.
    """),
            figs.fig_widths(
                _names,
                ("Level μ", "log decay, log κ", "log vol-of-vol, log σ"),
                _mean,
                _spread,
                title="Conditional width under each held quantity",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def widths_md(block_widths, coordinate_widths, mo):
    _path = coordinate_widths["Hold the path"].mean
    _translation = coordinate_widths["Chart, translation only"].mean
    _scale = coordinate_widths["Chart, scale only"].mean
    _both = coordinate_widths["Chart, both"].mean
    _shocks = coordinate_widths["Hold the daily shocks"].mean
    _block_path = block_widths.table["Hold the path"].mean
    _block_shocks = block_widths.table["Hold the daily shocks"].mean
    _block_chart = block_widths.table["Chart, both"].mean
    mo.md(rf"""
    Each part of the chart does one job. Translation takes level from {_path[0]:.2f} to {_translation[0]:.2f}
    of its marginal width and leaves vol-of-vol at {_translation[2]:.2f}. Scale takes vol-of-vol from
    {_path[2]:.2f} to {_scale[2]:.2f} and leaves level at {_scale[0]:.2f}. Together they reach
    {_both[0]:.2f} and {_both[2]:.2f}. Decay barely changes. Daily shocks do about as well for vol-of-vol
    ({_shocks[2]:.2f}), and nothing for level ({_shocks[0]:.2f}).

    These fix the other two parameters. A three-parameter kernel like NUTS can also follow correlations between
    them, so this next table is the fairer one. It comes from conditional NUTS runs at 16 posterior draws
    (the shortest of them still has about {block_widths.fewest_effective_draws:.0f} effective draws of each parameter):

    | Held quantity | Level | log decay | log vol-of-vol |
    |:---|---:|---:|---:|
    | The path | {_block_path[0]:.2f} | {_block_path[1]:.2f} | {_block_path[2]:.2f} |
    | Daily shocks | {_block_shocks[0]:.2f} | {_block_shocks[1]:.2f} | {_block_shocks[2]:.2f} |
    | The chart | {_block_chart[0]:.2f} | {_block_chart[1]:.2f} | {_block_chart[2]:.2f} |

    **Why this is the right direction.** Imagine a map under which the stored path's distribution doesn't depend
    on θ at all. Then holding it would cost nothing: the conditional of θ would equal its marginal. No such map is
    available, and the chart is a local approximation to it. In a jointly Gaussian posterior with an exact slope,
    the translation removes the mean's dependence on θ exactly. The structural scale is cruder: it isn't the true
    conditional covariance, it can over-resize deviations the readings pin down, and time correlations remain.

    **Shocks and chart compared.** Shocks are independent of θ under the prior, not the posterior, so informative
    readings tie them back to θ, which is the classic centred versus non-centred trade-off. And each shock moves all
    later days, so evaluating one density means rebuilding the path day by day, and the posterior over shocks isn't a
    chain that these smoothers can walk along. You can still pair shocks with a smoother, by decoding to $h$ for the
    path move and encoding back afterwards, and that stays exact, but then the path move is the same physical move as
    without shocks. The chart acts on each day separately, so particle and Kalman moves run directly on $z$, in the
    coordinates the parameters are held in, as a chain with the same time structure.

    **What the chart does to the path step.** With θ fixed, the chart is only a shift and a rescale of each day. Bootstrap
    is then exactly the same move on the physical path. Particle-mGRAD and auxiliary Kalman keep their form, but a fixed δ
    in $z$ means $\delta\,(\sigma/\sigma^*)^2$ in $h$, so their physical step size varies with σ. So NUTS + bootstrap gives the
    cleanest measure of what the chart does to the parameter step alone. With mGRAD and Kalman, the comparison also includes
    that rescaling.
    """)
    return


@app.cell(hide_code=True)
def chart_example_view(demos, method_labels, mo, runs):
    mo.vstack(
        [
            mo.md(r"""
    **The running example, with this one choice changed.** Bootstrap is the same physical move in both rows, so the difference
    comes from what the parameter move holds.
    """),
            mo.md(
                demos.markdown_table(
                    runs.example_rows(
                        method_labels, ("nuts_bootstrap_path", "nuts_bootstrap_chart")
                    )
                )
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def prediction_view(block_widths, demos, mo, runs):
    mo.vstack(
        [
            mo.md(r"""
    **Did the slices point the right way?** The opening rule of thumb says 2r² − 1 sweeps per effective sample, where r
    is the marginal sd over the conditional sd, and the three-parameter table above gives r for each held quantity. That's a
    formula for one scalar Gaussian with perfect draws, and this
    posterior has three parameters tied to a whole path, so it's only a reference. NUTS + bootstrap is the closest
    approximation we have to its assumptions: NUTS comes close to a fresh conditional draw of θ at a held path, and bootstrap
    comes close to a fresh path at known parameters. Neither is guaranteed throughout the joint chain.
    """),
            mo.md(
                demos.markdown_table(
                    demos.predicted_rows(
                        block_widths.table,
                        runs,
                        name_off="nuts_bootstrap_path",
                        name_on="nuts_bootstrap_chart",
                    )
                )
            ),
            mo.md(r"""
    Level and log σ land within a factor of about three of the rule, chart on or off. Log κ doesn't with the chart off: its
    own conditional is wide, yet it takes many sweeps, which fits it being tied to slower directions of the path and the other
    parameters. So narrowness is one limit on these samplers, and the rule doesn't see the others.
    """),
        ]
    )
    return


@app.cell(hide_code=True)
def table_c(coordinate_widths, mo):
    _path = coordinate_widths["Hold the path"].mean
    _translation = coordinate_widths["Chart, translation only"].mean
    _both = coordinate_widths["Chart, both"].mean
    _shocks = coordinate_widths["Hold the daily shocks"].mean
    mo.md(f"""
    | Question | The path itself | Chart, translation only | Chart, translation and scale | Daily shocks |
    |:---|:---|:---|:---|:---|
    | What is stored? | The log-variance path | Deviations from a centre that moves with θ | Deviations whose centre and size move with θ | $h_0$ and each day's shock |
    | Parameters change, stored values stay. What happens to $h$? | Nothing | It shifts | It shifts and its deviations resize | It's rebuilt through the dynamics |
    | What dependence does this reduce? | None by design | θ and the conditional mean path | That, and part of θ and the path's scale | θ and the prior driving noise |
    | What keeps it exact? | The original density | The density at the decoded path | The same, plus the Jacobian | The same, plus the Jacobian |
    | What stays difficult? | Narrow conditionals | Scale dependence, nonlinear mean response | Residual time correlation, imperfect scaling | Readings tying shocks back to θ |
    | Day-by-day structure for smoothers? | Yes | Yes | Yes | No: each shock moves all later days |
    | Conditional sd ÷ marginal sd: level | {_path[0]:.2f} | {_translation[0]:.2f} | {_both[0]:.2f} | {_shocks[0]:.2f} |
    | Conditional sd ÷ marginal sd: log σ | {_path[2]:.2f} | {_translation[2]:.2f} | {_both[2]:.2f} | {_shocks[2]:.2f} |
    """)
    return


@app.cell(hide_code=True)
def part_three_short(mo):
    mo.callout(
        mo.md(r"""
    **Part III in short.** What's held decides how far the parameters can go in one move. Translation frees level and
    scale frees vol-of-vol, which no mass matrix on θ could do, because this widens the conditional itself. Daily shocks free
    vol-of-vol too, at the cost of a day-by-day rebuild and of the local structure these smoothers use.

    **What it hands on: a stored path,** meaning a one-to-one decode to $h$ for each θ, and its Jacobian. The parameter move
    only sees a different density. The particle and Kalman moves need that density to stay day by day, which the chart's does
    and the shocks' doesn't.
    """),
        kind="neutral",
    )
    return


@app.cell(hide_code=True)
def part_four_md(mo):
    mo.md(r"""
    ## Part IV · Changing how the moves are organized

    So far the sampler takes turns. Holding anything means the parameters move against something fixed, however wide
    Part III made it. There are other ways to organize the moves, and here are three.

    ### IV.1 Move together: joint NUTS

    Put θ and the whole stored path into one vector and run NUTS on all of it. Nothing is held, so there is no
    conditional to be trapped in, and one trajectory can travel along a ridge. There's also no separate smoother.

    The price is that one sampler has to cope with every scale at once, and coordinates decide how hard that is. In
    $h$ coordinates the tight day-to-day dynamics are stiff, so steps stay small and trajectories get long, while
    vol-of-vol can only change by resizing all the steps together. In shock coordinates the prior is a plain
    standard normal, so steps can be much larger, and scaling σ with the shocks held resizes all the steps by
    itself. Each gradient then has to rebuild the path day by day. In chart coordinates the map is day by day, but the
    dynamics stay stiff, as in $h$.

    The first figure shows what one trajectory does in each system. These are newly simulated, fixed-length HMC
    trajectories, using the step size and mass matrix NUTS settled on in the saved runs, and as many steps as a typical
    NUTS move took. They aren't the NUTS trees themselves. The second figure shows real consecutive NUTS draws.
    """)
    return


@app.cell(hide_code=True)
def joint_view(demos, figs, history, mo, runs, sv):
    _names = {
        "joint_path": ("Coordinates: h", sv.PHYSICAL),
        "joint_shocks": ("Coordinates: daily shocks", sv.INNOVATIONS),
        "joint_chart": ("Coordinates: chart", runs.chart),
    }
    _trajectories = demos.joint_trajectories(
        runs, history.r2, {name: value[1] for name, value in _names.items()}
    )
    mo.vstack(
        [
            figs.fig_joint_trajectories(
                {
                    f"{label}: {_trajectories[name].steps} steps, "
                    f"acceptance {_trajectories[name].acceptance:.2f}": _trajectories[
                        name
                    ].summaries
                    for name, (label, _) in _names.items()
                },
                cloud=runs.cloud,
            ),
            figs.fig_joint_moves(
                {
                    f"{label}: {runs.leapfrog_steps(name):.0f} leapfrog steps per draw": runs.substeps(
                        name
                    )
                    for name, (label, _) in _names.items()
                },
                cloud=runs.cloud,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def joint_example_view(demos, method_labels, mo, runs):
    _s = runs.sweeps_per_effective_sample
    _taking_turns, _joint = _s("nuts_bootstrap_path"), _s("joint_path")
    mo.vstack(
        [
            mo.md(r"""
    **The running example, next to joint NUTS in each stored path.**
    """),
            mo.md(
                demos.markdown_table(
                    runs.example_rows(
                        method_labels,
                        (
                            "nuts_bootstrap_path",
                            "nuts_bootstrap_chart",
                            "joint_path",
                            "joint_chart",
                            "joint_shocks",
                        ),
                    )
                )
            ),
            mo.md(f"""
    With $h$ stored, moving together takes level from {_taking_turns[0]:.0f} to {_joint[0]:.0f} sweeps, while log σ goes from
    {_taking_turns[2]:.0f} to {_joint[2]:.0f}, a gap small enough to call a tie. The stored path does much more,
    even though nothing is held: {_s("joint_chart")[2]:.1f} sweeps for log σ in the chart and {_s("joint_shocks")[2]:.1f} in
    daily shocks. It's still a choice here, because it sets the geometry the trajectory runs in. A joint sweep doesn't cost the
    same as a sweep of taking turns, as the step counts above suggest, and Part VI compares time.
    """),
        ]
    )
    return


@app.cell(hide_code=True)
def mass_matrix_md(mo, np, rms_residuals):
    _span = np.quantile(rms_residuals, 0.95) / np.quantile(rms_residuals, 0.05)
    mo.md(rf"""
    ### Is the chart a mass matrix?

    Joint NUTS puts θ and the path under one mass matrix. A dense one can encode how they're coupled, which sounds a lot like
    the chart.

    **It isn't one, though one part of it overlaps with a dense one.** Four facts settle it. Convention: momentum has covariance
    $M$, and kinetic energy is $p^\top M^{{-1}} p / 2$.

    1. **A fixed linear change of coordinates is a dense mass matrix.** HMC on $w$, with $x = Lw$ and identity mass,
       traces the same path as HMC on $x$ with inverse mass $LL^\top$. This holds for any target, Gaussian or not.
    2. **So a fixed shift can live inside a dense mass matrix.** In a jointly Gaussian posterior the translation is
       $B = \mathrm{{Cov}}(h, \eta)\,\mathrm{{Var}}(\eta)^{{-1}}$. Choose $M^{{-1}}$ to be the joint covariance, parameters first. In its
       Cholesky factor, $L_{{h,\eta}} = B\,L_{{\eta,\eta}}$, so $B$ is in there. The raw cross-block, $\mathrm{{Cov}}(h, \eta)$, is a different object.
    3. **The scale can't.** Its right size changes with σ, and a mass matrix is one fixed matrix. Across this posterior,
       the path's RMS daily residual varies by a factor of {_span:.1f} between its 5th and 95th percentiles (the right panel
       near the top of the notebook). This part is closer to non-centring.
    4. **A metric on one block can't change that block's conditional.** NUTS on θ alone, with any mass matrix, still holds
       $h$. With the chart it holds $z$, and so targets a different, wider conditional. And a dense *joint* mass matrix
       can't help at all in a method whose parameter step never moves the path.

    They aren't alternatives: a chart can sit under any metric. The drawer checks facts 1 and 2 numerically.
    """)
    return


@app.cell(hide_code=True)
def mass_compute(demos, history, truth):
    mass_checks = demos.mass_matrix_checks(history.r2, truth)
    metric_table = demos.metric_versus_chart(history.r2)
    return mass_checks, metric_table


@app.cell(hide_code=True)
def mass_drawer(figs, mass_checks, mo):
    mo.accordion(
        {
            "Detail: the checks behind the four facts": mo.vstack(
                [
                    mo.md(f"""
    **Fact 1.** Fixed-length HMC on the three-parameter conditional, once with a dense inverse mass and once in
    coordinates whitened by its Cholesky factor, using the same random numbers. The largest difference between
    the two position trajectories, in the original coordinates, is {mass_checks.trajectory_difference:.1e}.

    **Fact 2.** On a linear-Gaussian twin with unknown level, the figure compares the translation, obtained by
    differentiating the conditional mean path, with the regression slope from the exact joint covariance. The largest
    difference is {mass_checks.slope_difference:.1e}. The block of the covariance's Cholesky factor, divided by the
    parameter's own entry, gives the same thing, to within {mass_checks.cholesky_difference:.1e}. The raw covariance block has a
    different scale.
    """),
                    figs.fig_slope_check(
                        mass_checks.slope_from_chart,
                        mass_checks.slope_from_covariance,
                        mass_checks.slope_from_cholesky,
                        mass_checks.raw_block,
                    ),
                ]
            )
        }
    )
    return


@app.cell(hide_code=True)
def metric_md(mo):
    mo.md(r"""
    **The direct experiment.** Joint NUTS on a 300-day prefix, where the two ideas can be compared head on, with each
    combination: 8 chains, 1,000 warmup and 6,000 retained draws each. "Worst R̂" compares the chains, and above about 1.01
    the sweep numbers beside it are rough. "Gradients per effective sample" multiplies sweeps by leapfrog steps, so it counts
    target gradients, which don't cost the same in each coordinate system. A dense metric on the path also has to store and
    multiply a 303 × 303 matrix at every step, and estimate it from warmup draws. At 3,000 days that would be about 9 million
    entries.
    """)
    return


@app.cell(hide_code=True)
def metric_view(demos, metric_table, mo):
    mo.md(demos.markdown_table(metric_table))
    return


@app.cell(hide_code=True)
def metric_reading(metric_table, mo):
    _row = {(row["Coordinates"], row["Mass matrix"]): row for row in metric_table}
    _level = "Sweeps per effective sample: level"
    _scale = "Sweeps per effective sample: log σ"
    _work = "Gradients per effective sample: log σ"
    mo.md(f"""
    In $h$ coordinates with a diagonal mass matrix, level takes
    {_row[("h", "diagonal")][_level]:.1f} sweeps per effective sample and log σ takes {_row[("h", "diagonal")][_scale]:.0f}.
    A dense mass matrix removes most of the level autocorrelation, leaving {_row[("h", "dense")][_level]:.1f} sweeps, which
    fits a largely linear relationship between level and path. Log σ stays slow, at about {_row[("h", "dense")][_scale]:.0f}.
    Both $h$ runs have R̂ above 1.01, so their log σ numbers are rough, and the gap between them may not mean much. The chart
    with a plain diagonal mass matrix gets level to
    {_row[("chart", "diagonal")][_level]:.1f} and log σ to {_row[("chart", "diagonal")][_scale]:.0f}, which fits its
    parameter-dependent rescaling.

    So coordinates and metrics address overlapping but different parts of this geometry, and they can be combined. Daily
    shocks use the fewest gradients here, {_row[("daily shocks", "diagonal")][_work]:,} per effective sample of log σ against
    {_row[("chart", "diagonal")][_work]:,} for the chart. That counts gradients, not time, since a gradient costs a
    different amount in each coordinate system. What the chart offers beyond shocks is the day-by-day structure Part I's
    smoothers need. All of this describes one dataset and these tuning budgets.
    """)
    return


@app.cell(hide_code=True)
def pmmh_md(mo):
    mo.md(r"""
    ### IV.2 A fresh path for each candidate: PMMH

    The other extreme integrates the path out. Particle marginal Metropolis–Hastings (Andrieu, Doucet & Holenstein
    2010) proposes a new θ, runs a whole particle filter at it, and accepts or rejects using the filter's estimate of
    the marginal likelihood. That estimate is nonnegative and unbiased, which is exactly what keeps the method exact
    despite the noise. With no path held there's no conditional limit, in principle: θ is judged on its marginal.

    The catch is noise in the estimate. The figure shows how much this particular estimator's logarithm varies at fixed
    θ, for a bootstrap filter. It grows with the number of days and shrinks with particles. When it's large, a chain that
    happens on a lucky estimate gets stuck there. More particles or better proposals reduce it, at a cost.

    PMMH isn't among the long runs here, because it isn't one of the families you asked about. For this estimator, these
    readings and θ at the truth, the figure suggests several hundred particles to bring the sd near 1 at 1,000 days, a rough
    tuning reference (Doucet, Pitt, Deligiannidis & Kohn 2015). That's about the noise alone, not PMMH's overall efficiency.
    """)
    return


@app.cell(hide_code=True)
def pmmh_view(demos, figs, history, truth):
    figs.fig_pmmh_noise(*demos.pmmh_noise(truth, history.r2))
    return


@app.cell(hide_code=True)
def mpgibbs_md(mo):
    mo.md(r"""
    ### IV.3 Update the path while undecided: MPGibbs

    Marginalized particle Gibbs (Corenflos 2025) sits between taking turns and integrating the path out. One sweep has
    four steps:

    1. **Noisy copy of the parameters.** On the sampling scale, draw $a \sim N(\eta, \Sigma/2)$, then a new candidate
       $\eta' \sim N(a, \Sigma/2)$. Now there are two candidates, and a hidden label says which one is real. Here $\Sigma$ is
       a pilot's estimate of the posterior covariance times a squared **candidate scale** $s^2$. So $s$ is the proposal's sd
       per coordinate, after whitening by that covariance.
    2. **Temporary target.** With the two candidates fixed, the stored path's target is a mixture of their complete
       joint densities: $G(z) = \sum_{l} \tilde\pi(\eta_l, z)$. Here $\tilde\pi$ is the exact density in the sampler's
       coordinates, including the parameter transform's Jacobian and, with the chart on, the chart's. The noisy-copy
       terms cancel, because the two halves are symmetric.
    3. **Move the path under the mixture.** Use any move that leaves $G$ invariant. Each candidate keeps a running
       density along the days, so a particle's weight asks how well it fits *either* candidate.
    4. **Flip.** Draw the label with probability proportional to $\tilde\pi(\eta_l, z_{\text{new}})$, and the winner is the next θ.

    The crucial difference from taking turns is the order. The path has already moved under both candidates when
    the flip happens. If step 3 were a perfect draw from $G$, the flip's chance would depend only on the two marginals,
    so it would be a Barker move on the marginal, with the path integrated out (Corenflos 2025, §3.3). Barker means
    picking each candidate in proportion to its density, rather than Metropolis's accept or reject. A finite-particle
    sweep is still exact, but it isn't that ideal move, and the second figure below compares it with the ideal.

    With the chart on, steps 2 to 4 happen in $z$ coordinates: both candidates share one stored path, and each decodes
    it into its own physical path. That's what the left panel shows when the chart switch is on. Each candidate's chart is
    still a shift and a rescale of each day, but they differ, so no single physical move matches this one, unlike in Part III.

    **This is where the path move stops being independent of the rest.** Taking turns hands it a density at one θ. MPGibbs
    hands it a mixture over two, and in the code that's the same call with two candidates instead of one. But the mixture is
    over whole paths, not one day at a time, because a candidate is global to the path. So it isn't a day-by-day chain any more:
    the particles need running densities, and the backward draw uses both past and future factors. Each move adapts in its own
    way:

    - **Particle-mGRAD** proposes from one Gaussian built at the candidates' mean, fixed by the two candidates. Its
      reading gradient depends only on that day's value, never on a particle's past. The exact mixture enters
      the weights and the backward draw.
    - **Bootstrap** proposes from the candidates' dynamics, mixed by each particle's running label probabilities, and
      the weight pays for that mixture.
    - **Whole-path auxiliary Kalman** uses one Gaussian at the candidates' mean, and its verdict uses $G$.
    - **Blocked auxiliary Kalman** first draws a temporary label given the current path, updates blocks under that one
      candidate, and still flips at the end. This can keep part of the conditional bottleneck the mixture is
      meant to ease. Updating blocks one at a time against the full mixture is also valid.

    The noisy copy of θ is the same trick as the noisy copy of the path, and it's tempting to see θ as one more day
    with two particles. That's a useful analogy only, because θ acts on every day, not one.

    **Try the chart switch below at the default settings.** With it off, the grey line, the new candidate's chance at the old
    path, collapses within days, because the old path fits only the current candidate. The path move rescues part of that (orange),
    but here the final chance still ends near zero. With it on, both lines stay near a half for most of the days, and the new
    candidate ends with a real chance. That's one sweep's random numbers, and the flip rates over many sweeps come further down.
    """)
    return


@app.cell(hide_code=True)
def mpgibbs_controls(mo):
    mpgibbs_chart = mo.ui.switch(value=False, label="Chart on")
    mpgibbs_scale = mo.ui.slider(
        steps=[0.25, 0.5, 1.0, 2.0, 4.0], value=1.0, label="Candidate scale", show_value=True
    )
    mpgibbs_seed = mo.ui.slider(start=0, stop=20, value=4, label="Random numbers")
    mo.hstack([mpgibbs_chart, mpgibbs_scale, mpgibbs_seed], justify="start", gap=2)
    return mpgibbs_chart, mpgibbs_scale, mpgibbs_seed


@app.cell(hide_code=True)
def mpgibbs_setup(
    chosen_draw,
    history,
    jax,
    jnp,
    k,
    reference_etas,
    reference_paths,
    runs,
    sv,
):
    mpgibbs_days = 200
    _r2 = history.r2[:mpgibbs_days]
    _eta = jnp.asarray(reference_etas[chosen_draw])
    _path = jnp.asarray(reference_paths[chosen_draw])[:mpgibbs_days]
    _cholesky = jnp.asarray(runs.tuning["posterior_cholesky"])
    _prefix_chart = sv.Chart(
        runs.chart.center[:mpgibbs_days],
        runs.chart.slopes[:mpgibbs_days],
        runs.chart.exponents[:mpgibbs_days],
        runs.chart.reference,
    )
    _spec = k.MGradSpec(particles=32, delta=1.0)

    def _move(key, scale, stored, coordinates):
        return k.mpgibbs_trace(key, _eta, stored, _r2, coordinates, _spec, scale * _cholesky)

    _off = jax.jit(lambda key, scale: _move(key, scale, _path, sv.PHYSICAL))
    _stored = sv.encode(_prefix_chart, _eta, _path)
    _on = jax.jit(lambda key, scale: _move(key, scale, _stored, _prefix_chart))
    mpgibbs_moves = {False: _off, True: _on}
    return mpgibbs_days, mpgibbs_moves


@app.cell(hide_code=True)
def mpgibbs_view(
    demos,
    figs,
    jax,
    mo,
    mpgibbs_chart,
    mpgibbs_days,
    mpgibbs_moves,
    mpgibbs_scale,
    mpgibbs_seed,
    np,
):
    _trace = mpgibbs_moves[mpgibbs_chart.value](
        jax.random.key(mpgibbs_seed.value), mpgibbs_scale.value
    )
    _values = [
        ("Current candidate", _trace.candidates[0]),
        ("Noisy copy a", _trace.parameter_auxiliary),
        ("New candidate", _trace.candidates[1]),
    ]
    mo.vstack(
        [
            mo.md(
                demos.markdown_table(
                    [
                        {
                            "Value": name,
                            "Level μ": round(float(value[0]), 3),
                            "Decay κ": round(float(np.exp(value[1])), 4),
                            "Vol-of-vol σ": round(float(np.exp(value[2])), 3),
                        }
                        for name, value in _values
                    ]
                )
            ),
            figs.fig_mpgibbs_stages(
                np.arange(mpgibbs_days),
                paths_before=np.asarray(_trace.candidate_physical_paths_before),
                paths_after=np.asarray(_trace.candidate_physical_paths_after),
                probability_before=np.asarray(_trace.label_probabilities_before[:, 1]),
                probability_after=np.asarray(_trace.label_probabilities_after[:, 1]),
                final_probability=float(_trace.final_label_probabilities[1]),
                selected=int(_trace.selected_label),
                title="One MPGibbs sweep",
            ),
            mo.accordion(
                {
                    "The code for this move": mo.ui.code_editor(
                        demos.kernel_source("ensemble_trace"), language="python", disabled=True
                    )
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def barker_compute(demos):
    barker = demos.barker_reference()
    return (barker,)


@app.cell(hide_code=True)
def barker_view(barker, figs, mo):
    mo.vstack(
        [
            mo.md(r"""
    **Compared with the ideal.** On a small linear-Gaussian oracle, Kalman filtering gives exact likelihoods and exact path
    draws, so we can run the reference chains too. The figure shows how often the new candidate wins, by number of
    particles, with ± 2 standard errors across independent chains. The dashed lines are references, themselves averages over
    chains: a flip after a perfect path draw from the mixture; Metropolis with the exact marginal likelihood, which is what
    PMMH approaches with many particles; and a flip at a path held from its conditional, which is taking turns. None of these
    lines bounds the others, and a finite-particle sweep can land above or below the ideal.
    """),
            figs.fig_barker_reference(
                barker.particles,
                mpgibbs=barker.mpgibbs,
                pmmh=barker.pmmh,
                ideal=barker.ideal,
                ideal_metropolis=barker.ideal_metropolis,
                held=barker.held,
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def walk_compute(demos):
    walk = demos.ideal_walk()
    return (walk,)


@app.cell(hide_code=True)
def walk_view(demos, figs, method_labels, mo, np, runs, walk):
    _smoothers = {
        "bootstrap": "Bootstrap conditional SMC",
        "mgrad": "Particle-mGRAD",
        "ak_whole": "Auxiliary Kalman, whole path",
        "ak_blocks": "Auxiliary Kalman, blocks",
    }
    _rows = runs.walk_rows(method_labels, walk)
    _particles = [row for row in _rows if "Kalman" not in str(row["Method"])]
    _gap = max(
        max(float(row["Measured: level"]), float(row["Measured: log σ"]))
        / float(row["Ideal walk at that scale"])
        for row in _particles
    )
    _best = int(np.argmin(walk.sweeps))
    mo.vstack(
        [
            mo.md(r"""
    **Over many sweeps: why the chart helps MPGibbs, and what's left after it.** After a perfect path draw, MPGibbs is a Barker
    random walk on the marginal. We can work out what that ideal does on a three-dimensional standard Gaussian, at each
    candidate scale. It gives two references: how often the new candidate should win, the grey curve below, and how many sweeps
    per effective sample the walk needs. The coloured curves are real flip rates from short pilot chains on the 1,000 days,
    which also chose the candidate scale of each saved run.
    """),
            figs.fig_flip_rates((walk.scales, walk.flip_rates), runs.flip_rates(_smoothers)),
            mo.md(r"""
    With the chart off, the new candidate rarely wins at larger candidate scales. The two candidates want different physical
    paths, and one sweep's path move can't bridge them. With the chart on, the same stored $z$ decodes to a different physical
    path for each candidate, so the rate stays much closer to the ideal. So the pilots pick small scales with the chart off and
    larger ones with it on. The scale then sets the pace of the walk, and the table puts each run next to an ideal walk at the
    scale it used.
    """),
            mo.md(demos.markdown_table(_rows)),
            mo.md(rf"""
    The two particle moves stay within {_gap:.1f}× of that ideal, chart on or off. So the random-walk candidate accounts for a
    large part of their cost, and the chart helps by allowing a larger scale. How the rest of the gap splits between imperfect path
    moves and a non-Gaussian marginal, we can't say from this. The Kalman moves land further away, which fits their path move having
    its own slow mode.

    Even at its best scale, {walk.scales[_best]:g}, the ideal walk needs about {walk.sweeps[_best]:.0f} sweeps, so NUTS on θ
    can take fewer simply because it isn't a one-step random walk. That's a property of the random-walk candidate on three
    parameters, not of updating the path while undecided. These references assume a Gaussian marginal, so read them as a
    scale, not a bound: a real run can land on either side.
    """),
        ]
    )
    return


@app.cell(hide_code=True)
def mpgibbs_example_view(demos, method_labels, mo, runs):
    mo.vstack(
        [
            mo.md(r"""
    **The running example, next to MPGibbs with the same path move.** MPGibbs takes more sweeps chart on or off, as the walk
    above suggests, and the chart helps both.
    """),
            mo.md(
                demos.markdown_table(
                    runs.example_rows(
                        method_labels,
                        (
                            "nuts_bootstrap_path",
                            "nuts_bootstrap_chart",
                            "mpgibbs_bootstrap_path",
                            "mpgibbs_bootstrap_chart",
                        ),
                    )
                )
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def table_p2(mo):
    mo.md(r"""
    ### Three different things called "marginal"

    | Name | What gets integrated | How |
    |:---|:---|:---|
    | **Marginal** posterior, as PMMH targets it | The whole path | By definition an integral. PMMH estimates the likelihood inside it with a particle filter |
    | **Marginalized** particle Gibbs | The candidate label, during the path move | Exactly: a sum over the two candidates |
    | **m** in Particle-mGRAD | The noisy copy, in the particle weights | Exactly: a Gaussian integral |

    ### Summary of the organizations

    | Question | Taking turns | Joint NUTS | MPGibbs | PMMH |
    |:---|:---|:---|:---|:---|
    | What moves together? | First parameters at a fixed stored path, then the path at fixed parameters | Parameters and path, along one trajectory | The path moves under two candidates, and a candidate is chosen afterwards | A proposed parameter and a particle-generated path, accepted or rejected together |
    | What does the path step target? | The path given the current parameters | There's no separate path step | A mixture of the candidates' complete path densities | There's no path step: a particle filter at the proposed parameter gives a path and a likelihood estimate |
    | Is the whole path integrated analytically? | No | No | No: it sums over labels | No: it uses a nonnegative unbiased estimate of the likelihood |
    | Can the path change when θ doesn't? | Yes | Yes | Yes: picking the old label doesn't undo the path move | Not in its basic form: a rejection keeps both |
    | What limits a θ move? | The conditional at the stored path | Coordinates and step size | The path move's reach under the mixture, and the random-walk step | The likelihood estimate's noise, and the random-walk step |
    | Does it need θ gradients? | NUTS and MALA do, a random walk doesn't | Yes | Not in its random-walk form | Not in its random-walk form |
    | What makes it exact? | Each part preserves its conditional | A valid HMC transition for the joint density | An augmented target, a mixture-invariant path move, the exact label conditional | The pseudo-marginal target, with the likelihood estimate kept |
    """)
    return


@app.cell(hide_code=True)
def part_four_short(mo):
    mo.callout(
        mo.md(r"""
    **Part IV in short.** Joint NUTS holds nothing and pays in trajectory length, which coordinates decide. PMMH integrates
    the path out with an estimate and pays in the estimate's noise. MPGibbs moves the path under two candidates before
    choosing between them, so the choice happens after the path has had a chance to adapt. It pays in a random-walk candidate
    and a path move that has to handle the mixture.
    """),
        kind="neutral",
    )
    return


@app.cell(hide_code=True)
def part_five_md(mo):
    mo.md(r"""
    ## Part V · The families, assembled

    All four choices are now on the table. Here's what each one takes and what it promises:

    | Choice | It takes | It promises | Part |
    |:---|:---|:---|:---|
    | **Stored path** | θ and the stored values | A one-to-one decode to $h$ for each θ, and the Jacobian, so the exact density is known in these coordinates | III |
    | **Path move** | A current path and a target density over it | To leave that target in place | I |
    | **Parameter move, when taking turns** | The density of θ, stored path fixed | To leave that density in place | II |
    | **Organization** | The moves it uses | To give each one its target, and combine them so that the posterior stays in place | II, IV |

    Those promises are why exactness composes: a combination that keeps them needs no new proof. Each path move also needs
    something from its target. The particle moves need a chain or a mixture of chains, auxiliary Kalman a usable Gaussian
    stand-in, and HMC a gradient. Joint NUTS and MPGibbs don't plug in a parameter move from Part II: joint NUTS runs one
    trajectory over everything, and MPGibbs has its own candidate and flip.

    **One sweep of each family, step by step,** with "smoother" standing for any path
    move from Part I that keeps the mixture or the conditional in place: auxiliary Kalman, Particle-mGRAD or bootstrap:

    | Step | Joint NUTS | NUTS + smoother | MPGibbs + smoother |
    |:---|:---|:---|:---|
    | **1. Fresh randomness** | Momentum for θ and the stored path | Momentum for θ | A noisy copy of θ, giving one new candidate |
    | **2. Parameters** | Trajectory through θ and the stored path, select a point | Trajectory through θ, stored path held, select a point | Nothing yet: θ stays undecided |
    | **3. Path** | Done in step 2 | The smoother at the new θ | The smoother, aimed at a mixture of both candidates |
    | **4. Settle θ** | Done in step 2 | Done in step 2 | A weighted flip, given the new stored path |
    | **Chart off: the stored path is** | $h$ | $h$ | $h$ |
    | **Chart on: the stored path is** | $z$; $h$ decodes at every leapfrog step | $z$; $h$ shifts and resizes along the θ trajectory | $z$; each candidate decodes it into its own $h$ |
    | **What this fitted chart does here** | How far one trajectory gets in a given number of steps | θ's held conditional is wider. The path move keeps its form, and a fixed δ in $z$ rescales with σ in $h$ | The candidates' path densities overlap more, so the new candidate wins from further away |
    | **Tuned before the retained draws** | Step size, mass matrix | NUTS's step size and mass matrix on θ, and the smoother's δ | Candidate covariance and scale, and the smoother's δ |

    **Which pieces can be combined?** Every organization (rows) with every path move (columns), including ideas beyond the ones you asked about.
    The bold cells are the ones you asked about.

    | | Auxiliary Kalman | Particle-mGRAD | Bootstrap conditional SMC | HMC on the path |
    |:---|:---|:---|:---|:---|
    | **NUTS on θ, stored path held** | **NUTS + auxiliary Kalman** | **NUTS + mGRAD** | Valid | Valid, though joint NUTS is the more natural way to use HMC for both |
    | **Random walk on θ, stored path held** | Valid | Valid | Particle Gibbs, the classic form | Valid |
    | **MPGibbs** | **MPGibbs + auxiliary Kalman** | **MPGibbs + mGRAD** | m-PGibbs as in the paper | Valid: HMC on the mixture's density |
    | **Joint NUTS** | Not separate: **joint NUTS** has no path step | Not separate | Not separate | Not separate |
    | **PMMH** | A true hole | A true hole | PMMH uses a plain particle filter, not a conditional move | A true hole |

    And every organization with every stored path:

    | | The path $h$ | The chart $z$ | Daily shocks |
    |:---|:---|:---|:---|
    | **Joint NUTS** | Yes | Yes | Yes |
    | **NUTS + smoother** | Yes | Yes | Yes: decode to $h$ for the path move and encode back, which is the same physical move as with $h$ held |
    | **MPGibbs + smoother** | Yes | Yes | Yes, but not directly: each candidate decodes the shocks through all earlier days, so the mixture has no day-by-day chain. Draw a temporary label, decode under it, run the smoother and encode back, as blocked Kalman does with its label. That can keep part of the conditional bottleneck. HMC on the mixture's density also works |

    PMMH holds no path during the parameter proposal, so this choice works differently there: a chart could only reshape the
    particle filter.

    **The true holes**, and some cells that only look like holes:

    | Proposed combination | Why |
    |:---|:---|
    | An uncorrected Kalman draw as an exact smoother | The nonlinear path posterior isn't Gaussian. Without the verdict it would target something else |
    | A separate parameter conditional inside a joint NUTS transition | That transition already moves both blocks |
    | Ordinary NUTS over the candidate label | NUTS needs continuous coordinates. Sum over the labels, or update the label separately |
    | A conditional path move supplying PMMH's likelihood estimate | PMMH needs a nonnegative unbiased likelihood estimate, and a path transition doesn't produce one |
    | Replacing MPGibbs's candidate step with a NUTS proposal on θ | Not a substitution: a path-dependent proposal breaks the construction, and it would need its own derivation. Adding a NUTS move after the sweep is valid, and is a different method |
    | Joint NUTS followed by a smoother | Valid, not a hole, just an extra hybrid |

    **Where the choices touch.** Exactness needs only the promises above, but some choices change what another one has to
    handle:

    | Choices | How they touch |
    |:---|:---|
    | Organization and path move | In MPGibbs the path move gets a mixture over two candidates instead of a density at one θ, so each move needs a way to handle it |
    | Organization and parameter move | MPGibbs's mixture and flip rely on its two symmetric half-steps, whose proposal terms cancel. Any other candidate would have to supply those terms, and a path-dependent one like NUTS breaks the construction. When taking turns, any kernel fits |
    | Organization and stored path | The fitted chart does a different job in each family, as the row "What this fitted chart does here" shows |
    | Stored path and parameter move | This is what the stored path is for: it sets the conditional the parameter move sees |
    | Stored path and path move | At one fixed θ, the chart keeps the day-by-day chain, leaves bootstrap's physical move alone, and rescales δ with σ for the other two. In MPGibbs the two candidates decode it differently, so no single physical move matches. Daily shocks leave no chain to walk along |

    Under taking turns, the parameter move and the path move themselves don't touch: each only sees the state the other leaves
    behind. What all this does to speed is for Part VI.
    """)
    return


@app.cell(hide_code=True)
def family_code_drawer(demos, mo):
    mo.accordion(
        {
            "The code: how a family is put together": mo.vstack(
                [
                    mo.md(r"""
    Every saved run comes out of one short function: three joint runs, then a loop over stored paths and smoothers.
    `smoother_trace` is the same call in NUTS + smoother and in MPGibbs + smoother, and only its `target` differs:
    `sv_target(updated.z[None], ...)` has one candidate when taking turns, and `make_target(candidates)` has two in MPGibbs.
    """),
                    mo.ui.code_editor(
                        "\n\n\n".join(
                            (
                                demos.schedule_source(),
                                demos.kernel_source(
                                    "JointNUTSSpec",
                                    "NUTSSmootherSpec",
                                    "MPGibbsSpec",
                                    "smoother_trace",
                                    "_nuts_smoother_run",
                                    "ensemble_trace",
                                ),
                            )
                        ),
                        language="python",
                        disabled=True,
                    ),
                ]
            )
        }
    )
    return


@app.cell(hide_code=True)
def replay_md(mo):
    mo.md(r"""
    **Watch them move.** Below, 40 real sweeps on one plane, from the saved runs. Each segment is one
    substep. Orange segments are parameter moves and blue segments are path moves. Pick a family and flip the chart. Taking
    turns walks in small steps along the ridge. In MPGibbs the path moves first and the flip comes after. So its
    middle state is an intermediate under the old label, not a posterior draw.
    """)
    return


@app.cell(hide_code=True)
def replay_controls(mo):
    replay_family = mo.ui.dropdown(
        options={
            "NUTS + auxiliary Kalman, blocks": "nuts_ak_blocks",
            "NUTS + auxiliary Kalman, whole path": "nuts_ak_whole",
            "NUTS + Particle-mGRAD": "nuts_mgrad",
            "NUTS + bootstrap conditional SMC": "nuts_bootstrap",
            "MPGibbs + auxiliary Kalman, blocks": "mpgibbs_ak_blocks",
            "MPGibbs + auxiliary Kalman, whole path": "mpgibbs_ak_whole",
            "MPGibbs + Particle-mGRAD": "mpgibbs_mgrad",
            "MPGibbs + bootstrap conditional SMC": "mpgibbs_bootstrap",
        },
        value="NUTS + Particle-mGRAD",
        label="Family",
    )
    replay_chart = mo.ui.switch(value=False, label="Chart on")
    replay_plane = mo.ui.dropdown(
        options={
            "log σ against RMS residual": "scale",
            "Level against the mean of h": "level",
        },
        value="log σ against RMS residual",
        label="Plane",
    )
    mo.hstack([replay_family, replay_chart, replay_plane], justify="start", gap=2)
    return replay_chart, replay_family, replay_plane


@app.cell(hide_code=True)
def replay_view(figs, method_labels, replay_chart, replay_family, replay_plane, runs):
    _suffix = "chart" if replay_chart.value else "path"
    _name = f"{replay_family.value}_{_suffix}"
    figs.fig_substeps(
        runs.substeps(_name),
        cloud=runs.cloud,
        plane=replay_plane.value,
        parameters_first=not _name.startswith("mpgibbs"),
        title=f"Forty sweeps of {method_labels[_name]}",
    )
    return


@app.cell(hide_code=True)
def sweep_controls(mo):
    sweep_index = mo.ui.slider(start=0, stop=39, value=0, label="Which sweep", show_value=True)
    mo.hstack([sweep_index], justify="start")
    return (sweep_index,)


@app.cell(hide_code=True)
def sweep_view(demos, mo, replay_chart, replay_family, runs, sweep_index):
    _suffix = "chart" if replay_chart.value else "path"
    mo.vstack(
        [
            mo.md(r"""
    **One sweep, state by state.** Compare the rows. In NUTS + smoother with the chart off, the parameter move leaves the
    mean of $h$ and middle-day $h$ exactly as they were, because $h$ is held. With the chart on they change too, because $h$
    follows θ. In MPGibbs the path moves first, and the flip changes θ only when the new candidate wins. With the chart on, that
    flip also changes physical $h$, because the winning candidate decodes the same $z$ differently. The RMS residual can change
    even with $h$ held, because it also depends on κ.
    """),
            mo.md(
                demos.markdown_table(
                    demos.sweep_rows(
                        runs.substeps(f"{replay_family.value}_{_suffix}"),
                        sweep_index.value,
                        parameters_first=not replay_family.value.startswith("mpgibbs"),
                    )
                )
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def part_six_md(mo, runs):
    mo.md(rf"""
    ## Part VI · Same posterior, different speed

    **First, the same answer.** Every method here is exact, so given enough sweeps every run should agree with the others.
    Each dot below is one run's posterior mean with ± 2 Monte Carlo standard errors. The dashed line is a separately seeded
    long run, joint NUTS in daily shocks with {runs.chains} chains of {runs.reference_draws:,} sweeps, and the thin band around
    it is its own ± 2 errors. The errors are only credible if a run actually explored the posterior: a chain that missed a
    region can show a tight, wrong bar. So the table further down also lists R̂, which compares chains.
    """)
    return


@app.cell(hide_code=True)
def agreement_view(figs, method_labels, np, runs):
    _shown = [name for name in method_labels if name in runs.names]
    _order = [runs.names.index(name) for name in _shown]
    figs.fig_agreement(
        [method_labels[name] for name in _shown],
        np.asarray(runs.means)[_order],
        np.asarray(runs.errors)[_order],
        runs.reference_means,
        runs.reference_errors,
        columns=(
            (0, "Level μ"),
            (1, "log κ"),
            (2, "log σ"),
            (3, "Mean of h"),
            (4, "RMS residual"),
            (5, "Middle-day h"),
        ),
    )
    return


@app.cell(hide_code=True)
def agreement_reading(demos, history, method_labels, mo, np, runs):
    _gaps = np.abs(runs.reference_gaps)
    _run, _column = np.unravel_index(np.argmax(_gaps), _gaps.shape)
    _quantity = ("level", "log κ", "log σ", "the mean of h", "the RMS residual", "middle-day h")[
        _column
    ]
    _repeat = np.abs(demos.repeat_run_gaps(runs, history.r2)).max()
    _worst_rhat = max(max(runs.metadata[name]["diagnostics"]["rhat"]) for name in runs.names)
    mo.md(f"""
    **How close is close?** Across {_gaps.size} comparisons of {len(runs.names)} runs with the reference, the largest gap is
    {_gaps[_run, _column]:.1f} combined Monte Carlo errors: {method_labels[runs.names[_run]]}, on {_quantity}. It is the only one
    above 3. The comparisons are correlated, and one such gap among this many isn't surprising by itself. A separately seeded
    repeat of that run stayed within {_repeat:.1f} on all six quantities. The worst R̂ across all runs is {_worst_rhat:.3f}.
    """)
    return


@app.cell(hide_code=True)
def speed_md(mo, runs):
    mo.md(rf"""
    **Then, the speed.** Sweeps per effective sample, by quantity, from each run's estimate of the effective sample size
    for that quantity's posterior mean, across {runs.chains} chains at T = {runs.days:,} days. Lower is better, and a value
    near 1 means roughly the same precision for that mean as independent draws. Slower methods ran longer, so that their estimates
    still rest on enough effective samples. The full table lists each run's length.

    The settings: N = 32 particles. δ was chosen per smoother in a short pilot at fixed parameters, by how far the path
    jumps per day. MPGibbs's candidate scale was chosen per method from short real chains, by the worst sweeps per
    effective sample across level, log σ and mean of h, then confirmed on separate seeds. Both are coarse-grid pilots, so a
    small gap between two methods can be a tuning effect.

    The prices come from separate runs on this laptop's CPU, one method at a time: {runs.chains} chains, 500 warmup and 500 retained
    updates each, after compilation, with NUTS adapting afresh. "ms per sweep" divides elapsed time by all those updates, and "ms per
    effective sample" multiplies it by the long run's sweeps per effective sample. That's a throughput estimate for this vectorized
    teaching code, so treat it as a rough scale. It isn't the total cost of a fresh fit, which would add compilation, pilot tuning
    and warmup.
    """)
    return


@app.cell(hide_code=True)
def speed_view(demos, figs, method_labels, mo, runs):
    _families = (
        ("Joint NUTS", "joint_path", "joint_chart"),
        ("NUTS + bootstrap", "nuts_bootstrap_path", "nuts_bootstrap_chart"),
        ("NUTS + mGRAD", "nuts_mgrad_path", "nuts_mgrad_chart"),
        ("NUTS + Kalman blocks", "nuts_ak_blocks_path", "nuts_ak_blocks_chart"),
        ("NUTS + Kalman whole path", "nuts_ak_whole_path", "nuts_ak_whole_chart"),
        ("MPGibbs + bootstrap", "mpgibbs_bootstrap_path", "mpgibbs_bootstrap_chart"),
        ("MPGibbs + mGRAD", "mpgibbs_mgrad_path", "mpgibbs_mgrad_chart"),
        ("MPGibbs + Kalman blocks", "mpgibbs_ak_blocks_path", "mpgibbs_ak_blocks_chart"),
        ("MPGibbs + Kalman whole path", "mpgibbs_ak_whole_path", "mpgibbs_ak_whole_chart"),
    )
    _nuts = [name for name in runs.names if not name.startswith("mpgibbs")]
    _limit = runs.tree_limit_share("joint_path")
    _most_divergences = max(runs.divergences(name) for name in _nuts)
    mo.vstack(
        [
            figs.fig_speed(
                [
                    (
                        label,
                        runs.sweeps_per_effective_sample(off),
                        runs.sweeps_per_effective_sample(on),
                    )
                    for label, off, on in _families
                ],
                columns=((0, "Level μ"), (2, "log σ"), (3, "Mean of h")),
                shocks=runs.sweeps_per_effective_sample("joint_shocks"),
            ),
            mo.accordion(
                {
                    "The full table": mo.vstack(
                        [
                            mo.md(demos.markdown_table(runs.speed_rows(method_labels))),
                            mo.md(f"""
    NUTS's limit is 511 leapfrog steps per sweep. Joint NUTS in $h$ hit it on {_limit:.0%} of its sweeps, so a higher limit
    would change that row, at a higher price per sweep. Retained divergences are rare: no run has more than
    {_most_divergences}.
    """),
                        ]
                    )
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def choices_view(demos, method_labels, mo, runs):
    mo.vstack(
        [
            mo.md(r"""
    **One choice at a time.** Start from the running example, NUTS + bootstrap with $h$ stored, and change one choice per row.
    Each number is per effective sample.
    """),
            mo.md(
                demos.markdown_table(
                    runs.choice_rows(
                        method_labels,
                        (
                            ("None: the running example", "nuts_bootstrap_path"),
                            ("Path move", "nuts_mgrad_path"),
                            ("Path move", "nuts_ak_blocks_path"),
                            ("Path move", "nuts_ak_whole_path"),
                            ("Stored path", "nuts_bootstrap_chart"),
                            ("Organization", "joint_path"),
                            ("Organization", "mpgibbs_bootstrap_path"),
                        ),
                    )
                )
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def choices_reading(mo, runs):
    _s = runs.sweeps_per_effective_sample
    _ms = runs.milliseconds_per_effective_sample
    _level, _log_vol = 0, 2
    mo.md(f"""
    - **The stored path does the most.** With the chart, level goes from {_s("nuts_bootstrap_path")[_level]:.0f} to
      {_s("nuts_bootstrap_chart")[_level]:.1f} sweeps and log σ from {_s("nuts_bootstrap_path")[_log_vol]:.0f} to
      {_s("nuts_bootstrap_chart")[_log_vol]:.0f}, at about the same price per sweep.
    - **A different path move doesn't help the sweeps here, though it can help the time.** Particle-mGRAD ties with bootstrap.
      The Kalman moves take more sweeps, most of all for level, which is tied to the path's mean, which they change slowly. But
      a Kalman sweep costs a fraction of a particle sweep, so they win on time: {_ms("nuts_ak_whole_path", _log_vol):.0f} ms for
      log σ on the whole path against {_ms("nuts_bootstrap_path", _log_vol):,.0f}.
    - **A different organization alone doesn't help log σ here.** With $h$ stored, joint NUTS ties on log σ and helps level, and
      MPGibbs takes more sweeps on both.
    """)
    return


@app.cell(hide_code=True)
def pairings_view(demos, mo, runs, smoother_labels):
    mo.vstack(
        [
            mo.md(r"""
    **The choices together: organization and path move.** The pairings you asked about, on log σ. Sweeps and milliseconds per
    effective sample, for NUTS + a smoother next to MPGibbs + the same smoother.
    """),
            mo.md(demos.markdown_table(runs.pairing_rows(smoother_labels, 2))),
        ]
    )
    return


@app.cell(hide_code=True)
def pairings_reading(mo, runs):
    _ms = runs.milliseconds_per_effective_sample
    _log_vol = 2
    mo.md(f"""
    - **In every pairing here, NUTS + a smoother takes fewer sweeps than MPGibbs + the same smoother.** IV.3 gave a large part of
      the reason: MPGibbs's candidate is a random walk, and NUTS on θ isn't.
    - **Which one wins on time depends on the path move.** A particle sweep costs about {runs.milliseconds_per_sweep("nuts_mgrad_chart"):.0f} ms
      here and a Kalman sweep a fraction of one. So MPGibbs's extra sweeps are expensive with particles:
      {_ms("nuts_mgrad_chart", _log_vol):.0f} ms against {_ms("mpgibbs_mgrad_chart", _log_vol):.0f} with mGRAD and the chart on. With
      Kalman moves a sweep is so cheap that NUTS's own gradient steps become a large part of it. That offsets MPGibbs's extra
      sweeps, and with the chart on the two families tie:
      {_ms("nuts_ak_blocks_chart", _log_vol):.0f} ms against {_ms("mpgibbs_ak_blocks_chart", _log_vol):.0f} with blocks, and
      {_ms("nuts_ak_whole_chart", _log_vol):.0f} against {_ms("mpgibbs_ak_whole_chart", _log_vol):.0f} with the whole path.
    """)
    return


@app.cell(hide_code=True)
def chart_gain_view(demos, mo, runs, smoother_labels):
    mo.vstack(
        [
            mo.md(r"""
    **The choices together: stored path and everything else.** How many times fewer sweeps each pairing needs with the chart on:
    """),
            mo.md(demos.markdown_table(runs.chart_gain_rows(smoother_labels))),
        ]
    )
    return


@app.cell(hide_code=True)
def chart_gain_reading(mo, runs):
    _s = runs.sweeps_per_effective_sample
    _ms = runs.milliseconds_per_effective_sample
    _level, _log_vol = 0, 2
    mo.md(f"""
    The chart helps everywhere, but by very different amounts.

    - **On log σ under NUTS, it helps the particle moves most.** With the chart off the narrow conditional limits them all. With
      it on, only the path move differs across these rows, and log σ takes {_s("nuts_bootstrap_chart")[_log_vol]:.0f} and
      {_s("nuts_mgrad_chart")[_log_vol]:.0f} sweeps with particles against {_s("nuts_ak_blocks_chart")[_log_vol]:.0f} and
      {_s("nuts_ak_whole_chart")[_log_vol]:.0f} with Kalman moves. So the path move now sets the pace, and a slower one leaves
      less for the chart to gain.
    - **On level, it's the other way round.** Level is tied to the path's mean, which Kalman moves change slowly on their own.
      With the chart off, NUTS + whole-path Kalman takes {_s("nuts_ak_whole_path")[_level]:,.0f} sweeps for level and
      {_s("nuts_ak_whole_path")[3]:,.0f} for the mean of $h$. With it on, a level move carries the path's mean along through the
      translation, and those fall to {_s("nuts_ak_whole_chart")[_level]:.0f} and {_s("nuts_ak_whole_chart")[3]:.0f}.
    - **It can change which path move you'd pick.** Under NUTS with the chart off, whole-path Kalman takes
      {_ms("nuts_ak_whole_path", _log_vol):.0f} ms for log σ and bootstrap {_ms("nuts_bootstrap_path", _log_vol):,.0f}. With it on
      they tie, at {_ms("nuts_ak_whole_chart", _log_vol):.0f} and {_ms("nuts_bootstrap_chart", _log_vol):.0f}.
    """)
    return


@app.cell(hide_code=True)
def across_md(mo, runs):
    _s = runs.sweeps_per_effective_sample
    _ms = runs.milliseconds_per_effective_sample
    _log_vol = 2
    mo.md(f"""
    **Across all of it, on log σ.** With the chart on, most routes under NUTS cost about the same: joint NUTS in the chart takes
    {_ms("joint_chart", _log_vol):.0f} ms, and NUTS + any of these smoothers {_ms("nuts_ak_blocks_chart", _log_vol):.0f} to
    {_ms("nuts_ak_whole_chart", _log_vol):.0f} ms. Joint NUTS in daily shocks is the cheapest by a wide margin:
    {_s("joint_shocks")[_log_vol]:.1f} sweeps and {_ms("joint_shocks", _log_vol):.1f} ms per effective sample. That's with one
    coordinate per day, on a CPU, and it needn't hold elsewhere.

    One dataset, one set of seeds, coarse-grid pilots. Treat gaps below about a factor of two as ties.
    """)
    return


@app.cell(hide_code=True)
def closing_md(mo):
    mo.md(r"""
    ## Where they agree, where they differ

    **The whole notebook in five sentences.** These methods all sample the same posterior, and they differ in how they
    move through it. With the parameters known, moving the path is the easy half on this model: a day's reading is weak,
    and even plain bootstrap particles nearly redraw the path. The hard half is the tie between parameters and path, because
    holding $h$ leaves θ a narrow conditional, and no kernel or mass matrix on θ alone can widen it. There are three
    ways out: change what's held (daily shocks, the chart), move everything together (joint NUTS), or update the path while
    the parameters are undecided (MPGibbs). The chart is a change in what's held, not a mass matrix: its translation
    overlaps with a dense metric, and its scale can't be one.

    **What composes, and what doesn't.** Exactness composes: each choice keeps its promise, so a valid combination needs no new
    proof, and every run here agrees. Speed doesn't: what the chart buys depends on the path move, the organization and the
    quantity you care about.

    **What they all share.**

    - The same exact posterior. Approximations shape proposals, and true densities make the final call. A bad
      approximation can cost speed, never the answer.
    - The same recipe for a move: add fresh randomness, propose candidates from what you know, correct exactly, and
      drop the extra randomness.
    - Auxiliary variables as the source of that randomness: momentum in HMC, a noisy copy of the path in the two
      smoothers, a noisy copy of θ in MPGibbs.
    - Tuning during warmup only, then frozen.

    **Where the path moves differ.**

    | | Particle-mGRAD | Auxiliary Kalman | HMC |
    |:---|:---|:---|:---|
    | Shared first step | Noisy copy of the path | Noisy copy of the path | (Momentum instead) |
    | Candidates | Many per day, recombined | One whole path or block | One trajectory of whole paths (NUTS: a tree) |
    | What the proposal knows | Dynamics exactly, one reading's gradient | Linearized dynamics and every reading, across the block | The gradient along the way |
    | Later readings enter through | The backward draw | The proposal | The trajectory |
    | The correction is | Per day | Per block | Per trajectory |

    **Where the parameter moves differ** is one question: what does the path do while θ changes?

    | | What the path does |
    |:---|:---|
    | NUTS + smoother, chart off | Nothing: $h$ is held, so θ stays inside its narrow conditional |
    | NUTS + smoother, chart on | Follows θ through the chart: $z$ is held, so the conditional is wider |
    | MPGibbs + smoother, chart off | Moves under both candidates before θ is chosen, and physical $h$ is shared at the flip |
    | MPGibbs + smoother, chart on | Moves under both candidates in $z$, and each candidate decodes it into its own $h$ |
    | Joint NUTS | Moves along the same trajectory as θ |

    **Check yourself.** Each question can be answered from what's above.

    - Can the path change when MPGibbs keeps the old θ? (Yes: the sweep's path move stands.)
    - Which thing does each use of "marginal" integrate? (The whole path, the label, the noisy copy.)
    - Would a dense three-parameter mass matrix fix a narrow conditional at a held $h$? (No: it can't widen it.)
    - Where would a missing chart Jacobian show up? (The posterior of log σ would be wrong, which Part VI's agreement check would catch.)
    - Why does a good smoother barely help vol-of-vol in NUTS + smoother with the chart off? (The limit is the
      conditional given $h$, not the path move.)
    - Swap Particle-mGRAD for auxiliary Kalman in NUTS + smoother. What has to be re-derived? (Nothing for exactness, since each
      keeps the same conditional in place. The speed can change a lot.)
    - Why can't NUTS replace MPGibbs's candidate step? (Its proposal would depend on the path, which breaks the construction.)
    """)
    return


@app.cell(hide_code=True)
def appendix_md(mo):
    mo.accordion(
        {
            "Sources": mo.md(r"""
    - Andrieu, Doucet & Holenstein (2010), particle MCMC: conditional SMC, particle Gibbs, PMMH. `docs/papers/particle-markov-chain-monte-carlo.pdf`
    - Whiteley (2010), backward sampling in conditional SMC.
    - Hoffman & Gelman (2014), NUTS.
    - Papaspiliopoulos, Roberts & Sköld (2007), centred and non-centred parameterizations.
    - Titsias & Papaspiliopoulos (2018), auxiliary gradient samplers.
    - Corenflos & Särkkä (2025), auxiliary MCMC and auxiliary Kalman. `docs/papers/auxiliary-kalman-samplers.pdf`
    - Corenflos & Finke (2024), Particle-MALA and Particle-mGRAD. `docs/papers/particle-mala-mgrad.pdf`
    - Corenflos (2025), marginalized particle Gibbs. `docs/papers/particle-gibbs-no-gibbs-bit.pdf`
    - Doucet, Pitt, Deligiannidis & Kohn (2015), tuning the noise of an unbiased likelihood estimate.
    """),
            "How this notebook was checked": mo.md(r"""
    The kernels here are short teaching versions, written from the papers, in `notebooks/support/sv_kernels.py`. They were
    checked against: an exact Kalman answer key for each path move; an exact two-candidate Gaussian mixture, with and without
    a chart, for label frequencies and path moments; refined quadrature on a small model for parameter moves; and agreement across
    all methods on the simulated nonlinear dataset, as shown in Part VI. The chart is frozen before retained draws. Each run is one dataset and one
    set of seeds, so small differences between methods shouldn't be read as general facts. With θ fixed, coupled runs of the
    particle and Kalman moves in $z$ and in $h$ matched to within $10^{-10}$, once δ was rescaled.

    The saved runs live in `notebooks/artifacts/sv_methods`, which git ignores. To regenerate them, on CPU, from
    `apps/grassmoth-server`:

    ```
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=true uv run python -m notebooks.support.sv_runs
    ```

    Run lengths are in `RUN_LENGTHS` in `notebooks/support/sv_runs.py`. The whole schedule takes several hours here. Smaller
    experiments save their results in the same folder, under a fingerprint of their code and inputs.
    """),
        }
    )
    return


if __name__ == "__main__":
    app.run()
