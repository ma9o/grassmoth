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
    # The sampler, piece by piece

    **What we are after.** A trustworthy posterior for France: about 50 parameters and
    4 unmeasured paths over 2,284 days, from mostly monthly readings. The posterior is
    over two things at once: the **parameters** θ (effects, noise scales, reading noise)
    and the **paths** x (the unmeasured states on the daily grid).

    **The two sampler configurations shown throughout this notebook:**

    | configuration | one complete transition | settings |
    |---|---|---|
    | **MPGibbs alone** | propose parameter labels, update the path under their mixture, select a label | `method: mpgibbs` |
    | **Joint NUTS** | update parameters and all free path coordinates together | `method: nuts_da`, `parameterization: non_centred` |

    The [MPGibbs kernel](../../../packages/grassmoth/src/grassmoth/models/ssm/inference/samplers/mpgibbs/kernel.py),
    and [joint NUTS](../../../packages/grassmoth/src/grassmoth/models/ssm/inference/samplers/nuts.py)
    own these moves. The smoother is inside the MPGibbs transition. Part 1 also checks
    joint NUTS in centred coordinates.

    **What decides between them** is one fact about the posterior, which Part 0 draws:
    some parameters are **loose** (a whole path says little more about them than the
    readings do) and some are **coupled** (a whole path pins them far more tightly than
    the readings do). Every piece below handles one kind well, and the comparison is
    about what that costs at France scale.

    **How this notebook works.** Everything is the production sampler, run through the
    study's public actions (`edit_model`, `simulate`, `fit`) on a small model in our own
    modelling language. The model is linear and Gaussian on purpose: a Kalman filter then
    gives the **exact** answer, so every sampler result can be checked against the truth.
    France and PARAM are nonlinear; the samplers target the exact nonlinear model all the
    same, so nothing here depends on linearity except the answer key.

    The first run fits five models on this machine; after that
    every call replays its saved result.
    """)
    return


@app.cell
def imports_lab():
    import numpy as np

    import notebooks.support.sampler_pieces as lab

    return lab, np


@app.cell(hide_code=True)
def model_md(mo):
    mo.md(r"""
    ## Part 0 · The model, and the reference picture

    One unmeasured level **x**, on a grid of 45-minute steps over 8 days (256 steps).
    Each step it is pulled back a little towards zero, pushed by five known on/off inputs
    through unknown **effects** β1…β5, and shaken by random noise of unknown size **σ**:

    $$x_{t+1} = x_t + \big(-0.5\,x_t + \textstyle\sum_j \beta_j u_j(t)\big)\,\Delta t + \sigma\sqrt{\Delta t}\;\varepsilon_t,
    \qquad \varepsilon_t \sim N(0,1).$$

    Every grid point is read with noise of SD 0.3, about three times one step's
    movement, so the readings don't pin the path step by step. That is PARAM's situation
    (every grid point read, noisily); France reads far more sparsely, monthly on a daily
    grid. Six unknowns: five effects (like PARAM's input and cross effects) and one noise
    scale (like PARAM's σ_x and σ_y). We author the model with `edit_model` and draw one
    history with `simulate` at β = (1.0, −0.6, 0.5, 0.8, −0.4) and σ = 0.5.
    """)
    return


@app.cell
def study_setup(lab):
    study = lab.Study()
    _question = study.question()
    truth_model = study.model(_question, lab.model_spec(reading_sd=lab.NOISY, free=False))
    free_model = study.model(_question, lab.model_spec(reading_sd=lab.NOISY, free=True))
    data, _body = study.simulate(truth_model)
    history = lab.history(_body, reading_sd=lab.NOISY)
    return data, free_model, history, study, truth_model


@app.cell
def model_view(history, lab):
    lab.fig_history(
        history, title="The simulated history: 257 noisy readings of x, and the five inputs"
    )
    return


@app.cell(hide_code=True)
def reference_md(mo):
    mo.md(r"""
    ### The reference picture

    The posterior covers the parameters and the whole path at once:

    $$p(\theta, x \mid y) \;\propto\; p(\theta)\; p(x \mid \theta)\; p(y \mid x).$$

    Squeeze the whole path to one axis, x, and draw each parameter against it. Every
    sampler in this notebook is a way of moving around this plane, and every difficulty
    below shows up on it. Two shapes matter, one for each kind of parameter:

    - **β, an effect, is loose.** Effects set how the path drifts while an input is on:
      its slow course, which the readings already follow. A whole path says little more
      about β than the readings do, so the cloud is round.
    - **log σ, the noise scale, is coupled.** σ is the size of the path's 45-minute
      steps. The readings can't see single steps, but a whole path holds all 256 of
      them, which fix σ to about 4%. So the cloud is a narrow ridge.

    Three marks recur on every figure:

    - **the shadow** (filled, along the bottom): θ given the readings, the path
      integrated out. It is what every sampler is after.
    - **the slice** (purple): θ given one whole path. Where θ can go while the path is
      held.
    - **the band** (green): the path given θ. Where the path can go while θ is held.
    """)
    return


@app.cell
def exact_answer(history, lab):
    exact = lab.exact_posterior(history)
    extremes = lab.scale_extremes(history, exact, seed=5)
    return exact, extremes


@app.cell
def cloud_compute(exact, history, lab):
    cloud = lab.posterior_cloud(history, exact, 2000, seed=3)
    chosen, panels = lab.shadow_and_slice(history, exact, cloud)
    return chosen, cloud, panels


@app.cell
def shapes_compute(lab, panels):
    shapes = tuple(lab.Shape(panel.label, panel.narrowing) for panel in panels)
    moves = tuple(
        lab.textbook_moves(shape, lab.start_state(shape), seed=seed)
        for seed, shape in enumerate(shapes, start=1)
    )
    return moves, shapes


@app.cell
def reference_view(lab, shapes):
    lab.fig_reference(shapes)
    return


@app.cell(hide_code=True)
def reference_takeaway(mo, panels):
    mo.md(f"""
    The shapes are cartoons, Gaussian in two dimensions, drawn to our model's exact
    answer: a whole path makes β's slice only {panels[0].narrowing:.2f}× narrower than
    its shadow, and log σ's {panels[1].narrowing:.1f}× narrower. We call the width of
    σ's slice the **pin**. Every figure below is this reference with one method drawn on
    it; the evidence behind each sits in a closed section underneath.
    """)
    return


@app.cell
def pin_compute(history, lab):
    _rows = []
    for _factor in (1, 4, 16):
        _refined = history.refined(_factor)
        _exact = lab.exact_posterior(_refined)
        _cloud = lab.posterior_cloud(_refined, _exact, 400, seed=3)
        _rows.append((_refined.times.size - 1, lab.shadow_and_slice(_refined, _exact, _cloud)[1]))
    pin_panels = tuple(_rows)
    return (pin_panels,)


@app.cell(hide_code=True)
def reference_evidence(chosen, cloud, extremes, history, lab, mo, panels, pin_panels):
    _spread = lab.step_noise(history, *extremes).std(axis=1)
    mo.accordion(
        {
            "Evidence: where the two shapes come from (exact, from the answer key)": mo.vstack(
                [
                    mo.md(f"""
    **What a whole path knows that the readings don't.** Two exact posterior paths, one
    with σ at the low end of its posterior range and one at the high end. Both thread the
    readings, which can't choose between them; their steps spread
    {_spread[1] / _spread[0]:.1f}× as wide as each other's.
    """),
                    lab.fig_what_paths_know(history, *extremes),
                    mo.md("""
    **The real clouds.** Each dot is one exact draw (θ, x): across, the parameter; up,
    what that draw's path alone says about it, the centre of p(θ | x). For σ that is the
    spread of the path's steps; for β1, how the path drifts while input 1 is on. Under
    each plane, the shadow (filled) and the slice of one path (purple), with the other
    parameters integrated out. These are the shapes the cartoon idealizes.
    """),
                    lab.fig_shadow_slice(cloud, chosen, panels),
                    mo.md("""
    **France's grid makes the ridge thinner.** Keep the same 257 readings and put more
    grid steps between them, as France does with daily steps between monthly readings.
    log σ's slice narrows like 1/√T while its shadow stays put; β's slice doesn't
    change. This is the classic trouble with diffusion parameters: on a fine enough
    grid, an alternating sampler can't move σ at all (Roberts & Stramer 2001).
    """),
                    lab.fig_pin(pin_panels),
                ]
            )
        }
    )
    return


@app.cell(hide_code=True)
def strategies_md(mo):
    mo.md(r"""
    ### Three ways to move on the reference

    You know the two tools: a particle filter handles the path for a given θ, and
    gradient MCMC (HMC, NUTS) moves continuous coordinates along the gradient of the log
    density. Every SSM sampler uses them to move θ in one of three ways, and each piece of
    our stack can be understood through these strategies:

    | strategy | how θ moves | classic form | here |
    |---|---|---|---|
    | **alternate** | the path within its band, then θ within its slice | particle Gibbs (Andrieu, Doucet & Holenstein 2010) | exact conditional draws on the reference only |
    | **integrate the path out** | θ′ judged with the path averaged out, a fresh path with it | particle marginal Metropolis–Hastings (same paper) | the MPGibbs label move builds on this idea, differently (Part 2) |
    | **move both** | one gradient trajectory through θ and the path together | HMC, NUTS | joint NUTS (Part 1) |

    Here is each at its textbook best on the two shapes, from the same start (open
    circle). These moves are exact for the cartoon; the parts below show where our real
    samplers fall short of them.
    """)
    return


@app.cell
def strategies_view(lab, moves, shapes):
    lab.fig_strategy_overview(shapes, moves)
    return


@app.cell(hide_code=True)
def strategies_takeaway(mo):
    mo.md(r"""
    - **Alternate** is held to the slice and the band. On β's round cloud both are wide,
      so each step crosses most of it. On σ's ridge both are one pin wide, so it zigzags:
      the narrower the ridge, the more sweeps per independent draw.
    - **Integrate the path out** isn't held to the slice. Judged on the shadow and
      bringing a fresh path, one move can land anywhere along the ridge; crosses are
      rejected proposals. In practice its limits are how well the path really is
      averaged out and how big a step it dares (Part 2).
    - **Move both** follows the gradient through θ and the path together, so in two
      dimensions one trajectory runs along either shape. In our 263 dimensions the
      coordinates decide whether it still can (Part 1).
    """)
    return


@app.cell
def walks_compute(exact, history, lab):
    alternating = lab.alternating_walk(history, exact, chains=16, sweeps=600, seed=11)
    integrated = lab.integrated_walk(history, exact, sweeps=600, seed=12)
    return alternating, integrated


@app.cell(hide_code=True)
def strategies_evidence(alternating, cloud, integrated, lab, mo, panels):
    _taus = [
        lab.sweeps_per_independent_draw(alternating.shown[:, 1:, panel.column]) for panel in panels
    ]
    _pins = panels[1].narrowing
    mo.accordion(
        {
            "Evidence: the first two strategies on the real σ ridge (exact)": mo.vstack(
                [
                    mo.md("""
    Both use the answer key, so each is its strategy at its best. Top: 25 sweeps of
    each on the real log σ ridge, either a perfect path given θ (green) then a perfect θ
    given that path (purple), or θ drawn straight from the shadow with a fresh path
    (orange). Bottom: 300 sweeps of each, for log σ and for β1, against the shadow's
    ±2 SD (grey band).
    """),
                    lab.fig_strategies(alternating, integrated, cloud, panels),
                    mo.md(f"""
    On the ridge the zigzag moves one pin at a time, a random walk across the shadow.
    Theory says that takes about 2 × {_pins:.1f}² ≈ {2 * _pins**2:.0f} sweeps per
    independent draw; this exact chain needed {_taus[1]:.0f}. On β1 it needed
    {_taus[0]:.1f}. On PARAM the coupled σ_y, about 4.8 pins per posterior SD, needed
    34–56 sweeps per effective draw.
    """),
                ]
            )
        }
    )
    return


@app.cell(hide_code=True)
def joint_md(mo):
    mo.md(r"""
    ## Part 1 · Joint NUTS

    **What it does.** NUTS treats the six parameters and all 257 path values as one point
    in 263 dimensions. Each draw follows the gradient of the log posterior along a
    trajectory of many small leapfrog steps and stops when the trajectory starts to turn
    back: the *move both* row of the reference. It needs no smoother and no special
    moves, the simplest option to own.

    **Where it gets stuck, on the reference.** The same posterior can be written in
    different coordinates, called charts, and the chart decides how far one draw travels
    along σ's ridge:

    - **centred**: NUTS moves the path values themselves. Travelling along σ's ridge
      means resizing all 256 steps together, and one draw manages only a local move.
    - **innovation** (`non_centred`): NUTS moves standardized shocks ε and rebuilds the
      path from them. Scaling σ with the shocks held resizes every step by itself, so a
      draw can travel along the ridge; only the few slow shocks that the readings see
      need adjusting. The price is longer trajectories, each gradient rebuilding the
      path step by step.

    On β's round cloud both charts cross freely. When the readings pin the path step by
    step, the centred chart wins instead; the **mixed** chart centres the coordinates
    that are read directly and uses innovations elsewhere.
    """)
    return


@app.cell
def joint_reference(lab, shapes):
    lab.fig_joint_schematic(shapes)
    return


@app.cell
def joint_fits(data, free_model, lab, study):
    joint_runs = {
        chart: lab.read_fit(study.fit(free_model, data, lab.joint_nuts(chart)))
        for chart in ("centred", "non_centred")
    }
    return (joint_runs,)


@app.cell(hide_code=True)
def joint_caption(joint_runs, lab, mo, np):
    def _row(chart, name):
        _run = joint_runs[chart]
        _effects = min(
            lab.effective_per_thousand(_run.column(f"beta{j + 1}"), _run.ess[f"beta{j + 1}"])
            for j in range(lab.EFFECTS)
        )
        _sigma = lab.effective_per_thousand(_run.column("sigma"), _run.ess["sigma"])
        return (
            f"| {name} | {_sigma:.0f} | {_effects:.0f} | "
            f"{np.median(_run.extra['num_steps']):.0f} | {lab.energy_check(_run)[2]:.2f} |"
        )

    mo.md(f"""
    **Measured on the real fits** (`fit` with `method: nuts_da`, four chains of 1,000
    warmup and 1,000 retained draws):

    | chart | log σ, effective per 1,000 | worst effect | gradients per draw | E-BFMI |
    |---|---|---|---|---|
    {_row("centred", "centred")}
    {_row("non_centred", "innovation")}

    E-BFMI is NUTS's own energy check, and Stan warns below 0.3. In the centred chart one
    draw's change of energy is small against the range that σ's ridge spans, so the
    chain random-walks through it. **For France** the readings are monthly on a daily
    grid, the innovation chart's case, which is why the agreed comparison runs France in
    it. What France will measure is the price: gradients per draw times a sequential
    rebuild of 2,284 days × 4 paths.
    """)
    return


@app.cell(hide_code=True)
def joint_evidence(cloud, exact, history, joint_runs, lab, mo):
    _spread, _kick, _ = lab.energy_check(joint_runs["centred"])
    mo.accordion(
        {
            "Evidence: the real joint NUTS chains": mo.vstack(
                [
                    lab.fig_joint_runs(joint_runs, exact),
                    mo.md("""
    The same chains on the real σ ridge, consecutive retained draws joined. The lines
    are not trajectories; they show how far one draw moves.
    """),
                    lab.fig_ridge_runs(
                        cloud,
                        history,
                        exact.log_scales,
                        {
                            "joint NUTS, centred chart": (joint_runs["centred"], lab.C_JOINT),
                            "joint NUTS, innovation chart": (
                                joint_runs["non_centred"],
                                lab.C_JOINT,
                            ),
                        },
                    ),
                    mo.md(f"""
    The energy check itself: the spread of energy across draws (grey) against one
    draw's change of energy (blue). In the centred chart the spread ({_spread:.0f}) is
    {_spread / _kick:.1f}× one draw's change ({_kick:.0f}); in the innovation chart they
    match.
    """),
                    lab.fig_energy(joint_runs),
                ]
            )
        }
    )
    return


@app.cell(hide_code=True)
def mpgibbs_md(mo):
    mo.md(r"""
    ## Part 2 · MPGibbs: the label move

    **What it does**, once per sweep (Corenflos 2025, "Particle Gibbs without the Gibbs
    bit"):

    1. Propose θ′ near θ: a random-walk step shaped by a covariance learned in warmup.
    2. Run the smoother (Part 3) on a **mixture**: half the weight says the path comes
       from θ, half from θ′. Today's path stays in the particle system as the reference;
       the other particles try new paths.
    3. Pick a new path from the particles, then the label, θ or θ′, by how well each
       explains that path (a Barker choice).

    **Where it gets stuck, on the reference.** Each label has its band, the paths it
    allows, and the move succeeds when the new path lands in θ′'s band. On β's round
    cloud the two bands overlap: any new path suits both labels, the vote is easy, and
    the limit is the size of the random-walk step. On σ's ridge the bands barely touch:
    θ′ wins only if the particles bring a path as rough as σ′ calls for. If the new path
    stays in θ's band, θ is kept, though the path still changed. This is not PMMH:
    nothing estimates the shadow, and the vote is decided by the paths the particles
    find.
    """)
    return


@app.cell
def mpgibbs_reference(lab, shapes):
    lab.fig_label_schematic(shapes)
    return


@app.cell
def mpgibbs_fit(data, free_model, lab, study):
    mpgibbs_run = lab.read_fit(study.fit(free_model, data, lab.mpgibbs()))
    return (mpgibbs_run,)


@app.cell(hide_code=True)
def mpgibbs_caption(lab, mo, mpgibbs_run, np):
    _effects = [
        lab.effective_per_thousand(
            mpgibbs_run.column(f"beta{j + 1}"), mpgibbs_run.ess[f"beta{j + 1}"]
        )
        for j in range(lab.EFFECTS)
    ]
    _scale = lab.effective_per_thousand(mpgibbs_run.column("sigma"), mpgibbs_run.ess["sigma"])
    _odds = np.exp(mpgibbs_run.extra["final_label_log_probs"][..., 1]).mean()
    mo.md(f"""
    **Measured on the real fit** (`method: mpgibbs`, twisted
    Particle-aGRAD with 128 particles): the effects get {min(_effects):.0f}–{max(_effects):.0f}
    effective draws per 1,000 and log σ {_scale:.0f}. θ′ won {_odds:.0%} of the votes,
    the step size having adapted towards a 35% target.

    **What PARAM showed about σ.** More path refreshes per move raised the chance of
    taking a one-SD scale move. A mixture proposal that explores each label's paths
    better passed 3 of 4 two-SD scale tests. Taking the loose coordinates out of the
    labels was not a general fix. So exploring the paths behind both labels is the lead,
    though what drives the gap here isn't pinned down.
    """)
    return


@app.cell
def vote_rows(exact, history, lab, np):
    _distances = np.array([-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0])
    scale_votes = lab.votes(history, exact, _distances, coordinate=-1)
    effect_votes = lab.votes(history, exact, _distances, coordinate=0)
    return effect_votes, scale_votes


@app.cell(hide_code=True)
def mpgibbs_evidence(effect_votes, exact, lab, mo, mpgibbs_run, scale_votes):
    mo.accordion(
        {
            "Evidence: the vote, and the real label move": mo.vstack(
                [
                    mo.md("""
    **The vote, computed exactly.** The chance that θ′ wins against its distance from θ,
    judged by today's path alone or with the path integrated out. For β1 the two agree:
    today's path suits θ′ about as well as any path. For σ, today's path votes against
    any move beyond a fraction of a posterior SD, so a σ move needs a *new* path.
    """),
                    lab.fig_votes(scale_votes, effect_votes),
                    mo.md("""
    **Inside the real label move.** Each sweep ends with the label posterior given the
    chosen path; most sweeps are nearly decided one way or the other.
    """),
                    lab.fig_label_odds(mpgibbs_run),
                    mo.md("""
    **How that moves θ.** The exact joint posterior of β1 and log σ (other effects
    integrated out), with one chain's first retained draws joined.
    """),
                    lab.fig_plane(exact, {"MPGibbs label move alone": (mpgibbs_run, lab.C_LABEL)}),
                ]
            )
        }
    )
    return


@app.cell(hide_code=True)
def smoother_md(mo):
    mo.md(r"""
    ## Part 3 · The smoother: redrawing the path

    **What it does.** Given θ, redraw the whole path: the green move, along the band. It
    runs forward in time with 128 particles: at each step every particle proposes where x
    goes next, particles that explain the readings get more weight, and poor ones are
    replaced. Today's path always survives as one of them. At the end a path is assembled
    backwards in time from the particles. The weights use the exact model, so the result
    is exact whatever the proposals; better proposals mean more of the path changes per
    sweep. Inside MPGibbs it runs on the label mixture; here θ is fixed at its true value
    to see it alone.

    **The two path updates we have:**

    - **twisted Particle-aGRAD** (the policy): each proposal looks ahead to every later
      reading, through a backward filter built around a jittered copy of today's path.
      It runs step by step in time.
    - **dSMC with Particle-RWM** (the older default): each time step's candidates are a
      random-walk jitter of today's path, proposed separately so that all time steps can
      run in parallel.

    **On the reference (top).** θ never moves, so the smoother alone can't move a
    parameter. On σ's ridge its band is one pin tall: at fixed σ, every new path says the
    same σ. **What one axis can't show (bottom):** how much of the 257-point path a sweep
    actually renews. The real draws show it.
    """)
    return


@app.cell
def smoother_fits(data, lab, study, truth_model):
    smoother_runs = {
        "twisted Particle-aGRAD (policy)": lab.read_fit(
            study.fit(truth_model, data, lab.mpgibbs())
        ),
        "dSMC, Particle-RWM": lab.read_fit(
            study.fit(truth_model, data, lab.mpgibbs(path_update="particle_rwm"))
        ),
    }
    return (smoother_runs,)


@app.cell
def smoother_band(history, lab, np):
    _truth = np.append(np.asarray(lab.TRUE_EFFECTS), np.log(lab.TRUE_SCALE))
    _paths = lab.path_draws(history, np.repeat(_truth[None], 600, axis=0), np.random.default_rng(0))
    path_band = (np.quantile(_paths, 0.05, axis=0), np.quantile(_paths, 0.95, axis=0))
    return (path_band,)


@app.cell
def smoother_reference(history, lab, path_band, shapes, smoother_runs):
    lab.fig_smoother_schematic(shapes, history, path_band, smoother_runs)
    return


@app.cell(hide_code=True)
def smoother_caption(mo, smoother_runs):
    _lines = "\n".join(
        f"- **{label}**: {run.path_ess / run.paths[..., 0].size * 1000:.0f} effective draws per "
        f"1,000 at the path's worst point; at a given time point a sweep keeps today's value "
        f"{run.extra['reference_path_hit_rate'].mean():.0%} of the time."
        for label, run in smoother_runs.items()
    )
    mo.md(f"""
    **Measured on the real fits** at the true θ:

    {_lines}

    At fixed θ the policy's smoother renews almost the whole path every sweep: given θ,
    the path is the easy part in this example. Two cautions
    for France: the twisted update runs step by step in time, so one pass over 2,284
    days is sequential; and France's deterministic accumulators make it trace ancestors
    instead of assembling backwards, which renews early times less (C8 measured 38%
    against 99% late).
    """)
    return


@app.cell(hide_code=True)
def smoother_evidence(history, lab, mo, smoother_runs):
    mo.accordion(
        {
            "Evidence: how much of the path each sweep renews": lab.fig_smoother_refresh(
                history, smoother_runs
            )
        }
    )
    return


@app.cell(hide_code=True)
def comparison_md(mo):
    mo.md(r"""
    ## Part 4 · Comparing MPGibbs and joint NUTS

    MPGibbs proposes parameter labels and updates a path under their mixture. Joint
    NUTS moves parameters and path coordinates together. Compare retained movement,
    effective draws and computation on the same exact reference; a draw alone does not
    measure the work each transition spends.
    """)
    return


@app.cell
def arms(joint_runs, lab, mpgibbs_run):
    arm_runs = {
        "MPGibbs alone": (mpgibbs_run, lab.C_LABEL),
        "joint NUTS, innovation chart": (joint_runs["non_centred"], lab.C_JOINT),
    }
    return (arm_runs,)


@app.cell(hide_code=True)
def comparison_evidence(arm_runs, cloud, exact, history, lab, mo):
    mo.accordion(
        {
            "Evidence: the two real chains side by side": mo.vstack(
                [
                    lab.fig_arm_traces(arm_runs, exact),
                    lab.fig_autocorrelation(arm_runs),
                    lab.fig_plane(exact, arm_runs),
                    mo.md("""
    The same two chains on the real σ ridge, consecutive retained draws joined; the
    lines show how far one draw moves, not trajectories.
    """),
                    lab.fig_ridge_runs(cloud, history, exact.log_scales, arm_runs),
                ]
            )
        }
    )
    return


@app.cell
def summary_table(joint_runs, lab, mo, mpgibbs_run):
    mo.ui.table(
        lab.summary_rows(
            {
                "MPGibbs alone": mpgibbs_run,
                "joint NUTS, innovation": joint_runs["non_centred"],
                "joint NUTS, centred": joint_runs["centred"],
            },
            gradients={
                "MPGibbs alone": "",
                "joint NUTS, innovation": "num_steps",
                "joint NUTS, centred": "num_steps",
            },
        ),
        selection=None,
    )
    return


@app.cell(hide_code=True)
def closing(mo):
    mo.md(r"""
    ## What remains to measure

    The reference separates loose parameters from parameters tightly coupled to the
    path. MPGibbs must move both kinds through its label move; joint NUTS moves the
    complete state in its chosen chart. Parts 0, 2 and 3 explain where each spends work.

    The conditional parameter updates have been removed from MPGibbs. PARAM is expected
    to fail its convergence gate until joint parameter/path transport inside the label
    move is implemented. That transport is planned, not part of the sampler shown here.

    Compare the current exact samplers on convergence and cost under the same declared
    budget. This Gaussian example supplies an answer key; it does not establish their
    relative performance on nonlinear PARAM or France.
    """)
    return


if __name__ == "__main__":
    app.run()
