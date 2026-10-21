import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def imports_marimo():
    from nof1_causal_service.bootstrap import initialize_numerical_runtime as _initialize_runtime

    _initialize_runtime()
    import marimo as mo

    return (mo,)


@app.cell
def imports_lab():
    import support.particle_gibbs_plots as plots
    import support.particle_gibbs_surface as lab

    from nof1_causal_lab.models.ssm.inference.warmup.outcomes import InitializationFailure

    return InitializationFailure, lab, plots


@app.cell
def display_theme(mo, plots):
    theme = plots.theme_for(mo.app_meta().theme)
    return (theme,)


@app.cell(hide_code=True)
def intro(mo):
    mo.md(r"""
    # Particle Gibbs, mPGibbs and LAPS on one surface

    The 14 October benchmark (`scratchpad/warmup_optimized_rerun_20261014/report.html`)
    samples one synthetic
    state-space model with $T = 512$ steps. It has three global parameters (baseline $b$,
    input effect $\beta_u$, cross effect $\beta_x$) and 1,023 latent path coordinates.
    Seven particle arms and one joint-NUTS reference all sample the same joint posterior
    $p(\theta, \text{path} \mid y)$. They differ in **which coordinates move together**:

    | Arm | Parameter move | Path move | θ and path move… |
    | --- | --- | --- | --- |
    | A, B, C, E | mPGibbs: propose θ′, let the smoother choose between θ and θ′ | conditional DSMC against both labels | together, through the label |
    | D | adjusted LAPS steps on θ given the path | conditional DSMC given θ | in turn |
    | F | NUTS on θ given the path | conditional DSMC given θ | in turn |
    | G | exact draw of θ given the path | conditional DSMC given θ | in turn |
    | golden | NUTS on (θ, path) jointly | (same move) | together, by gradients |

    A, B, C and E use the same transition family with different initialization and
    adaptation budgets. The points below compare minimum parameter and latent ESS
    per retained draw. They do not compare equal computational cost or isolate one cause.
    """)
    return


@app.cell
def benchmark_view(plots, theme):
    plots.benchmark_figure(theme)
    return


@app.cell(hide_code=True)
def toy_md(mo):
    mo.md(r"""
    ## A posterior with two coordinates

    To see why, shrink the posterior to two coordinates: a parameter $\theta$ and a latent
    $x$. Then every move can be drawn on the surface $p(\theta, x \mid y)$.

    - **$x$ stands in for the path.** Given the path, the benchmark's parameters enter
      Gaussian transition means linearly. So the parameter update reads the path only
      through a few sums, for example the average drift the path implies for $b$. Here $x$
      plays that summary.
    - **The samplers are production code.** The particle kernels, LAPS and NUTS run through
      the production inference calls (`notebooks/support/particle_gibbs_surface.py`).
      The parameter-conditional draw in the Gibbs references is a mathematical control:
      closed form for the ridge, numerical inverse-CDF quadrature for the bounded-scale model.
      This control is also composed with the production DSMC smoother for the PG comparison.
    - **Particle-chain tuning is frozen at the values you set.** Their first 200 transitions
      are discarded without adaptation. LAPS and NUTS retain their production adaptation.
      Their optional run buttons keep costly work separate from the view controls.

    **What a one-step toy cannot show.** It has no time axis, so it cannot show:

    - genealogy: earlier states collapsing onto a few ancestors;
    - DSMC's tree stitching of neighbouring time blocks;
    - interval factors that span several steps;
    - how any acceptance rate scales with $T$;
    - parameters on very different scales, where one isotropic proposal can be too small
      for some coordinates and too large for others.

    What carries over is the geometry between the parameters and the latent, and how each
    sampler crosses it.
    """)
    return


@app.cell
def world_controls(mo):
    world_kind = mo.ui.dropdown(
        options={
            "Baseline ridge (location parameter)": "ridge",
            "Bounded noise scale (nonlinear)": "bounded_scale",
        },
        value="Baseline ridge (location parameter)",
        label="World",
    )
    ridge_tau = mo.ui.slider(
        steps=[0.15, 0.2, 0.25, 0.3, 0.35, 0.45, 0.6, 0.8, 1.0, 1.4],
        value=0.35,
        label="τ: how far the latent may wander from θ",
        show_value=True,
        debounce=True,
    )
    ridge_sigma = mo.ui.slider(
        steps=[0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.5],
        value=0.8,
        label="σ: observation noise",
        show_value=True,
        debounce=True,
    )
    scale_sigma = mo.ui.slider(
        steps=[0.25, 0.5, 1.0, 2.0, 4.0],
        value=2.0,
        label="σ: observation noise",
        show_value=True,
        debounce=True,
    )
    scale_prior_sd = mo.ui.slider(
        steps=[0.5, 1.0, 1.5, 2.0],
        value=1.5,
        label="prior sd of the scale-control θ",
        show_value=True,
        debounce=True,
    )
    return scale_prior_sd, scale_sigma, ridge_sigma, ridge_tau, world_kind


@app.cell
def world_panel(scale_prior_sd, scale_sigma, mo, ridge_sigma, ridge_tau, world_kind):
    _controls = (
        [ridge_tau, ridge_sigma] if world_kind.value == "ridge" else [scale_sigma, scale_prior_sd]
    )
    mo.hstack([world_kind, *_controls], justify="start", gap=2)
    return


@app.cell
def world_state(scale_prior_sd, scale_sigma, lab, ridge_sigma, ridge_tau, world_kind):
    world = (
        lab.Ridge(tau=ridge_tau.value, sigma=ridge_sigma.value)
        if world_kind.value == "ridge"
        else lab.BoundedScale(sigma=scale_sigma.value, prior_sd=scale_prior_sd.value)
    )
    grid = lab.surface(world)
    coupling = lab.coupling(world)
    # s^k and (1 - s)/(1 + s) are exact only for a Gaussian surface.
    gaussian_share = coupling.share if world_kind.value == "ridge" else None
    return coupling, gaussian_share, grid, world


@app.cell(hide_code=True)
def world_md(coupling, mo, world_kind):
    _ridge = r"""
    **Baseline ridge.** $\theta \sim N(0, 1)$ is a baseline. The latent wanders around it,
    $x \mid \theta \sim N(\theta, \tau^2)$, and one reading $y = 1.5 \sim N(x, \sigma^2)$
    sees the latent. Small $\tau$ ties the latent to the parameter, and large $\sigma$
    blurs the reading. Together they tilt the surface into a ridge along $x \approx \theta$.
    Try $\tau = \sigma = 0.6$: the toy then has coupling $s \approx 0.37$.
    """
    _scale = r"""
    **Bounded noise scale.** The same two coordinates now have a nonlinear relationship:

    $$
    \theta \sim N(-0.5, s_\theta^2), \qquad
    a(\theta) = \exp\!\bigl(\log(2)\tanh\theta\bigr), \qquad
    x \mid \theta \sim N(0, a(\theta)^2), \qquad y = 0 \sim N(x, \sigma^2).
    $$

    The prior standard deviation $s_\theta$ is the slider above. The latent standard
    deviation $a(\theta)$ varies smoothly between $1/2$ and $2$. This keeps scale
    dependence while removing the funnel's arbitrarily narrow neck. With $y = 0$ the
    surface is symmetric in $x$, so $\theta$ and $x$ have zero linear correlation, yet
    their conditional distributions depend on each other. Every sampler uses this
    same two-dimensional target; the scale link is part of the model itself.
    """
    mo.md(
        (_ridge if world_kind.value == "ridge" else _scale)
        + rf"""

    Coupling share here: **$s = {coupling.share:.2f}$**. Section 1 defines it.
    """
    )
    return


@app.cell
def surface_view(grid, plots, theme):
    plots.surface_figure(grid, theme)
    return


@app.cell(hide_code=True)
def surface_md(mo):
    mo.md(r"""
    The height is the joint posterior density relative to its peak on the grid.
    Wall curves are each scaled to their own peak so their shapes remain visible.
    The black curve on the back wall is
    $p(\theta \mid y)$, the parameter's **shadow**: the surface integrated over $x$, which
    is the answer we want. The grey curve on the left wall is the latent's shadow,
    $p(x \mid y)$. Drag to rotate.
    """)
    return


@app.cell(hide_code=True)
def cuts_md(mo):
    mo.md(r"""
    ## 1. Two cuts and a shadow: how smoothing feeds the parameter

    - **Cut at fixed $\theta$ (orange): the smoothing slice $p(x \mid \theta, y)$.** This
      is what the path update targets: the latent given the parameter and the data. For a
      real path it is the smoothing distribution. The orange dashed line joins the slice
      centres $E[x \mid \theta, y]$, so it shows how the smoothed latent follows $\theta$.
    - **Cut at fixed $x$ (blue): the parameter slice $p(\theta \mid x, y)$.** This is what
      the parameter update targets given the current path. Arm G draws it exactly, arm F
      steps through it with NUTS and arm D with adjusted LAPS. The blue dashed line joins
      its centres $E[\theta \mid x, y]$.
    - **The shadow $p(\theta \mid y)$ (black)** is what we want. A sampler that integrated
      the latent out would explore this curve directly.
    """)
    return


@app.cell
def cut_controls(grid, mo):
    theta_cut = mo.ui.slider(
        start=float(grid.theta[10]),
        stop=float(grid.theta[-11]),
        step=float((grid.theta[-1] - grid.theta[0]) / 100),
        value=float(grid.theta[grid.theta.size // 2]),
        label="cut at θ",
        show_value=True,
    )
    x_cut = mo.ui.slider(
        start=float(grid.x[10]),
        stop=float(grid.x[-11]),
        step=float((grid.x[-1] - grid.x[0]) / 100),
        value=float(grid.x[grid.x.size // 2]),
        label="cut at x",
        show_value=True,
    )
    mo.hstack([theta_cut, x_cut], justify="start", gap=2)
    return theta_cut, x_cut


@app.cell
def cuts_view(grid, mo, plots, theme, theta_cut, x_cut):
    mo.vstack(
        [
            plots.surface_figure(grid, theme, theta_cut=theta_cut.value, x_cut=x_cut.value),
            plots.cuts_figure(grid, theme, theta_cut=theta_cut.value, x_cut=x_cut.value),
        ]
    )
    return


@app.cell(hide_code=True)
def share_md(coupling, mo):
    mo.md(rf"""
    On average, conditioning on $x$ reduces the variance of $\theta$. Individual
    nonlinear slices can be wider or narrower. The fraction explained by the
    conditional mean $E[\theta \mid x,y]$ is the **coupling share**

    $$
    s \;=\; 1 - \frac{{E[\operatorname{{Var}}(\theta \mid x, y)]}}{{\operatorname{{Var}}(\theta \mid y)}}
    \;=\; 1 - \frac{{{coupling.conditional_var:.4f}}}{{{coupling.var:.4f}}} \;=\; {coupling.share:.2f}.
    $$

    For a Gaussian surface $s = \rho^2$: the two straight conditional-mean lines cross
    at the posterior mean, and a narrow ridge has $s$ near one. In the bounded-scale model, $\rho = 0$
    while $s > 0$ because the conditional mean of $\theta$ depends nonlinearly on $x$.
    This statistic does not capture every possible form of dependence. The bounded-scale
    value is estimated by deterministic quadrature on a finite grid, so it has numerical error.
    """)
    return


@app.cell(hide_code=True)
def gibbs_md(mo):
    mo.md(r"""
    ## 2. Gibbs reference: a staircase between the two cuts

    Alternate the two cuts. Draw $\theta$ from the parameter slice at the current $x$ (a
    horizontal move). Then draw $x$ from the smoothing slice at the new $\theta$ (a
    vertical move). Each $\theta$ lands near the blue line and each $x$ near the orange
    line, so the chain zig-zags between them. Arm G is this, except that its path move is
    a DSMC sweep rather than an exact draw.

    On the ridge these are exact conditional draws. On the bounded-scale model the parameter draw
    is a numerical approximation on a finite grid, not an exact continuous sampler.

    **The Gaussian echo.** Take one sweep:

    1. Move $\theta$ away from the posterior mean $\mu$.
    2. The smoother pulls $x$ along by a fraction of that move.
    3. The next parameter draw reads $x$ back.

    For the Gaussian ridge,
    $E[\theta_{k+1} - \mu \mid \theta_k] = s\,(\theta_k - \mu)$,
    lag $k$ gives autocorrelation $s^k$, and the asymptotic effective sample size
    per sweep is $(1 - s)/(1 + s)$.

    For exact two-block Gibbs at stationarity, lag-1 correlation is $s$ more generally.
    The geometric sequence and linear conditional expectation are Gaussian results.
    In the scatter below, the line is the best linear predictor; for a nonlinear target
    it need not be the conditional-mean curve.
    """)
    return


@app.cell
def chain_settings(grid, mo, plots):
    start = plots.start_point(grid)
    sweeps = 1500
    warmup = 200
    path_sweeps = mo.ui.slider(
        start=5, stop=120, step=5, value=30, label="sweeps drawn on the surface", show_value=True
    )
    return path_sweeps, start, sweeps, warmup


@app.cell
def gibbs_run(lab, start, sweeps, warmup, world):
    gibbs_chain = lab.gibbs(world, sweeps, start, seed=1, warmup=warmup)
    return (gibbs_chain,)


@app.cell
def gibbs_view(coupling, gibbs_chain, grid, mo, path_sweeps, plots, theme):
    _gibbs = plots.Series("Gibbs reference", gibbs_chain, theme.gibbs)
    mo.vstack(
        [
            path_sweeps,
            plots.paths_figure(grid, theme, [_gibbs], sweeps=path_sweeps.value),
            plots.echo_figure(
                theme, gibbs_chain, mean=coupling.mean, share=coupling.share, name="Gibbs reference"
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def gibbs_benchmark_md(coupling, gaussian_share, mo):
    _theory = (
        rf"On this ridge, $(1-s)/(1+s) = {coupling.gibbs_ess_per_sweep:.3f}$ effective draws per sweep asymptotically."
        if gaussian_share is not None
        else "The bounded-scale model has no geometric ESS formula here; use its measured autocorrelation and ESS."
    )
    mo.md(
        _theory
        + r"""

    **Back to the benchmark.** Arm G uses exact parameter-conditional draws, yet its
    minimum parameter ESS is about 993 and its minimum latent ESS about 302 out of
    2,000 retained draws. Exact parameter updates alone therefore do not make the
    joint draws independent. The path update and its coupling to the parameters still
    matter (saved diagnostics (`scratchpad/warmup_optimized_rerun_20261014/report.json`)).
    """
    )
    return


@app.cell(hide_code=True)
def dsmc_md(mo):
    mo.md(r"""
    ## 3. The production smoother: one conditional DSMC sweep

    The path move in every arm is a conditional DSMC sweep
    (`samplers/particle/dsmc.py`). At one time step it reduces to its leaf:

    1. Form the aMALA+ auxiliary mean $b(x) = x + \kappa\,\tfrac{\delta}{2}\,
       \nabla_x \log\text{target}(x)$, with $\kappa = 0$ by default or $\kappa = 1$ for local drift.
    2. Draw $z \sim N(b(x_{\text{ref}}), \delta/2)$ around the drifted reference.
    3. Propose $N-1$ particles $x_i \sim N(z, \delta/2)$. Particle 0 is the reference.
    4. Weight each by $\text{target}(x_i)\,N(z; b(x_i), \delta/2) / N(x_i; z, \delta/2)$.
    5. Apply the modified Gibbs selection to those weights.

    As $N$ grows, the draw becomes an exact draw from

    $$
    \pi(x \mid z) \;\propto\; p(x \mid \theta, y)\; N(z;\, b(x),\, \delta/2),
    $$

    the thick curve below, **not** from the slice $p(x \mid \theta, y)$. The auxiliary
    point contributes a locality scale of $\sqrt{\delta/2}$; the target also shapes the move.
    Production adapts $\delta$ in warmup towards a 0.75 move rate. A move rate does not
    by itself measure how far a chain travels. Here $\delta$ stays where you put it.

    **The smoother's own echo.** The auxiliary can retain dependence even with infinitely
    many particles. If the Gaussian slice has variance $v$, set $h=\delta/2$ and
    $a=1-\kappa h/v$. Then
    $E[x_{\text{new}} - m \mid x_{\text{ref}}] = \psi\,(x_{\text{ref}} - m)$ with

    $$
    \psi \;=\; \frac{a^2v}{a^2v + h}.
    $$

    This is the infinite-particle echo at fixed $\theta$ for these Gaussian latent
    conditionals. It is distinct from $s$, although both depend on the model.
    Finite-particle selection can retain the reference and add further dependence;
    the formula is not an exact autocorrelation for the full alternating chain.
    """)
    return


@app.cell
def dsmc_controls(mo):
    particles = mo.ui.slider(
        steps=[2, 4, 8, 16, 32, 64, 128, 256], value=16, label="particles N", show_value=True
    )
    delta = mo.ui.slider(
        steps=[0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
        value=0.05,
        label="δ (each auxiliary/proposal variance is δ/2)",
        show_value=True,
    )
    sweep_seed = mo.ui.slider(
        start=0, stop=50, step=1, value=0, label="sweep seed", show_value=True
    )
    mo.hstack([particles, delta, sweep_seed], justify="start", gap=2)
    return delta, particles, sweep_seed


@app.cell
def dsmc_sweep_view(coupling, delta, lab, mo, particles, plots, sweep_seed, theme, world):
    _theta = coupling.mean
    _x_mean, _x_var = world.x_given_theta(_theta)
    _sweep = lab.dsmc_sweep(
        world,
        _theta,
        float(_x_mean - _x_var**0.5),
        seed=sweep_seed.value,
        particles=particles.value,
        delta=delta.value,
    )
    _echo = plots.smoother_echo(float(_x_var), delta.value, kappa=_sweep.kappa)
    mo.vstack(
        [
            mo.md(
                f"θ is fixed at its posterior mean {_theta:.2f}, and the reference latent "
                f"sits one slice sd below the slice centre. At this δ the infinite-particle echo "
                f"is **ψ = {_echo:.2f}**: the slice variance is v = {float(_x_var):.4f} "
                f"and δ/2 = {delta.value / 2:.4f}. This is a fixed-θ toy calculation, "
                f"not a prediction for a multi-time-point chain."
            ),
            plots.sweep_figure(
                world,
                _sweep,
                theme,
                delta=delta.value,
                color=theme.dsmc,
                x_range=world.x_box(),
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def pg_md(mo):
    mo.md(r"""
    Put the sweep into the chain: draw $\theta$ from its conditional given $x$, then
    make one production DSMC sweep for $x$. The parameter draw is exact on the ridge
    and uses grid quadrature on the bounded-scale model. This illustrates arm G's update order.
    The smoother contributes dependence beyond exact two-block Gibbs.
    """)
    return


@app.cell
def pg_run(delta, lab, particles, start, sweeps, warmup, world):
    pg_chain = lab.particle_gibbs(
        world,
        sweeps,
        start,
        seed=2,
        particles=particles.value,
        delta=delta.value,
        warmup=warmup,
    )
    return (pg_chain,)


@app.cell
def pg_view(gaussian_share, gibbs_chain, grid, mo, path_sweeps, pg_chain, plots, theme):
    _series = [
        plots.Series("Gibbs reference", gibbs_chain, theme.gibbs),
        plots.Series("conditional θ + DSMC", pg_chain, theme.dsmc),
    ]
    mo.vstack(
        [
            plots.paths_figure(grid, theme, _series, sweeps=path_sweeps.value),
            plots.mixing_figure(theme, _series, share=gaussian_share),
        ]
    )
    return


@app.cell(hide_code=True)
def mpgibbs_md(mo):
    mo.md(r"""
    ## 4. mPGibbs: let the smoother choose the parameter

    Marginalized particle Gibbs (Corenflos 2025,
    [Particle Gibbs without the Gibbs bit](https://arxiv.org/abs/2505.04611);
    `samplers/mpgibbs/kernel.py`) moves $\theta$ and the latent in one step:

    1. **Propose.** Draw $u \sim N(\theta + 2\,\text{step}\,\Sigma\, g(\theta),\;
       2\,\text{step}\,\Sigma)$, then $\theta' \sim N(u, 2\,\text{step}\,\Sigma)$. For the
       random walk $g = 0$. For pseudo-Langevin, $g$ is the complete-data gradient at
       one fixed path, so it does not move with the chain.
    2. **Label prior.** Each label gets $w_l \propto p(\theta_l)\, q(u \mid \theta_l) /
       q(\theta_l \mid u)$. This Barker correction keeps the move exact.
    3. **Smooth against both labels.** The DSMC sweep targets the mixture
       $\sum_l w_l\, p(x, y \mid \theta_l)$, so each particle is scored under $\theta$ and
       under $\theta'$.
    4. **Pick the label.** Given the new latent, keep $\theta$ or move to $\theta'$ with
       probability $\propto w_l\, p(x_{\text{new}}, y \mid \theta_l)$.

    So $\theta$ moves only if the new latent is plausible under $\theta'$. The auxiliary
    point ties that latent to the old one. The label is therefore judged on a latent that
    still mostly reflects $\theta$, unless the reach $\sqrt{\delta}$ covers the shift
    between the two slices.
    """)
    return


@app.cell
def mpgibbs_controls(mo):
    parameter_step = mo.ui.slider(
        steps=[0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
        value=0.02,
        label="param_step_size (production default 0.02)",
        show_value=True,
    )
    proposal = mo.ui.dropdown(
        options={
            "random walk": "random_walk",
            "pseudo-Langevin (production default)": "pseudo_langevin",
        },
        value="random walk",
        label="parameter proposal",
    )
    return parameter_step, proposal


@app.cell
def mpgibbs_panel(coupling, mo, parameter_step, plots, proposal):
    _sd = plots.proposal_sd(parameter_step.value)
    mo.vstack(
        [
            mo.hstack([parameter_step, proposal], justify="start", gap=2),
            mo.md(
                f"With Σ = I (a cold production chain), θ′ − θ has standard deviation "
                f"2√step = {_sd:.3f}, which is **{_sd / coupling.var**0.5:.2f}×** the "
                f"posterior sd of θ."
            ),
        ]
    )
    return


@app.cell
def mpgibbs_sweep_view(
    coupling,
    delta,
    grid,
    lab,
    mo,
    parameter_step,
    particles,
    plots,
    proposal,
    start,
    sweep_seed,
    theme,
    world,
):
    _theta = coupling.mean
    _x_mean, _ = world.x_given_theta(_theta)
    _sweep = lab.mpgibbs_sweep(
        world,
        _theta,
        float(_x_mean),
        seed=sweep_seed.value,
        particles=particles.value,
        delta=delta.value,
        step=parameter_step.value,
        proposal=proposal.value,
        anchor_x=start[1],
    )
    mo.vstack(
        [
            plots.sweep_figure(
                world,
                _sweep,
                theme,
                delta=delta.value,
                color=theme.mpgibbs,
                x_range=world.x_box(),
            ),
            plots.mpgibbs_move_figure(grid, _sweep, theme),
        ]
    )
    return


@app.cell
def mpgibbs_run(delta, lab, parameter_step, particles, proposal, start, sweeps, warmup, world):
    # A cold production chain: Σ = I, and pseudo-Langevin's fixed path is the initial one.
    mpgibbs_chain = lab.mpgibbs(
        world,
        sweeps,
        start,
        seed=3,
        particles=particles.value,
        delta=delta.value,
        step=parameter_step.value,
        proposal=proposal.value,
        anchor_x=start[1],
        warmup=warmup,
    )
    return (mpgibbs_chain,)


@app.cell
def mpgibbs_view(
    gaussian_share,
    gibbs_chain,
    grid,
    mo,
    mpgibbs_chain,
    path_sweeps,
    pg_chain,
    plots,
    theme,
):
    _series = [
        plots.Series("Gibbs reference", gibbs_chain, theme.gibbs),
        plots.Series("conditional θ + DSMC", pg_chain, theme.dsmc),
        plots.Series("mPGibbs + DSMC", mpgibbs_chain, theme.mpgibbs),
    ]
    _switch = float(mpgibbs_chain.theta_moved.mean())
    mo.vstack(
        [
            mo.md(f"mPGibbs moved θ in **{_switch:.0%}** of sweeps."),
            plots.paths_figure(grid, theme, _series[::2], sweeps=path_sweeps.value),
            plots.mixing_figure(theme, _series, share=gaussian_share),
        ]
    )
    return


@app.cell(hide_code=True)
def mpgibbs_benchmark_md(mo):
    mo.md(r"""
    mPGibbs balances two mechanisms:

    - **Joint movement.** The smoother can propose a latent that supports a different
      parameter label, reducing the cost of alternating narrow conditionals.
    - **Local proposals.** The parameter step and auxiliary smoother reach limit how far
      the pair moves. Pseudo-Langevin adds drift but still requires the exact correction.

    Increasing a step can improve exploration or lower selection probability; decreasing
    it can make accepted moves too small. Coupling, preconditioning, particle count and
    smoother reach all matter. There is no universal coupling threshold at which mPGibbs wins.

    In the saved benchmark (`scratchpad/warmup_optimized_rerun_20261014/report.json`),
    the mPGibbs arms have minimum parameter ESS of 4–36 per 2,000 draws. Arm C improves
    over cold arm B, but the experiment changes both initialization and adaptation.
    These results motivate the toy mechanisms; the toy does not establish their causal
    contribution in a 512-step model.
    """)
    return


@app.cell(hide_code=True)
def laps_md(mo):
    mo.md(r"""
    ## 5. LAPS: an ensemble that moves both coordinates at once (initialization only)

    LAPS, the Late Adjusted Parallel Sampler from BlackJAX (`warmup/laps_ensemble.py`),
    runs many chains on the **joint** density in two phases:

    1. **Unadjusted burn-in.** Every member follows isokinetic Langevin dynamics, so its
       velocity carries it along the ridge, and $\theta$ and $x$ move together. There is
       no accept/reject step. After each step the ensemble retunes everyone:
       - the **step size**, to hit an energy-error target that shrinks with the ensemble's
         equipartition bias, so steps anneal as the cloud settles;
       - the momentum decoherence length;
       - a **diagonal preconditioner** from each coordinate's ensemble variance.

       Burn-in can stop early when BlackJAX's ensemble bias criterion is met;
       this is a warmup stopping rule, not a convergence guarantee.
    2. **Late adjustment.** A few Metropolis-adjusted (MAMS) steps follow, with the
       learned preconditioner and a step size bisected towards 0.7 acceptance (0.9 above
       200 dimensions).

    Production derives paired chain starts $(\theta, x)$, parameter covariance
    (stored through its Cholesky factor `parameter_chol`) and fixed pilot path moments.
    Initialization history is discarded, not pooled with posterior draws. The notebook
    opts into a trace so it can show those discarded transitions. A diagonal preconditioner
    rescales the axes but cannot rotate a tilted ridge. Momentum helps members travel
    along it. How quickly they settle depends on the target.
    """)
    return


@app.cell
def laps_controls(mo):
    laps_button = mo.ui.run_button(label="Run LAPS initialization")
    mo.hstack([laps_button], justify="start")
    return (laps_button,)


@app.cell
def laps_run(InitializationFailure, lab, laps_button, mo, world):
    mo.stop(not laps_button.value, mo.md("Press the button to run LAPS on this surface."))
    laps = lab.run_laps(world, seed=4)
    if isinstance(laps, InitializationFailure):
        mo.stop(True, mo.md(f"LAPS returned no initialization: {laps.message}"))
    return (laps,)


@app.cell
def laps_scrubber(laps, mo):
    _last = int(laps.positions.shape[0])
    laps_step = mo.ui.slider(
        start=1,
        stop=_last,
        step=1,
        value=_last,
        label="executed step (the last one shows the handoff)",
        show_value=True,
    )
    mo.hstack([laps_step], justify="start")
    return (laps_step,)


@app.cell
def laps_view(grid, laps, laps_step, plots, theme):
    plots.laps_figure(grid, laps, theme, step=laps_step.value - 1)
    return


@app.cell(hide_code=True)
def handoff_md(mo):
    mo.md(r"""
    **The handoff.** mPGibbs starts from the selected LAPS members, with the LAPS
    parameter covariance as its proposal scale. Better starts and a better scale help
    convergence: in the benchmark diagnostics (`scratchpad/warmup_optimized_rerun_20261014/report.json`),
    R̂ fell from 4.19 to 1.10 between arms B and C, alongside other adaptation changes.
    Better starts reduce transient bias. The preconditioner also changes proposal
    geometry and can improve mixing, but it does not remove the smoother's auxiliary locality.
    """)
    return


@app.cell
def handoff_run(delta, lab, laps, parameter_step, particles, proposal, sweeps, warmup, world):
    # As production: start at the first selected member, scale proposals by the LAPS
    # parameter covariance, and anchor pseudo-Langevin on that member's path.
    _start = (float(laps.starts[0, 0]), float(laps.starts[0, 1]))
    handoff_chain = lab.mpgibbs(
        world,
        sweeps,
        _start,
        seed=5,
        particles=particles.value,
        delta=delta.value,
        step=parameter_step.value,
        scale=laps.parameter_chol,
        proposal=proposal.value,
        anchor_x=_start[1],
        warmup=warmup,
    )
    return (handoff_chain,)


@app.cell
def handoff_view(
    gaussian_share,
    handoff_chain,
    mpgibbs_chain,
    plots,
    theme,
):
    plots.mixing_figure(
        theme,
        [
            plots.Series("mPGibbs, cold start", mpgibbs_chain, theme.mpgibbs),
            plots.Series("mPGibbs from LAPS", handoff_chain, theme.laps),
        ],
        share=gaussian_share,
    )
    return


@app.cell(hide_code=True)
def nuts_md(mo):
    mo.md(r"""
    ## 6. Joint NUTS: the golden reference's move

    NUTS on $(\theta, x)$ jointly, through production's `run_nuts_da` with its own
    initialization and adaptation. Its gradient trajectories run along the ridge.
    In the benchmark, joint NUTS sampled all 1,026 coordinates in 33 s on a CPU (float64),
    with minimum bulk ESS of 4,673 over 4,000 draws and no divergences
    (saved reference checks in `scratchpad/warmup_optimized_rerun_20261014/report.html`).
    ESS can exceed the draw count when successive draws are anticorrelated.
    """)
    return


@app.cell
def nuts_controls(mo):
    nuts_button = mo.ui.run_button(label="Run joint NUTS")
    mo.hstack([nuts_button], justify="start")
    return (nuts_button,)


@app.cell
def nuts_run(InitializationFailure, lab, mo, nuts_button, world):
    mo.stop(not nuts_button.value, mo.md("Press the button to run joint NUTS on this surface."))
    nuts = lab.run_nuts(world, seed=6)
    if isinstance(nuts, InitializationFailure):
        mo.stop(True, mo.md(f"NUTS returned no initialization: {nuts.message}"))
    return (nuts,)


@app.cell
def nuts_view(gaussian_share, gibbs_chain, grid, mo, nuts, path_sweeps, plots, theme):
    _series = [
        plots.Series("Gibbs reference", gibbs_chain, theme.gibbs),
        plots.Series("joint NUTS", nuts.chain, theme.nuts),
    ]
    mo.vstack(
        [
            mo.md(
                f"NUTS took **{nuts.integration_steps.mean():.1f}** integration steps per "
                f"draw, with **{int(nuts.divergences.sum())} divergences**. It starts from its own "
                f"production initialization."
                + (
                    " **Divergences can bias these draws; this run is not a reliable reference.**"
                    if nuts.divergences.any()
                    else ""
                )
            ),
            plots.paths_figure(grid, theme, _series, sweeps=path_sweeps.value),
            plots.mixing_figure(theme, _series, share=gaussian_share),
        ]
    )
    return


@app.cell(hide_code=True)
def compare_md(mo):
    mo.md(r"""
    ## 7. Which recipe wins, and when

    Effective draws per sweep at the current setting. The costs differ: a DSMC sweep
    weighs $N$ particles under one label (two for mPGibbs), and a NUTS draw costs its
    gradient evaluations.
    """)
    return


@app.cell
def scoreboard_view(
    coupling,
    gaussian_share,
    gibbs_chain,
    lab,
    mpgibbs_chain,
    pg_chain,
    plots,
    theme,
):
    def _score(name, color, chain, note):
        size = chain.theta.size
        return plots.Score(name, color, lab.ess(chain.theta) / size, lab.ess(chain.x) / size, note)

    plots.scoreboard_figure(
        theme,
        [
            _score("Gibbs reference", theme.gibbs, gibbs_chain, "two conditional draws"),
            _score("conditional θ + DSMC", theme.dsmc, pg_chain, "one DSMC sweep, one label"),
            _score("mPGibbs + DSMC", theme.mpgibbs, mpgibbs_chain, "one DSMC sweep, two labels"),
        ],
        gibbs_theory=None if gaussian_share is None else coupling.gibbs_ess_per_sweep,
    )
    return


@app.cell(hide_code=True)
def sweep_md(mo):
    mo.md(r"""
    **Across couplings.** The run below repeats the comparison on the ridge for several
    values of $\tau$ at the current $\sigma$ and DSMC settings, and plots the parameter's
    ESS per sweep against the coupling share. The dashed curve is exact Gibbs's theory.
    These short single-chain ESS estimates are noisy and are not a timing benchmark.
    """)
    return


@app.cell
def coupling_sweep_controls(mo):
    coupling_button = mo.ui.run_button(label="Run the coupling sweep")
    mo.hstack([coupling_button], justify="start")
    return (coupling_button,)


@app.cell
def coupling_sweep_run(
    coupling_button,
    delta,
    lab,
    mo,
    parameter_step,
    particles,
    plots,
    proposal,
    ridge_sigma,
    theme,
):
    mo.stop(not coupling_button.value, mo.md("Press the button to run the sweep."))
    _rows = lab.coupling_sweep(
        sigma=ridge_sigma.value,
        taus=(0.15, 0.25, 0.35, 0.5, 0.7, 1.0, 1.4),
        particles=particles.value,
        delta=delta.value,
        step=parameter_step.value,
        proposal=proposal.value,
    )
    plots.coupling_figure(theme, plots.coupling_curves(_rows, theme))
    return


@app.cell(hide_code=True)
def takeaways(mo):
    mo.md(r"""
    ## Takeaways

    1. **Coupling leaves memory.** On the Gaussian ridge, exact two-block Gibbs has
       autocorrelation $s^k$. The bounded-scale model demonstrates nonlinear dependence, where that
       geometric formula no longer applies.
    2. **More particles do not remove the auxiliary tether.** At fixed parameter and
       infinite particle count, the local smoother still conditions on $z$. For these
       Gaussian latent conditionals its echo is $a^2v/(a^2v+\delta/2)$, where
       $a=1-\kappa\delta/(2v)$.
    3. **mPGibbs chooses a parameter together with a latent.** How effectively it moves
       depends on proposal geometry and smoother reach as well as coupling.
    4. **LAPS supplies initialization.** Its ensemble yields paired starts and proposal
       scales. Its adaptive history is not posterior output.
    5. **NUTS explores the joint by gradients.** It has its own adaptation and variable
       integration cost. Inspect divergences alongside ESS.
    6. **One time point isolates geometry.** Genealogy, tree stitching, interval factors
       and scaling with time-series length require a multi-time-point study.

    ESS here is a short single-chain estimate, not a convergence certificate. Compare
    autocorrelation, movement and the reference density as well as the score.
    """)
    return


if __name__ == "__main__":
    app.run()
