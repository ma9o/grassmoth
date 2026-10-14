import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def imports_marimo():
    from nof1_causal_lab.bootstrap import initialize_numerical_runtime as _initialize_runtime

    _initialize_runtime()
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def intro(mo):
    mo.md(r"""
    # Reachability checks as controlled failures

    A prior can be perfectly legal and still describe a model that cannot be computed, cannot
    be resolved by the study design, or cannot put observable variation where the data live.
    The reachability battery catches those failures **before fitting**.

    This notebook keeps one toy N-of-1 study fixed and changes one modeling choice at a time.
    Every red verdict is produced by the production functions in
    `nof1_causal_lab.models.ssm.diagnostics.reachability`; the notebook only manufactures small,
    deterministic prior-predictive arrays and draws the evidence, so each failure appears in
    isolation, with nothing fitted.

    Each failure panel answers three questions in turn:

    1. **What statistic crossed which band?**
    2. **What modeling choice created that geometry?**
    3. **What does the finding mean, and which modeling choice should be revisited?**
    """)
    return


@app.cell
def imports():
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np

    from nof1_causal_lab.models.ssm.diagnostics.reachability import (
        check_confinement,
        check_coverage,
        check_edge_share,
        check_resolvability,
        check_saturation,
        check_scale,
        check_transmission,
    )
    from nof1_causal_lab.observations.quality.rules import indicator_findings

    return (
        indicator_findings,
        check_confinement,
        check_coverage,
        check_edge_share,
        check_resolvability,
        check_saturation,
        check_scale,
        check_transmission,
        np,
        nx,
        plt,
    )


@app.cell
def utilities(mo, np):
    def sigmoid(values):
        return 1.0 / (1.0 + np.exp(-np.asarray(values)))

    def simulate_ar1(rng, n_draws, n_times, persistence, marginal_scale):
        paths = np.empty((n_draws, n_times))
        paths[:, 0] = rng.normal(0.0, marginal_scale, n_draws)
        innovation_scale = marginal_scale * np.sqrt(1.0 - persistence**2)
        for time_index in range(1, n_times):
            paths[:, time_index] = persistence * paths[:, time_index - 1] + rng.normal(
                0.0, innovation_scale, n_draws
            )
        return paths

    def robust_scale(values, axis=None):
        q75 = np.percentile(values, 75, axis=axis)
        q25 = np.percentile(values, 25, axis=axis)
        return (q75 - q25) / 1.349

    def style_axes(axes):
        for axis in np.atleast_1d(axes).flat:
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis="y", color="#ececec", linewidth=0.7, zorder=0)

    def result_panel(title, result, increment, mechanism, revision, standing):
        diagnosis = "\n".join(f"- {line}" for line in result.diagnosis)
        markdown = "\n\n".join(
            [
                f"### {title}",
                f"**Increment.** {increment}",
                "| statistic | required band | verdict |\n|---|---|---|\n"
                f"| {result.value} | {result.band} | "
                f"{'🟢 green' if result.passed else '🔴 red'} |",
                f"**What it catches.** {mechanism}",
                "**Production diagnosis.**\n\n" + diagnosis,
                f"**Revise.** {revision}",
                f"**Can the finding stand?** {standing} The next edit remains the modeler's "
                "decision.",
            ]
        )
        return mo.md(markdown)

    return result_panel, robust_scale, sigmoid, simulate_ar1, style_axes


@app.cell(hide_code=True)
def toy_md(mo):
    mo.md(r"""
    ## 1. One toy, held fixed

    One person self-tracks for **60 once-daily observations**. `AutonomicArousal` is an
    explicit unobserved common cause of caffeine load and perceived stress. Caffeine can affect
    stress through a Hill edge. Stress is measured on a 0–100 slider using a Beta/logit
    emission; caffeine has a Poisson count indicator. The last increment (§6) proposes HRV as an
    arousal indicator but gives it no rows.

    The healthy fragment uses 512 prior draws. Its stress relaxation time is about four days,
    its marginal latent scale is anchored at 0.8, the Hill EC50 sits near the caffeine parent's
    scale, and the slider's inverse-link median sets its intercept.
    """)
    return


@app.cell(hide_code=True)
def toy_model_diagram(mo, nx, plt):
    _graph = nx.DiGraph()
    _causal_edges = [
        ("Autonomic\nArousal (U)", "Caffeine\nLoad"),
        ("Autonomic\nArousal (U)", "Perceived\nStress"),
        ("Caffeine\nLoad", "Perceived\nStress"),
    ]
    _measurement_edges = [
        ("Caffeine\nLoad", "Caffeine\ncount"),
        ("Perceived\nStress", "Stress\nslider"),
        ("Autonomic\nArousal (U)", "HRV\n(optional)"),
    ]
    _graph.add_edges_from([*_causal_edges, *_measurement_edges])
    _positions = {
        "Autonomic\nArousal (U)": (0.0, 1.25),
        "Caffeine\nLoad": (-1.15, 0.25),
        "Perceived\nStress": (1.15, 0.25),
        "Caffeine\ncount": (-1.15, -0.85),
        "Stress\nslider": (1.15, -0.85),
        "HRV\n(optional)": (0.0, -0.85),
    }
    _fig, _ax = plt.subplots(figsize=(10.5, 5.0))
    nx.draw_networkx_edges(
        _graph,
        _positions,
        edgelist=_causal_edges,
        ax=_ax,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=18,
        width=1.8,
        edge_color="#505050",
        node_size=2600,
    )
    nx.draw_networkx_edges(
        _graph,
        _positions,
        edgelist=_measurement_edges,
        ax=_ax,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=15,
        width=1.3,
        style="dashed",
        edge_color="#8b8b8b",
        node_size=2600,
    )
    nx.draw_networkx_nodes(
        _graph,
        _positions,
        nodelist=["Caffeine\nLoad", "Perceived\nStress"],
        ax=_ax,
        node_color="#3b6ea5",
        node_size=3000,
        edgecolors="white",
        linewidths=1.5,
    )
    nx.draw_networkx_nodes(
        _graph,
        _positions,
        nodelist=["Autonomic\nArousal (U)"],
        ax=_ax,
        node_color="white",
        node_size=3000,
        edgecolors="#c0504d",
        linewidths=2.5,
    )
    nx.draw_networkx_nodes(
        _graph,
        _positions,
        nodelist=["Caffeine\ncount", "Stress\nslider"],
        ax=_ax,
        node_color="#dce8f3",
        node_shape="s",
        node_size=2300,
        edgecolors="#3b6ea5",
        linewidths=1.4,
    )
    nx.draw_networkx_nodes(
        _graph,
        _positions,
        nodelist=["HRV\n(optional)"],
        ax=_ax,
        node_color="#f4f4f4",
        node_shape="s",
        node_size=2300,
        edgecolors="#aaaaaa",
        linewidths=1.4,
    )
    _label_colors = {
        _node: (
            "#c0504d"
            if _node == "Autonomic\nArousal (U)"
            else "white"
            if _node in {"Caffeine\nLoad", "Perceived\nStress"}
            else "#333333"
        )
        for _node in _graph.nodes
    }
    for _node, (_x, _y) in _positions.items():
        _ax.text(
            _x,
            _y,
            _node,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=_label_colors[_node],
            zorder=5,
        )
    _ax.text(0.0, 0.39, "Hill edge", ha="center", va="bottom", fontsize=9, color="#333333")
    _ax.text(
        0.0,
        1.58,
        "solid = causal DAG     dashed = measurement",
        ha="center",
        fontsize=9,
        color="#666666",
    )
    _ax.set_xlim(-1.85, 1.85)
    _ax.set_ylim(-1.25, 1.75)
    _ax.axis("off")
    _fig.tight_layout()
    mo.as_html(_fig)
    return


@app.cell
def generate_shared_toy(np, sigmoid, simulate_ar1):
    _rng = np.random.default_rng(20260713)
    n_draws = 512
    times = np.arange(60, dtype=float)

    arousal_paths = simulate_ar1(_rng, n_draws, times.size, 0.88, 1.0)
    _caffeine_innovation = simulate_ar1(_rng, n_draws, times.size, 0.45, 1.0)
    _caffeine_latent = 0.45 * arousal_paths + 0.893 * _caffeine_innovation
    caffeine_paths = np.exp(0.32 * _caffeine_latent)

    own_stress_paths = simulate_ar1(_rng, n_draws, times.size, 0.78, 0.72)
    ec50_draws = np.exp(_rng.normal(np.log(np.median(caffeine_paths)), 0.18, size=n_draws))
    hill_n_draws = np.exp(_rng.normal(np.log(2.0), 0.10, size=n_draws))
    _caffeine_power = caffeine_paths ** hill_n_draws[:, None]
    hill_occupancy = _caffeine_power / (
        _caffeine_power + ec50_draws[:, None] ** hill_n_draws[:, None]
    )
    edge_component = 0.28 * (hill_occupancy - np.mean(hill_occupancy, axis=1, keepdims=True))
    stress_without_edge = own_stress_paths + 0.18 * arousal_paths
    stress_paths = stress_without_edge + edge_component

    stress_signal = sigmoid(-0.45 + 0.85 * stress_paths)
    _concentration = 8.0
    stress_pp = _rng.beta(
        stress_signal * _concentration,
        (1.0 - stress_signal) * _concentration,
    )
    stress_observed = stress_pp[37].copy()
    tau_draws = np.exp(_rng.normal(np.log(4.0), 0.25, size=n_draws))

    return (
        arousal_paths,
        caffeine_paths,
        ec50_draws,
        edge_component,
        hill_n_draws,
        hill_occupancy,
        n_draws,
        own_stress_paths,
        stress_observed,
        stress_paths,
        stress_pp,
        stress_signal,
        stress_without_edge,
        tau_draws,
        times,
    )


@app.cell(hide_code=True)
def healthy_md(mo):
    mo.md(r"""
    ## 2. Establish the healthy reference

    Before any failure is manufactured, the unchanged toy must pass every check. C5d is not
    emitted here because every declared indicator has data; it appears only when the empty HRV
    channel is proposed.
    """)
    return


@app.cell
def evaluate_healthy_model(
    caffeine_paths,
    check_confinement,
    check_coverage,
    check_edge_share,
    check_resolvability,
    check_saturation,
    check_scale,
    check_transmission,
    ec50_draws,
    hill_n_draws,
    stress_observed,
    stress_paths,
    stress_pp,
    stress_signal,
    stress_without_edge,
    tau_draws,
    times,
):
    _results = [
        *check_confinement("PerceivedStress", stress_paths, times),
        check_scale(
            "PerceivedStress",
            stress_paths,
            scale_anchor=0.8,
            anchor_src="stress-slider anchor",
            anchor_detail="inverse-logit IQR / 1.349 with unit reference loading",
        ),
        check_resolvability("PerceivedStress", tau_draws, times),
        *check_edge_share("CaffeineLoad -> PerceivedStress", stress_paths, stress_without_edge),
        check_saturation(
            "CaffeineLoad -> PerceivedStress",
            ec50_draws,
            hill_n_draws,
            caffeine_paths,
        ),
        *check_coverage(
            "StressSlider",
            stress_pp,
            stress_observed,
            distribution="beta",
        ),
        check_transmission(
            "StressSlider",
            stress_signal,
            stress_signal * (1.0 - stress_signal) / 9.0,
        ),
    ]
    healthy_results = {_result.check: _result for _result in _results}
    assert all(_result.passed for _result in _results)
    return (healthy_results,)


@app.cell(hide_code=True)
def healthy_dashboard(healthy_results, mo, plt):
    _order = [
        "C1a finiteness",
        "C1b confinement",
        "C2 latent scale",
        "C3 resolvability",
        "C4b edge overwhelm",
        "C4c saturation",
        "C5a location reach",
        "C5b width",
        "C5c transmission",
    ]
    _rows = "\n".join(
        f"| {_name} | {healthy_results[_name].value} | {healthy_results[_name].band} | 🟢 |"
        for _name in _order
    )
    _fig, _ax = plt.subplots(figsize=(9.5, 3.4))
    _groups = ["C1", "C1", "C2", "C3", "C4", "C4", "C5", "C5", "C5"]
    _x = range(len(_order))
    _ax.scatter(_x, [1] * len(_order), s=420, color="#4a9d5b", edgecolor="white", linewidth=2)
    for _index, _name in enumerate(_order):
        _ax.text(_index, 1, "✓", color="white", ha="center", va="center", fontweight="bold")
        _ax.text(_index, 0.84, _name.split()[0], ha="center", va="top", fontsize=9)
        _ax.text(
            _index, 1.17, _groups[_index], ha="center", va="bottom", fontsize=8, color="#666666"
        )
    _ax.set_ylim(0.68, 1.32)
    _ax.set_xlim(-0.6, len(_order) - 0.4)
    _ax.axis("off")
    _ax.set_title(
        "Healthy reference: every applicable check is green", fontsize=11, fontweight="bold"
    )
    _fig.tight_layout()
    mo.vstack(
        [
            mo.as_html(_fig),
            mo.md("| check | healthy statistic | band | verdict |\n|---|---|---|---|\n" + _rows),
        ]
    )
    return


@app.cell(hide_code=True)
def containment_md(mo):
    mo.md(r"""
    ## 3. Numerical containment: can the latent path exist and settle?

    C1a and C1b inspect the same paths but answer different questions. **Finiteness is a hard
    executability requirement.** Confinement is a soft tail-behavior judgment: finite paths may
    still keep amplifying over the study window.
    """)
    return


@app.cell
def make_c1a_case(check_confinement, stress_paths, times):
    _nonfinite_paths = stress_paths.copy()
    _nonfinite_paths[:4, 38:] = float("nan")
    _results = {r.check: r for r in check_confinement("PerceivedStress", _nonfinite_paths, times)}
    assert not _results["C1a finiteness"].passed
    assert _results["C1b confinement"].passed
    c1a_case = {"paths": _nonfinite_paths, "result": _results["C1a finiteness"]}
    return (c1a_case,)


@app.cell(hide_code=True)
def show_c1a(c1a_case, mo, np, plt, result_panel, stress_paths, style_axes, times):
    _fig, _axes = plt.subplots(1, 2, figsize=(10.8, 3.5))
    for _row in stress_paths[10:24]:
        _axes[0].plot(times, _row, color="#c9c9c9", linewidth=0.7)
    for _row in c1a_case["paths"][:4]:
        _axes[0].plot(times, _row, color="#c0504d", linewidth=1.5)
    _axes[0].axvline(38, color="#c0504d", linestyle="--", linewidth=1)
    _axes[0].text(38.8, 2.3, "solver output becomes NaN", color="#c0504d", fontsize=8)
    _axes[0].set(xlabel="day", ylabel="stress latent", title="Four explosive prior draws")

    _rates = [0.0, 100.0 * np.mean(~np.isfinite(c1a_case["paths"]))]
    _axes[1].bar(["healthy", "positive-feedback issue"], _rates, color=["#4a9d5b", "#c0504d"])
    _axes[1].axhline(0.0, color="#111111", linewidth=1.2, label="required: exactly 0%")
    _axes[1].set(ylabel="non-finite values (%)", title="Any non-zero mass is a hard red")
    _axes[1].legend(frameon=False, fontsize=8)
    style_axes(_axes)
    _fig.tight_layout()
    mo.vstack(
        [
            result_panel(
                "C1a finiteness — the exact solve fails",
                c1a_case["result"],
                "Keep the toy fixed, but let four wide-prior draws enter a strong positive "
                "stress feedback regime. Their represented exact-solver output becomes NaN at "
                "day 38.",
                "A fragment with non-finite prior predictive output cannot be evaluated; all "
                "later summaries of those draws are undefined.",
                "Add a confining self-limit or tighten the feedback/diffusion priors, then rerun "
                "the exact predictive solve.",
                "No. A draw without a finite path is not an evaluable world, so C1a is a hard "
                "requirement.",
            ),
            mo.as_html(_fig),
        ]
    )
    return


@app.cell(hide_code=True)
def c1a_epistemology(mo):
    mo.accordion(
        {
            "Why a non-finite path is a hard failure": mo.md(r"""
            A prior draw is a sentence the model utters about a possible world; a NaN trajectory
            is not an improbable world but *no world at all* — on that draw the fragment fails to
            denote a distribution over trajectories that can be evaluated. Whether the cause is
            numerical (the discretized solve diverged where the true solution exists) or genuine
            finite-time explosion (the SDE asserts the construct leaves every compact set and
            ceases to be defined mid-window), both are semantic breakdowns rather than bold
            claims: a latent whose identity is "a standardized quantity that modulates these
            indicators through these links" has no measurement relation at infinity. That is why
            C1a is hard and non-negotiable — one can argue with a strange belief, but one cannot
            assign a probability to a sentence that does not parse. The claim is not false; it is
            not truth-apt.
            """)
        }
    )
    return


@app.cell
def make_c1b_case(check_confinement, np, stress_paths, times):
    _growing_paths = stress_paths.copy()
    _multipliers = 1.0 + np.linspace(-0.05, 0.05, 12)
    _growing_paths[:12] = 0.04 * _multipliers[:, None] * np.exp(0.11 * times)[None, :]
    _results = {r.check: r for r in check_confinement("PerceivedStress", _growing_paths, times)}
    assert _results["C1a finiteness"].passed
    assert not _results["C1b confinement"].passed
    c1b_case = {"paths": _growing_paths, "result": _results["C1b confinement"]}
    return (c1b_case,)


@app.cell(hide_code=True)
def show_c1b(c1b_case, healthy_results, mo, np, plt, result_panel, style_axes, times):
    _fig, _axes = plt.subplots(1, 2, figsize=(10.8, 3.5))
    for _row in c1b_case["paths"][20:34]:
        _axes[0].plot(times, _row, color="#c9c9c9", linewidth=0.7)
    for _row in c1b_case["paths"][:6]:
        _axes[0].plot(times, _row, color="#c0504d", linewidth=1.1)
    _axes[0].set(xlabel="day", ylabel="stress latent", title="Finite, but still growing late")

    _healthy_growth = healthy_results["C1b confinement"].evidence["growth"]
    _issue_growth = c1b_case["result"].evidence["growth"]
    _threshold = c1b_case["result"].evidence["growth_ratio"]
    _bins = np.linspace(0.0, 12.0, 48)
    _axes[1].hist(
        np.clip(_healthy_growth, 0, 12), bins=_bins, alpha=0.55, color="#4a9d5b", label="healthy"
    )
    _axes[1].hist(
        np.clip(_issue_growth, 0, 12), bins=_bins, alpha=0.55, color="#c0504d", label="issue"
    )
    _axes[1].axvline(
        _threshold, color="#333333", linestyle="--", label=f"late/early = {_threshold:g}"
    )
    _axes[1].set(xlabel="late / early amplitude", ylabel="draws", title="Tail frequency crosses 1%")
    _axes[1].legend(frameon=False, fontsize=8)
    style_axes(_axes)
    _fig.tight_layout()
    mo.vstack(
        [
            result_panel(
                "C1b confinement — rare paths never settle",
                c1b_case["result"],
                "Replace 12 of 512 otherwise healthy paths with finite positive-feedback paths "
                "whose late amplitude keeps growing.",
                "The statistic is self-calibrating within each draw: it asks whether the late "
                "quarter exceeds five times that draw's own early amplitude, then counts how "
                "often this happens. Both constants — the growth ratio and the tolerated "
                "explosive fraction — are design calibration on DesignInfo, not part of the "
                "statistic: they encode the model class's confinement commitment and an "
                "intrinsically trending domain raises them there.",
                "Strengthen the confining well or reduce the tail of the incoming-edge and "
                "diffusion priors.",
                "Only if the excursion is a substantively intended, negligible-frequency prior "
                "tail and the posterior will be explicitly re-checked.",
            ),
            mo.as_html(_fig),
        ]
    )
    return


@app.cell(hide_code=True)
def c1b_epistemology(mo):
    mo.accordion(
        {
            "Why unsettled growth is a judgment call": mo.md(r"""
            Unlike a NaN, a finite growing path *is* a genuine belief: "some of my credence is
            on worlds where this construct grows five-fold past its own early amplitude and
            never settles" is a substantive commitment about instability. The check flags a
            coherence problem between two parts of one's knowledge — the priors as written
            versus the (usually firmer) background knowledge that psychological and
            physiological constructs are homeostatically bounded; confronting the joint prior
            with what is known outside the model is Box's (1980) sense of model criticism. In de
            Finetti's terms, a prior is a betting disposition, and explosive mass means offering
            odds one would never actually take: the prior written down is not the prior held.

            But sometimes the belief *is* held — tipping points, a manic episode, a relapse
            spiral — and the check cannot distinguish an incoherent prior from a deliberately
            non-stationary one; only the proposer can. Letting the finding stand is therefore a
            precise epistemic act: declaring that region of belief intentional while
            acknowledging it will never be data-vetted, because the data lives near the bulk and
            the explosive tail only re-enters in unconditioned forward runs (forecasts,
            counterfactuals). Whatever those tails contribute downstream is testimony from the
            prior, not evidence from the world. A quieter corollary: far outside the range its
            links exercise, the emissions saturate and further latent differences make no
            observable difference — probability mass spent on distinctions the indicators could
            never confirm or refute, the same disease C5c (transmission) diagnoses from the
            measurement end.
            """)
        }
    )
    return


@app.cell(hide_code=True)
def latent_design_md(mo):
    mo.md(r"""
    ## 4. Latent scale and design: is the state plausible and observable?

    The next two screens do not ask whether a posterior can estimate a parameter. C2 asks
    whether latent dynamics and the reference emission share a coherent scale convention. C3
    asks whether the **prior timescale** lies inside what the actual schedule can resolve.
    """)
    return


@app.cell
def make_c2_case(check_scale, stress_paths):
    _wide_paths = 5.0 * stress_paths
    _result = check_scale(
        "PerceivedStress",
        _wide_paths,
        scale_anchor=0.8,
        anchor_src="stress-slider anchor",
        anchor_detail="inverse-logit IQR / 1.349 with unit reference loading",
    )
    assert not _result.passed
    c2_case = {"paths": _wide_paths, "result": _result}
    return (c2_case,)


@app.cell(hide_code=True)
def show_c2(c2_case, healthy_results, mo, np, plt, result_panel, style_axes):
    _fig, _axes = plt.subplots(1, 2, figsize=(10.8, 3.5))
    _healthy = healthy_results["C2 latent scale"].evidence["marginal_scales"]
    _issue = c2_case["result"].evidence["marginal_scales"]
    _bins = np.linspace(0.0, max(4.6, float(np.max(_issue))), 45)
    _axes[0].hist(_healthy, bins=_bins, color="#4a9d5b", alpha=0.65, label="healthy diffusion")
    _axes[0].hist(_issue, bins=_bins, color="#c0504d", alpha=0.65, label="5× latent scale")
    _axes[0].axvspan(
        c2_case["result"].evidence["lo"],
        c2_case["result"].evidence["hi"],
        color="#4a9d5b",
        alpha=0.10,
        label="allowed anchor / 3 … 3 × anchor",
    )
    _axes[0].axvline(0.8, color="#333333", linestyle="--", label="anchor = 0.8")
    _axes[0].set(
        xlabel="marginal robust scale across draws",
        ylabel="time points",
        title="C2 reads the stationary half",
    )
    _axes[0].legend(frameon=False, fontsize=7)

    _axes[1].plot(c2_case["paths"][:20].T, color="#c0504d", alpha=0.18, linewidth=0.8)
    _axes[1].axhspan(-0.8, 0.8, color="#4a9d5b", alpha=0.10, label="± one data anchor")
    _axes[1].set(
        xlabel="day",
        ylabel="stress latent",
        title="Wide diffusion overwhelms the emission convention",
    )
    _axes[1].legend(frameon=False, fontsize=8)
    style_axes(_axes)
    _fig.tight_layout()
    mo.vstack(
        [
            result_panel(
                "C2 latent scale — dynamics and emission disagree",
                c2_case["result"],
                "Multiply only the stress latent's marginal scale by five, representing a "
                "diffusion prior that ignored the slider-derived anchor.",
                "The state wanders on a scale inconsistent with the unit reference loading and "
                "inverse-logit spread. The same red could instead mean that the loading is "
                "mis-scaled: this is a dynamics–emission contract.",
                "For OU-style elicitation, set diffusion near "
                "`sigma = anchor × sqrt(2 / tau)`, or revise the reference loading and anchor "
                "together.",
                "Only if magnitude claims may stay convention-bound and that limitation is "
                "carried explicitly into interpretation.",
            ),
            mo.as_html(_fig),
        ]
    )
    return


@app.cell
def make_c3_case(check_resolvability, n_draws, np, times):
    _rng = np.random.default_rng(303)
    _fast_tau = np.exp(_rng.normal(np.log(0.18), 0.25, size=n_draws))
    _result = check_resolvability("PerceivedStress", _fast_tau, times)
    assert not _result.passed
    c3_case = {"tau": _fast_tau, "result": _result}
    return (c3_case,)


@app.cell(hide_code=True)
def show_c3(c3_case, mo, np, plt, result_panel, style_axes, tau_draws, times):
    _fig, _axes = plt.subplots(1, 2, figsize=(10.8, 3.5))
    _lo = np.median(np.diff(times)) / 3.0
    _hi = np.ptp(times) / 4.0
    _bins = np.geomspace(0.08, 20.0, 55)
    _axes[0].hist(tau_draws, bins=_bins, color="#4a9d5b", alpha=0.65, label="healthy τ prior")
    _axes[0].hist(c3_case["tau"], bins=_bins, color="#c0504d", alpha=0.65, label="fast τ prior")
    _axes[0].axvspan(_lo, _hi, color="#4a9d5b", alpha=0.10, label="regular-design window")
    _axes[0].axvline(_lo, color="#333333", linestyle="--")
    _axes[0].axvline(_hi, color="#333333", linestyle=":")
    _axes[0].set_xscale("log")
    _axes[0].set(
        xlabel="self-relaxation τ (days, log scale)",
        ylabel="draws",
        title="Most fast prior mass lies below gap / 3",
    )
    _axes[0].legend(frameon=False, fontsize=7)

    _tau_values = np.array([0.18, 4.0, 30.0])
    _decay = np.exp(-1.0 / _tau_values)
    _axes[1].bar(
        ["τ = 0.18 d\ntoo fast", "τ = 4 d\nresolved", "τ = 30 d\ntoo slow"],
        _decay,
        color=["#c0504d", "#4a9d5b", "#e08a3c"],
    )
    _axes[1].set_ylim(0, 1.04)
    _axes[1].set(ylabel="fraction persisting across one day", title="What daily sampling can see")
    style_axes(_axes)
    _fig.tight_layout()
    mo.vstack(
        [
            result_panel(
                "C3 resolvability — daily samples miss a sub-day process",
                c3_case["result"],
                "Move only the stress relaxation prior from about four days to 0.18 day while "
                "keeping the 60-day, once-daily schedule.",
                "The process relaxes several times between adjacent observations. C3 uses every "
                "actual gap plus the total span, so irregular schedules need not be collapsed to "
                "one cadence number.",
                "Collect denser measurements or move the prior timescale only if the construct's "
                "semantics support a slower process.",
                "Only if the construct is genuinely fast and the modeler labels its timescale "
                "and trajectory as prior-set, then confirms posterior contraction.",
            ),
            mo.as_html(_fig),
        ]
    )
    return


@app.cell(hide_code=True)
def structural_md(mo):
    mo.md(r"""
    ## 5. Structural edges: does the child remain a state, and is nonlinearity earned?

    Counterfactual prior-predictive paths with an edge on and off isolate how much temporal
    variation the parent supplies. A separate draw-paired occupancy calculation checks whether
    the proposed Hill bend is ever visited. Both are about the prior's structural geometry, not
    post-fit edge detectability.
    """)
    return


@app.cell
def make_c4b_case(check_edge_share, edge_component, stress_without_edge):
    _overwhelming_paths = stress_without_edge + 100.0 * edge_component
    _result = check_edge_share(
        "CaffeineLoad -> PerceivedStress", _overwhelming_paths, stress_without_edge
    )[0]
    assert not _result.passed
    c4b_case = {
        "edge_on": _overwhelming_paths,
        "edge_off": stress_without_edge,
        "result": _result,
    }
    return (c4b_case,)


@app.cell(hide_code=True)
def show_c4b(c4b_case, healthy_results, mo, np, plt, result_panel, style_axes, times):
    _fig, _axes = plt.subplots(1, 2, figsize=(10.8, 3.5))
    _issue_evidence = c4b_case["result"].evidence
    _axes[0].plot(times, _issue_evidence["on"], color="#c0504d", linewidth=1.6, label="edge on")
    _axes[0].plot(
        times,
        _issue_evidence["off"],
        color="#3b6ea5",
        linewidth=1.3,
        linestyle="--",
        label="same draw, edge off",
    )
    _axes[0].set(xlabel="day", ylabel="stress latent", title="A high-displacement paired draw")
    _axes[0].legend(frameon=False, fontsize=8)

    _healthy_e = healthy_results["C4b edge overwhelm"].evidence["e"]
    _issue_e = _issue_evidence["e"]
    _bins = np.linspace(0.0, 1.05, 45)
    _axes[1].hist(
        np.clip(_healthy_e, 0, 1.05), bins=_bins, alpha=0.65, color="#4a9d5b", label="healthy edge"
    )
    _axes[1].hist(
        np.clip(_issue_e, 0, 1.05), bins=_bins, alpha=0.65, color="#c0504d", label="wide edge prior"
    )
    _axes[1].axvline(0.95, color="#333333", linestyle="--", label="median cap = 95%")
    _axes[1].set(
        xlabel="edge-off displacement / child scale",
        ylabel="draws",
        title="The parent supplies almost the whole path",
    )
    _axes[1].legend(frameon=False, fontsize=7)
    style_axes(_axes)
    _fig.tight_layout()
    mo.vstack(
        [
            result_panel(
                "C4b edge overwhelm — stress is slaved to caffeine",
                c4b_case["result"],
                "Widen only the caffeine→stress edge prior. Paired simulations reuse the same "
                "self-dynamics and noise; the edge-off path is therefore a clean contrast.",
                "When the edge displacement matches the child's entire temporal scale, the "
                "child's own stiffness and diffusion contribute little distinguishable "
                "variation.",
                "Scale the edge prior with the child's relaxation rate: roughly "
                "`edge scale ∝ (1 / tau_child) × child_anchor / parent_anchor`.",
                "Only if a parent-driven child is substantively intended and weak information "
                "about the child's own dynamics is an accepted consequence.",
            ),
            mo.as_html(_fig),
        ]
    )
    return


@app.cell
def make_c4c_case(
    caffeine_paths,
    check_saturation,
    hill_n_draws,
    n_draws,
    np,
):
    _rng = np.random.default_rng(404)
    _far_ec50 = np.exp(_rng.normal(np.log(10.0), 0.10, size=n_draws))
    _result = check_saturation(
        "CaffeineLoad -> PerceivedStress",
        _far_ec50,
        hill_n_draws,
        caffeine_paths,
    )
    assert not _result.passed
    c4c_case = {"ec50": _far_ec50, "result": _result}
    return (c4c_case,)


@app.cell(hide_code=True)
def show_c4c(
    c4c_case,
    caffeine_paths,
    ec50_draws,
    mo,
    np,
    plt,
    result_panel,
    style_axes,
):
    _fig, _axes = plt.subplots(1, 2, figsize=(10.8, 3.5))
    _parent_flat = caffeine_paths.ravel()
    _bins = np.linspace(0.0, 12.0, 55)
    _axes[0].hist(
        _parent_flat[:: _parent_flat.size // 12000],
        bins=_bins,
        density=True,
        color="#bdbdbd",
        alpha=0.65,
        label="parent values",
    )
    _axes[0].hist(
        ec50_draws, bins=_bins, density=True, color="#4a9d5b", alpha=0.60, label="healthy EC50"
    )
    _axes[0].hist(
        c4c_case["ec50"],
        bins=_bins,
        density=True,
        color="#c0504d",
        alpha=0.60,
        label="EC50 near 10",
    )
    _axes[0].set(
        xlabel="caffeine latent input / EC50",
        ylabel="density",
        title="The proposed bend misses the parent range",
    )
    _axes[0].legend(frameon=False, fontsize=7)

    _x = np.linspace(0.0, 3.0, 300)
    _healthy_curve = _x**2 / (_x**2 + np.median(ec50_draws) ** 2)
    _dead_curve = _x**2 / (_x**2 + np.median(c4c_case["ec50"]) ** 2)
    _axes[1].plot(_x, _healthy_curve, color="#4a9d5b", linewidth=2.0, label="healthy Hill edge")
    _axes[1].plot(_x, _dead_curve, color="#c0504d", linewidth=2.0, label="dead-low arm")
    _axes[1].axhspan(0.1, 0.9, color="#7d6bb0", alpha=0.08, label="Hill bend occupancy")
    _axes[1].set(
        xlabel="caffeine parent",
        ylabel="Hill occupancy",
        title="Extra Hill parameters never become active",
    )
    _axes[1].legend(frameon=False, fontsize=7)
    style_axes(_axes)
    _fig.tight_layout()
    mo.vstack(
        [
            result_panel(
                "C4c saturation — a Hill edge that never bends",
                c4c_case["result"],
                "Move only the Hill EC50 prior from the caffeine parent's scale (about 1) to "
                "about 10, preserving draw-wise EC50, Hill exponent, and parent-path pairing.",
                "Almost every draw remains on the dead-low arm. The extra EC50 and Hill-power "
                "parameters are prior baggage rather than exercised nonlinearity.",
                "Center EC50 on the parent's latent scale anchor, or author a linear edge when "
                "the bend is not substantively expected inside the realized range.",
                "Only if the edge is treated as effectively linear and the extra Hill "
                "parameters are acknowledged as weakly informed.",
            ),
            mo.as_html(_fig),
        ]
    )
    return


@app.cell(hide_code=True)
def measurement_md(mo):
    mo.md(r"""
    ## 6. Measurement reach: can the indicator live where the data live?

    C5a, C5b, and C5c deliberately separate location, dispersion, and signal transmission.
    The same observed slider stays fixed in all three increments. C5d then asks a simpler but
    easily overlooked question: did the proposed channel contribute any likelihood terms at
    all?
    """)
    return


@app.cell
def make_c5a_case(check_coverage, check_transmission, np, sigmoid, stress_observed, stress_paths):
    _rng = np.random.default_rng(501)
    _wrong_signal = sigmoid(0.60 + 0.85 * stress_paths)
    _wrong_pp = _rng.beta(_wrong_signal * 8.0, (1.0 - _wrong_signal) * 8.0)
    _results = {
        r.check: r
        for r in [
            *check_coverage(
                "StressSlider",
                _wrong_pp,
                stress_observed,
                distribution="beta",
            ),
            check_transmission(
                "StressSlider",
                _wrong_signal,
                _wrong_signal * (1.0 - _wrong_signal) / 9.0,
            ),
        ]
    }
    assert not _results["C5a location reach"].passed
    assert _results["C5b width"].passed
    assert _results["C5c transmission"].passed
    c5a_case = {
        "pp": _wrong_pp,
        "signal": _wrong_signal,
        "result": _results["C5a location reach"],
    }
    return (c5a_case,)


@app.cell(hide_code=True)
def show_c5a(c5a_case, mo, np, plt, result_panel, stress_observed, style_axes, times):
    _fig, _axes = plt.subplots(1, 2, figsize=(10.8, 3.5))
    _replicate_medians = np.median(c5a_case["pp"], axis=1)
    _observed_median = np.median(stress_observed)
    _axes[0].hist(_replicate_medians, bins=42, color="#c0504d", alpha=0.70)
    _axes[0].axvline(_observed_median, color="#3b6ea5", linewidth=2.0, label="observed median")
    _axes[0].set(
        xlabel="replicate median slider fraction",
        ylabel="prior replicates",
        title="Observed location misses the 1–99% envelope",
    )
    _axes[0].legend(frameon=False, fontsize=8)

    for _row in c5a_case["pp"][:45]:
        _axes[1].plot(times, _row, color="#c0504d", alpha=0.10, linewidth=0.7)
    _axes[1].plot(times, stress_observed, color="#3b6ea5", linewidth=1.7, label="observed slider")
    _axes[1].set(xlabel="day", ylabel="slider fraction", title="Right width, wrong center")
    _axes[1].legend(frameon=False, fontsize=8)
    style_axes(_axes)
    _fig.tight_layout()
    mo.vstack(
        [
            result_panel(
                "C5a location reach — the emission intercept misses the data",
                c5a_case["result"],
                "Change only the stress manifest intercept from −0.45 to +0.60 on the logit "
                "scale; keep the latent, loading, and Beta concentration fixed.",
                "The observed median falls outside the distribution of prior-replicate medians. "
                "Continuous support does not rescue a prior that puts negligible practical "
                "mass near the data.",
                "Elicit the intercept from the inverse-link observed median, then rerun the "
                "replicated-data checks.",
                "Only if the prior–data location tension is scientifically intentional and a "
                "prior-sensitive posterior adaptation is accepted.",
            ),
            mo.as_html(_fig),
        ]
    )
    return


@app.cell
def make_c5b_case(check_coverage, check_transmission, np, stress_observed, stress_signal):
    _rng = np.random.default_rng(502)
    _wide_pp = _rng.beta(stress_signal * 2.0, (1.0 - stress_signal) * 2.0)
    _results = {
        r.check: r
        for r in [
            *check_coverage(
                "StressSlider",
                _wide_pp,
                stress_observed,
                distribution="beta",
            ),
            check_transmission(
                "StressSlider",
                stress_signal,
                stress_signal * (1.0 - stress_signal) / 3.0,
            ),
        ]
    }
    assert _results["C5a location reach"].passed
    assert not _results["C5b width"].passed
    assert _results["C5c transmission"].passed
    c5b_case = {"pp": _wide_pp, "result": _results["C5b width"]}
    return (c5b_case,)


@app.cell(hide_code=True)
def show_c5b(
    c5b_case, mo, np, plt, result_panel, robust_scale, stress_observed, stress_pp, style_axes
):
    _fig, _axes = plt.subplots(1, 2, figsize=(10.8, 3.5))
    _healthy_width = robust_scale(stress_pp, axis=1)
    _issue_width = robust_scale(c5b_case["pp"], axis=1)
    _observed_width = float(robust_scale(stress_observed))
    _bins = np.linspace(0.0, 0.60, 48)
    _axes[0].hist(
        _healthy_width, bins=_bins, color="#4a9d5b", alpha=0.65, label="healthy concentration = 8"
    )
    _axes[0].hist(
        _issue_width, bins=_bins, color="#c0504d", alpha=0.65, label="diffuse concentration = 2"
    )
    _axes[0].axvline(_observed_width, color="#3b6ea5", linewidth=2.0, label="observed robust scale")
    _axes[0].set(
        xlabel="replicate robust scale",
        ylabel="prior replicates",
        title="Replicate width is much too large",
    )
    _axes[0].legend(frameon=False, fontsize=7)

    _axes[1].hist(
        stress_observed, bins=18, density=True, color="#3b6ea5", alpha=0.70, label="observed"
    )
    _axes[1].hist(
        c5b_case["pp"].ravel()[::20],
        bins=40,
        density=True,
        color="#c0504d",
        alpha=0.38,
        label="wide prior predictive",
    )
    _axes[1].set(
        xlabel="slider fraction", ylabel="density", title="Location can pass while width fails"
    )
    _axes[1].legend(frameon=False, fontsize=8)
    style_axes(_axes)
    _fig.tight_layout()
    mo.vstack(
        [
            result_panel(
                "C5b width — measurement noise is too diffuse",
                c5b_case["result"],
                "Change only the Beta concentration from 8 to 2. The latent path, logit link, "
                "intercept, and loading remain healthy.",
                "The prior-replicate robust scale misses the observed robust scale even though "
                "the replicate median still reaches the data. Count channels analogously add "
                "variance and zero-fraction comparisons; categorical channels use entropy.",
                "Retune the family-specific dispersion/noise prior. If C4b is also red, first "
                "check whether an overwhelming parent is inflating the emission through the "
                "link.",
                "Only if the width imbalance is intentional and the expected weak "
                "regularization or slow warmup is explicitly accepted.",
            ),
            mo.as_html(_fig),
        ]
    )
    return


@app.cell
def make_c5c_case(check_coverage, check_transmission, np, sigmoid, stress_observed, stress_paths):
    _rng = np.random.default_rng(503)
    _flat_signal = sigmoid(-0.45 + 0.025 * stress_paths)
    _noise_dominated_pp = _rng.beta(_flat_signal * 8.0, (1.0 - _flat_signal) * 8.0)
    _results = {
        r.check: r
        for r in [
            *check_coverage(
                "StressSlider",
                _noise_dominated_pp,
                stress_observed,
                distribution="beta",
            ),
            check_transmission(
                "StressSlider",
                _flat_signal,
                _flat_signal * (1.0 - _flat_signal) / 9.0,
            ),
        ]
    }
    assert _results["C5a location reach"].passed
    assert _results["C5b width"].passed
    assert not _results["C5c transmission"].passed
    c5c_case = {
        "pp": _noise_dominated_pp,
        "signal": _flat_signal,
        "result": _results["C5c transmission"],
    }
    return (c5c_case,)


@app.cell(hide_code=True)
def show_c5c(c5c_case, mo, np, plt, result_panel, stress_observed, style_axes, times):
    _fig, _axes = plt.subplots(1, 2, figsize=(10.8, 3.5))
    _draw_index = 37
    _axes[0].plot(
        times,
        c5c_case["signal"][_draw_index],
        color="#4a9d5b",
        linewidth=2.0,
        label="noise-free mean",
    )
    _axes[0].scatter(
        times,
        c5c_case["pp"][_draw_index],
        color="#c0504d",
        s=13,
        alpha=0.65,
        label="one noisy replicate",
    )
    _axes[0].plot(
        times, stress_observed, color="#3b6ea5", linewidth=1.0, alpha=0.60, label="observed"
    )
    _axes[0].set(
        xlabel="day",
        ylabel="slider fraction",
        title="Noise moves; the structural signal barely does",
    )
    _axes[0].legend(frameon=False, fontsize=7)

    _signal_fraction = c5c_case["result"].evidence["signal_fraction"]
    _axes[1].hist(_signal_fraction, bins=np.linspace(0, 0.12, 45), color="#c0504d", alpha=0.70)
    _minimum = c5c_case["result"].evidence["min_signal_fraction"]
    _axes[1].axvline(_minimum, color="#333333", linestyle="--", label=f"minimum {_minimum:.0%}")
    _axes[1].set(
        xlabel="temporal signal variance / total predictive variance",
        ylabel="draws",
        title="C5b passes, but data barely ground the latent",
    )
    _axes[1].legend(frameon=False, fontsize=8)
    style_axes(_axes)
    _fig.tight_layout()
    mo.vstack(
        [
            result_panel(
                "C5c transmission — a live latent behind a flat emission",
                c5c_case["result"],
                "Change only the stress loading from 0.85 to 0.025. Beta noise, intercept, "
                "latent paths, and observed data remain fixed.",
                "Noise alone can reproduce the observed width, so C5b stays green. C5c exposes "
                "that the construct trajectory scarcely moves the indicator. Parking a logit "
                "mean in a saturated tail creates the same flat-link geometry.",
                "Move the link operating point out of saturation or elicit a loading large "
                "enough to transmit substantively plausible latent movement.",
                "Only if the indicator is knowingly noise-dominated and trajectory claims will "
                "be labeled weakly grounded in data.",
            ),
            mo.as_html(_fig),
        ]
    )
    return


@app.cell
def make_c5d_case(indicator_findings):
    import polars as pl

    from nof1_causal_lab.artifacts.identity import IndicatorId

    _result = indicator_findings(
        IndicatorId("indicator:hrv"), pl.DataFrame(schema={"value": pl.Float64}), {}
    )[0]
    c5d_case = {"result": _result}
    return (c5d_case,)


@app.cell(hide_code=True)
def show_c5d(c5d_case, mo, np, plt, style_axes, times):
    _fig, _axes = plt.subplots(1, 2, figsize=(10.8, 3.3))
    _availability = np.vstack([np.ones(times.size), np.zeros(times.size)])
    _axes[0].imshow(
        _availability, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest"
    )
    _axes[0].set_yticks([0, 1], ["StressSlider", "HRV"])
    _axes[0].set_xticks([0, 14, 29, 44, 59], ["1", "15", "30", "45", "60"])
    _axes[0].set(xlabel="study day", title="Declared channels versus actual rows")
    _axes[0].spines[:].set_visible(False)

    _axes[1].bar(["StressSlider", "HRV"], [times.size, 0], color=["#4a9d5b", "#c0504d"])
    _axes[1].set(
        ylabel="observed values", title="Forward simulation still runs; likelihood terms do not"
    )
    style_axes(_axes[1])
    _fig.tight_layout()
    mo.vstack(
        [
            mo.md(
                "### C5d model/data compatibility — a proposed channel with no rows\n\n"
                "**Increment.** Declare HRV as an indicator of `AutonomicArousal` but supply no "
                f"observations for it.\n\n**Finding.** {c5d_case['result'].evidence}\n\n"
                "Unlike C1–C5c, this check runs when the model and data are submitted together "
                "and needs no generated trajectories."
            ),
            mo.as_html(_fig),
        ]
    )
    return


@app.cell(hide_code=True)
def closing_md(mo):
    mo.md("""
    ## 7. How findings are used

    Each check reports a measurement and its applicable band. Failed checks stay attached to
    the saved model and guide the modeler's next edit. A check whose prerequisites are missing
    reports `not_evaluated`, while an implementation failure fails the action itself. Fitting
    and causal reporting enforce their own execution and evidence requirements.
    """)
    return


if __name__ == "__main__":
    app.run()
