import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def imports_notebook():
    import importlib.metadata

    import marimo as mo
    import networkx as nx

    import notebooks.support.two_slice_identification as lab

    return importlib, lab, mo, nx


@app.cell(hide_code=True)
def introduction(mo):
    mo.md(r"""
    # Causal identification can change when we restore the past

    **A truncated graph can identify an effect that a longer unrolling cannot.**
    The first example changes verdict between two and three slices. Section 6 adds
    a second example that changes verdict between **three and four slices**.
    In both, truncation hides the history of an unmeasured common cause.

    We hold the question fixed throughout:

    $$\boxed{P(Y_1\mid do(X_0=x))}.$$

    “Identified” means uniquely determined by the observed joint distribution and
    the causal graph, for the nonparametric model class. It does not mean that one
    selected parameterization can be fitted. All results below are computed with
    **y0's ID algorithm**. A recorded Grassmoth public API call independently confirms
    the current two-slice result.
    """)
    return


@app.cell(hide_code=True)
def state_space_model(mo):
    mo.md(r"""
    ## 1. Three evolving states, two indicators

    $U$ is a hidden common cause, $X$ is the treatment, and $Y$ is the outcome.
    Take $X$ and $Y$ to have exact indicators, so measurement identification is
    settled. $U$ has no indicator.

    A first-order state-space causal model with this structure is

    $$\begin{aligned}
    U_{s+1} &= f_{U,s}(U_s,\epsilon^U_{s+1}),\\
    X_{s+1} &= f_{X,s}(X_s,U_s,\epsilon^X_{s+1}),\\
    Y_{s+1} &= f_{Y,s}(Y_s,X_s,U_s,\epsilon^Y_{s+1}),\\
    Z^X_s &= X_s,\qquad Z^Y_s=Y_s.
    \end{aligned}$$

    The innovations are independent across states and time. The functions may be
    nonlinear; no linearity, Gaussianity, stationarity, or equality of parameters
    across time is imposed by this graph calculation. Each displayed window begins
    with independent roots at its first slice. Moving that artificial boundary
    backwards reveals the dependence it previously erased.

    The construct DAG below is acyclic. Its arrows become next-slice arrows in the
    identifier, and every evolving state also carries over to its next value.
    """)
    return


@app.cell
def construct_dag(lab):
    lab.construct_figure()
    return


@app.cell
def fixed_graphs(lab):
    two_slice = lab.unroll(0)
    three_slice = lab.unroll(-1)
    short_result = lab.identify(two_slice)
    long_result = lab.identify(three_slice)
    lab.comparison_figure()
    return long_result, short_result, three_slice, two_slice


@app.cell(hide_code=True)
def why_the_verdict_changes(mo):
    mo.md(r"""
    ## 2. The omitted path

    In the two-slice graph, $X_0$ has no parents: everything before time 0 was
    deleted. It therefore behaves as an unconfounded treatment in that graph.
    One valid estimand is

    $$P(Y_1\mid do(X_0=x))
      =\int P(Y_1\mid X_0=x,Y_0=y_0)\,dP(y_0).$$

    With slice $-1$ restored, the graph contains

    $$\color{#dc2626}{X_0\leftarrow U_{-1}\rightarrow U_0\rightarrow Y_1}.$$

    Both internal nodes on this path are hidden non-colliders. Neither observing
    $Y_0$ nor observing earlier $X,Y$ states blocks it. The causal edge
    $X_0\rightarrow Y_1$ is still present. The intervention and its one-step horizon
    have not changed.

    The internal latent projection now has a **hedge** with node sets
    $F=\{X_0,Y_1\}$ and $F'=\{Y_1\}$, rooted at $Y_1$.
    ID returns failure. The hedge establishes non-identification even with the full
    observed joint distribution in the longer window; it is stronger evidence than
    a failed search for an adjustment set. It persists when still earlier slices
    are added. See [Shpitser & Pearl (2006), Definitions 6–7 and Theorem 4](https://ftp.cs.ucla.edu/pub/stat_ser/r327.pdf).
    """)
    return


@app.cell
def computed_results(lab, long_result, mo, short_result):
    corrected_boundary = lab.identify(lab.history_marginalized())
    no_persistence = lab.identify(lab.unroll(-1, persistent_hidden=False))
    measured_u = lab.identify(lab.unroll(-1, observe_hidden=True))
    scenarios = (
        ("Two slices; preceding history deleted", short_result),
        ("Three slices; U hidden and persistent", long_result),
        ("Only two slices observed; earlier history marginalized", corrected_boundary),
        ("Three slices; U has no carryover", no_persistence),
        ("Three slices; U is observed", measured_u),
    )
    mo.ui.table(
        [
            {
                "Graph / assumption": label,
                "ID verdict": "Identified" if result.identified else "Not identified",
                "Two-node hedge": "Present" if result.has_hedge else "Absent",
            }
            for label, result in scenarios
        ],
        selection=None,
        pagination=False,
    )
    return corrected_boundary, measured_u, no_persistence


@app.cell(hide_code=True)
def boundary_explanation(mo):
    mo.md(r"""
    **Deleting history changes the model.** In the original process, $X_0$, $Y_0$
    and $U_0$ have dependent initial values because of their shared past.
    Correctly marginalizing that past preserves those dependencies, even if only
    times 0 and 1 are observed. The third row above verifies this distinction.

    A two-slice *transition template* can describe a first-order SSM. Its sufficiency
    as a transition template does not make a stand-alone two-slice DAG with
    independent boundary roots sufficient for causal identification.
    """)
    return


@app.cell(hide_code=True)
def explorer_intro(mo):
    mo.md(r"""
    ## 3. Move the boundary, keep the question

    Add earlier slices while keeping $X_0$ and $Y_1$ fixed. Then remove $U$'s
    persistence, or give $U$ an indicator. These controls change the graph assumptions
    and recompute ID; they do not change the query.
    """)
    return


@app.cell
def explorer_controls(mo):
    history = mo.ui.slider(0, 4, value=1, step=1, label="Earlier slices before time 0")
    persistence = mo.ui.checkbox(value=True, label="U persists across slices")
    observe_u = mo.ui.checkbox(value=False, label="U has an exact indicator")
    mo.hstack([history, persistence, observe_u], justify="start", gap=2)
    return history, observe_u, persistence


@app.cell
def explorer_result(history, lab, mo, observe_u, persistence):
    explorer_graph = lab.unroll(
        -history.value,
        persistent_hidden=persistence.value,
        observe_hidden=observe_u.value,
    )
    explorer_verdict = lab.identify(explorer_graph)
    _label = "Identified in this graph" if explorer_verdict.identified else "Not identified"
    mo.vstack(
        [
            mo.callout(
                mo.md(f"**{_label}.** Window: {2 + history.value} slices; query: $X_0$ to $Y_1$."),
                kind="success" if explorer_verdict.identified else "warn",
            ),
            lab.explorer_figure(explorer_graph),
        ]
    )
    return


@app.cell
def verification(lab, long_result, measured_u, mo, no_persistence, nx, short_result):
    assert short_result.identified
    assert not short_result.has_hedge
    assert not long_result.identified
    assert long_result.has_hedge
    assert no_persistence.identified
    assert measured_u.identified
    assert not lab.identify(lab.history_marginalized()).identified
    for _first in range(-4, 1):
        _graph = lab.unroll(_first)
        assert nx.is_directed_acyclic_graph(_graph)
        _result = lab.identify(_graph)
        assert _result.identified == (_first == 0)
        assert _result.has_hedge == (_first < 0)
        assert lab.identify(lab.unroll(_first, observe_hidden=True)).identified
        assert lab.identify(lab.unroll(_first, persistent_hidden=False)).identified
    mo.callout(
        "Assertions passed: 2–6 slices, the fixed query, DAG validity, the hedge, "
        "both identifying controls, and correct marginalization of the boundary.",
        kind="success",
    )
    return


@app.cell(hide_code=True)
def production_intro(mo):
    mo.md(r"""
    ## 4. What our current framework actually returned

    The evidence below is from public HTTP `edit_question` and `edit_model` actions,
    using the construct DAG above and exact indicators on $X,Y$. It is a graph-only
    model revision: numerical dynamics were left unspecified because this check
    needs no fit or simulation.

    Our identifier uses $X_{t-1}$ and $Y_t$; these are $X_0$ and $Y_1$ here.
    The notebook independently constructs the longer DAG and uses the same y0 ID
    algorithm. It imports no Grassmoth internals and changes no production behavior.
    """)
    return


@app.cell
def production_evidence(importlib, lab, mo, short_result):
    api_evidence = lab.read_api_evidence()
    _response = api_evidence["model_response"]
    _finding = _response["body"]["identification"]["treatments"]["construct:treatment-x"]
    assert _response["status"] == "success"
    assert _finding["status"] == "identified"
    assert short_result.identified
    _version = importlib.metadata.version("y0")
    mo.vstack(
        [
            mo.md(f"""
            **Recorded production verdict: `{_finding["status"]}`.**

            - Recorded: `{api_evidence["recorded_at"]}`
            - Study: `{api_evidence["workspace_id"]}`
            - Model revision: `{_response["commit_id"]}`
            - Repository revision: `{api_evidence["repository_revision"]}`
            - y0: recorded `{api_evidence["y0_version"]}`; executing `{_version}`
            """),
            mo.accordion(
                {
                    "Exact production estimand": mo.md(f"```text\n{_finding['estimand']}\n```"),
                    "Independent two-slice estimand": mo.md(
                        f"```text\n{short_result.estimand}\n```"
                    ),
                    "Saved requests, response, and source fingerprint": mo.json(api_evidence),
                    "Replay the public calls": mo.md(r"""
                        From the repository root, with the study server running:

                        ```bash
                        EVIDENCE=apps/grassmoth-server/notebooks/data/two_slice_identification.json
                        STUDY=two-slice-id-replay
                        jq '.question_request' "$EVIDENCE" | curl -sS \
                          "http://localhost:8100/api/studies/$STUDY/edit_question" \
                          -H 'Content-Type: application/json' -H 'Accept: application/json' \
                          --data-binary @-
                        ```

                        Poll the returned `call_id` at the same action URL with
                        `/<call_id>` appended until `status` is `success`. Then:

                        ```bash
                        jq '.model_request' "$EVIDENCE" | curl -sS \
                          "http://localhost:8100/api/studies/$STUDY/edit_model" \
                          -H 'Content-Type: application/json' -H 'Accept: application/json' \
                          --data-binary @-
                        ```

                        Poll this call the same way. Its identification report is at
                        `body.identification`. The saved response is evidence from its
                        recorded revision; these commands check the running server.
                    """),
                }
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def implications_and_sources(mo):
    mo.md(r"""
    ## 5. Scope of the counterexample

    The result concerns **nonparametric identification in the temporal graph used
    by our causal checker**. It does not establish non-identifiability of every
    restricted continuous-time SDE, or of a particular parameterization with known
    coefficients. Our continuous-time dynamics and their numerical transition law
    are separate from this two-slice graph convention.

    The independent-root boundary is an actual assumption. If time 0 really is a
    fresh, independently initialized experiment, the two-slice answer can be right.
    For an ongoing process with a persistent hidden cause, simply deleting the past
    removes dependence that the original process retains.

    Temporal identification must account for relevant history and its induced
    dependencies. [Jahn, Karnik & Schulman (2025), Section 3](https://arxiv.org/html/2504.20172v1#S3)
    studies finite-window bounds under specific assumptions; it does not justify a
    universal two-slice check. A persistent hidden state can induce arbitrarily
    long latent connections in the observed projection, so finite-lag hypotheses
    must be checked at the level to which a theorem applies.

    Repository owners: `models/causal/identification.py` defines the two-slice
    unrolling; `docs/assumptions.md`, “Scope of the two-slice check,” states its limit.
    The response and requests live beside this notebook in
    `data/two_slice_identification.json`.
    """)
    return


@app.cell(hide_code=True)
def three_to_four_model(mo):
    mo.md(r"""
    ## 6. Three slices still miss a hidden cause

    Insert a second unmeasured state $V$ between $U$ and $X$. The construct DAG is

    $$U\rightarrow V\rightarrow X\rightarrow Y,\qquad U\rightarrow Y.$$

    $U,V$ have no indicators; $X,Y$ are observed. Every state has carryover. Every
    cross-construct arrow still joins **adjacent slices**, so this example does not
    introduce a longer direct lag. The transition structure is

    $$\begin{aligned}
    U_{s+1}&=f_{U,s}(U_s,\epsilon^U_{s+1}),\\
    V_{s+1}&=f_{V,s}(V_s,U_s,\epsilon^V_{s+1}),\\
    X_{s+1}&=f_{X,s}(X_s,V_s,\epsilon^X_{s+1}),\\
    Y_{s+1}&=f_{Y,s}(Y_s,X_s,U_s,\epsilon^Y_{s+1}).
    \end{aligned}$$

    Keep the query $P(Y_1\mid do(X_0=x))$ fixed. Three slices mean $-1,0,1$;
    four slices mean $-2,-1,0,1$. The added slice is in the **past**, not an extension
    of the outcome horizon.

    These are direct y0 calculations for the proposed three-slice check. The
    production API evidence in Section 4 remains a two-slice result for the first
    example.
    """)
    return


@app.cell
def three_to_four_comparison(lab):
    delayed_three = lab.unroll_delayed(-1)
    delayed_four = lab.unroll_delayed(-2)
    delayed_short_result = lab.identify(delayed_three)
    delayed_long_result = lab.identify(delayed_four)
    lab.delayed_comparison_figure()
    return delayed_four, delayed_long_result, delayed_short_result, delayed_three


@app.cell(hide_code=True)
def three_to_four_proof(mo):
    mo.md(r"""
    **On three slices**, $V_{-1}$ and $U_{-1}$ are independent boundary roots.
    The parent $V_{-1}$ of $X_0$ has no hidden route to $Y_1$ that avoids $X_0$.
    Adjusting for $X_{-1}$ blocks the remaining backdoor paths, giving

    $$P(Y_1\mid do(X_0=x))
       =\int P(Y_1\mid X_0=x,X_{-1}=a)\,dP(a).$$

    **On four slices**, the missing root $U_{-2}$ appears and creates the path

    $$\color{#dc2626}{
      X_0\leftarrow V_{-1}\leftarrow U_{-2}
      \rightarrow U_{-1}\rightarrow U_0\rightarrow Y_1}.$$

    Every internal node is hidden. The directed edge $X_0\rightarrow Y_1$ plus
    this hidden connection produces the same two-node hedge as before,
    $F=\{X_0,Y_1\}$, $F'=\{Y_1\}$. Thus the effect is not nonparametrically
    identified, even with all observed history in the four-slice graph.
    [The hedge criterion](https://ftp.cs.ucla.edu/pub/stat_ser/r327.pdf) establishes
    the failure; the assertions below also check ID's verdict.

    The fourth slice invalidates the assumption that the boundary values
    $V_{-1}$ and $U_{-1}$ are independent. Adding a single hidden mediator has
    pushed the first visible identification failure back by one more slice.
    """)
    return


@app.cell
def three_to_four_results(delayed_short_result, lab, mo):
    _rows = []
    for _first in (-1, -2, -3, -4):
        _result = lab.identify(lab.unroll_delayed(_first))
        _rows.append(
            {
                "Slices": 2 - _first,
                "Window": f"{_first}, …, 1",
                "Query": "P(Y₁ | do(X₀=x))",
                "ID verdict": "Identified" if _result.identified else "Not identified",
                "Hedge": "Present" if _result.has_hedge else "Absent",
            }
        )
    mo.vstack(
        [
            mo.ui.table(_rows, selection=None, pagination=False),
            mo.accordion(
                {
                    "Exact y0 estimand for the three-slice graph": mo.md(
                        f"```text\n{delayed_short_result.estimand}\n```"
                    )
                }
            ),
        ]
    )
    return


@app.cell
def delayed_controls(mo):
    delayed_history = mo.ui.slider(1, 4, value=2, step=1, label="Earlier slices before time 0")
    delayed_persistence = mo.ui.checkbox(value=True, label="U persists across slices")
    observe_v = mo.ui.checkbox(value=False, label="V has an exact indicator")
    mo.hstack([delayed_history, delayed_persistence, observe_v], justify="start", gap=2)
    return delayed_history, delayed_persistence, observe_v


@app.cell
def delayed_explorer(delayed_history, delayed_persistence, lab, mo, observe_v):
    _graph = lab.unroll_delayed(
        -delayed_history.value,
        persistent_hidden=delayed_persistence.value,
        observe_relay=observe_v.value,
    )
    _result = lab.identify(_graph)
    _label = "Identified in this graph" if _result.identified else "Not identified"
    mo.vstack(
        [
            mo.callout(
                f"{_label}. Window: {2 + delayed_history.value} slices; query: X₀ to Y₁.",
                kind="success" if _result.identified else "warn",
            ),
            lab.delayed_explorer_figure(_graph),
        ]
    )
    return


@app.cell
def delayed_verification(
    delayed_four, delayed_long_result, delayed_short_result, delayed_three, lab, mo, nx
):
    assert delayed_short_result.identified
    assert not delayed_short_result.has_hedge
    assert not delayed_long_result.identified
    assert delayed_long_result.has_hedge
    for _graph, _expected_adjustment in ((delayed_three, True), (delayed_four, False)):
        _backdoor = _graph.copy()
        _backdoor.remove_edges_from(list(_backdoor.out_edges("X_0")))
        assert nx.is_d_separator(_backdoor, {"X_0"}, {"Y_1"}, {"X_-1"}) == _expected_adjustment
    for _first in range(-4, 0):
        _graph = lab.unroll_delayed(_first)
        assert nx.is_directed_acyclic_graph(_graph)
        assert all(
            _graph.nodes[effect]["time"] - _graph.nodes[cause]["time"] == 1
            for cause, effect in _graph.edges
        )
        _result = lab.identify(_graph)
        assert _result.identified == (_first == -1)
        assert _result.has_hedge == (_first <= -2)
        assert lab.identify(lab.unroll_delayed(_first, observe_relay=True)).identified
        assert lab.identify(lab.unroll_delayed(_first, persistent_hidden=False)).identified
    mo.callout(
        "Verified: three slices identify the effect; four through six do not. "
        "Every edge has lag one. The three-slice adjustment set is valid only in "
        "that truncated graph. Observing V or removing U's persistence restores identification.",
        kind="success",
    )
    return


if __name__ == "__main__":
    app.run()
