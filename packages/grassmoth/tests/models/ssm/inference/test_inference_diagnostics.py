"""Scientific identities, retained draws, and sampler evidence in diagnostic reports."""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest
from pydantic import ValidationError

from grassmoth.artifacts.identity import ParameterElementId, ParameterId, ParameterRef
from grassmoth.artifacts.model.parameter import ParameterCoordinate
from grassmoth.artifacts.reports.posterior_diagnostics import PosteriorMarginal
from grassmoth.models.ssm.inference.diagnostics.plots import (
    build_trace_data,
    compute_posterior_marginals,
)
from grassmoth.models.ssm.inference.types import ProductionDiagnostics


def references(coordinates):
    return {
        ParameterCoordinate(site_name=name, indices=indices): (
            ParameterCoordinate(site_name=name, indices=indices).label,
            ParameterRef(
                parameter_id=ParameterId("parameter:" + str(index).zfill(64)),
                element_id=ParameterElementId("element:" + str(index).zfill(64)),
            ),
        )
        for index, (name, indices) in enumerate(coordinates)
    }


@pytest.mark.contract
def test_traces_preserve_all_draws_chains_and_scientific_subjects():
    scalar = np.arange(16).reshape(2, 8)
    matrix = np.arange(64).reshape(2, 8, 2, 2)
    expected = [("scalar", (), scalar)] + [
        ("matrix", ij, matrix[:, :, *ij]) for ij in np.ndindex(2, 2)
    ]
    refs = references([(name, indices) for name, indices, _ in expected])
    traces = build_trace_data({"scalar": jnp.asarray(scalar), "matrix": jnp.asarray(matrix)}, refs)
    assert len(traces) == len(expected)
    for trace, (name, indices, values) in zip(traces, expected, strict=True):
        _, subject = refs[ParameterCoordinate(site_name=name, indices=indices)]
        assert trace.subject == subject
        assert trace.chains == tuple(tuple(row) for row in values.tolist())


@pytest.mark.contract
@pytest.mark.parametrize(
    "invalid",
    [{"lower": 3.0}, {"interval_mass": 0.0}, {"interval_mass": 1.0}, {"mean": float("nan")}],
)
def test_posterior_estimate_rejects_invalid_intervals(invalid):
    payload = {
        "parameter": "test",
        "subject": {"parameter_id": "parameter:" + "0" * 64, "element_id": "element:" + "0" * 64},
        "sd": 0.5,
        "density_curve": {"x": [], "density": []},
        "mean": 1.0,
        "lower": 0.0,
        "upper": 2.0,
        "interval_kind": "hdi",
        "interval_mass": 0.94,
    }
    with pytest.raises(ValidationError):
        PosteriorMarginal.model_validate({**payload, **invalid})


@pytest.fixture
def particle_posterior():
    """Fixed input draws for report integration; no inference or shared mutable results."""
    from grassmoth.models.ssm.inference.mcmc_state import TrajectoryMCMCResult
    from grassmoth.models.ssm.inference.types import (
        JointPosteriorDraws,
        MCMCPosterior,
    )

    rng = np.random.default_rng(1)
    samples = {
        "alpha": jnp.asarray(1.0 + 0.15 * rng.normal(size=(2, 64))),
        "beta": jnp.asarray([2.5, -0.2]) + jnp.asarray(0.08 * rng.normal(size=(2, 64, 2))),
        "sigma": jnp.asarray(0.5 + 0.04 * rng.normal(size=(2, 64))),
    }
    parameter_moves = np.zeros((2, 64))
    parameter_moves[0, 4] = 1.0
    mcmc = TrajectoryMCMCResult(
        chain_samples=samples,
        chain_extra_fields={
            "parameter_accept_prob": jnp.asarray(parameter_moves),
            "latent_accept_prob": jnp.full((2, 64), 0.84),
        },
        num_chains=2,
        num_samples=64,
    )
    return MCMCPosterior.from_run(
        draws=JointPosteriorDraws(parameters=mcmc.get_samples()),
        diagnostics=ProductionDiagnostics(
            mcmc=mcmc, observation_log_probs=jnp.zeros((mcmc.num_chains, mcmc.num_samples, 0))
        ),
    )


@pytest.mark.inference(concern="sampling")
def test_mcmc_report_preserves_coordinate_metrics_chains_and_sampler_statistics(particle_posterior):
    coordinates = [("alpha", ()), ("beta", (0,)), ("beta", (1,)), ("sigma", ())]
    refs = references(coordinates)
    report = particle_posterior.get_mcmc_diagnostics(refs)
    traces, ranks = particle_posterior.get_chain_detail(refs)
    assert report.num_chains == 2
    assert report.num_samples_per_chain == 64
    assert report.parameter_accept_prob_mean == pytest.approx(1 / 128)
    assert report.latent_accept_prob_mean == pytest.approx(0.84)
    assert len(report.per_parameter) == len(traces) == len(ranks) == 4
    for metric, trace, hist, (name, indices) in zip(
        report.per_parameter, traces, ranks, coordinates, strict=True
    ):
        coordinate = ParameterCoordinate(site_name=name, indices=indices)
        assert metric.parameter == coordinate.label
        for entry in (metric, trace, hist):
            assert entry.subject == refs[coordinate][1]
        for key in ("r_hat", "ess_bulk", "ess_tail", "mcse_mean"):
            assert np.isfinite(getattr(metric, key))
            assert getattr(metric, key) > 0
        expected = np.asarray(
            particle_posterior.get_samples()[name][(slice(None), *indices)]
        ).reshape(2, 64)
        assert trace.chains == tuple(tuple(row) for row in expected.tolist())
        n_bins = len(hist.chains[0])
        assert hist.expected_per_bin == 64 / n_bins
        assert len(hist.chains) == 2
        assert all(len(row) == n_bins and min(row) >= 0 and sum(row) == 64 for row in hist.chains)


@pytest.mark.inference(concern="sampling")
def test_saved_host_telemetry_is_shared_and_gives_the_same_diagnostics(particle_posterior):
    from grassmoth.artifacts.arrays import NumericalArray

    native = particle_posterior.diagnostics.mcmc
    retained = {
        name: NumericalArray.from_numpy(np.asarray(values))
        for name, values in native.chain_extra_fields.items()
    }
    mcmc = replace(
        native, chain_extra_fields={name: value.values for name, value in retained.items()}
    )
    for name, values in mcmc.chain_extra_fields.items():
        assert values is retained[name].values
        assert not values.flags.writeable
    host = replace(
        particle_posterior, diagnostics=replace(particle_posterior.diagnostics, mcmc=mcmc)
    )
    refs = references([("alpha", ()), ("beta", (0,)), ("beta", (1,)), ("sigma", ())])
    assert host.get_mcmc_diagnostics(refs) == particle_posterior.get_mcmc_diagnostics(refs)


@pytest.mark.inference(concern="sampling")
def test_posterior_marginals_and_traces_preserve_joint_draws_and_extra_fields(particle_posterior):
    coords = [("alpha", ()), ("beta", (0,)), ("beta", (1,)), ("sigma", ())]
    refs = references(coords)
    marginals = compute_posterior_marginals(particle_posterior.draws.parameters, refs, n_bins=8)
    assert len(marginals) == 4
    by_subject = {subject.element_id: coordinate for coordinate, (_, subject) in refs.items()}
    for marginal in marginals:
        coordinate = by_subject[marginal.subject.element_id]
        values = particle_posterior.get_samples()[coordinate.site_name][
            (slice(None), *coordinate.indices)
        ]
        assert marginal.mean == pytest.approx(float(jnp.mean(values)))
        assert marginal.lower < marginal.mean < marginal.upper
        assert marginal.interval_kind == "hdi"
        assert marginal.interval_mass == 0.94
        assert len(marginal.density_curve.x) == len(marginal.density_curve.density) == 8
        assert min(marginal.density_curve.density) >= 0
    expected_moves = [0.0] * 128
    expected_moves[4] = 1.0
    traces, _ = particle_posterior.get_chain_detail(refs)
    columns = {trace.subject: np.asarray(trace.chains).reshape(-1) for trace in traces}
    extra = particle_posterior.diagnostics.mcmc.get_extra_fields()
    assert tuple(float(value) for value in extra["parameter_accept_prob"].reshape(-1)) == tuple(
        expected_moves
    )
    for coordinate, (_, subject) in refs.items():
        values = particle_posterior.get_samples()[coordinate.site_name][
            (slice(None), *coordinate.indices)
        ]
        np.testing.assert_array_equal(columns[subject], values)


@pytest.mark.inference(concern="sampling")
@pytest.mark.parametrize("constant", [False, True])
def test_host_plot_summaries_use_full_draws_and_average_ties(constant):
    from arviz_stats.base.array import array_stats
    from scipy.stats import rankdata

    from grassmoth.models.ssm.inference.diagnostics.plots import build_rank_histograms

    values = np.round(np.random.default_rng(719).normal(size=(4, 101)), 1).astype(np.float32)
    if constant:
        values[:] = 2.0
    refs = references([("x", ())])
    (ranks,) = build_rank_histograms({"x": jnp.asarray(values)}, refs)
    native_ranks = rankdata(values.reshape(-1), method="average").reshape(values.shape)
    counts, _ = array_stats.histogram(
        native_ranks, bins=20, range=(1, values.size + 1), axis=1, density=False
    )
    np.testing.assert_array_equal(ranks.chains, counts)
    (marginal,) = compute_posterior_marginals({"x": jnp.asarray(values.reshape(-1))}, refs)
    np.testing.assert_allclose(
        (marginal.lower, marginal.upper), array_stats.hdi(values.reshape(-1), prob=0.94, axis=-1)
    )
    assert marginal.mean == pytest.approx(float(values.mean()), rel=1e-6, abs=1e-7)
    assert marginal.sd == pytest.approx(float(values.std()), rel=1e-6, abs=1e-7)
    unique, counts = np.unique(values, return_counts=True)
    np.testing.assert_array_equal(marginal.empirical.values.values, unique)
    np.testing.assert_allclose(
        marginal.empirical.probabilities.values, counts.cumsum() / counts.sum(), rtol=1e-6
    )


@pytest.mark.inference(concern="sampling")
def test_latent_convergence_detects_stuck_paths_despite_well_mixed_parameters(
    particle_posterior, monkeypatch
):
    from dataclasses import replace
    from unittest.mock import Mock

    from grassmoth.artifacts.identity import ConstructId, ConstructRef
    from grassmoth.models.ssm.inference.diagnostics import latent
    from grassmoth.models.ssm.inference.diagnostics.convergence import chain_convergence
    from grassmoth.models.ssm.inference.diagnostics.latent import LatentChainEvidence

    states = tuple(
        (ConstructRef(id=ConstructId("construct:" + name)), name)
        for name in ("moving", "stuck", "input")
    )
    rng = np.random.default_rng(23)
    paths = rng.normal(size=(2, 64, 3, 3))
    paths[:, :, :, 1] = np.array([0.0, 1.0])[:, None, None]
    paths[:, :, :, 2] = 10.0
    free = np.ones((3, 3), dtype=bool)
    free[:, 2] = False
    posterior = replace(
        particle_posterior,
        diagnostics=replace(
            particle_posterior.diagnostics,
            latent_chains=LatentChainEvidence.from_draws(paths, free, states),
        ),
    )
    monkeypatch.setattr(
        latent, "chain_statistics", Mock(side_effect=AssertionError("Repeated latent reduction"))
    )
    refs = references([("alpha", ()), ("beta", (0,)), ("beta", (1,)), ("sigma", ())])
    report = posterior.get_mcmc_diagnostics(refs)
    assert len(report.per_latent) == 2
    assert report.per_latent[0].ess_bulk > 0
    assert report.per_latent[1].ess_bulk == report.per_latent[1].ess_tail == 0
    assert chain_convergence(report).status == "failed"
    no_parameters = replace(
        posterior,
        diagnostics=replace(
            posterior.diagnostics,
            mcmc=replace(posterior.diagnostics.mcmc, chain_samples={}),
        ),
    )
    assert no_parameters.get_mcmc_diagnostics({}).per_parameter == ()


@pytest.mark.inference(concern="sampling")
def test_production_gate_detects_stationary_sd_chain_shift():
    from grassmoth.models.ssm.inference.diagnostics.convergence import chain_convergence
    from grassmoth.models.ssm.inference.mcmc_state import TrajectoryMCMCResult
    from grassmoth.models.ssm.inference.types import JointPosteriorDraws, MCMCPosterior

    rng = np.random.default_rng(712)
    chains = np.empty((4, 8192))
    chains[:, 0] = rng.normal(size=4)
    # Unit stationary SD, not unit innovation SD.
    for index in range(1, chains.shape[1]):
        chains[:, index] = 0.8 * chains[:, index - 1] + np.sqrt(1 - 0.8**2) * rng.normal(size=4)

    def gate(values):
        result = TrajectoryMCMCResult(
            chain_samples={"x": jnp.asarray(values)},
            chain_extra_fields={},
            num_chains=4,
            num_samples=values.shape[1],
        )
        posterior = MCMCPosterior.from_run(
            draws=JointPosteriorDraws(parameters=result.get_samples()),
            diagnostics=ProductionDiagnostics(
                mcmc=result, observation_log_probs=jnp.empty((4, values.shape[1], 0))
            ),
        )
        diagnostics = posterior.get_mcmc_diagnostics(references([("x", ())])).revised(per_latent=())
        return chain_convergence(diagnostics)

    assert gate(chains).status == "passed"
    shifted = chains.copy()
    shifted[0] += 0.6
    assert gate(shifted).status == "failed"


@pytest.mark.contract
def test_joint_latent_paths_share_owned_atoms_with_parameter_stride():
    from grassmoth.artifacts.arrays import NumericalArray
    from grassmoth.artifacts.identity import ConstructId, ConstructRef
    from grassmoth.artifacts.probability.joint_layout import JointLawLayout
    from grassmoth.models.ssm.inference.diagnostics.latent import (
        LatentChainEvidence,
        joint_latent_paths,
    )

    element = ParameterElementId("element:" + "a" * 64)
    layout = JointLawLayout(
        parameters=((ParameterId("parameter:" + "a" * 64), (element,)),),
        constructs=(ConstructId("construct:a"), ConstructId("construct:b")),
        time_points=(0.0, 1.0, 2.0),
        labels={element: "scale"},
    )
    owner = NumericalArray.from_numpy(np.arange(14 * 7, dtype=np.float32).reshape(14, 7))
    paths = joint_latent_paths(layout, owner.values, 2)
    evidence = LatentChainEvidence.from_draws(
        paths,
        np.ones((3, 2), dtype=bool),
        tuple((ConstructRef(id=identity), str(identity)) for identity in layout.constructs),
    )
    assert evidence.paths.shape == (2, 7, 3, 2)
    assert np.shares_memory(evidence.paths, owner.values)
    assert isinstance(evidence.paths, np.ndarray)
    assert not evidence.paths.flags.writeable
    for dimension, span in enumerate(layout.trajectory_slices.values()):
        np.testing.assert_array_equal(
            evidence.paths[..., dimension], owner.values[:, span].reshape(2, 7, 3)
        )


@pytest.mark.inference(concern="sampling")
@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("device", [False, True])
def test_latent_diagnostics_bound_selection_and_detect_late_frozen_coordinate(
    monkeypatch, frozen, device
):
    from grassmoth.artifacts.identity import ConstructId, ConstructRef
    from grassmoth.models.ssm.inference.diagnostics import latent
    from grassmoth.models.ssm.inference.diagnostics.chain_stats import chain_statistics

    rng = np.random.default_rng(137)
    paths = rng.normal(size=(2, 64, 263, 2)).astype(np.float32)
    paths[:, :, 1, 0] = np.nan
    if frozen:
        paths[0, :, -1, 0] = 1.0
    free = np.zeros((263, 2), dtype=bool)
    free[:, 0] = True
    free[1, 0] = False
    selected_shapes = []

    def record_selection(samples):
        assert isinstance(samples, np.ndarray)
        selected_shapes.append(samples.shape)
        return chain_statistics(samples)

    monkeypatch.setattr(latent, "chain_statistics", record_selection)
    source = jnp.asarray(paths) if device else paths
    if device:
        getitem = type(source).__getitem__

        def bounded_getitem(value, index):
            if value is source:
                raise AssertionError("Retained paths must transfer before coordinate selection")
            return getitem(value, index)

        monkeypatch.setattr(type(source), "__getitem__", bounded_getitem)
    evidence = latent.LatentChainEvidence.from_draws(
        source,
        free,
        tuple(
            (ConstructRef(id=ConstructId("construct:" + name)), name)
            for name in ("state", "observed")
        ),
    )
    (report,) = evidence.diagnostics
    assert isinstance(evidence.paths, np.ndarray)
    assert not evidence.paths.flags.writeable
    np.testing.assert_array_equal(evidence.paths, paths)
    assert report.coordinates == 262
    assert selected_shapes
    assert max(shape[-1] for shape in selected_shapes) <= 256
    if frozen:
        assert report.ess_bulk == report.ess_tail == 0
        assert report.r_hat is report.mcse_mean is None
    else:
        expected = chain_statistics(paths[:, :, free[:, 0], 0])
        assert report.r_hat is not None
        assert report.ess_bulk is not None
        assert report.ess_tail is not None
        assert report.mcse_mean is not None
        np.testing.assert_allclose(
            (report.r_hat, report.ess_bulk, report.ess_tail, report.mcse_mean),
            (
                expected[:, 0].max(),
                expected[:, 1].min(),
                expected[:, 2].min(),
                expected[:, 3].max(),
            ),
            rtol=5e-5,
        )


@pytest.mark.inference(concern="sampling")
def test_retained_reports_do_not_upload_histories_to_jax(monkeypatch, particle_posterior):
    """Report consumers may copy device inputs to host, but cannot create device work."""
    import jax

    from grassmoth.models.ssm.inference.diagnostics.plots import build_rank_histograms

    def no_device_work(*_args, **_kwargs):
        raise AssertionError("Reporting must not place retained draws on an accelerator")

    monkeypatch.setattr(jnp, "asarray", no_device_work)
    monkeypatch.setattr(jax, "device_put", no_device_work)
    monkeypatch.setattr(jax, "jit", no_device_work)
    refs = references([("alpha", ()), ("beta", (0,)), ("beta", (1,)), ("sigma", ())])
    report = particle_posterior.get_mcmc_diagnostics(refs)
    ranks = build_rank_histograms(particle_posterior.diagnostics.mcmc.chain_samples, refs)
    marginals = compute_posterior_marginals(particle_posterior.draws.parameters, refs)
    assert len(report.per_parameter) == len(ranks) == len(marginals) == 4
    assert report.num_samples_per_chain == 64
    assert all(sum(chain) == 64 for rank in ranks for chain in rank.chains)


@pytest.mark.inference(concern="sampling")
def test_host_projection_preserves_native_transforms_across_chunks(monkeypatch):
    import jax
    import numpyro.distributions as dist

    from grassmoth.models.ssm.inference.parameter_tracing import extract_constrained_samples
    from grassmoth.models.ssm.inference.parameter_transform import ParameterTransform

    transform = dist.transforms.StickBreakingTransform()

    def constrain(position):
        simplex = transform(position)
        return {"simplex": simplex, "positive": jnp.exp(position), "derived": simplex[0] ** 2}

    positions = np.random.default_rng(418).normal(size=(259, 3)).astype(np.float32)
    expected = jax.device_get(jax.vmap(constrain)(positions))
    original_jit = jax.jit
    shapes = []

    def cpu_jit(function, **options):
        assert options["backend"] == "cpu"
        compiled = original_jit(function, **options)

        def call(batch):
            assert isinstance(batch, np.ndarray)
            assert batch.shape[0] <= 256
            shapes.append(batch.shape)
            result = compiled(batch)
            assert all(
                device.platform == "cpu" for value in result.values() for device in value.devices()
            )
            return result

        return call

    monkeypatch.setattr(jax, "jit", cpu_jit)
    parameters = ParameterTransform(
        initial_position=jnp.zeros(3),
        unravel=lambda _: {},
        constrain=constrain,
        log_prior=lambda _: jnp.asarray(0.0),
    )
    actual = extract_constrained_samples(positions, parameters, frozenset(expected))
    assert shapes == [(256, 3), (3, 3)]
    for name, values in actual.items():
        assert isinstance(values, np.ndarray)
        assert values.dtype == np.float32
        np.testing.assert_allclose(values, expected[name], rtol=1e-6, atol=1e-7)


@pytest.mark.inference(concern="sampling")
@pytest.mark.parametrize("family", ["beta", "student", "transformed"])
def test_prior_curves_keep_native_density_on_cpu(family):
    import numpyro.distributions as dist

    from grassmoth.study.projections.priors import _native_curve, _viewport_draws, prior_density

    law = {
        "beta": dist.Beta(2.0, 3.0),
        "student": dist.StudentT(4.0, 0.3, 1.2),
        "transformed": dist.TransformedDistribution(
            dist.Normal(0.2, 0.7), dist.transforms.ExpTransform()
        ),
    }[family]
    viewport = _viewport_draws(law)
    assert viewport.dtype == np.float32
    assert all(device.platform == "cpu" for device in viewport.devices())
    low, high = np.quantile(np.asarray(viewport), [0.01, 0.99])
    grid, density = _native_curve(law, np.float32(low), np.float32(high))
    assert all(device.platform == "cpu" for array in (grid, density) for device in array.devices())
    curve = prior_density(law)
    np.testing.assert_allclose(curve.x, grid, rtol=1e-6)
    np.testing.assert_allclose(curve.density, np.exp(np.asarray(law.log_prob(grid))), rtol=1e-6)


@pytest.mark.contract
@pytest.mark.parametrize(
    ("r_hat", "status", "advisory"),
    [
        (1.009, "passed", False),
        (1.01, "passed", True),
        (1.049, "passed", True),
        (1.05, "failed", False),
        (None, "not_evaluated", False),
    ],
)
def test_convergence_findings_keep_mixing_advisories_separate_from_verdict(r_hat, status, advisory):
    from grassmoth.artifacts.identity import ConstructRef
    from grassmoth.artifacts.reports.checks import ConvergenceSubject, LatentConvergenceSubject
    from grassmoth.artifacts.reports.posterior_diagnostics import (
        ChainDiagnostics,
        LatentDiagnostics,
        ParameterDiagnostics,
    )
    from grassmoth.models.ssm.inference.diagnostics.convergence import chain_convergence

    parameter = references([("scale", ())])[ParameterCoordinate(site_name="scale", indices=())][1]
    latent = ConstructRef(id="construct:state")
    report = chain_convergence(
        ChainDiagnostics(
            num_chains=4,
            num_samples_per_chain=1000,
            per_parameter=(
                ParameterDiagnostics(
                    parameter="scale",
                    subject=parameter,
                    r_hat=r_hat,
                    ess_bulk=800.0,
                    ess_tail=600.0,
                    mcse_mean=None,
                ),
            ),
            per_latent=(
                LatentDiagnostics(
                    subject=latent,
                    label="state",
                    coordinates=3,
                    r_hat=r_hat,
                    ess_bulk=800.0,
                    ess_tail=600.0,
                    mcse_mean=None,
                ),
            ),
        )
    )
    assert report.status == status
    assert report.checked == 2
    mixing = tuple(item for item in report.findings if item.code == "fit_mixing")
    assert len(mixing) == (2 if advisory else 0)
    for item in mixing:
        assert item.kind == "evaluated"
        assert item.outcome == "failed"
        assert item.evidence.criterion == "r_hat"
        assert item.evidence.value == r_hat
        assert item.evidence.upper == 1.01
        assert not item.evidence.accepted
    if mixing:
        assert isinstance(mixing[0].subject, ConvergenceSubject)
        assert isinstance(mixing[1].subject, LatentConvergenceSubject)
        assert mixing[0].subject.parameter == parameter
        assert mixing[1].subject.state == latent
    assert set(report.model_dump()) == {"findings", "checked", "status"}
