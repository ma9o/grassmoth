"""Scientific identities, retained draws, and sampler evidence in diagnostic reports."""

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
    divergences = np.zeros((2, 64), dtype=bool)
    divergences[0, 4] = True
    mcmc = TrajectoryMCMCResult(
        chain_samples=samples,
        chain_extra_fields={
            "diverging": jnp.asarray(divergences),
            "num_steps": jnp.full((2, 64), 4),
            "tree_depth": jnp.full((2, 64), 3),
            "accept_prob": jnp.full((2, 64), 0.84),
            "energy": jnp.asarray(4.0 + rng.normal(size=(2, 64))),
        },
        num_chains=2,
        num_samples=64,
        backend="marginal_particle_gibbs",
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
    assert report.num_divergences == 1
    assert report.divergence_rate == pytest.approx(1 / 128)
    assert report.tree_depth_mean == report.tree_depth_max == 3
    assert report.accept_prob_mean == pytest.approx(0.84)
    assert len(report.energy.bfmi) == 2
    assert all(np.isfinite(value) and value > 0 for value in report.energy.bfmi)
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
def test_posterior_marginals_and_traces_preserve_joint_draws_and_divergences(particle_posterior):
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
    expected_divergences = [False] * 128
    expected_divergences[4] = True
    traces, _ = particle_posterior.get_chain_detail(refs)
    columns = {trace.subject: np.asarray(trace.chains).reshape(-1) for trace in traces}
    assert tuple(
        bool(value)
        for value in particle_posterior.diagnostics.mcmc.get_extra_fields()["diverging"].reshape(-1)
    ) == tuple(expected_divergences)
    for coordinate, (_, subject) in refs.items():
        values = particle_posterior.get_samples()[coordinate.site_name][
            (slice(None), *coordinate.indices)
        ]
        np.testing.assert_array_equal(columns[subject], values)


@pytest.mark.inference(concern="sampling")
def test_latent_convergence_detects_stuck_paths_despite_well_mixed_parameters(particle_posterior):
    from dataclasses import replace

    from grassmoth.artifacts.identity import ConstructId, ConstructRef
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
            latent_chains=LatentChainEvidence(paths, free, states),
        ),
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
