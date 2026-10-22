"""Initializer quality uses qualified scientific marginals, never its own density."""

from datetime import UTC, datetime

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from evaluation.comparison import initializer_scores, qualify_reference
from evaluation.declarations import Reference
from grassmoth.actions.contracts.bodies import FitOutput
from grassmoth.actions.fit.conditioning import condition_model
from grassmoth.artifacts.arrays import NumericalArray
from grassmoth.artifacts.identity import ConstructRef, ParameterRef
from grassmoth.artifacts.reports.data_quality import DataProfileReport
from grassmoth.artifacts.reports.inference import (
    FitCheckReport,
    FitRuntime,
    FitSegment,
    InferenceEvidence,
    InferenceMetadata,
    InferencePhase,
    InferenceReport,
    InferenceReportCore,
    InferenceReportDetail,
)
from grassmoth.artifacts.reports.model_checks import QuestionCheckReport
from grassmoth.artifacts.reports.posterior_diagnostics import (
    ChainDiagnostics,
    LatentDiagnostics,
    MCMCEvidence,
    ParameterDiagnostics,
)
from grassmoth.artifacts.sampling import SamplerSpec
from grassmoth.models.ssm.inference.diagnostics.convergence import chain_convergence
from grassmoth.models.ssm.inference.extraction import project_initialization
from grassmoth.models.ssm.inference.problem import build_particle_problem
from grassmoth.models.ssm.inference.reparameterization import AutoReparam
from grassmoth.models.ssm.inference.types import JointPosteriorDraws
from tests.support.inference import (
    bind_panel_fixture,
    compile_fit_fixture,
    parameter_draws,
    particle_posterior,
)
from tests.support.model import one_state_gaussian_model

pytestmark = pytest.mark.inference(concern="warmup")


@pytest.fixture
def scored_starts():
    model = one_state_gaussian_model()
    compiled = compile_fit_fixture(model).compiled_dynamical_model
    z = np.random.default_rng(4).permutation(np.linspace(-1, 1, 1024))
    samples = {
        name: jnp.asarray(np.abs(values) + 1 + z.reshape((-1, *([1] * (values.ndim - 1)))))
        for name, values in parameter_draws(model, len(z)).items()
    }
    paths = np.column_stack((2 + z, 4 + 2 * z, np.full_like(z, 7)))[..., None]
    conditioned, identity = condition_model(
        model,
        compiled,
        particle_posterior(
            JointPosteriorDraws(parameters=samples, latent_paths=jnp.asarray(paths))
        ),
        times=np.arange(3.0),
        time_origin=datetime(2026, 1, 1, tzinfo=UTC),
    )
    diagnostics = ChainDiagnostics(
        num_chains=4,
        num_samples_per_chain=256,
        per_parameter=tuple(
            ParameterDiagnostics(
                parameter=label,
                subject=ParameterRef(parameter_id=binding.parameter_id, element_id=element),
                r_hat=1.0,
                ess_bulk=1024.0,
                ess_tail=1024.0,
                mcse_mean=None,
            )
            for binding in compiled.bindings
            for element, label in binding.elements.items()
        ),
        per_latent=tuple(
            LatentDiagnostics(
                subject=ConstructRef(id=state.id),
                label=state.name,
                coordinates=2,
                r_hat=1.0,
                ess_bulk=1024.0,
                ess_tail=1024.0,
                mcse_mean=None,
            )
            for state in compiled.states
        ),
    )
    phases = tuple(
        InferencePhase(
            sampler="mpgibbs",
            stage=stage,
            start_seconds=start,
            end_seconds=stop,
            outcome="complete",
        )
        for stage, start, stop in (
            ("preparing_initialization", 1, 4),
            ("initializing_chains", 3, 5),
            ("compiling_sweeps", 0, 8),
        )
    )
    metadata = InferenceMetadata(
        distribution=identity,
        num_samples_total=1024,
        num_chains=4,
        duration_seconds=8,
        engine=MCMCEvidence(),
        sampler_diagnostics=None,
        segments=(
            FitSegment(
                start=0,
                stop=256,
                settings=SamplerSpec(num_chains=4, num_samples_per_chain=256),
                runtime=FitRuntime(
                    source_digest="test", versions={}, compiler={}, devices=("cpu",)
                ),
                phases=phases,
                diagnostics=diagnostics,
                convergence=chain_convergence(diagnostics),
            ),
        ),
    )
    reference = FitOutput(
        dynamical_model_spec=conditioned,
        checks=FitCheckReport(
            data=DataProfileReport(indicators={}, findings=()),
            preflight=(),
            question=QuestionCheckReport(findings=()),
        ),
        inference=InferenceReport(
            evidence=InferenceEvidence(),
            core=InferenceReportCore(
                inference_metadata=metadata,
                inference_diagnostics=diagnostics,
                convergence=chain_convergence(diagnostics),
                posterior_marginals=(),
                prior_densities={},
            ),
            detail=InferenceReportDetail(),
        ),
    )
    qualify_reference(
        reference,
        Reference(
            workspace="test", case_manifest="test.json", call_id="call:" + "1" * 64, max_mcse_sd=0.1
        ),
    )
    coordinates = {
        str(element): np.asarray(samples[coordinate.site_name][(slice(None), *coordinate.indices)])
        for binding in compiled.bindings
        for element, coordinate in binding.coordinates.items()
    }
    coordinates.update({str(compiled.states[0].id): paths[..., 0]})

    def starts(spread, log_joint=0.0):
        initial = {
            "time_points": NumericalArray.from_numpy(np.arange(3.0)),
            "log_joint": NumericalArray.from_numpy(np.full(2, log_joint)),
            **{
                name: NumericalArray.from_numpy(
                    np.stack(
                        (
                            values.mean(axis=0) - spread * values.std(axis=0, ddof=1),
                            values.mean(axis=0) + spread * values.std(axis=0, ddof=1),
                        )
                    )
                )
                for name, values in coordinates.items()
            },
        }
        return reference.revised(
            dynamical_model_spec=None,
            inference=reference.inference.revised(
                evidence=InferenceEvidence(phase_extra_fields={"initialization": initial}),
                core=reference.inference.core.revised(
                    inference_metadata=metadata.revised(
                        distribution=None,
                        num_samples_total=0,
                        num_chains=2,
                        segments=(
                            metadata.segments[0].revised(
                                stop=0,
                                diagnostics=None,
                                convergence=chain_convergence(None),
                                settings=SamplerSpec(
                                    num_chains=2,
                                    num_warmup=0,
                                    num_samples_per_chain=0,
                                ),
                            ),
                        ),
                    )
                ),
            ),
        )

    return reference, starts


def test_collapsed_high_density_starts_lose_reference_coverage(scored_starts):
    reference, starts = scored_starts
    covered = initializer_scores(starts(1), reference)
    collapsed = initializer_scores(starts(0, log_joint=1e9), reference)
    assert covered["initializer"] == collapsed["initializer"] == "scored"
    assert covered["initializer_preparation_seconds"] == 4.0
    assert collapsed == initializer_scores(starts(0, log_joint=-1e9), reference)
    for name in ("initializer_parameters", "initializer_paths"):
        covered_items, collapsed_items = covered[name], collapsed[name]
        assert isinstance(covered_items, dict)
        assert isinstance(collapsed_items, dict)
        for identity, scores in covered_items.items():
            assert isinstance(scores, dict)
            np.testing.assert_allclose(
                np.asarray(scores["start_rms_reference_sd"]), [1, 1], atol=1e-6
            )
            assert scores["min_spread_reference_sd"] == pytest.approx(1)
            narrow = collapsed_items[identity]
            assert isinstance(narrow, dict)
            assert narrow["min_spread_reference_sd"] == 0
            assert np.asarray(scores["reference_mass_covered"])[0] > 0.5
            assert np.asarray(narrow["reference_mass_covered"])[0] == 0
    paths = covered["initializer_paths"]
    assert isinstance(paths, dict)
    path = next(iter(paths.values()))
    assert isinstance(path, dict)
    spread = path["spread_reference_sd"]
    assert isinstance(spread, list)
    assert spread[-1] is None
    assert path["start_fixed_coordinates_equal"] == [True, True]


@pytest.mark.inference(concern="sampling")
def test_segment_scoring_cannot_borrow_healthy_anchor_draws(scored_starts):
    from evaluation.comparison import posterior_arrays, posterior_equivalence
    from evaluation.declarations import Tolerances
    from evaluation.segments import segment_summary
    from grassmoth.actions.fit.diagnostics import segment_diagnostics
    from grassmoth.artifacts.probability.serialization import (
        empirical_atoms,
        empirical_distribution,
    )

    reference, _ = scored_starts
    model = reference.dynamical_model_spec
    metadata = reference.inference.core.inference_metadata
    identity = metadata.distribution
    assert model is not None
    assert identity is not None
    atoms = empirical_atoms(model.distributions[identity]).copy()
    free = np.ptp(atoms, axis=0) > 0
    atoms.reshape(4, 256, -1)[:, 224:, free] = 100
    segment = metadata.segments[0].revised(
        start=224,
        stop=256,
        settings=metadata.segments[0].settings.revised(num_samples_per_chain=32),
        phases=(
            InferencePhase(
                sampler="mpgibbs",
                stage="sampling",
                start_seconds=0,
                end_seconds=10,
                outcome="complete",
            ),
        ),
    )
    result = reference.revised(
        dynamical_model_spec=model.with_entities(
            distributions={
                **model.distributions,
                identity: empirical_distribution(atoms),
            }
        ),
        inference=reference.inference.revised(
            core=reference.inference.core.revised(
                inference_metadata=metadata.revised(
                    segments=(metadata.segments[0].revised(stop=224), segment),
                )
            ),
            evidence=InferenceEvidence(
                latent_free_mask=NumericalArray.from_numpy(np.array([[True], [True], [False]])),
                phase_extra_fields={
                    "post_warmup": {
                        "diverging": NumericalArray.from_numpy(np.zeros((4, 32), dtype=bool))
                    }
                },
            ),
        ),
    )
    assert result.inference.core.convergence.status == "passed"
    current_model = model.with_entities(
        distributions={
            **model.distributions,
            identity: empirical_distribution(atoms.reshape(4, 256, -1)[:, 224:].reshape(128, -1)),
        }
    )
    diagnostics = segment_diagnostics(
        current_model, identity, result.inference.evidence, 4, metadata.engine
    )
    segment = segment.revised(diagnostics=diagnostics, convergence=chain_convergence(diagnostics))
    result = result.revised(
        inference=result.inference.revised(
            core=result.inference.core.revised(
                inference_metadata=result.inference.core.inference_metadata.revised(
                    segments=(*result.inference.core.inference_metadata.segments[:-1], segment)
                )
            )
        )
    )
    summary = segment_summary(result)
    assert summary["gate"] == "failed"
    assert summary["min_ess_per_sampling_second"] == 0
    np.testing.assert_array_equal(posterior_arrays(result, segment)[1][free], 100)
    assert np.all(posterior_arrays(result)[1][free] < 100)
    compared = posterior_equivalence(
        result, reference, Tolerances(mean_sd=0.5, sd_relative=0.3), segment
    )
    assert compared["comparison_status"] == "discrepant"


def test_no_reference_has_no_initializer_quality_proxy(scored_starts):
    _, starts = scored_starts
    assert initializer_scores(starts(0, log_joint=1e9), None) == {
        "initializer": "not scored",
        "initializer_reason": "No qualified reference",
        "initializer_preparation_seconds": 4.0,
    }


def test_initializer_paths_align_time_and_check_fixed_reference_coordinates(scored_starts):
    reference, starts = scored_starts
    result = starts(1)
    initial = result.inference.evidence.phase_extra_fields["initialization"]
    model = reference.dynamical_model_spec
    distribution = reference.inference.core.inference_metadata.distribution
    assert model is not None
    assert distribution is not None
    identity = model.law_layouts[distribution].constructs[0]
    wrong = initial[str(identity)].values.copy()
    wrong[1, 2] += 1

    def with_fields(fields):
        return result.revised(
            inference=result.inference.revised(
                evidence=InferenceEvidence(
                    phase_extra_fields={"initialization": {**initial, **fields}},
                )
            )
        )

    score = initializer_scores(
        with_fields({str(identity): NumericalArray.from_numpy(wrong)}), reference
    )
    paths = score["initializer_paths"]
    assert isinstance(paths, dict)
    path = paths[str(identity)]
    assert isinstance(path, dict)
    assert path["start_fixed_coordinates_equal"] == [
        True,
        False,
    ]
    with pytest.raises(ValueError, match="different time grids"):
        initializer_scores(
            with_fields({"time_points": NumericalArray.from_numpy(np.arange(3.0) + 1)}), reference
        )


def test_public_initialization_reuses_constrained_projection_and_scientific_identity():
    model = one_state_gaussian_model()
    prepared = compile_fit_fixture(model)
    panel = bind_panel_fixture(prepared.compiled_dynamical_model, np.zeros((3, 1)), np.arange(3.0))
    bundle = build_particle_problem(
        prepared.prior_runtime_bundle,
        panel,
        scheme="euler_maruyama",
        trace_key=jax.random.key(5),
        reparam=AutoReparam(centered=0.0),
    )
    positions = jnp.stack((bundle.runtime.initial_position, bundle.runtime.initial_position + 0.2))
    paths = jnp.arange(6.0).reshape(2, 3, 1)
    projected = project_initialization(
        {"positions": positions, "paths": paths},
        bundle=bundle,
        compiled_dynamical_model=prepared.compiled_dynamical_model,
    )
    for binding in prepared.compiled_dynamical_model.bindings:
        for identity, coordinate in binding.coordinates.items():
            expected = jnp.stack(
                tuple(
                    bundle.runtime.parameters.constrain(point)[coordinate.site_name][
                        coordinate.indices
                    ]
                    for point in positions
                )
            )
            np.testing.assert_array_equal(projected[str(identity)], expected)
    np.testing.assert_array_equal(
        projected[str(prepared.compiled_dynamical_model.states[0].id)], paths[..., 0]
    )
    np.testing.assert_array_equal(projected["time_points"], panel.times)
