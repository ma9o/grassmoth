"""Launch-only automatic counts never enter numerical sampler contracts."""

import pytest
from pydantic import ValidationError

from grassmoth.artifacts.sampling import (
    SamplerLaunchSpec,
    SamplerSpec,
    UnsupportedSamplerLaunch,
)

pytestmark = pytest.mark.contract


@pytest.mark.parametrize(
    ("payload", "axis"),
    [
        ({"num_chains": "auto"}, "chains"),
        ({"algorithm": {"method": "mpgibbs", "n_parameter_particles": "auto"}}, "parameters"),
        (
            {
                "algorithm": {
                    "method": "mpgibbs",
                    "path_update": {"method": {"kind": "particle_mgrad", "num_particles": "auto"}},
                }
            },
            "particles",
        ),
    ],
)
def test_one_auto_round_trips_and_resolves_to_numeric_owner(payload, axis):
    launch = SamplerLaunchSpec.model_validate(payload)
    assert launch.automatic_dimension == axis
    assert SamplerLaunchSpec.model_validate_json(launch.model_dump_json()) == launch
    resolved = launch.resolve(8)
    assert "auto" not in resolved.model_dump_json()
    assert SamplerSpec.model_validate_json(resolved.model_dump_json()) == resolved
    with pytest.raises(ValidationError):
        SamplerSpec.model_validate(payload)


@pytest.mark.parametrize("auto_axis", ["parameters", "particles"])
def test_two_auto_counts_are_rejected(auto_axis):
    algorithm = (
        {"method": "mpgibbs", "n_parameter_particles": "auto"}
        if auto_axis == "parameters"
        else {
            "method": "mpgibbs",
            "path_update": {"method": {"kind": "particle_mgrad", "num_particles": "auto"}},
        }
    )
    with pytest.raises(ValidationError, match="At most one"):
        SamplerLaunchSpec.model_validate({"num_chains": "auto", "algorithm": algorithm})


def test_launch_reuses_numerical_defaults_and_constraints():
    assert SamplerLaunchSpec().resolve(2) == SamplerSpec()
    for payload in [
        {"num_warmup": -1},
        {"num_samples_per_chain": -1},
        {"time_limit_seconds": 0},
        {"num_chains": 0},
        {"algorithm": {"method": "mpgibbs", "n_parameter_particles": 3}},
        {
            "algorithm": {
                "method": "mpgibbs",
                "path_update": {"method": {"kind": "particle_mgrad", "scan_unroll": 0}},
            }
        },
    ]:
        with pytest.raises(ValidationError):
            SamplerLaunchSpec.model_validate(payload)


@pytest.mark.parametrize(
    "algorithm",
    [
        {"method": "nuts_da"},
        {"method": "mpgibbs", "path_update": {"method": {"kind": "auxiliary_kalman"}}},
        {"method": "mpgibbs", "path_update": {"method": {"kind": "twisted_particle_agrad"}}},
        {
            "method": "mpgibbs",
            "path_update": {"update": {"kind": "blocked", "block_length": 10}},
        },
        {"method": "mpgibbs", "path_update": {"proposal": {"kind": "parameter_mean"}}},
        {"method": "mpgibbs", "path_update": {"proposal": {"kind": "conditional_parameter"}}},
        {
            "method": "mpgibbs",
            "path_update": {"proposal": {"kind": "parameter_mixture", "weights": "prior"}},
        },
    ],
)
def test_launch_rejects_alternatives_but_numerical_specs_remain_available(algorithm):
    payload = {"algorithm": algorithm}
    numerical = SamplerSpec.model_validate(payload)
    with pytest.raises(ValidationError):
        SamplerLaunchSpec.model_validate(payload)
    inherited = SamplerLaunchSpec.from_resolved(numerical)
    assert isinstance(inherited, UnsupportedSamplerLaunch)
    assert inherited.message
    assert SamplerSpec.model_validate_json(numerical.model_dump_json()) == numerical


@pytest.mark.parametrize("kappa", [0, 1])
@pytest.mark.parametrize("parameter_proposal", ["random_walk", "pseudo_langevin"])
def test_supported_inheritance_preserves_every_control(kappa, parameter_proposal):
    numerical = SamplerSpec.model_validate(
        {
            "num_warmup": 17,
            "num_samples_per_chain": 23,
            "num_chains": 3,
            "thinning": False,
            "seed": 19,
            "hardware": "local",
            "time_limit_seconds": 37,
            "algorithm": {
                "method": "mpgibbs",
                "n_parameter_particles": 16,
                "path_update": {
                    "method": {
                        "kind": "particle_mgrad",
                        "num_particles": 32,
                        "kappa": kappa,
                        "scan_unroll": 2,
                    },
                    "latent_step": {"kind": "fixed", "delta": 0.31},
                },
                "parameter_proposal": parameter_proposal,
                "diagnostic_metrics": ["particle_weights"],
                "param_target_accept": 0.3,
                "adaptation_scheme": "simple",
                "initialization": {"scale": 0.7},
            },
        }
    )
    inherited = SamplerLaunchSpec.from_resolved(numerical)
    assert isinstance(inherited, SamplerLaunchSpec)
    assert inherited.automatic_dimension is None
    assert inherited.resolve(128) == numerical
    assert SamplerLaunchSpec.model_validate_json(inherited.model_dump_json()) == inherited
