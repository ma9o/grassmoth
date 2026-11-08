"""Launch-only automatic counts never enter numerical sampler contracts."""

import pytest
from pydantic import ValidationError

from grassmoth.artifacts.sampling import SamplerLaunchSpec, SamplerSpec

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
