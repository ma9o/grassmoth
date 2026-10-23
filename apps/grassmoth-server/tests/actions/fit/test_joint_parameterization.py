"""Joint parameterization settings and saved identities have one strict production contract."""

import pytest
from pydantic import ValidationError

from grassmoth.artifacts.sampling import NutsDASpec, SamplerSpec
from grassmoth_server.actions.contracts.settings import FitSettingsSpec

pytestmark = pytest.mark.contract


def test_parameterization_is_a_joint_only_setting_and_part_of_continuation_identity():
    assert NutsDASpec().parameterization == "centred"
    identities = []
    for name in ("centred", "non_centred", "mixed"):
        settings = FitSettingsSpec.model_validate(
            {"algorithm": {"method": "nuts_da", "parameterization": name}}
        )
        assert isinstance(settings.algorithm, NutsDASpec)
        assert settings.algorithm.parameterization == name
        identities.append(SamplerSpec(algorithm=settings.algorithm).kernel_identity)
    assert len(set(identities)) == 3
    with pytest.raises(ValidationError, match="Extra inputs"):
        FitSettingsSpec.model_validate(
            {"algorithm": {"method": "mpgibbs", "parameterization": "non_centred"}}
        )
    with pytest.raises(ValidationError):
        NutsDASpec.model_validate({"parameterization": "automatic"})
    with pytest.raises(ValidationError):
        NutsDASpec.model_validate({"chart": "centred"})
    with pytest.raises(ValidationError):
        NutsDASpec.model_validate({"parameterization": "innovation"})
