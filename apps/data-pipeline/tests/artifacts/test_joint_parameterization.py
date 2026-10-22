"""Joint parameterization settings and saved identities have one strict production contract."""

import pytest
from pydantic import TypeAdapter, ValidationError

from nof1_causal_lab.artifacts.continuation import JointTuning
from nof1_causal_lab.artifacts.joint_parameterization import (
    JointParameterizationRecord,
    MixedParameterizationRecord,
)
from nof1_causal_lab.artifacts.sampling import NutsDASpec, SamplerSpec
from nof1_causal_service.actions.contracts.settings import FitSettingsSpec

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
    for method in ("pgibbs", "mpgibbs"):
        with pytest.raises(ValidationError, match="Extra inputs"):
            FitSettingsSpec.model_validate(
                {"algorithm": {"method": method, "parameterization": "non_centred"}}
            )
    with pytest.raises(ValidationError):
        NutsDASpec.model_validate({"parameterization": "automatic"})
    with pytest.raises(ValidationError):
        NutsDASpec.model_validate({"chart": "centred"})
    with pytest.raises(ValidationError):
        NutsDASpec.model_validate({"parameterization": "innovation"})


def test_actual_mask_identity_survives_serialization_and_reports_equivalent_parameterization():
    record = MixedParameterizationRecord(centred_mask=((False, True), (False, False)))
    saved = JointTuning(parameterization=record)
    restored = JointTuning.model_validate_json(saved.model_dump_json())
    assert restored == saved
    assert isinstance(restored.parameterization, MixedParameterizationRecord)
    assert restored.parameterization.mask_identity == record.mask_identity
    assert not record.equals_non_centred
    empty = MixedParameterizationRecord(centred_mask=((False, False), (False, False)))
    assert empty.equals_non_centred
    assert empty.mask_identity != record.mask_identity
    assert (
        empty.mask_identity
        != MixedParameterizationRecord(centred_mask=((False, False, False, False),)).mask_identity
    )
    assert (
        TypeAdapter(JointParameterizationRecord).validate_json(record.model_dump_json()) == record
    )
