"""Joint parameterization settings and saved identities have one strict production contract."""

import pytest
from pydantic import TypeAdapter

from grassmoth.artifacts.continuation import JointTuning
from grassmoth.artifacts.joint_parameterization import (
    JointParameterizationRecord,
    MixedParameterizationRecord,
)

pytestmark = pytest.mark.contract


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
