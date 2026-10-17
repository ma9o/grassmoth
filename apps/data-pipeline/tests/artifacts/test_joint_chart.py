"""Joint chart settings and saved identities have one strict production contract."""

import pytest
from pydantic import TypeAdapter, ValidationError

from nof1_causal_lab.artifacts.continuation import JointTuning, sampler_identity
from nof1_causal_lab.artifacts.joint_chart import JointChartRecord, MixedChartRecord
from nof1_causal_lab.artifacts.sampling import FitSettingsSpec, NutsDASpec, SamplerSpec

pytestmark = pytest.mark.contract


def test_chart_is_a_joint_only_setting_and_part_of_continuation_identity():
    assert NutsDASpec().chart == "centred"
    identities = []
    for name in ("centred", "innovation", "mixed"):
        settings = FitSettingsSpec.model_validate(
            {"algorithm": {"method": "nuts_da", "chart": name}}
        )
        assert isinstance(settings.algorithm, NutsDASpec)
        assert settings.algorithm.chart == name
        identities.append(sampler_identity(SamplerSpec(algorithm=settings.algorithm)))
    assert len(set(identities)) == 3
    for method in ("pgibbs", "mpgibbs"):
        with pytest.raises(ValidationError, match="Extra inputs"):
            FitSettingsSpec.model_validate({"algorithm": {"method": method, "chart": "innovation"}})
    with pytest.raises(ValidationError):
        NutsDASpec.model_validate({"chart": "automatic"})


def test_actual_mask_identity_survives_serialization_and_reports_equivalent_chart():
    record = MixedChartRecord(centred_mask=((False, True), (False, False)))
    saved = JointTuning(chart=record)
    restored = JointTuning.model_validate_json(saved.model_dump_json())
    assert restored == saved
    assert isinstance(restored.chart, MixedChartRecord)
    assert restored.chart.mask_identity == record.mask_identity
    assert not record.equals_innovation
    empty = MixedChartRecord(centred_mask=((False, False), (False, False)))
    assert empty.equals_innovation
    assert empty.mask_identity != record.mask_identity
    assert (
        empty.mask_identity
        != MixedChartRecord(centred_mask=((False, False, False, False),)).mask_identity
    )
    assert TypeAdapter(JointChartRecord).validate_json(record.model_dump_json()) == record
