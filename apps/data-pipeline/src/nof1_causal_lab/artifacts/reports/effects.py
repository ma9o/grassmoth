"""Reported causal-effect summaries and their plotting data."""

from pydantic import ConfigDict, Field

from nof1_causal_lab.artifacts.base import Value


class EffectSummary(Value):
    """An effect summary reports posterior location, uncertainty, and sign probability."""

    model_config = ConfigDict(allow_inf_nan=False)

    mean: float
    median: float
    lower_95: float
    upper_95: float
    prob_positive: float = Field(ge=0, le=1)


class HistogramBin(Value):
    """A histogram bin gives its interval, center, and number of posterior draws."""

    model_config = ConfigDict(allow_inf_nan=False)

    bin_center: float
    bin_start: float
    bin_end: float
    count: int = Field(ge=0)
