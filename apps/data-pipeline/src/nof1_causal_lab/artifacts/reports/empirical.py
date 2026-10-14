"""Recorded empirical distribution coordinates shared by data and posterior reports."""

from pydantic import Field, FiniteFloat

from nof1_causal_lab.artifacts.base import Value


class EmpiricalPoint(Value):
    """A step of an empirical cumulative distribution."""

    value: FiniteFloat = Field(description="Distinct finite observed or sampled value.")
    probability: FiniteFloat = Field(
        description="Fraction of finite values less than or equal to `value`."
    )
