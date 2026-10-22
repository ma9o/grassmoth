"""Recorded checks of a model and its selected observation design."""

from __future__ import annotations

from grassmoth.artifacts.base import Value
from grassmoth.artifacts.reports.checks import (
    Assessment,
    QuestionSubject,
    SpecificationAssessment,
)

type QuestionAssessment = Assessment[QuestionSubject, str]


class QuestionCheckReport(Value):
    """The study question checked against the model and, once prepared, the record."""

    findings: tuple[QuestionAssessment, ...]


class ModelCheckReport(Value):
    """Findings evaluated by the action against its pinned scientific inputs."""

    specification: tuple[SpecificationAssessment, ...]
    question: QuestionCheckReport
