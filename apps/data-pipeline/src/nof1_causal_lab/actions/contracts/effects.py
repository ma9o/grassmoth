"""What an executed action produced; the workflow publishes it with the attempt."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import Field

from nof1_causal_lab.artifacts.base import Value
from nof1_causal_lab.artifacts.identity import GitOid
from nof1_causal_lab.study.state import ArtifactRecord, RetractedArtifact

type ActionReportName = Literal[
    "checks",
    "identification",
    "validation",
    "data-profile",
    "inference",
    "simulation",
    "data-diff",
    "model-diff",
]


class ActionEffects(Value):
    """What an executed action did to the store: the workflow installs this."""

    produced: tuple[ArtifactRecord, ...] = Field(default_factory=tuple)
    retracted: tuple[RetractedArtifact, ...] = Field(default_factory=tuple)
    reports: Mapping[ActionReportName, GitOid] = Field(default_factory=dict)
