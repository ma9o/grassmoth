"""HTTP dependency injection for the study service's connection owner."""

from typing import cast

from fastapi import Request

from grassmoth_server.study.interface import TemporalClientProvider


def study_clients(request: Request) -> TemporalClientProvider:
    """Retrieve the application's shared Temporal client provider."""
    return cast("TemporalClientProvider", request.app.state.study_clients)
