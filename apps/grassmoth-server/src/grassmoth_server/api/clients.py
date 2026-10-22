"""HTTP dependency injection for the study service's connection owner."""

from typing import cast

from fastapi import Request

from grassmoth_server.study.calls import TemporalClientProvider


def temporal_client(request: Request) -> TemporalClientProvider:
    """Retrieve the application's shared Temporal client provider."""
    return cast("TemporalClientProvider", request.app.state.temporal_client)
