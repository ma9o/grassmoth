"""HTTP clients boundary."""

import asyncio
from typing import TYPE_CHECKING, cast

from fastapi import Request

if TYPE_CHECKING:
    from temporalio.client import Client


class TemporalClientProvider:
    """One lazily connected Temporal client owned by the facade."""

    def __init__(self) -> None:
        """Initialize lazy Temporal connection state protected by an asynchronous lock."""
        self._lock = asyncio.Lock()
        self._client: Client | None = None

    async def get(self) -> "Client":
        """Connect to Temporal once and share the resulting client across concurrent requests."""
        async with self._lock:
            if self._client is None:
                from nof1_causal_lab.actions.temporal.client import connect_client

                self._client = await connect_client()
            return self._client


def study_clients(request: Request) -> TemporalClientProvider:
    """Retrieve the application's shared Temporal client provider for request dependency injection."""
    return cast("TemporalClientProvider", request.app.state.study_clients)
