"""Study actions for native Python callers using the configured framework runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING

from nof1_causal_service.study import interface as _operations
from nof1_causal_service.study.interface import ActionResult as ActionResult
from nof1_causal_service.study.interface import StudyError as StudyError

if TYPE_CHECKING:
    from nof1_causal_service.public.contracts import ActionId, ActionInput, CallId, TimelineResponse

__all__ = ["ActionResult", "StudyError", "get_timeline", "poll_action", "submit_action"]


async def submit_action(workspace_id: str, request: ActionInput) -> ActionResult | StudyError:
    """Resolve and submit an action through the production lifecycle, reusing saved calls."""
    return await _operations.submit_action(
        workspace_id, request, _operations.TemporalClientProvider()
    )


async def poll_action(
    workspace_id: str, action: ActionId, call_id: CallId
) -> ActionResult | StudyError:
    """Read an existing action without starting or retrying work."""
    return await _operations.poll_action(
        workspace_id, action, call_id, _operations.TemporalClientProvider()
    )


async def get_timeline(workspace_id: str) -> TimelineResponse | StudyError:
    """Read saved request lineage and the running action through the configured runtime."""
    return await _operations.get_timeline(workspace_id, _operations.TemporalClientProvider())
