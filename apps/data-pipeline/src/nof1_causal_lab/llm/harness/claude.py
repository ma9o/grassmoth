"""AgentSession backed by the ``claude -p`` subprocess.

Each :class:`ClaudeHarnessSession` owns a single Claude Code session
(identified by a UUID we mint and pass via ``--session-id``). The first
``turn()`` call spawns ``claude -p`` with that ID; subsequent turns
spawn ``claude -p --resume <id>`` so conversation state is preserved
across turns.

Tools are served to Claude through an in-process streamable-HTTP MCP
server (see :mod:`.mcp_server`). The MCP server URL is written into a
per-session ``.mcp.json`` file that we pass via ``--mcp-config``; we
also force ``--strict-mcp-config`` so Claude ignores any user-level
MCP config that might conflict.

Stream-json output is parsed incrementally into the shared
:class:`~.events.common.SessionStreamRuntime`; the per-turn ``TurnResult``
and the cumulative ``AgentResult`` are derived from that.
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from nof1_causal_lab.actions.errors import execution_failure_handler
from nof1_causal_lab.llm.harness.events.claude import (
    apply_claude_event,
    format_claude_event_for_log,
)
from nof1_causal_lab.llm.harness.events.common import (
    SessionStreamRuntime,
    event_object,
    finalize_trace,
)
from nof1_causal_lab.llm.harness.mcp_server import serve_tools_http
from nof1_causal_lab.llm.harness.streaming import finish_harness_process, handle_stream_event
from nof1_causal_lab.llm.sessions import AgentResult, TurnResult

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from nof1_causal_lab.json_types import JsonObject
    from nof1_causal_lab.llm.openrouter import Tool

logger = logging.getLogger(__name__)
# Stream events from the claude subprocess are at INFO level; make sure they
# clear the root logger's default WARNING threshold so they propagate up to
# the Prefect APILogHandler attached to ``nof1_causal_lab``.
logger.setLevel(logging.INFO)

MCP_SERVER_NAME = "pipeline-tools"
# Tool allowlist glob pattern for --allowedTools. Claude prefixes MCP
# tools as ``mcp__<server>__<tool>``; the ``*`` suffix matches any tool
# exposed by this server so the pipeline doesn't have to enumerate them.
MCP_TOOL_ALLOWLIST = f"mcp__{MCP_SERVER_NAME}__*"

_DEFAULT_TIMEOUT_SECONDS = 900


def build_mcp_config_json(url: str, server_name: str = MCP_SERVER_NAME) -> str:
    """Render a ``.mcp.json`` payload pointing Claude at our HTTP MCP server."""
    payload = {
        "mcpServers": {
            server_name: {
                "type": "http",
                "url": url,
            }
        }
    }
    return json.dumps(payload)


def build_claude_argv(
    *,
    executable: str,
    user_message: str,
    session_id: str,
    resume: bool,
    mcp_config_path: str | Path,
    system_prompt: str | None,
    model: str,
    effort: str | None,
    max_turns: int | None,
    max_budget_usd: float | None,
    fallback_model: str | None,
    tool_allowlist: str = MCP_TOOL_ALLOWLIST,
) -> list[str]:
    """Build the argv for one ``claude -p`` invocation.

    Pipeline-fixed flags: ``--strict-mcp-config`` (only our MCP server
    is loaded), ``--permission-mode bypassPermissions``,
    ``--tools ""`` + ``--allowedTools mcp__pipeline-tools__*``
    (disable built-in tools; only our MCP tools are reachable),
    ``--disable-slash-commands`` (no user-level skills),
    ``--output-format stream-json``, ``--verbose``,
    ``--exclude-dynamic-system-prompt-sections`` (stable prompt cache).

    Deliberately does NOT pass ``--bare``: bare mode skips OAuth /
    keychain reads, so Claude Max / Pro subscription auth is lost. We
    accept the minor hermeticity loss (user's ``~/.claude`` hooks and
    ``CLAUDE.md`` auto-memory may load) in exchange for subscription
    auth working out of the box.
    """
    argv: list[str] = [
        str(executable),
        "-p",
        user_message,
        "--mcp-config",
        str(mcp_config_path),
        "--strict-mcp-config",
        "--permission-mode",
        "bypassPermissions",
        "--tools",
        "",
        "--allowedTools",
        tool_allowlist,
        "--disable-slash-commands",
        "--output-format",
        "stream-json",
        "--verbose",
        "--exclude-dynamic-system-prompt-sections",
        "--model",
        model,
    ]
    if resume:
        argv.extend(["--resume", session_id])
    else:
        argv.extend(["--session-id", session_id])
    if system_prompt:
        argv.extend(["--append-system-prompt", system_prompt])
    if effort is not None:
        argv.extend(["--effort", effort])
    if max_turns is not None:
        argv.extend(["--max-turns", str(max_turns)])
    if max_budget_usd is not None:
        argv.extend(["--max-budget-usd", str(max_budget_usd)])
    if fallback_model is not None:
        argv.extend(["--fallback-model", fallback_model])
    return argv


class ClaudeHarnessSession:
    """:class:`AgentSession` implemented over ``claude -p`` subprocesses."""

    def __init__(
        self,
        *,
        tools: list[Tool],
        mcp_config_path: Path,
        system_prompt: str | None,
        model: str,
        executable: str = "claude",
        effort: str | None = None,
        max_turns: int | None = None,
        max_budget_usd: float | None = None,
        fallback_model: str | None = None,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        log_label: str | None = None,
        session_id: str | None = None,
        initial_events: list[JsonObject] | None = None,
        turn_index: int = 0,
    ) -> None:
        """Bind Claude process settings and restore trace state for a new or resumed harness session."""
        self._tools = list(tools)
        self._tool_stop_map = {t.name: t.success_output for t in tools if t.stop_on_success}
        self._mcp_config_path = mcp_config_path
        self._system_prompt = system_prompt
        self._model = model
        self._bin = executable
        self._effort = effort
        self._max_turns = max_turns
        self._max_budget_usd = max_budget_usd
        self._fallback_model = fallback_model
        self._timeout_seconds = timeout_seconds
        self._log_label = log_label

        self._session_id = session_id or str(uuid.uuid4())
        self._state = SessionStreamRuntime()
        for event in initial_events or []:
            apply_claude_event(self._state, event)
        self._turn_index = turn_index
        self._terminal_tool: tuple[str, str] | None = None

    @property
    def session_id(self) -> str:
        """Session identity passed to Claude when starting or resuming the conversation."""
        return self._session_id

    @property
    def raw_events(self) -> list[JsonObject]:
        """A copy of the raw Claude events accumulated across this session's turns."""
        return list(self._state.raw_events)

    @execution_failure_handler
    async def turn(self, user_message: str) -> TurnResult:
        """Run the next Claude prompt in this session and return only that turn's trace and outcome."""
        self._turn_index += 1
        pre_event_count = len(self._state.raw_events)

        argv = build_claude_argv(
            executable=self._bin,
            user_message=user_message,
            session_id=self._session_id,
            resume=self._turn_index > 1,
            mcp_config_path=self._mcp_config_path,
            system_prompt=self._system_prompt if self._turn_index == 1 else None,
            model=self._model,
            effort=self._effort,
            max_turns=self._max_turns,
            max_budget_usd=self._max_budget_usd,
            fallback_model=self._fallback_model,
        )

        proc = await asyncio.create_subprocess_exec(
            *argv,
            # The user message is passed via argv (-p); don't inherit the
            # parent's stdin or ``claude -p`` may block reading additional
            # input when the parent (e.g. a Prefect worker) has stdin open.
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        await finish_harness_process(
            proc,
            timeout_seconds=self._timeout_seconds,
            backend="claude",
            handle_stdout=self._handle_claude_line,
            handle_stderr=lambda _text: None,
        )

        turn_events = self._state.raw_events[pre_event_count:]
        return self._build_turn_result(turn_events)

    def _handle_claude_line(self, raw: bytes) -> None:
        handle_stream_event(
            raw,
            self._state,
            error_label=f"[{self._log_label}] claude",
            log_label=self._log_label,
            logger=logger,
            format_event=format_claude_event_for_log,
            apply_event=apply_claude_event,
        )

    def _build_turn_result(self, turn_events: list[JsonObject]) -> TurnResult:
        tool_calls_fired: list[str] = []
        terminal: tuple[str, str] | None = None
        for event in turn_events:
            if event.get("type") == "assistant":
                message = event_object(event.get("message"))
                content = message.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            tool_calls_fired.append(str(block.get("name", "")))
            elif event.get("type") == "user":
                message = event_object(event.get("message"))
                content = message.get("content")
                if not isinstance(content, list):
                    continue
                for block in content:
                    if not isinstance(block, dict) or block.get("type") != "tool_result":
                        continue
                    result_text = _tool_result_text(block.get("content"))
                    matched = self._match_terminal(tool_calls_fired, result_text, block)
                    if matched is not None:
                        terminal = matched
        if terminal is not None:
            self._terminal_tool = terminal

        return TurnResult(
            terminal_tool_name=terminal[0] if terminal else None,
            tool_calls_fired=tool_calls_fired,
        )

    def _match_terminal(
        self,
        tool_calls_fired: list[str],
        result_text: str,
        block: JsonObject,
    ) -> tuple[str, str] | None:
        """Match a tool-result block against stop_on_success tool metadata."""
        # Claude's MCP tool names arrive as ``mcp__<server>__<name>``. Strip
        # the prefix before checking against our tools' stop_on_success map.
        candidate_names: list[str] = []
        for fired in reversed(tool_calls_fired):
            short = fired.split("__")[-1] if "__" in fired else fired
            candidate_names.append(short)
            if len(candidate_names) >= len(tool_calls_fired):
                break
        # Also consider any stop_on_success tool by name: if the result_text
        # matches its success_output, that's a terminal hit regardless of
        # which tool the stream says fired (handles edge cases where the
        # assistant event lags behind the tool_result).
        for tool_name, success_output in self._tool_stop_map.items():
            if success_output is None:
                # Any non-error result counts as success for this shape.
                if not bool(block.get("is_error", False)):
                    return tool_name, result_text
            elif result_text.strip() == success_output:
                return tool_name, result_text
        return None

    @property
    def result(self) -> AgentResult:
        """Assemble the complete session trace and the last accepted terminal tool name."""
        trace = finalize_trace(self._state)
        return AgentResult(
            trace=trace,
            terminal_tool_name=self._terminal_tool[0] if self._terminal_tool else None,
        )

    async def aclose(self) -> None:
        """Complete the session interface; each turn already owns and closes its subprocess."""
        return


def _tool_result_text(content: object) -> str:
    """Flatten a tool_result ``content`` value to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    if content is None:
        return ""
    return json.dumps(content)


@asynccontextmanager
async def open_claude_harness_session(
    *,
    tools: list[Tool],
    system_prompt: str | None,
    model: str,
    executable: str = "claude",
    effort: str | None = None,
    max_turns: int | None = None,
    max_budget_usd: float | None = None,
    fallback_model: str | None = None,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    log_label: str | None = None,
    session_id: str | None = None,
    initial_events: list[JsonObject] | None = None,
    turn_index: int = 0,
) -> AsyncIterator[ClaudeHarnessSession]:
    """Open a Claude-backed agent session scoped to an ``async with`` block.

    Starts an in-process MCP server for ``tools``, writes a
    per-invocation ``.mcp.json`` file pointing at it, then yields a
    :class:`ClaudeHarnessSession`. On exit, the MCP server shuts down
    and the temp MCP config file is deleted.
    """
    from nof1_causal_lab.config import ensure_harness_prereqs

    ensure_harness_prereqs("claude-code")
    async with serve_tools_http(tools, name=MCP_SERVER_NAME) as mcp_url:
        mcp_config_payload = build_mcp_config_json(mcp_url)
        with tempfile.TemporaryDirectory(prefix="claude-harness-") as tmpdir:
            mcp_config_path = Path(tmpdir) / "mcp.json"
            mcp_config_path.write_text(mcp_config_payload)
            session = ClaudeHarnessSession(
                tools=tools,
                mcp_config_path=mcp_config_path,
                system_prompt=system_prompt,
                model=model,
                executable=executable,
                effort=effort,
                max_turns=max_turns,
                max_budget_usd=max_budget_usd,
                fallback_model=fallback_model,
                timeout_seconds=timeout_seconds,
                log_label=log_label,
                session_id=session_id,
                initial_events=initial_events,
                turn_index=turn_index,
            )
            try:
                yield session
            finally:
                await session.aclose()
