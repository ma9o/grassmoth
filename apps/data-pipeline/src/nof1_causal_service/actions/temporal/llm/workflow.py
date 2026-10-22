"""Temporal workflow for one tool-validated LLM subroutine."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError, ApplicationError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from nof1_causal_service.llm.specs import EmbeddedLLMSpec

with workflow.unsafe.imports_passed_through():
    from nof1_causal_service.actions.temporal.client import OPENROUTER_TASK_QUEUE
    from nof1_causal_service.actions.temporal.llm.messages import (
        AppendLLMRepairMessageInput,
        AppendLLMRepairMessageResult,
        AppendLLMUserMessageInput,
        AppendLLMUserMessageResult,
        LLMSubroutineInput,
        LLMSubroutineResult,
        LLMSubroutineStart,
        LLMSubroutineTraceInput,
        LLMSubroutineTraceResult,
        LLMToolExecutionInput,
        LLMToolExecutionResult,
        OpenRouterCallInput,
        OpenRouterCallResult,
    )

_LOCAL_TIMEOUT = timedelta(minutes=5)
_TRACE_TIMEOUT = timedelta(minutes=5)
_LOCAL_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=10),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(minutes=5),
    maximum_attempts=3,
)
_OPENROUTER_CALL_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    backoff_coefficient=1.0,
    maximum_attempts=2,
)


def _provider_call_timeout(llm: EmbeddedLLMSpec) -> timedelta:
    return timedelta(seconds=(llm.timeout or 300) + 30)


async def _append_user_message(
    workflow_input: LLMSubroutineInput,
    conversation_ref: str,
    user_message_index: int,
) -> str:
    appended: AppendLLMUserMessageResult = await workflow.execute_activity(
        "append_llm_user_message_activity",
        AppendLLMUserMessageInput(
            subroutine=workflow_input.subroutine,
            conversation_ref=conversation_ref,
            user_message_index=user_message_index,
        ),
        result_type=AppendLLMUserMessageResult,
        start_to_close_timeout=_LOCAL_TIMEOUT,
        retry_policy=_LOCAL_RETRY,
        summary=(
            f"Append LLM user message {workflow_input.subroutine.subroutine_id} "
            f"#{user_message_index + 1}"
        ),
    )
    return appended.conversation_ref


def _activity_error_text(exc: ActivityError) -> str:
    cause = exc.cause
    if cause is not None:
        return str(cause)
    return str(exc)


async def _execute_openrouter_call(
    workflow_input: LLMSubroutineInput,
    start: LLMSubroutineStart,
    conversation_ref: str,
    turn_label: str,
    llm: EmbeddedLLMSpec,
    retain_conversation: Callable[[str], Awaitable[None]],
) -> tuple[OpenRouterCallResult, int]:
    def _call(label: str, source_conversation_ref: str) -> OpenRouterCallInput:
        return OpenRouterCallInput(
            conversation_ref=source_conversation_ref,
            next_conversation_ref=f"{start.conversation_ref_base}/{label}-assistant.json",
            call_ref=f"{start.call_ref_base}/{label}.json",
            assistant_ref=f"{start.assistant_ref_base}/{label}.json",
            llm=llm,
            tools=start.tools,
            log_label=f"{workflow_input.subroutine.subroutine_id}/{label}",
        )

    try:
        call: OpenRouterCallResult = await workflow.execute_activity(
            "call_openrouter_activity",
            _call(turn_label, conversation_ref),
            result_type=OpenRouterCallResult,
            task_queue=OPENROUTER_TASK_QUEUE,
            start_to_close_timeout=_provider_call_timeout(llm),
            retry_policy=_OPENROUTER_CALL_RETRY,
            summary=f"OpenRouter {workflow_input.subroutine.subroutine_id} {turn_label}",
        )
        return call, 1
    except ActivityError as exc:
        if not start.tools:
            raise
        repair_label = f"{turn_label}-repair-001"
        repaired: AppendLLMRepairMessageResult = await workflow.execute_activity(
            "append_llm_repair_message_activity",
            AppendLLMRepairMessageInput(
                subroutine=workflow_input.subroutine,
                conversation_ref=conversation_ref,
                next_conversation_ref=f"{start.conversation_ref_base}/{repair_label}.json",
                error_text=_activity_error_text(exc),
                tools=start.tools,
            ),
            result_type=AppendLLMRepairMessageResult,
            start_to_close_timeout=_LOCAL_TIMEOUT,
            retry_policy=_LOCAL_RETRY,
            summary=(
                f"Append LLM repair message {workflow_input.subroutine.subroutine_id} {turn_label}"
            ),
        )
        await retain_conversation(repaired.conversation_ref)
        repaired_call: OpenRouterCallResult = await workflow.execute_activity(
            "call_openrouter_activity",
            _call(repair_label, repaired.conversation_ref),
            result_type=OpenRouterCallResult,
            task_queue=OPENROUTER_TASK_QUEUE,
            start_to_close_timeout=_provider_call_timeout(llm),
            retry_policy=_OPENROUTER_CALL_RETRY,
            summary=f"OpenRouter {workflow_input.subroutine.subroutine_id} {repair_label}",
        )
        return repaired_call, 2


@workflow.defn
class LLMSubroutineWorkflow:
    """Durable extraction conversation that coordinates model turns and tool execution."""

    @workflow.run
    async def run(self, workflow_input: LLMSubroutineInput) -> LLMSubroutineResult:
        """Drive the extraction conversation to a validated terminal result.

        Args:
            workflow_input: Subroutine context, model backend, and allowed tool-turn budget.

        Returns:
            Stored result and trace references with the number of provider calls.

        Raises:
            ApplicationError: The conversation ends without a valid terminal result.
        """
        start: LLMSubroutineStart = await workflow.execute_activity(
            "start_llm_subroutine_activity",
            workflow_input.subroutine,
            result_type=LLMSubroutineStart,
            start_to_close_timeout=_LOCAL_TIMEOUT,
            retry_policy=_LOCAL_RETRY,
            summary=f"Start LLM subroutine {workflow_input.subroutine.subroutine_id}",
        )

        conversation_ref = start.conversation_ref
        last_result_ref: str | None = None
        n_llm_calls = 0
        terminal_error: str | None = None

        async def retain_trace() -> LLMSubroutineTraceResult:
            result: LLMSubroutineTraceResult = await workflow.execute_activity(
                "finalize_llm_subroutine_trace_activity",
                LLMSubroutineTraceInput(
                    subroutine=workflow_input.subroutine,
                    conversation_ref=conversation_ref,
                    call_ref_base=start.call_ref_base,
                ),
                result_type=LLMSubroutineTraceResult,
                start_to_close_timeout=_TRACE_TIMEOUT,
                retry_policy=_LOCAL_RETRY,
                summary=f"Retain LLM trace {workflow_input.subroutine.subroutine_id}",
            )
            return result

        async def retain_conversation(reference: str) -> None:
            nonlocal conversation_ref
            conversation_ref = reference
            await retain_trace()

        try:
            for user_message_index in range(start.user_message_count):
                conversation_ref = await _append_user_message(
                    workflow_input, conversation_ref, user_message_index
                )
                await retain_trace()
                user_label = f"user-{user_message_index + 1:03d}"

                for turn in range(1, workflow_input.max_tool_turns + 1):
                    turn_label = f"{user_label}-turn-{turn:03d}"
                    call, call_count = await _execute_openrouter_call(
                        workflow_input,
                        start,
                        conversation_ref,
                        turn_label,
                        workflow_input.llm,
                        retain_conversation,
                    )
                    n_llm_calls += call_count
                    conversation_ref = call.conversation_ref
                    await retain_trace()

                    if not start.tools:
                        break
                    if not call.tool_calls:
                        break

                    tool_execution: LLMToolExecutionResult = await workflow.execute_activity(
                        "execute_llm_tool_calls_activity",
                        LLMToolExecutionInput(
                            subroutine=workflow_input.subroutine,
                            conversation_ref=conversation_ref,
                            assistant_ref=call.assistant_ref,
                            execution_ref=f"{start.tool_execution_ref_base}/{turn_label}.json",
                            result_ref=f"{start.result_ref_base}/{turn_label}.json",
                            tools=start.tools,
                        ),
                        result_type=LLMToolExecutionResult,
                        retry_policy=_LOCAL_RETRY,
                        summary=(
                            f"Execute LLM tools {workflow_input.subroutine.subroutine_id} "
                            f"{turn_label}"
                        ),
                        start_to_close_timeout=_LOCAL_TIMEOUT,
                    )
                    conversation_ref = tool_execution.conversation_ref
                    await retain_trace()
                    if tool_execution.terminal_success:
                        if tool_execution.result_ref is not None:
                            last_result_ref = tool_execution.result_ref
                        else:
                            terminal_error = (
                                f"LLM subroutine {workflow_input.subroutine.subroutine_id} "
                                "terminal tool succeeded without a result ref"
                            )
                        break
                else:
                    terminal_error = (
                        f"LLM subroutine {workflow_input.subroutine.subroutine_id} exceeded "
                        f"{workflow_input.max_tool_turns} turns without validation success."
                    )

                if terminal_error is not None:
                    break
        finally:
            trace = await retain_trace()

        if terminal_error is not None:
            raise ApplicationError(
                terminal_error,
                type="LLMSubroutineError",
                non_retryable=True,
            )
        if last_result_ref is None:
            raise ApplicationError(
                f"LLM subroutine {workflow_input.subroutine.subroutine_id} "
                "produced no valid result",
                type="LLMSubroutineError",
                non_retryable=True,
            )

        return LLMSubroutineResult(
            result_ref=last_result_ref,
            conversation_ref=conversation_ref,
            trace_ref=trace.trace_ref,
            n_llm_calls=n_llm_calls,
        )
