/**
 * Convert pipeline LLMTrace → Vercel AI SDK UIMessage[].
 *
 * The pipeline stores a flat list of TraceMessages (system, user, assistant, tool).
 * The AI SDK uses a parts-based UIMessage format where:
 * - Each assistant message contains text, reasoning, and tool-call parts
 * - Tool results are merged into the assistant message's tool parts (state: "output-available")
 * - System/user messages become simple text-part messages
 */
import type { LLMTrace, TraceMessage } from "@grassmoth/api-types";
import type { DynamicToolUIPart, UIMessage } from "ai";
import { normalizeTraceToolCall } from "./trace-tool-call";

/**
 * A tool message's result as the dynamic-tool part of the call it answers.
 */
function toolResultPart(
  toolMsg: TraceMessage,
  toolCallId: string,
  input: unknown,
): DynamicToolUIPart {
  return toolMsg.tool_is_error
    ? {
        type: "dynamic-tool",
        toolCallId,
        toolName: toolMsg.tool_name ?? "unknown",
        state: "output-error",
        input,
        errorText: toolMsg.tool_result ?? toolMsg.content,
      }
    : {
        type: "dynamic-tool",
        toolCallId,
        toolName: toolMsg.tool_name ?? "unknown",
        state: "output-available",
        input,
        output: toolMsg.tool_result ?? toolMsg.content,
      };
}

/**
 * Tool calls — start as input-available, then look ahead for matching tool results.
 */
function toolCallParts(msg: TraceMessage): DynamicToolUIPart[] {
  return (msg.tool_calls ?? []).flatMap((tc): DynamicToolUIPart[] => {
    const normalized = normalizeTraceToolCall(tc);
    return normalized
      ? [
          {
            type: "dynamic-tool",
            toolCallId: normalized.toolCallId,
            toolName: normalized.toolName,
            state: "input-available",
            input: normalized.input,
          },
        ]
      : [];
  });
}

/**
 * The tool messages directly following an assistant message.
 */
function followingToolMessages(
  traceMessages: readonly TraceMessage[],
  index: number,
): readonly TraceMessage[] {
  const following = traceMessages.slice(index + 1);
  const end = following.findIndex((toolMsg) => toolMsg.role !== "tool");
  return end === -1 ? following : following.slice(0, end);
}

/**
 * An assistant message's text, reasoning and tool-call parts, with its tool messages folded in.
 */
function assistantParts(
  msg: TraceMessage,
  toolMessages: readonly TraceMessage[],
): UIMessage["parts"] {
  const parts: UIMessage["parts"] = [];

  // Reasoning
  if (msg.reasoning) {
    parts.push({ type: "reasoning", text: msg.reasoning });
  }

  // Text content
  if (msg.content) {
    parts.push({ type: "text", text: msg.content });
  }

  const toolParts = toolCallParts(msg);
  const toolCallIds = new Set(toolParts.map((part) => part.toolCallId));
  parts.push(...toolParts);

  // Consume following tool messages and merge them into the tool parts
  for (const toolMsg of toolMessages) {
    const callId = toolMsg.tool_call_id;
    if (callId && toolCallIds.has(callId)) {
      // Find the part in the parts array and replace it
      const idx = parts.findIndex(
        (p) => p.type === "dynamic-tool" && "toolCallId" in p && p.toolCallId === callId,
      );
      const part = parts[idx];
      if (idx !== -1 && part?.type === "dynamic-tool") {
        parts[idx] = toolResultPart(toolMsg, callId, part.input);
      }
    }
  }

  return parts;
}

/**
 * Convert a pipeline LLMTrace to an array of AI SDK UIMessages.
 *
 * Strategy:
 * - system/user → one UIMessage with a single text part
 * - assistant → one UIMessage with text + reasoning + dynamic-tool parts
 * - tool messages following an assistant are folded into the preceding
 *   assistant's tool parts as output-available / output-error
 */
export function traceToUIMessages(trace: LLMTrace): UIMessage[] {
  const messages: UIMessage[] = [];
  const traceMessages = trace.messages;

  let nextIndex = 0;
  for (const [i, msg] of traceMessages.entries()) {
    if (i < nextIndex) continue;
    // A stable id from the index
    const id = `trace-${i}`;

    if (msg.role === "system" || msg.role === "user") {
      messages.push({ id, role: msg.role, parts: [{ type: "text", text: msg.content }] });
      continue;
    }

    if (msg.role === "assistant") {
      const toolMessages = followingToolMessages(traceMessages, i);
      messages.push({ id, role: "assistant", parts: assistantParts(msg, toolMessages) });
      nextIndex = i + 1 + toolMessages.length; // skip past consumed tool messages
      continue;
    }

    // Orphan tool message (shouldn't happen, but handle gracefully)
    if (msg.role === "tool") {
      const toolCallId = msg.tool_call_id ?? `orphan-${i}`;
      messages.push({ id, role: "assistant", parts: [toolResultPart(msg, toolCallId, {})] });
      continue;
    }

    // Other provider role names have no UI conversation turn.
  }

  return messages;
}
