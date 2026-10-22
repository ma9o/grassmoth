export interface NormalizedTraceToolCall {
  toolCallId: string;
  toolName: string;
  input: unknown;
}

export function normalizeTraceToolCall(toolCall: unknown): NormalizedTraceToolCall | null {
  const record = recordValue(toolCall);
  if (record === null) return null;
  const toolCallId = typeof record.id === "string" && record.id.length > 0 ? record.id : null;
  if (!toolCallId) {
    return null;
  }

  const nestedFunction = recordValue(record.function);
  const toolName =
    typeof record.name === "string" && record.name.length > 0
      ? record.name
      : typeof nestedFunction?.name === "string" && nestedFunction.name.length > 0
        ? nestedFunction.name
        : null;
  if (!toolName) {
    return null;
  }

  const rawArguments = record.arguments ?? nestedFunction?.arguments;
  if (typeof rawArguments !== "string") {
    return {
      toolCallId,
      toolName,
      input: rawArguments ?? {},
    };
  }

  try {
    return {
      toolCallId,
      toolName,
      input: JSON.parse(rawArguments),
    };
  } catch (err) {
    console.warn("Failed to parse tool call arguments:", err);
    return {
      toolCallId,
      toolName,
      input: rawArguments,
    };
  }
}
import { recordValue } from "@/lib/model-asset/action-presentation";
