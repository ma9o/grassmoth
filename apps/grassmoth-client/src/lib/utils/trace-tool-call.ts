import { recordValue } from "@/lib/model-asset/action-presentation";

export interface NormalizedTraceToolCall {
  readonly toolCallId: string;
  readonly toolName: string;
  readonly input: unknown;
}

function nonEmptyString(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

export function normalizeTraceToolCall(toolCall: unknown): NormalizedTraceToolCall | null {
  const record = recordValue(toolCall);
  if (record === null) return null;
  const toolCallId = nonEmptyString(record.id);
  if (toolCallId === null) {
    return null;
  }

  const nestedFunction = recordValue(record.function);
  const toolName = nonEmptyString(record.name) ?? nonEmptyString(nestedFunction?.name);
  if (toolName === null) {
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
  } catch {
    // Models can emit malformed argument JSON; the raw text is what was recorded.
    return {
      toolCallId,
      toolName,
      input: rawArguments,
    };
  }
}
