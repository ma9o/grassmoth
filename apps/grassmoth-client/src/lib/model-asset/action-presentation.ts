export function recordValue(value: unknown): Record<string, unknown> | null {
  const isRecord = (input: unknown): input is Record<string, unknown> =>
    input !== null && typeof input === "object" && !Array.isArray(input);
  return isRecord(value) ? value : null;
}
