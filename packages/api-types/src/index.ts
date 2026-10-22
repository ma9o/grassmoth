// ---------------------------------------------------------------------------
// Generated from Python, using the same type names in both languages
// ---------------------------------------------------------------------------

export type * from "./generated/models";

export interface ArtifactData<T = unknown> {
  artifactId: string;
  data: T;
  context: string;
}

// Named type aliases inlined in generated types but needed as standalone exports
export type ValidationSeverity = "error" | "warning" | "info";
export type CellStatus = "ok" | "warning" | "error" | "not_evaluated";
export type CausalGranularity = "hourly" | "daily" | "weekly" | "monthly" | "yearly";

export { readNumericalArray } from "./arrays";
export { createModelClient } from "./client";

export { decodeModelMessage } from "./msgpack";
