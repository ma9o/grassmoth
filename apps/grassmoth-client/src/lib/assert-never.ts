/** A closed alternative must be handled before it reaches this branch. */
export function assertNever(value: never): never {
  throw new Error(`Unhandled alternative: ${String(value)}`);
}
