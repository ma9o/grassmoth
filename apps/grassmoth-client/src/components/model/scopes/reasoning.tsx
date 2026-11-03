import type { ReasoningSpec } from "@grassmoth/api-types";
import { Hint, Prose } from "../scope-primitives";

/** Render the citation markers already resolved by the reasoning owner. */
export function ReasoningText({ value, maxLength }: { value: ReasoningSpec; maxLength?: number }) {
  const text =
    maxLength !== undefined && value.text.length > maxLength
      ? `${value.text.slice(0, maxLength - 1)}…`
      : value.text;
  return text.split(/(\[[0-9]+\])/).map((part, index) =>
    index % 2 === 0 ? (
      part
    ) : (
      <a
        key={index}
        href={value.reference_urls[Number(part.slice(1, -1)) - 1]}
        target="_blank"
        rel="noreferrer"
        className="underline underline-offset-2"
      >
        {part}
      </a>
    ),
  );
}

/** The scientific reasoning recorded on an authored entity. */
export function Reasoning({ value, muted = false }: { value: ReasoningSpec; muted?: boolean }) {
  const Text = muted ? Hint : Prose;
  return (
    <Text>
      <ReasoningText value={value} />
    </Text>
  );
}
