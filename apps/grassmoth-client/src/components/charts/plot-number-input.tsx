import type { ComponentProps } from "react";

/** Commit a complete number on blur/Enter, so partial edits never move the plot. */
export function PlotNumberInput({
  value,
  onValue,
  ...props
}: Readonly<
  // An input is a void element, so it takes no inner HTML.
  Omit<ComponentProps<"input">, "value" | "onChange" | "type" | "dangerouslySetInnerHTML">
> & {
  readonly value: number;
  readonly onValue: (value: number) => void;
}) {
  return (
    <input
      {...props}
      key={value}
      type="number"
      defaultValue={value}
      onBlur={(event) => {
        const input = event.currentTarget;
        if (input.value !== "" && input.validity.valid && Number.isFinite(input.valueAsNumber)) {
          onValue(input.valueAsNumber);
        }
        input.value = String(value);
      }}
      onKeyDown={(event) => {
        if (event.key === "Enter") {
          event.preventDefault();
          event.currentTarget.blur();
        }
      }}
    />
  );
}
