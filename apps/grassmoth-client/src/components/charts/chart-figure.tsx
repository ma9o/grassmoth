"use client";

import { Maximize2 } from "lucide-react";
import { type ReactNode, useState } from "react";
import { DAY_MS, type Domain } from "./plot-geometry";
import { PlotNumberInput } from "./plot-number-input";

export interface ChartView {
  readonly height: number;
  readonly timeWindow: Domain | null;
}

const isoDay = (day: number, origin: string) =>
  new Date(Date.parse(origin) + day * DAY_MS).toISOString().slice(0, 10);
const dayOf = (iso: string, origin: string) =>
  (Date.parse(`${iso}T00:00:00Z`) - Date.parse(origin)) / DAY_MS;

/** Narrow a chart over time to a window, by calendar date when the axis has an origin. */
function WindowControls({
  domain,
  origin,
  value,
  onChange,
  label,
}: {
  domain: Domain;
  origin: string | null;
  value: Domain | null;
  onChange: (value: Domain | null) => void;
  label: string;
}) {
  const [from, to] = value ?? domain;
  const set = (next: Domain) => {
    if (next[0] < next[1]) onChange(next);
  };
  return (
    <div className="flex flex-wrap items-center gap-2 text-xs">
      {origin ? (
        <>
          <label className="flex items-center gap-1">
            From
            <input
              type="date"
              aria-label={`${label}: from`}
              className="rounded border bg-background px-1"
              value={isoDay(from, origin)}
              onChange={(event) => {
                if (event.target.value) set([dayOf(event.target.value, origin), to]);
              }}
            />
          </label>
          <label className="flex items-center gap-1">
            To
            <input
              type="date"
              aria-label={`${label}: to`}
              className="rounded border bg-background px-1"
              value={isoDay(to, origin)}
              onChange={(event) => {
                if (event.target.value) set([from, dayOf(event.target.value, origin)]);
              }}
            />
          </label>
        </>
      ) : (
        <>
          <label className="flex items-center gap-1">
            From
            <PlotNumberInput
              aria-label={`${label}: from`}
              className="w-20 rounded border bg-background px-1"
              value={from}
              step="any"
              onValue={(next) => set([next, to])}
            />
          </label>
          <label className="flex items-center gap-1">
            To
            <PlotNumberInput
              aria-label={`${label}: to`}
              className="w-20 rounded border bg-background px-1"
              value={to}
              step="any"
              onValue={(next) => set([from, next])}
            />
          </label>
        </>
      )}
      <button type="button" className="underline" onClick={() => onChange(null)}>
        Whole range
      </button>
    </div>
  );
}

/**
 * A chart with its caption. Expanding opens the same chart larger, where a chart over time
 * can also be narrowed to a window.
 */
export function ChartFigure({
  title,
  legend,
  note,
  height,
  timeline,
  children,
}: {
  title: ReactNode;
  legend?: ReactNode;
  note?: ReactNode;
  height: number;
  /** The chart's time span, which enables narrowing it when expanded. */
  timeline?: { readonly domain: Domain; readonly origin: string | null };
  children: (view: ChartView) => ReactNode;
}) {
  const [expanded, setExpanded] = useState(false);
  const [timeWindow, setTimeWindow] = useState<Domain | null>(null);
  const name = typeof title === "string" ? title : "Chart";
  return (
    <figure className="m-0 flex min-w-0 flex-col gap-1">
      <figcaption className="flex min-w-0 flex-wrap items-start justify-between gap-x-2 gap-y-0.5 text-[11px]">
        <span className="font-medium">{title}</span>
        <span className="ml-auto flex flex-none items-center gap-2 text-[10px] text-muted-foreground">
          {legend}
          <button
            type="button"
            className="rounded p-0.5 hover:bg-secondary"
            aria-label={`Expand ${name}`}
            onClick={() => setExpanded(true)}
          >
            <Maximize2 className="size-3" aria-hidden="true" />
          </button>
        </span>
      </figcaption>
      {children({ height, timeWindow: null })}
      {note && <p className="m-0 text-[10px] leading-relaxed text-muted-foreground">{note}</p>}
      {expanded && (
        <dialog
          ref={(dialog) => {
            if (dialog && !dialog.open) dialog.showModal();
          }}
          aria-label={name}
          className="fixed inset-4 z-50 m-auto w-[min(1100px,94vw)] max-w-none space-y-3 rounded-xl border bg-card p-5 shadow-2xl backdrop:bg-black/40"
          onCancel={() => setExpanded(false)}
        >
          <div className="flex items-start justify-between gap-3 text-sm">
            <span className="font-medium">{title}</span>
            <button type="button" autoFocus onClick={() => setExpanded(false)}>
              Close
            </button>
          </div>
          {timeline && (
            <WindowControls
              domain={timeline.domain}
              origin={timeline.origin}
              value={timeWindow}
              onChange={setTimeWindow}
              label={name}
            />
          )}
          {children({ height: 440, timeWindow })}
          {legend && <div className="text-xs text-muted-foreground">{legend}</div>}
          {note && <p className="m-0 text-xs leading-relaxed text-muted-foreground">{note}</p>}
        </dialog>
      )}
    </figure>
  );
}
