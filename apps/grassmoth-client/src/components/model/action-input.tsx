"use client";

import {
  type JsonScalar,
  type NumericalArray,
  readNumericalArray,
  type TimelineRevision,
} from "@grassmoth/api-types";
import { Ellipsis } from "lucide-react";
import Link from "next/link";
import { type ReactNode, useEffect, useRef, useState } from "react";
import { timelineTickLabel } from "@/lib/model-asset/timeline-presentation";
import { ACTION_STYLE, ActionMark } from "./action-mark";
import { Hint, RecordSection } from "./scope-primitives";

/** A request value as the API serializes it; MessagePack keeps numerical arrays as NPY bytes. */
type InputValue = JsonScalar | NumericalArray | InputList | InputRecord;
type InputList = readonly InputValue[];
interface InputRecord {
  readonly [key: string]: InputValue | undefined;
}
type InputObject = NumericalArray | InputList | InputRecord;
type InputEntry = readonly [string, InputValue];

/** Where each input reference came from, opened the way the timeline opens its nodes. */
export interface InputSources {
  /** The calls that published each reference, keyed by the reference's field name. */
  readonly calls: ReadonlyMap<string, readonly TimelineRevision[]>;
  readonly hrefAt: (seq: number) => string;
  readonly select: (seq: number) => void;
}

/**
 * Top-level arguments start closed. Once one is opened, nested arguments up to this size start
 * open, about a pane's height of wrapped text, so a small value reads at once while a whole model
 * specification still opens one level at a time.
 */
const OPEN_UP_TO_CHARACTERS = 1200;

function isList(value: InputObject): value is InputList {
  return Array.isArray(value);
}

function isNumericalArray(value: NumericalArray | InputRecord): value is NumericalArray {
  return "npy" in value && value.npy instanceof Uint8Array;
}

/** The value's entries to open, or null when it shows in one row. */
function groupOf(value: InputValue): InputList | InputRecord | null {
  if (value === null || typeof value !== "object") return null;
  if (isList(value)) return value.length > 0 ? value : null;
  if (isNumericalArray(value)) return null;
  return Object.values(value).some((item) => item !== undefined) ? value : null;
}

/**
 * A list keeps its order. A record lists its one-row fields first and the fields that open
 * last, each part in the API's order.
 */
function entriesOf(value: InputList | InputRecord): readonly InputEntry[] {
  return isList(value)
    ? value.map((item, index): InputEntry => [String(index), item])
    : Object.entries(value)
        .flatMap(([key, item]): InputEntry[] => (item === undefined ? [] : [[key, item]]))
        .toSorted(([, a], [, b]) => Number(groupOf(a) !== null) - Number(groupOf(b) !== null));
}

/** A long value kept to one line; the control at its end shows the rest in place. */
function OneLine({ children, breaks }: { children: ReactNode; breaks: boolean }) {
  const [expanded, setExpanded] = useState(false);
  const [clipped, setClipped] = useState(breaks);
  const text = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    const element = text.current;
    if (element === null || expanded) return;
    const observer = new ResizeObserver(() =>
      setClipped(breaks || element.scrollWidth > element.clientWidth),
    );
    observer.observe(element);
    return () => observer.disconnect();
  }, [breaks, expanded]);
  const toggle = (
    <button
      type="button"
      aria-label="Show full value"
      aria-expanded={expanded}
      onClick={() => setExpanded((current) => !current)}
      className="inline-flex flex-none cursor-pointer items-center rounded-sm border bg-card px-0.5 align-middle text-muted-foreground outline-none hover:border-slate-400 hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring"
    >
      <Ellipsis className="size-3" aria-hidden="true" />
    </button>
  );
  return (
    <span className="flex min-w-0 items-center gap-1">
      <span
        ref={text}
        className={expanded ? "min-w-0 break-words whitespace-pre-wrap" : "min-w-0 truncate"}
      >
        {children}
        {expanded && <> {toggle}</>}
      </span>
      {!expanded && clipped && toggle}
    </span>
  );
}

/** A value with no entries to open, written as the API's JSON form would read. */
function LeafValue({ value }: { value: InputValue }) {
  if (value === null) return <span className="text-muted-foreground">null</span>;
  if (typeof value === "string")
    return value === "" ? '""' : <OneLine breaks={value.includes("\n")}>{value}</OneLine>;
  if (typeof value !== "object") return String(value);
  if (isList(value)) return <span className="text-muted-foreground">[]</span>;
  if (isNumericalArray(value)) {
    const array = readNumericalArray(value);
    return (
      <OneLine breaks={false}>
        <span className="text-muted-foreground">
          {array.dtype}[{array.shape.join(", ")}]{" "}
        </span>
        {Array.from(array.values, String).join(", ")}
      </OneLine>
    );
  }
  return <span className="text-muted-foreground">{"{}"}</span>;
}

/** The timeline node that published an input, drawn as the timeline draws it. */
function SourceChip({ call, sources }: { call: TimelineRevision; sources: InputSources }) {
  const style = ACTION_STYLE[call.record.attempt.action];
  return (
    <Link
      href={sources.hrefAt(call.record.seq)}
      prefetch={false}
      aria-label={timelineTickLabel(call)}
      title={timelineTickLabel(call)}
      onNavigate={(event) => {
        event.preventDefault();
        sources.select(call.record.seq);
      }}
      className="inline-flex flex-none items-center gap-1 rounded-md border bg-card px-1.5 py-0.5 align-middle text-foreground shadow-xs outline-none hover:border-slate-400 focus-visible:ring-2 focus-visible:ring-ring"
    >
      <ActionMark icon={style.icon} color={style.color} size={12} />
      {call.commit_id.slice(0, 7)}
    </Link>
  );
}

function SourceChips({
  calls,
  sources,
}: {
  calls: readonly TimelineRevision[];
  sources: InputSources;
}) {
  return (
    <span className="inline-flex flex-wrap gap-1 align-middle">
      {calls.map((call) => (
        <SourceChip key={call.record.seq} call={call} sources={sources} />
      ))}
    </span>
  );
}

/**
 * A reference shows only the timeline nodes that published it; a structured reference, such as
 * a data selection's replicates, starts closed beside them.
 */
function InputGroup({
  name,
  value,
  entries,
  calls,
  sources,
  depth,
}: {
  name: string;
  value: InputList | InputRecord;
  entries: readonly InputEntry[];
  calls: readonly TimelineRevision[];
  sources: InputSources;
  depth: number;
}) {
  const [open, setOpen] = useState(
    () => depth > 0 && calls.length === 0 && JSON.stringify(value).length <= OPEN_UP_TO_CHARACTERS,
  );
  return (
    <details open={open} onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary className="cursor-pointer truncate rounded-sm outline-none focus-visible:ring-2 focus-visible:ring-ring">
        <span className="text-muted-foreground">{name}</span>
        {calls.length > 0 ? (
          <span className="ml-1.5">
            <SourceChips calls={calls} sources={sources} />
          </span>
        ) : (
          !open && (
            <span className="ml-2 text-muted-foreground/70">
              {isList(value)
                ? `${entries.length} ${entries.length === 1 ? "item" : "items"}`
                : entries.map(([key]) => key).join(", ")}
            </span>
          )
        )}
      </summary>
      {open && (
        <div className="mt-1 ml-1 border-l pl-3">
          <InputFields entries={entries} sources={sources} depth={depth + 1} />
        </div>
      )}
    </details>
  );
}

function InputField({
  name,
  value,
  sources,
  depth,
}: {
  name: string;
  value: InputValue;
  sources: InputSources;
  depth: number;
}) {
  const group = groupOf(value);
  const calls = sources.calls.get(name) ?? [];
  if (group !== null)
    return (
      <InputGroup
        name={name}
        value={group}
        entries={entriesOf(group)}
        calls={calls}
        sources={sources}
        depth={depth}
      />
    );
  return (
    <div className="flex min-w-0 items-baseline gap-2">
      <span className="flex-none text-muted-foreground">{name}</span>
      {calls.length > 0 ? (
        <SourceChips calls={calls} sources={sources} />
      ) : (
        <span className="min-w-0 flex-1">
          <LeafValue value={value} />
        </span>
      )}
    </div>
  );
}

function InputFields({
  entries,
  sources,
  depth,
}: {
  entries: readonly InputEntry[];
  sources: InputSources;
  depth: number;
}) {
  return (
    <div className="space-y-1">
      {entries.map(([name, value]) => (
        <InputField key={name} name={name} value={value} sources={sources} depth={depth} />
      ))}
    </div>
  );
}

/**
 * A call's request, keyed as the API names it. Reasoning, the field every action records, stays
 * pinned above the action's own arguments, which fill the rest. References to earlier calls
 * show the timeline nodes that published them.
 */
export function ActionInput({
  request,
  sources,
}: {
  request: TimelineRevision["record"]["attempt"]["request"];
  sources: InputSources;
}) {
  if (request === null)
    return (
      <RecordSection title="Input" aria-label="Action input">
        <Hint>This attempt has no retained call arguments.</Hint>
      </RecordSection>
    );
  return (
    <RecordSection
      title="Input"
      aria-label="Action input"
      pinned={
        <div className="max-h-[8lh] overflow-y-auto font-mono text-[10px]">
          <InputField name="reasoning" value={request.reasoning} sources={sources} depth={0} />
        </div>
      }
    >
      <div className="font-mono text-[10px]">
        <InputFields entries={entriesOf(request.input)} sources={sources} depth={0} />
      </div>
    </RecordSection>
  );
}
