"use client";

import { attemptError } from "@/lib/model-asset/journal";

import type { RecordDependency } from "@grassmoth/api-types";
import { type KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";
import type { TimelineRevision } from "@grassmoth/api-types";
import { revisionTimeline, type RevisionTimelineNode } from "@/lib/model-asset/revision-timeline";
import {
  ACTION_STYLE,
  type ActionGlyph,
  FAILED_COLOR,
  FAILED_MARK,
  MARK,
  PITCH,
  timelineLayout,
  timelineLinkPath,
  timelinePoint,
  timelineTickLabel,
} from "@/lib/model-asset/timeline-presentation";
import { cn } from "@/lib/utils";

const HOVER_INTENT_MS = 300;
/** Sweeping across ticks on the way to the readout must not change what it shows. */
const READOUT_INTENT_MS = 120;
const COMPARED_COLOR = "#b45309";

/** One action's mark on the rail; a failed attempt takes the same slot as a cross. */
function ActionMark({
  glyph,
  color,
  failed,
  size,
}: {
  glyph: ActionGlyph;
  color: string;
  failed: boolean;
  size: number;
}) {
  return (
    <svg
      viewBox="0 0 16 16"
      width={size}
      height={size}
      aria-hidden="true"
      className="block flex-none"
    >
      <circle cx={8} cy={8} r={8} className="fill-card" />
      {failed ? (
        <path
          d="M4.5 4.5 11.5 11.5M11.5 4.5 4.5 11.5"
          stroke={FAILED_COLOR}
          strokeWidth={2.8}
          strokeLinecap="round"
        />
      ) : glyph === "dot" ? (
        <circle cx={8} cy={8} r={6} fill={color} />
      ) : glyph === "diamond" ? (
        <rect x={3} y={3} width={10} height={10} rx={1.5} transform="rotate(45 8 8)" fill={color} />
      ) : glyph === "ring" ? (
        <circle cx={8} cy={8} r={4.75} fill="none" stroke={color} strokeWidth={3.5} />
      ) : (
        <path d="M4.5 2.75 13.25 8 4.5 13.25Z" fill={color} stroke={color} strokeLinejoin="round" />
      )}
    </svg>
  );
}

/**
 * The study's actions in execution order, one narrow column each, in lanes by what they produce.
 * Links follow the served dependencies: each action connects to the earlier actions whose outputs
 * its request named, and dotted links mark outputs only its checks read. A failed attempt hangs
 * off its lane's track, since nothing can depend on it. The readout names the action under the
 * pointer, else the viewed one, and holds it while the pointer travels over to Compare.
 */
export function VersionScrubber({
  ticks,
  dependencies,
  playhead,
  latest,
  comparedSeq,
  onPlayhead,
  onPreviewComparison,
  onEndPreview,
  onRetainPreview,
  onKeepComparison,
}: {
  ticks: readonly TimelineRevision[];
  dependencies: readonly RecordDependency[];
  playhead: number;
  latest: number;
  comparedSeq: number | null;
  onPlayhead: (seq: number) => void;
  onPreviewComparison: (seq: number) => void;
  onEndPreview: () => void;
  onRetainPreview: () => void;
  onKeepComparison: (seq: number) => void;
}) {
  const viewport = useRef<HTMLDivElement>(null);
  const buttons = useRef(new Map<number, HTMLButtonElement>());
  const timeline = useMemo(() => revisionTimeline(ticks, dependencies), [ticks, dependencies]);
  const layout = useMemo(() => timelineLayout(timeline), [timeline]);
  // The readout follows the pointer after it settles, or keyboard focus; the column highlight
  // follows the pointer at once.
  const [focusSeq, setFocusSeq] = useState<number | null>(null);
  const [hoverSeq, setHoverSeq] = useState<number | null>(null);
  const selectedNode = timeline.nodes.find((node) => node.tick.record.seq === playhead);
  const focusNode =
    timeline.nodes.find((node) => node.tick.record.seq === focusSeq) ?? selectedNode;
  const hoverNode =
    hoverSeq === playhead
      ? undefined
      : timeline.nodes.find((node) => node.tick.record.seq === hoverSeq);
  const latestNode = timeline.nodes.find((node) => node.tick.record.seq === latest);
  const tabStop = selectedNode ?? timeline.nodes.at(-1);
  const canCompare = (node: RevisionTimelineNode) =>
    !node.failed &&
    node.tick.record.attempt.action !== "data_diff" &&
    node.tick.record.attempt.action !== "model_diff" &&
    node.tick.record.seq !== playhead;
  const hasComparisons = timeline.nodes.some(canCompare);
  // A comparison costs a backend diff: start it only once the pointer rests on a tick.
  const hoverIntent = useRef<ReturnType<typeof setTimeout> | null>(null);
  const readoutIntent = useRef<ReturnType<typeof setTimeout> | null>(null);
  const cancelIntents = () => {
    if (hoverIntent.current) clearTimeout(hoverIntent.current);
    if (readoutIntent.current) clearTimeout(readoutIntent.current);
    hoverIntent.current = null;
    readoutIntent.current = null;
  };
  useEffect(() => cancelIntents, []);

  useEffect(() => {
    const frame = viewport.current;
    const node = buttons.current.get(playhead);
    if (!frame || !node) return;
    const left = node.offsetLeft;
    if (left < frame.scrollLeft || left + node.offsetWidth > frame.scrollLeft + frame.clientWidth) {
      frame.scrollLeft = Math.max(0, left - frame.clientWidth / 2 + node.offsetWidth / 2);
    }
  }, [playhead, selectedNode?.column]);

  const moveFocus = (event: KeyboardEvent<HTMLButtonElement>, column: number) => {
    const target =
      event.key === "ArrowRight"
        ? column + 1
        : event.key === "ArrowLeft"
          ? column - 1
          : event.key === "Home"
            ? 0
            : event.key === "End"
              ? timeline.nodes.length - 1
              : null;
    if (target === null) return;
    event.preventDefault();
    const node = timeline.nodes[target];
    if (node) buttons.current.get(node.tick.record.seq)?.focus();
  };

  if (timeline.nodes.length === 0)
    return (
      <nav
        aria-label="Action history"
        className="flex flex-none items-center gap-4 border-b bg-card px-5 py-2.5 text-xs"
      >
        <span className="font-semibold">Timeline</span>
        <p className="text-muted-foreground">No actions yet.</p>
      </nav>
    );

  const focusError = focusNode ? attemptError(focusNode.tick.record.attempt.outcome) : null;
  return (
    <nav
      aria-label="Action history"
      className="flex flex-none flex-wrap border-b bg-card md:flex-nowrap"
      onPointerLeave={() => setFocusSeq(null)}
      onBlur={(event) => {
        const next = event.relatedTarget;
        if (!(next instanceof Node && event.currentTarget.contains(next))) setFocusSeq(null);
      }}
    >
      {selectedNode && (
        <span aria-live="polite" className="sr-only">
          Viewing {timelineTickLabel(selectedNode.tick)}.
        </span>
      )}
      {hasComparisons && selectedNode && (
        <span id="model-version-comparison-instructions" className="sr-only">
          Preview topology differences with {timelineTickLabel(selectedNode.tick)}. Tab to Compare
          to keep the comparison open.
        </span>
      )}
      <div className="relative w-38 flex-none" style={{ height: layout.height }}>
        <span
          className="absolute left-5 text-xs leading-4 font-semibold"
          style={{ top: (layout.height - 16) / 2 }}
        >
          Timeline
        </span>
        <span aria-hidden="true" className="absolute top-1.5 bottom-1.5 left-19 w-px bg-border" />
        {layout.lanes.map((lane) => (
          <span
            key={lane.name}
            className="absolute right-2 font-mono text-[9px] leading-2.5 text-muted-foreground"
            style={{ top: lane.track - 5 }}
          >
            {lane.name}
          </span>
        ))}
      </div>
      <div ref={viewport} className="min-w-0 flex-1 overflow-x-auto overflow-y-hidden">
        <div className="relative" style={{ width: layout.width, height: layout.height }}>
          {[selectedNode, hoverNode].map(
            (node) =>
              node && (
                <span
                  key={node.tick.record.seq}
                  aria-hidden="true"
                  className={cn(
                    "absolute top-0.5 bottom-0.5 rounded",
                    node === selectedNode ? "bg-foreground/6" : "bg-foreground/3",
                  )}
                  style={{ left: timelinePoint(layout, node).x - PITCH / 2, width: PITCH }}
                />
              ),
          )}
          <svg
            aria-hidden="true"
            width={layout.width}
            height={layout.height}
            className="pointer-events-none absolute inset-0 overflow-visible"
          >
            {layout.lanes.map((lane) => (
              <line
                key={lane.name}
                x1={0}
                x2={layout.width}
                y1={lane.track}
                y2={lane.track}
                className="stroke-border"
                strokeDasharray="1 3"
              />
            ))}
            {timeline.links
              .map((link) => ({
                link,
                touches: link.from === selectedNode || link.to === selectedNode,
              }))
              .sort((a, b) => Number(a.touches) - Number(b.touches))
              .map(({ link, touches }) => (
                <path
                  key={`${link.from.tick.record.seq}:${link.to.tick.record.seq}:${link.argument}`}
                  data-argument={link.argument}
                  d={timelineLinkPath(layout, link)}
                  fill="none"
                  className={touches ? "stroke-muted-foreground" : "stroke-border"}
                  strokeWidth={touches ? 1.5 : 1}
                  strokeDasharray={link.to.failed ? "2 3" : undefined}
                />
              ))}
          </svg>
          {latestNode && (
            <span
              aria-hidden="true"
              className="absolute size-1 rounded-full bg-muted-foreground"
              style={{ left: timelinePoint(layout, latestNode).x - 2, top: layout.height - 5 }}
            />
          )}
          {timeline.nodes.map((node) => {
            const seq = node.tick.record.seq;
            const point = timelinePoint(layout, node);
            const style = ACTION_STYLE[node.tick.record.attempt.action];
            const current = node === selectedNode;
            const compared = seq === comparedSeq;
            const comparable = canCompare(node);
            const size = node.failed ? FAILED_MARK : MARK;
            const label = `${timelineTickLabel(node.tick)}${node.failed ? " · failed" : ""}${seq === latest ? " · latest" : ""}`;
            const error = attemptError(node.tick.record.attempt.outcome);
            return (
              <button
                key={seq}
                ref={(element) => {
                  if (element) buttons.current.set(seq, element);
                  else buttons.current.delete(seq);
                }}
                type="button"
                tabIndex={node === tabStop ? 0 : -1}
                aria-label={label}
                aria-current={current ? "step" : undefined}
                aria-describedby={comparable ? "model-version-comparison-instructions" : undefined}
                aria-expanded={comparable ? compared : undefined}
                aria-controls={compared ? "model-comparison-preview" : undefined}
                data-compared={compared || undefined}
                title={`${label}\n${new Date(node.tick.record.ts).toLocaleString()}${error ? `\n${error}` : ""}`}
                className="absolute top-0 cursor-pointer rounded outline-none focus-visible:ring-2 focus-visible:ring-ring"
                style={{ left: point.x - PITCH / 2, width: PITCH, height: layout.height }}
                onPointerEnter={() => {
                  cancelIntents();
                  setHoverSeq(seq);
                  readoutIntent.current = setTimeout(() => setFocusSeq(seq), READOUT_INTENT_MS);
                  if (!comparable) return onEndPreview();
                  hoverIntent.current = setTimeout(() => onPreviewComparison(seq), HOVER_INTENT_MS);
                }}
                onPointerLeave={() => {
                  cancelIntents();
                  setHoverSeq(null);
                  onEndPreview();
                }}
                onFocus={() => {
                  setFocusSeq(seq);
                  if (comparable) onPreviewComparison(seq);
                }}
                onBlur={onEndPreview}
                onClick={() => {
                  cancelIntents();
                  onPlayhead(seq);
                }}
                onKeyDown={(event) => moveFocus(event, node.column)}
              >
                <span
                  className={cn(
                    "absolute grid place-items-center rounded-full",
                    (current || compared) && "outline-[1.5px] outline-offset-[1.5px]",
                  )}
                  style={{
                    left: (PITCH - size) / 2,
                    top: point.y - size / 2,
                    outlineColor: compared
                      ? COMPARED_COLOR
                      : node.failed
                        ? FAILED_COLOR
                        : style.color,
                  }}
                >
                  <ActionMark
                    glyph={style.glyph}
                    color={style.color}
                    failed={node.failed}
                    size={size}
                  />
                </span>
              </button>
            );
          })}
        </div>
      </div>
      <div
        className="flex w-full items-center gap-2.5 border-t px-3.5 py-1.5 md:w-80 md:flex-none md:border-t-0 md:border-l md:py-0 lg:w-95"
        onPointerEnter={onRetainPreview}
        onPointerLeave={onEndPreview}
        onFocus={onRetainPreview}
        onBlur={onEndPreview}
      >
        {focusNode && (
          <div className="flex min-w-0 flex-1 flex-col">
            <div className="flex min-w-0 items-center gap-1.5 whitespace-nowrap">
              <ActionMark
                glyph={ACTION_STYLE[focusNode.tick.record.attempt.action].glyph}
                color={ACTION_STYLE[focusNode.tick.record.attempt.action].color}
                failed={focusNode.failed}
                size={12}
              />
              <span
                className={cn(
                  "truncate text-xs leading-4",
                  focusNode === selectedNode ? "font-semibold" : "font-medium",
                  focusNode.tick.record.seq === comparedSeq
                    ? "text-amber-900"
                    : focusNode.failed && "text-muted-foreground",
                )}
              >
                {focusNode.tick.record.attempt.action}
              </span>
              <span
                className={cn(
                  "flex-none font-mono text-[10.5px] leading-4",
                  focusNode === selectedNode ? "text-foreground/70" : "text-muted-foreground",
                )}
              >
                {focusNode.tick.commit_id.slice(0, 7)}
              </span>
              {focusNode.tick.record.seq === latest && (
                <span
                  title="Latest action"
                  className="size-1.5 flex-none rounded-full bg-muted-foreground"
                />
              )}
            </div>
            <p className="truncate text-[10.5px] leading-3.5 text-muted-foreground">
              {focusNode === selectedNode
                ? "Viewing · "
                : focusNode.tick.record.seq === comparedSeq
                  ? "Comparing · "
                  : ""}
              {layout.lanes[focusNode.lane]?.name} ·{" "}
              {new Date(focusNode.tick.record.ts).toLocaleString()}
              {focusError && <span className="text-red-700"> · {focusError}</span>}
            </p>
          </div>
        )}
        {focusNode && selectedNode && canCompare(focusNode) && (
          <button
            type="button"
            aria-label={`Compare ${timelineTickLabel(selectedNode.tick)} with ${timelineTickLabel(focusNode.tick)}`}
            onClick={() => onKeepComparison(focusNode.tick.record.seq)}
            className={cn(
              "flex-none cursor-pointer rounded-md border px-2 py-0.5 text-xs hover:bg-muted",
              focusNode.tick.record.seq === comparedSeq &&
                "border-amber-500 text-amber-900 hover:bg-amber-50",
            )}
          >
            Compare
          </button>
        )}
        {playhead !== latest && (
          <button
            type="button"
            onClick={() => onPlayhead(latest)}
            className="flex-none cursor-pointer rounded-md border px-2 py-0.5 text-xs hover:bg-muted"
          >
            Return to latest
          </button>
        )}
      </div>
    </nav>
  );
}
