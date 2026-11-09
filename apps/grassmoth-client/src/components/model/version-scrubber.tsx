"use client";

import type { RecordDependency, TimelineRevision } from "@grassmoth/api-types";
import { ChevronDown, ChevronUp } from "lucide-react";
import Link from "next/link";
import { Fragment, type KeyboardEvent, useEffect, useId, useMemo, useRef, useState } from "react";
import { attemptError } from "@/lib/model-asset/journal";
import { type RevisionTimelineNode, revisionTimeline } from "@/lib/model-asset/revision-timeline";
import { type TimelineComparison, timelineComparison } from "@/lib/model-asset/timeline-comparison";
import type { TimelineNavigation } from "@/lib/model-asset/timeline-location";
import {
  NODE_HEIGHT,
  NODE_WIDTH,
  PITCH,
  timelineLayout,
  timelineLinkPath,
  timelinePoint,
  timelineTickLabel,
} from "@/lib/model-asset/timeline-presentation";
import { cn } from "@/lib/utils";
import { ACTION_STYLE, ActionMark } from "./action-mark";
import { PinToggle } from "./pin-toggle";

const HOVER_INTENT_MS = 300;
/** Sweeping across nodes must not move the contextual Compare control. */
const COMPARE_INTENT_MS = 120;
const COLLAPSED_TRACK = NODE_HEIGHT / 2 + 6;
const COLLAPSED_HEIGHT = COLLAPSED_TRACK + NODE_HEIGHT / 2 + 20;

/** The column a navigation key moves focus to, or null for keys the timeline leaves alone. */
function keyTarget(key: string, column: number, columns: number): number | null {
  if (key === "ArrowRight") return column + 1;
  if (key === "ArrowLeft") return column - 1;
  if (key === "Home") return 0;
  if (key === "End") return columns - 1;
  return null;
}

/** A node's name adds whether its attempt failed and whether it is the latest action. */
function nodeLabel(node: RevisionTimelineNode, latest: number): string {
  return `${timelineTickLabel(node.tick)}${node.failed ? " · failed" : ""}${node.tick.record.seq === latest ? " · latest" : ""}`;
}

interface ComparisonControls {
  readonly comparedSeq: number | null;
  readonly onPreview: (comparison: TimelineComparison) => void;
  readonly onEndPreview: () => void;
  readonly onRetainPreview: () => void;
  readonly onKeep: (comparison: TimelineComparison) => void;
}

/**
 * The study's named actions in execution order, in lanes by what they produce.
 * Links follow the served dependencies: each action connects to the earlier actions whose outputs
 * its request named, and dotted links lead to failed attempts. A failed attempt hangs
 * off its lane's track, since nothing can depend on it. Compare appears beneath the hovered or
 * focused node, keeping actions beside the node they apply to. Collapsing flattens the same
 * actions into one sequence and hides dependency links and lane labels.
 */
export function VersionScrubber({
  ticks,
  dependencies,
  playhead,
  latest,
  navigation,
  onPlayhead,
  compare,
}: {
  readonly ticks: readonly TimelineRevision[];
  readonly dependencies: readonly RecordDependency[];
  readonly playhead: number;
  readonly latest: number;
  readonly navigation: TimelineNavigation;
  readonly onPlayhead: (seq: number | null) => void;
  /** Absent while the viewed version has no model state, which leaves nothing to compare. */
  readonly compare?: ComparisonControls;
}) {
  const timelineId = useId();
  const [collapsed, setCollapsed] = useState(false);
  const viewport = useRef<HTMLDivElement>(null);
  const links = useRef(new Map<number, HTMLAnchorElement>());
  const timeline = useMemo(() => revisionTimeline(ticks, dependencies), [ticks, dependencies]);
  const layout = useMemo(() => timelineLayout(timeline), [timeline]);
  const height = collapsed ? COLLAPSED_HEIGHT : layout.height;
  // Compare follows the pointer after it settles, or keyboard focus; the column highlight
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
  const tabStop = selectedNode ?? timeline.nodes.at(-1);
  const offersComparisons = compare !== undefined;
  const comparisons = useMemo(
    () =>
      new Map(
        timeline.nodes.flatMap((node) => {
          const comparison = offersComparisons
            ? timelineComparison(selectedNode?.tick, node.tick)
            : null;
          return comparison ? [[node.tick.record.seq, comparison] as const] : [];
        }),
      ),
    [timeline, selectedNode, offersComparisons],
  );
  const hasComparisons = comparisons.size > 0;
  // A comparison costs a backend diff: start it only once the pointer rests on a tick.
  const hoverIntent = useRef<ReturnType<typeof setTimeout> | null>(null);
  const compareIntent = useRef<ReturnType<typeof setTimeout> | null>(null);
  const cancelIntents = () => {
    if (hoverIntent.current) clearTimeout(hoverIntent.current);
    if (compareIntent.current) clearTimeout(compareIntent.current);
    hoverIntent.current = null;
    compareIntent.current = null;
  };
  useEffect(() => cancelIntents, [playhead, navigation.seq]);

  useEffect(() => {
    const frame = viewport.current;
    const node = links.current.get(playhead);
    if (!frame || !node) return;
    const left = node.offsetLeft;
    if (left < frame.scrollLeft || left + node.offsetWidth > frame.scrollLeft + frame.clientWidth) {
      frame.scrollLeft = Math.max(0, left - frame.clientWidth / 2 + node.offsetWidth / 2);
    }
  }, [playhead, selectedNode?.column, collapsed]);

  const moveFocus = (event: KeyboardEvent<HTMLElement>, column: number) => {
    const target = keyTarget(event.key, column, timeline.nodes.length);
    if (target === null) return;
    event.preventDefault();
    const node = timeline.nodes[target];
    if (node) links.current.get(node.tick.record.seq)?.focus();
  };

  if (timeline.nodes.length === 0) {
    return (
      <nav
        aria-label="Action history"
        className="flex flex-none items-center gap-4 border-b bg-card px-5 py-2.5 text-xs"
      >
        <p className="text-muted-foreground">No actions yet.</p>
      </nav>
    );
  }

  return (
    <nav
      aria-label="Action history"
      className={cn(
        "grid flex-none border-b bg-card",
        collapsed ? "grid-cols-[40px_minmax(0,1fr)]" : "grid-cols-[96px_minmax(0,1fr)]",
      )}
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
      <div
        className="relative col-start-1 row-start-1 border-r border-border/60 bg-muted/20"
        style={{ height }}
      >
        <button
          type="button"
          aria-label={collapsed ? "Expand timeline" : "Collapse timeline"}
          aria-expanded={!collapsed}
          aria-controls={timelineId}
          title={collapsed ? "Expand timeline" : "Collapse timeline"}
          className="absolute top-1.5 left-1.5 flex size-7 cursor-pointer items-center justify-center rounded text-muted-foreground hover:bg-muted hover:text-foreground focus-visible:outline-ring"
          onClick={() => {
            cancelIntents();
            setHoverSeq(null);
            setFocusSeq(null);
            compare?.onEndPreview();
            setCollapsed((value) => !value);
          }}
        >
          {collapsed ? (
            <ChevronDown size={16} aria-hidden="true" />
          ) : (
            <ChevronUp size={16} aria-hidden="true" />
          )}
        </button>
        {!collapsed &&
          layout.lanes.map((lane) => (
            <span
              key={lane.name}
              className="absolute right-3 text-[11px] leading-4 font-medium text-muted-foreground"
              style={{ top: lane.track - 8 }}
            >
              {lane.name}
            </span>
          ))}
      </div>
      <div
        id={timelineId}
        ref={viewport}
        className="col-start-2 row-start-1 min-w-0 overflow-x-auto overflow-y-hidden"
      >
        <div className="relative min-w-full" style={{ width: layout.width, height }}>
          {[selectedNode, hoverNode].map(
            (node) =>
              node && (
                <span
                  key={node.tick.record.seq}
                  aria-hidden="true"
                  className={cn(
                    "absolute inset-y-1 rounded-md",
                    node === selectedNode ? "bg-muted/70" : "bg-muted/35",
                  )}
                  style={{ left: timelinePoint(layout, node).x - PITCH / 2 + 4, width: PITCH - 8 }}
                />
              ),
          )}
          <svg
            aria-hidden="true"
            width="100%"
            height={height}
            className="pointer-events-none absolute inset-0 overflow-visible"
          >
            {collapsed ? (
              <line
                x1={0}
                x2="100%"
                y1={COLLAPSED_TRACK}
                y2={COLLAPSED_TRACK}
                className="stroke-slate-300"
                strokeWidth={1.25}
              />
            ) : (
              <>
                {layout.lanes.map((lane) => (
                  <line
                    key={lane.name}
                    x1={0}
                    x2="100%"
                    y1={lane.track}
                    y2={lane.track}
                    className="stroke-border"
                    strokeDasharray="2 5"
                  />
                ))}
                {timeline.links
                  .map((link) => ({
                    link,
                    touches: link.from === selectedNode || link.to === selectedNode,
                  }))
                  .toSorted((a, b) => Number(a.touches) - Number(b.touches))
                  .map(({ link, touches }) => (
                    <path
                      key={`${link.from.tick.record.seq}:${link.to.tick.record.seq}:${link.argument}`}
                      data-argument={link.argument}
                      d={timelineLinkPath(layout, link)}
                      fill="none"
                      className={touches ? "stroke-slate-500" : "stroke-slate-300"}
                      strokeWidth={touches ? 1.75 : 1.25}
                      strokeDasharray={link.to.failed ? "2 3" : undefined}
                    />
                  ))}
              </>
            )}
          </svg>
          {timeline.nodes.map((node) => {
            const seq = node.tick.record.seq;
            const point = timelinePoint(layout, node);
            const track = collapsed ? COLLAPSED_TRACK : point.y;
            const style = ACTION_STYLE[node.tick.record.attempt.action];
            const current = node === selectedNode;
            const pinned = navigation.seq === seq;
            const compared = seq === compare?.comparedSeq;
            const comparison = comparisons.get(seq);
            const comparable = comparison !== undefined;
            const label = nodeLabel(node, latest);
            const error = attemptError(node.tick.record.attempt.outcome);
            return (
              <Fragment key={seq}>
                <Link
                  ref={(element) => {
                    if (element) links.current.set(seq, element);
                    else links.current.delete(seq);
                  }}
                  href={navigation.hrefAt(seq)}
                  prefetch={false}
                  tabIndex={node === tabStop ? 0 : -1}
                  aria-label={label}
                  aria-current={current ? "step" : undefined}
                  aria-describedby={
                    comparable ? "model-version-comparison-instructions" : undefined
                  }
                  aria-expanded={comparable ? compared : undefined}
                  aria-controls={compared ? "model-comparison-preview" : undefined}
                  data-compared={compared || undefined}
                  title={`${label}\n${new Date(node.tick.record.ts).toLocaleString()}${error ? `\n${error}` : ""}`}
                  className={cn(
                    "absolute flex cursor-pointer items-center gap-2 rounded-md border bg-card pr-8 pl-2 text-left whitespace-nowrap shadow-xs transition-[border-color,background-color,box-shadow] outline-none hover:border-slate-400 focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
                    current && "border-slate-600 ring-1 ring-slate-600",
                    compared && "border-amber-600 ring-1 ring-amber-600",
                    node.failed && "border-red-300 bg-red-50",
                  )}
                  style={{
                    left: point.x - NODE_WIDTH / 2,
                    top: track - NODE_HEIGHT / 2,
                    width: NODE_WIDTH,
                    height: NODE_HEIGHT,
                  }}
                  onPointerEnter={() => {
                    cancelIntents();
                    setHoverSeq(seq);
                    compareIntent.current = setTimeout(() => setFocusSeq(seq), COMPARE_INTENT_MS);
                    if (!comparison) {
                      compare?.onEndPreview();
                      return;
                    }
                    hoverIntent.current = setTimeout(
                      () => compare?.onPreview(comparison),
                      HOVER_INTENT_MS,
                    );
                  }}
                  onPointerLeave={() => {
                    cancelIntents();
                    setHoverSeq(null);
                    compare?.onEndPreview();
                  }}
                  onFocus={() => {
                    setFocusSeq(seq);
                    if (comparison) compare?.onPreview(comparison);
                  }}
                  onBlur={compare?.onEndPreview}
                  onNavigate={(event) => {
                    event.preventDefault();
                    cancelIntents();
                    onPlayhead(seq);
                  }}
                  onKeyDown={(event) => moveFocus(event, node.column)}
                >
                  <ActionMark icon={style.icon} color={style.color} failed={node.failed} />
                  <span className={cn("font-mono text-[11px]", current && "font-semibold")}>
                    {node.tick.commit_id.slice(0, 7)}
                  </span>
                </Link>
                <PinToggle
                  tick={node.tick}
                  pinned={pinned}
                  tabbable={node === focusNode || node === tabStop}
                  x={point.x}
                  y={track}
                  onEnter={() => {
                    cancelIntents();
                    compare?.onEndPreview();
                  }}
                  onToggle={() => {
                    cancelIntents();
                    onPlayhead(pinned ? null : seq);
                  }}
                  onKeyDown={(event) => moveFocus(event, node.column)}
                />
                <div
                  className="absolute flex h-4 items-center justify-center gap-2 text-[10px] leading-3"
                  style={{
                    left: point.x - NODE_WIDTH / 2,
                    top: track + NODE_HEIGHT / 2 + 4,
                    width: NODE_WIDTH,
                  }}
                >
                  {node === focusNode && selectedNode && comparison && (
                    <button
                      type="button"
                      aria-label={`Compare ${timelineTickLabel(selectedNode.tick)} with ${timelineTickLabel(node.tick)}`}
                      onPointerEnter={compare?.onRetainPreview}
                      onPointerLeave={compare?.onEndPreview}
                      onFocus={compare?.onRetainPreview}
                      onBlur={compare?.onEndPreview}
                      onClick={() => compare?.onKeep(comparison)}
                      className={cn(
                        "cursor-pointer rounded bg-card px-1 underline-offset-2 hover:underline focus-visible:outline-ring",
                        compared ? "text-amber-900" : "text-muted-foreground",
                      )}
                    >
                      Compare
                    </button>
                  )}
                </div>
              </Fragment>
            );
          })}
        </div>
      </div>
      <ul
        aria-label="Timeline symbol legend"
        className="col-span-2 row-start-2 flex flex-wrap items-center gap-x-5 gap-y-2 border-t border-border/40 px-4 py-2 text-[11px] text-muted-foreground"
      >
        {Object.entries(ACTION_STYLE).map(([action, style]) => (
          <li key={action} className="flex items-center gap-1.5">
            <ActionMark icon={style.icon} color={style.color} />
            <span className="font-mono">{action}</span>
          </li>
        ))}
      </ul>
    </nav>
  );
}
