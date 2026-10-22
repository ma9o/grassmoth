import type { ActionId } from "@grassmoth/api-types";
import type { TimelineRevision } from "@grassmoth/api-types";
import {
  TIMELINE_LANES,
  type RevisionTimeline,
  type RevisionTimelineLink,
  type RevisionTimelineNode,
} from "./revision-timeline";

/** Horizontal pitch between consecutive actions; ticks carry no labels, so columns stay narrow. */
export const PITCH = 18;
/** Room before the first column and after the last. */
const PAD = 8;
/** Height of the first track from the top of the strip. */
const TOP = 8;
/** From a track to the corridor beside it, where links between lanes run. */
const CORRIDOR = 5;
/** From a lane's track to the side track its failed attempts hang on. */
const SIDE = 5;
/** Mark diameters; a failure is a dead end, so its mark is smaller. */
export const MARK = 10;
export const FAILED_MARK = 8;

export type ActionGlyph = "dot" | "diamond" | "ring" | "arrow";

/** One mark and colour per scientific action; the readout names the action. */
export const ACTION_STYLE: Record<ActionId, { color: string; glyph: ActionGlyph }> = {
  edit_question: { color: "#b45309", glyph: "diamond" },
  edit_model: { color: "#475569", glyph: "dot" },
  prepare_data: { color: "#0f766e", glyph: "diamond" },
  fit: { color: "#2563eb", glyph: "ring" },
  data_diff: { color: "#0369a1", glyph: "ring" },
  model_diff: { color: "#b45309", glyph: "ring" },
  simulate: { color: "#7c3aed", glyph: "arrow" },
};
export const FAILED_COLOR = "#dc2626";

/** Vertical positions in one lane: its track, its failures' side track and the corridors beside it. */
interface TimelineLanePlacement {
  name: string;
  track: number;
  side: number;
  above: number;
  below: number;
}

interface TimelineLayout {
  lanes: TimelineLanePlacement[];
  width: number;
  height: number;
}

/**
 * Stacks the lanes. A lane with a failed attempt gains a side track toward the lane its inputs
 * come from, so links reach the failure without crossing the track that later actions use.
 */
export function timelineLayout(timeline: RevisionTimeline): TimelineLayout {
  const failing = new Set(timeline.nodes.filter((node) => node.failed).map((node) => node.lane));
  const lanes: TimelineLanePlacement[] = [];
  let bottom = TOP - 2 * CORRIDOR;
  TIMELINE_LANES.forEach((lane, index) => {
    const above = bottom + CORRIDOR;
    const first = bottom + 2 * CORRIDOR;
    const failures = failing.has(index) ? lane.failures : null;
    const track = failures === "above" ? first + SIDE : first;
    const side = failures === "above" ? first : failures === "below" ? track + SIDE : track;
    bottom = Math.max(track, side);
    lanes.push({ name: lane.name, track, side, above, below: bottom + CORRIDOR });
  });
  return {
    lanes,
    width: 2 * PAD + Math.max(1, timeline.nodes.length) * PITCH,
    // Room under the lowest track for the latest-action marker.
    height: bottom + MARK / 2 + 6,
  };
}

function placement(layout: TimelineLayout, node: RevisionTimelineNode) {
  const lane = layout.lanes[node.lane];
  if (!lane) throw new Error(`${node.tick.record.attempt.action} has no timeline lane`);
  return lane;
}

export function timelinePoint(layout: TimelineLayout, node: RevisionTimelineNode) {
  const lane = placement(layout, node);
  return { x: PAD + node.column * PITCH + PITCH / 2, y: node.failed ? lane.side : lane.track };
}

/**
 * Within a lane, runs along its track and curves off it into a failure. Between lanes, leaves the
 * producer vertically, runs along the corridor beside the consumer's lane and enters the consumer
 * vertically. Producers never failed, and consumers always sit to the right of their producers.
 */
export function timelineLinkPath(layout: TimelineLayout, link: RevisionTimelineLink) {
  const from = timelinePoint(layout, link.from);
  const to = timelinePoint(layout, link.to);
  if (link.from.lane === link.to.lane) {
    if (from.y === to.y) return `M ${from.x} ${from.y} H ${to.x}`;
    const bend = Math.min(10, to.x - from.x);
    return `M ${from.x} ${from.y} H ${to.x - bend} C ${to.x - bend / 2} ${from.y}, ${to.x - bend / 2} ${to.y}, ${to.x} ${to.y}`;
  }
  const lane = placement(layout, link.to);
  const corridor = link.from.lane > link.to.lane ? lane.below : lane.above;
  const bend = Math.min(3, (to.x - from.x) / 2);
  return `M ${from.x} ${from.y} C ${from.x} ${corridor}, ${from.x} ${corridor}, ${from.x + bend} ${corridor} H ${to.x - bend} C ${to.x} ${corridor}, ${to.x} ${corridor}, ${to.x} ${to.y}`;
}

export function timelineTickLabel(tick: TimelineRevision) {
  return `${tick.record.attempt.action} · ${tick.commit_id.slice(0, 7)}`;
}
