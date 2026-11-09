import type { TimelineRevision } from "@grassmoth/api-types";
import {
  type RevisionTimeline,
  type RevisionTimelineLink,
  type RevisionTimelineNode,
  TIMELINE_LANES,
} from "./revision-timeline";

/** Each action has room for its icon, short commit hash and pin toggle. */
export const PITCH = 148;
export const NODE_WIDTH = 124;
export const NODE_HEIGHT = 28;
/** Room before the first column and after the last. */
const PAD = 12;
/** Height of the first track from the top of the strip. */
const TOP = 20;
/** From a track to the corridor beside it, where links between lanes run. */
const CORRIDOR = 20;
/** From a lane's track to the side track its failed attempts hang on. */
const SIDE = 36;

/** Vertical positions in one lane: its track, its failures' side track and the corridors beside it. */
interface TimelineLanePlacement {
  readonly name: string;
  readonly track: number;
  readonly side: number;
  readonly above: number;
  readonly below: number;
}

interface TimelineLayout {
  readonly lanes: readonly TimelineLanePlacement[];
  readonly width: number;
  readonly height: number;
}

function laneTracks(
  failures: (typeof TIMELINE_LANES)[number]["failures"] | null,
  first: number,
): { track: number; side: number } {
  if (failures === "above") return { track: first + SIDE, side: first };
  if (failures === "below") return { track: first, side: first + SIDE };
  return { track: first, side: first };
}

/**
 * Stacks the lanes. A lane with a failed attempt gains a side track toward the lane its inputs
 * come from, so links reach the failure without crossing the track that later actions use.
 */
export function timelineLayout(timeline: RevisionTimeline): TimelineLayout {
  const failing = new Set(timeline.nodes.filter((node) => node.failed).map((node) => node.lane));
  const lanes: TimelineLanePlacement[] = [];
  let bottom = TOP - 2 * CORRIDOR;
  for (const [index, lane] of TIMELINE_LANES.entries()) {
    const above = bottom + CORRIDOR;
    const first = bottom + 2 * CORRIDOR;
    const failures = failing.has(index) ? lane.failures : null;
    const { track, side } = laneTracks(failures, first);
    bottom = Math.max(track, side);
    lanes.push({ name: lane.name, track, side, above, below: bottom + CORRIDOR });
  }
  return {
    lanes,
    width: 2 * PAD + Math.max(1, timeline.nodes.length) * PITCH,
    // Keep status labels beneath the cards inside the scrollable strip.
    height: bottom + NODE_HEIGHT / 2 + 20,
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
 * Connect the right edge of the producer to the left edge of its consumer. Cross-lane links
 * run through the space between rows, outside the cards. Failed calls have no outgoing links.
 */
export function timelineLinkPath(layout: TimelineLayout, link: RevisionTimelineLink) {
  const source = timelinePoint(layout, link.from);
  const target = timelinePoint(layout, link.to);
  const from = { x: source.x + NODE_WIDTH / 2, y: source.y };
  const to = { x: target.x - NODE_WIDTH / 2, y: target.y };
  if (link.from.lane === link.to.lane) {
    if (from.y === to.y) return `M ${from.x} ${from.y} H ${to.x}`;
    const bend = Math.min(10, to.x - from.x);
    return `M ${from.x} ${from.y} H ${to.x - bend} C ${to.x - bend / 2} ${from.y}, ${to.x - bend / 2} ${to.y}, ${to.x} ${to.y}`;
  }
  const lane = placement(layout, link.to);
  const corridor = link.from.lane > link.to.lane ? lane.below : lane.above;
  const bend = Math.min(6, (to.x - from.x) / 4);
  return `M ${from.x} ${from.y} C ${from.x + bend} ${from.y}, ${from.x + bend} ${corridor}, ${from.x + 2 * bend} ${corridor} H ${to.x - 2 * bend} C ${to.x - bend} ${corridor}, ${to.x - bend} ${to.y}, ${to.x} ${to.y}`;
}

export function timelineTickLabel(tick: TimelineRevision) {
  return `${tick.record.attempt.action} · ${tick.commit_id.slice(0, 7)}`;
}
