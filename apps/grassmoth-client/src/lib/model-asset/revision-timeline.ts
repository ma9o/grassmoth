import type { ActionId, RecordDependency } from "@grassmoth/api-types";
import type { TimelineRevision } from "@grassmoth/api-types";

/**
 * Lanes group actions by what they produce: comparisons read histories, data makes them, and the
 * model lane starts from the question every model is built to answer. A failed attempt produces
 * nothing, so it hangs off its lane on the side its inputs arrive from.
 */
export const TIMELINE_LANES: ReadonlyArray<{
  name: string;
  actions: readonly ActionId[];
  failures: "above" | "below";
}> = [
  { name: "Comparisons", actions: ["data_diff", "model_diff"], failures: "below" },
  { name: "Data", actions: ["prepare_data", "simulate"], failures: "below" },
  { name: "Model", actions: ["edit_question", "edit_model", "fit"], failures: "above" },
];

export interface RevisionTimelineNode {
  tick: TimelineRevision;
  column: number;
  lane: number;
  failed: boolean;
}

export interface RevisionTimelineLink {
  from: RevisionTimelineNode;
  to: RevisionTimelineNode;
  argument: string;
}

export interface RevisionTimeline {
  nodes: RevisionTimelineNode[];
  links: RevisionTimelineLink[];
}

/** Presentation only: execution order gives columns; the served dependencies give links. */
export function revisionTimeline(
  ticks: readonly TimelineRevision[],
  dependencies: readonly RecordDependency[],
): RevisionTimeline {
  const nodes: RevisionTimelineNode[] = ticks.map((tick, column) => ({
    tick,
    column,
    lane: TIMELINE_LANES.findIndex((lane) => lane.actions.includes(tick.record.attempt.action)),
    failed: tick.record.attempt.outcome.status !== "applied",
  }));
  const bySeq = new Map(nodes.map((node) => [node.tick.record.seq, node]));
  const links: RevisionTimelineLink[] = dependencies.flatMap((dependency) => {
    const from = bySeq.get(dependency.source_seq);
    const to = bySeq.get(dependency.seq);
    return from && to ? [{ from, to, argument: dependency.argument }] : [];
  });
  return { nodes, links };
}
