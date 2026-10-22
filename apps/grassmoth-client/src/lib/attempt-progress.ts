import type {
  ExtractionSnapshotEvent,
  ExtractionWorkerEvent,
  ProgressEvent,
  StepEvent,
} from "@grassmoth/api-types";

/**
 * What one running attempt has reported so far. Each field holds the latest value an event
 * carried, and the cursor pages forward through that attempt's retained stream.
 */
export interface AttemptProgressView {
  cursor: string | null;
  extraction: StepEvent | null;
  /** The latest absolute worker counts; a later snapshot restores them after older events expire. */
  snapshot: ExtractionSnapshotEvent | null;
  workers: Record<number, ExtractionWorkerEvent>;
}

export const EMPTY_ATTEMPT_PROGRESS: AttemptProgressView = {
  cursor: null,
  extraction: null,
  snapshot: null,
  workers: {},
};

/** Events replace the values they carry, so a duplicate or a re-read changes nothing. */
export function applyProgressEvents(
  view: AttemptProgressView,
  events: readonly ProgressEvent[],
): AttemptProgressView {
  return events.reduce<AttemptProgressView>((next, event) => {
    const { cursor } = event;
    switch (event.event) {
      case "grassmoth.step":
        return { ...next, cursor, extraction: event };
      case "grassmoth.extraction.snapshot":
        return { ...next, cursor, snapshot: event };
      case "grassmoth.extraction.worker":
        return { ...next, cursor, workers: { ...next.workers, [event.worker_id]: event } };
      case "grassmoth.extraction.plan":
        // The snapshot emitted with the plan carries its totals.
        return { ...next, cursor };
      default:
        throw new Error(`Unhandled progress event: ${event satisfies never}`);
    }
  }, view);
}
