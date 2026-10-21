"use client";

import {
  createModelClient,
  type RecordDependency,
  type RunningAction,
  type TimelineRevision,
} from "@nof1-causal-lab/api-types";
import { useQuery, useQueryClient } from "@tanstack/react-query";

const client = createModelClient();
export function journalPollIntervalMs(active: boolean): number {
  return active ? 500 : 10_000;
}

/** The slim call log and all executing calls; scientific results are read by replay. */
export interface StudyJournal {
  seq: number;
  running: readonly RunningAction[];
  attempts: readonly TimelineRevision[];
  dependencies: readonly RecordDependency[];
}

export function getStudyJournalQueryKey(workspaceId: string) {
  return ["study", workspaceId, "journal"] as const;
}

export async function readStudyJournal(
  workspaceId: string,
  held?: StudyJournal,
  signal?: AbortSignal,
): Promise<StudyJournal> {
  const { data, error, response } = await client.GET("/api/studies/{workspace_id}/timeline", {
    params: { path: { workspace_id: workspaceId } },
    ...(signal === undefined ? {} : { signal }),
  });
  if (data === undefined)
    throw new Error(`Cannot read the journal (${response.status}): ${JSON.stringify(error)}`);
  const seq = Math.max(0, ...data.attempts.map((entry) => entry.record.seq));
  return {
    seq,
    running: data.running,
    attempts:
      held?.seq === seq && held.attempts.length === data.attempts.length
        ? held.attempts
        : data.attempts,
    dependencies: data.dependencies,
  };
}

export function useStudyJournal(workspaceId: string) {
  const queryClient = useQueryClient();
  return useQuery({
    queryKey: getStudyJournalQueryKey(workspaceId),
    queryFn: ({ signal }) =>
      readStudyJournal(
        workspaceId,
        queryClient.getQueryData<StudyJournal>(getStudyJournalQueryKey(workspaceId)),
        signal,
      ),
    refetchInterval: (query) =>
      journalPollIntervalMs(!query.state.data || query.state.data.running.length > 0),
    staleTime: 0,
    gcTime: 0,
    retry: false,
  });
}
