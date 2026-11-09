"use client";

import { modelReference, producingCall } from "./call-dependencies";

import type { ActionSuccess, ModelSnapshot, TimelineRevision } from "@grassmoth/api-types";

import { useEffect, useMemo, useRef, useState } from "react";
import type { ModelComparison } from "@/lib/dag/comparison-overlay";
import { useModelDiff } from "@/lib/hooks/use-model-diff";
import { indexModel } from "./entities";
import { latestSeq } from "./journal";
import type { ScopeContext } from "./scope";
import type { EntitySelection } from "./selection";
import type { TimelineComparison } from "./timeline-comparison";

export type SnapshotReader = (commitId: string | undefined) => {
  data: ModelSnapshot | undefined;
  error: Error | null;
  isPlaceholderData?: boolean;
  result?: ActionSuccess | undefined;
};

export function useWorkbenchSnapshots(
  attempts: readonly TimelineRevision[],
  useSnapshot: SnapshotReader,
  requestedSeq: number | null,
) {
  const latest = latestSeq(attempts);
  const playhead = requestedSeq ?? latest;
  const record = attempts.find((item) => item.record.seq === playhead);
  // Applied leaves own their results; failed attempts inspect their unchanged parent state.
  const commitId = record
    ? record.record.attempt.outcome.status === "applied" && record.record.attempt.request !== null
      ? record.commit_id
      : modelReference(record, attempts)
    : attempts.find((entry) => entry.record.seq === latest)?.commit_id;
  const selected = useSnapshot(commitId);
  const hasSelectedState = producingCall(attempts, commitId) !== undefined;
  const hasCurrentState = attempts.some(
    (entry) =>
      entry.record.seq === latest &&
      entry.record.attempt.outcome.status === "applied" &&
      entry.record.attempt.request !== null &&
      entry.record.attempt.action !== "data_diff" &&
      entry.record.attempt.action !== "model_diff",
  );
  return { selected, focusSeq: playhead, hasSelectedState, hasCurrentState };
}

interface WorkbenchOptions {
  useSnapshot: SnapshotReader;
  workspaceId: string;
  attempts: readonly TimelineRevision[];
  modelSnapshot: ModelSnapshot;
  focusSeq: number;
  navigationKey: string;
  result: ActionSuccess | undefined;
  viewAt: (seq: number | null) => void;
}

/** Coordinate scoped details, chat, version navigation and comparison previews. */
export function useWorkbench({
  useSnapshot,
  workspaceId,
  attempts,
  modelSnapshot,
  focusSeq,
  navigationKey,
  result,
  viewAt,
}: WorkbenchOptions) {
  const [selection, select] = useState<EntitySelection | null>(null);
  const [comparison, setComparison] = useState<
    (TimelineComparison & { readonly pinned: boolean }) | null
  >(null);
  const [previousNavigation, setPreviousNavigation] = useState(navigationKey);
  if (previousNavigation !== navigationKey) {
    setPreviousNavigation(navigationKey);
    select(null);
    setComparison(null);
  }
  const previewTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(
    () => () => {
      if (previewTimer.current) clearTimeout(previewTimer.current);
    },
    [navigationKey],
  );
  const entities = useMemo(() => indexModel(modelSnapshot.dynamical_model_spec), [modelSnapshot]);
  const ticks = attempts;
  const latest = latestSeq(attempts);
  const playhead = focusSeq;
  const activeComparison = comparison?.before.record.seq === playhead ? comparison : null;
  const comparedCommit = modelReference(activeComparison?.after, attempts);
  const selectedCall = attempts.find((record) => record.record.seq === playhead);
  const selectedCommit = modelReference(selectedCall, attempts);
  const compared = useModelDiff(
    workspaceId,
    selectedCommit ?? null,
    comparedCommit ?? null,
    attempts,
  );
  const request = selectedCall?.record.attempt.request;
  const recorded =
    request?.action === "model_diff" && result?.action === "model_diff"
      ? { input: request.input, report: result.body }
      : null;
  const beforeView = useSnapshot(
    activeComparison ? selectedCommit : (recorded?.input.before_ref ?? modelSnapshot.commit_id),
  );
  const afterView = useSnapshot(
    activeComparison ? comparedCommit : (recorded?.input.after_ref ?? modelSnapshot.commit_id),
  );
  const report = activeComparison ? compared.data : recorded?.report;
  const modelComparison: ModelComparison | null =
    report && beforeView.data && afterView.data
      ? {
          ...report,
          beforeDynamicalModelSpec: beforeView.data.dynamical_model_spec,
          afterDynamicalModelSpec: afterView.data.dynamical_model_spec,
        }
      : null;
  const comparisonError = beforeView.error ?? afterView.error;
  const retainPreview = () => {
    if (previewTimer.current) clearTimeout(previewTimer.current);
  };
  const endPreview = () => {
    retainPreview();
    previewTimer.current = setTimeout(
      () => setComparison((value) => (value?.pinned ? value : null)),
      350,
    );
  };
  const previewComparison = (pair: TimelineComparison, pinned = false) => {
    retainPreview();
    setComparison((value) =>
      !pinned && value?.pinned && value.before.record.seq === playhead
        ? value
        : { ...pair, pinned },
    );
  };
  const dismissComparison = () => {
    retainPreview();
    setComparison(null);
  };
  const selectVersion = (seq: number | null) => {
    dismissComparison();
    viewAt(seq);
  };
  const question = modelSnapshot.dynamical_model_spec?.question;
  const simulationResult = modelSnapshot.simulation;
  const context: ScopeContext = {
    modelSnapshot,
    entities,
    select,
    ticks,
    dataDiff: null,
    result,
  };

  const toggleComparison = () => {
    if (activeComparison) setComparison({ ...activeComparison, pinned: !activeComparison.pinned });
  };

  return {
    selection,
    ticks,
    latest,
    playhead,
    activeComparison,
    compared,
    modelComparison,
    comparisonError,
    retainPreview,
    endPreview,
    previewComparison,
    dismissComparison,
    toggleComparison,
    selectVersion,
    question,
    simulationResult,
    select,
    context,
  };
}
