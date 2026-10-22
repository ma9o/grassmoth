"use client";

import {
  type ActionSuccess,
  type ModelSnapshot,
  type RunningAction,
  type TimelineRevision,
} from "@grassmoth/api-types";
import Link from "next/link";
import { graphEntities } from "@/lib/dag/layered-model";
import { useEffect, useMemo, useRef } from "react";
import { LayeredCausalGraph } from "@/components/dag/layered/layered-causal-graph";
import { Button } from "@/components/ui/button";
import { useLLMTraceForAction } from "@/lib/hooks/use-llm-trace";
import { useDataComparison } from "@/lib/hooks/use-data-comparison";
import { useModelSnapshot } from "@/lib/hooks/use-model-snapshot";
import type { StudyJournal } from "@/lib/hooks/use-study-journal";
import { useSimulationPaths } from "@/lib/hooks/use-visuals";
import {
  useWorkbench,
  useWorkbenchSnapshots,
  type SnapshotReader,
} from "@/lib/model-asset/use-workbench";
import { ActionRecord, type ActionTraceState, type UseActionTrace } from "./action-record";
import { DetailsPane } from "./details-pane";
import { VersionScrubber } from "./version-scrubber";

export interface CausalModelAssetViewProps {
  workspaceId: string;
  question: string | undefined;
  useSnapshot: SnapshotReader;
  attempts: readonly TimelineRevision[];
  dependencies: StudyJournal["dependencies"];
  useActionTrace: UseActionTrace;
  running: readonly RunningAction[];
}

/**
 * The viewer answers two questions about the action selected on the timeline. The action record
 * on the right says what that action did and what came of it. The graph, with the details pane
 * below it, shows the state the action left: the graph one part at a glance, the details pane in
 * depth. The agent harness makes every change, so no pane offers writes.
 * A data_diff leaf records a comparison without changing that state: the graph and details
 * overlay its saved evidence on the parent version, and the record states what was compared.
 */
export function CausalModelAssetView(props: CausalModelAssetViewProps) {
  const { selected, viewAt, focusSeq, hasSelectedState, hasCurrentState } = useWorkbenchSnapshots(
    props.attempts,
    props.useSnapshot,
  );
  if (selected.error)
    return (
      <div role="alert" className="p-6">
        {selected.error.message}
      </div>
    );
  if (!hasSelectedState || !hasCurrentState) {
    const tick =
      props.attempts.find((entry) => entry.record.seq === focusSeq) ?? props.attempts.at(-1);
    return (
      <div className="flex min-h-screen flex-col gap-4 bg-muted/20">
        <header className="border-b bg-card p-4">
          <Link href="/" className="text-sm font-semibold">
            Grassmoth
          </Link>
          <p className="text-sm">{props.question ?? props.workspaceId}</p>
        </header>
        <VersionScrubber
          ticks={props.attempts}
          dependencies={props.dependencies}
          playhead={tick?.record.seq ?? 0}
          latest={0}
          comparedSeq={null}
          onPlayhead={viewAt}
          onPreviewComparison={() => {}}
          onEndPreview={() => {}}
          onRetainPreview={() => {}}
          onKeepComparison={() => {}}
        />
        <section className="mx-4 flex min-h-64 flex-col rounded-2xl border bg-card">
          <ActionRecord
            workspaceId={props.workspaceId}
            context={null}
            tick={tick}
            running={props.running}
            useActionTrace={props.useActionTrace}
          />
        </section>
      </div>
    );
  }
  if (!selected.data)
    return (
      <div role="status" className="p-6">
        Loading model version…
      </div>
    );
  return (
    <ModelRevision
      {...props}
      result={selected.result}
      modelSnapshot={selected.data}
      loadingRevision={selected.isPlaceholderData === true}
      viewAt={viewAt}
      focusSeq={focusSeq}
    />
  );
}

function ModelRevision({
  useSnapshot,
  workspaceId,
  question: initialQuestion,
  attempts,
  dependencies,
  useActionTrace,
  modelSnapshot,
  result,
  loadingRevision,
  viewAt,
  focusSeq,
  running,
}: CausalModelAssetViewProps & {
  modelSnapshot: ModelSnapshot;
  result: ActionSuccess | undefined;
  loadingRevision: boolean;
  viewAt: (seq: number | null) => void;
  focusSeq: number;
}) {
  const {
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
    context: versionContext,
  } = useWorkbench({
    useSnapshot,
    workspaceId,
    question: initialQuestion,
    attempts,
    modelSnapshot,
    focusSeq,
    result,
    viewAt,
  });
  const recordedPaths = useSimulationPaths(modelSnapshot);
  const tick = ticks.find((item) => item.record.seq === focusSeq);
  const dataReport =
    tick?.record.attempt.action === "data_diff" && tick.record.attempt.outcome.status === "applied"
      ? result?.action === "data_diff"
        ? result.body.report
        : null
      : null;
  const dataComparison = useDataComparison(workspaceId, dataReport, attempts);
  const dataDiff = dataComparison.data;
  const context = { ...versionContext, dataDiff };
  const recordedComparison = result?.action === "model_diff" ? result.body : null;
  // Nodes chart what the viewed version's action produced.
  const step = tick?.record.attempt.action ?? null;
  const comparisonPane = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (activeComparison?.pinned) {
      comparisonPane.current?.focus({ preventScroll: true });
      comparisonPane.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  }, [activeComparison?.pinned, activeComparison?.before, activeComparison?.after]);

  return (
    <div
      className="flex min-h-screen flex-col bg-muted/20 md:h-dvh md:overflow-hidden"
      onKeyDown={(event) => {
        if (event.key === "Escape" && activeComparison) {
          event.stopPropagation();
          dismissComparison();
        }
      }}
    >
      <header className="flex flex-none flex-wrap items-center gap-x-4 gap-y-2 border-b bg-card px-4 py-3 sm:px-5">
        <Link href="/" className="text-sm font-semibold tracking-tight">
          Grassmoth
        </Link>
        <span className="rounded-md border px-2 py-1 font-mono text-[10px] tracking-widest text-muted-foreground">
          {workspaceId}
        </span>
        <div className="w-full text-sm leading-relaxed">{question ?? "Untitled causal model"}</div>
      </header>
      <VersionScrubber
        ticks={ticks}
        dependencies={dependencies}
        playhead={focusSeq}
        latest={latest}
        comparedSeq={activeComparison?.after ?? null}
        onPlayhead={selectVersion}
        onPreviewComparison={(seq) => previewComparison(seq)}
        onEndPreview={endPreview}
        onRetainPreview={retainPreview}
        onKeepComparison={(seq) => previewComparison(seq, true)}
      />
      <main
        aria-busy={loadingRevision}
        className="relative grid min-h-0 flex-1 gap-3 p-3 md:grid-cols-[minmax(0,1fr)_320px] lg:px-5 lg:pb-4 xl:grid-cols-[minmax(0,1fr)_400px]"
      >
        <div className="flex min-h-0 min-w-0 flex-col gap-3">
          <section
            aria-label="Causal graph"
            className="relative flex h-[480px] flex-none flex-col overflow-hidden rounded-2xl border bg-card md:h-auto md:min-h-0 md:flex-1"
          >
            <div className="flex min-h-11 flex-none flex-wrap items-center justify-between gap-2 border-b px-4 py-1.5">
              <h1 className="text-sm font-semibold">Causal model</h1>
              {activeComparison ? (
                <div
                  ref={comparisonPane}
                  id="model-comparison-preview"
                  role="region"
                  aria-label="Model comparison preview"
                  tabIndex={-1}
                  className="flex min-w-0 flex-wrap items-center gap-2 text-[11px]"
                  onPointerEnter={retainPreview}
                  onPointerLeave={endPreview}
                  onFocus={retainPreview}
                >
                  <span className="whitespace-nowrap font-medium text-amber-700">
                    Spec · {playhead} → {activeComparison.after}
                  </span>
                  {compared.data &&
                    (Object.keys(compared.data.changes).length === 0 ? (
                      <span className="whitespace-nowrap text-muted-foreground">
                        No spec changes
                      </span>
                    ) : (
                      <span className="hidden items-center gap-2 text-[10px] text-muted-foreground sm:flex">
                        <span className="text-amber-700">~ changed</span>
                        <span className="text-emerald-700">+ added</span>
                        <span className="text-rose-700">− removed</span>
                      </span>
                    ))}
                  <Button
                    type="button"
                    size="sm"
                    variant={activeComparison.pinned ? "secondary" : "ghost"}
                    aria-label="Keep comparison"
                    aria-pressed={activeComparison.pinned}
                    onClick={toggleComparison}
                  >
                    Keep
                  </Button>
                  <Button
                    type="button"
                    size="icon-sm"
                    variant="ghost"
                    aria-label="Close comparison"
                    onClick={dismissComparison}
                  >
                    ×
                  </Button>
                </div>
              ) : null}
            </div>
            <div
              className="relative min-h-0 flex-1"
              onPointerEnter={retainPreview}
              onPointerLeave={endPreview}
            >
              {dataReport && !dataDiff && (
                <p
                  role={dataComparison.error ? "alert" : "status"}
                  className="absolute top-3 left-3 z-20 rounded border bg-card px-3 py-2 text-xs"
                >
                  {dataComparison.error?.message ?? "Reading compared histories…"}
                </p>
              )}
              {activeComparison && !modelComparison && (
                <p
                  role={compared.error || comparisonError ? "alert" : "status"}
                  className="absolute top-3 left-3 z-20 rounded border bg-card px-3 py-2 text-xs"
                >
                  {compared.error || comparisonError
                    ? (compared.error ?? comparisonError)?.message
                    : compared.isLoading || compared.data
                      ? "Reading differences…"
                      : "No saved comparison for these model versions."}
                </p>
              )}
              {graphEntities(context.entities).constructs.length > 0 ||
              activeComparison ||
              recordedComparison ? (
                <LayeredCausalGraph
                  modelSnapshot={modelSnapshot}
                  entities={context.entities}
                  simulation={step === "simulate" ? simulationResult : null}
                  simulationPaths={step === "simulate" ? (recordedPaths.data ?? null) : null}
                  dataDiff={dataDiff}
                  step={step}
                  selection={selection}
                  onSelect={select}
                  comparison={modelComparison}
                  variant="asset"
                />
              ) : (
                <div className="flex h-full flex-col items-center justify-center gap-3 px-8 text-center">
                  <p className="text-sm font-medium">No structure defined yet</p>
                  <p className="max-w-sm text-xs leading-relaxed text-muted-foreground">
                    The graph will appear when the agent harness records a model definition.
                  </p>
                </div>
              )}
            </div>
          </section>
          <DetailsPane
            selection={selection}
            context={context}
            loading={loadingRevision || dataComparison.isLoading}
            tick={tick}
          />
        </div>
        <aside
          aria-label="Action record"
          className="relative flex h-[480px] min-h-0 min-w-0 flex-col overflow-hidden rounded-2xl border bg-card md:h-auto"
        >
          {loadingRevision ? (
            <p role="status" className="p-3 text-xs">
              Loading action record…
            </p>
          ) : (
            <ActionRecord
              workspaceId={workspaceId}
              context={context}
              tick={tick}
              running={
                // Work still running has no completed timeline node yet.
                focusSeq === latest ? running : []
              }
              useActionTrace={useActionTrace}
            />
          )}
        </aside>
      </main>
    </div>
  );
}

/** The workbench reads semantic snapshots, committed history and recorded traces. */
export function CausalModelAsset({
  workspaceId,
  question,
  journal,
}: {
  workspaceId: string;
  question: string | undefined;
  journal: StudyJournal;
}) {
  const useSnapshot = useMemo(
    () =>
      function useWorkspaceSnapshot(commitId: string | undefined) {
        return useModelSnapshot(workspaceId, commitId);
      },
    [workspaceId],
  );
  const useActionTrace = useMemo<UseActionTrace>(
    () =>
      function useWorkspaceActionTrace(seq, enabled): ActionTraceState {
        const record = journal.attempts.find((attempt) => attempt.record.seq === seq);
        const traceIds = record?.record.trace_ids ?? [];
        const query = useLLMTraceForAction(workspaceId, record, traceIds, enabled);
        if (
          !enabled ||
          record?.record.attempt.outcome.status !== "applied" ||
          record.record.attempt.request === null ||
          traceIds.length === 0 ||
          query.isError
        )
          return { status: "absent" };
        if (query.data) return { status: "ready", trace: query.data };
        return { status: "loading" };
      },
    [workspaceId, journal.attempts],
  );
  return (
    <CausalModelAssetView
      workspaceId={workspaceId}
      question={question}
      useSnapshot={useSnapshot}
      attempts={journal.attempts}
      dependencies={journal.dependencies}
      useActionTrace={useActionTrace}
      running={journal.running}
    />
  );
}
