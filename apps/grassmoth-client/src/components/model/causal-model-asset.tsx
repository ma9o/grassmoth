"use client";

import type {
  ActionSuccess,
  ModelSnapshot,
  RunningAction,
  TimelineRevision,
} from "@grassmoth/api-types";
import Link from "next/link";
import { useMemo } from "react";
import { LayeredCausalGraph } from "@/components/dag/layered/layered-causal-graph";
import { graphEntities } from "@/lib/dag/layered-model";
import { type UseActionPoll, useActionPoll as useActionPollFor } from "@/lib/hooks/use-action-poll";
import { useDataComparison } from "@/lib/hooks/use-data-comparison";
import { useModelSnapshot } from "@/lib/hooks/use-model-snapshot";
import type { StudyJournal } from "@/lib/hooks/use-study-journal";
import { useTimelineNavigation } from "@/lib/hooks/use-timeline-navigation";
import { useSimulationPaths } from "@/lib/hooks/use-visuals";
import type { TimelineNavigation } from "@/lib/model-asset/timeline-location";
import {
  type SnapshotReader,
  useWorkbench,
  useWorkbenchSnapshots,
} from "@/lib/model-asset/use-workbench";
import { ActionRecord } from "./action-record";
import { ComparisonPreview } from "./comparison-preview";
import { DetailsPane } from "./details-pane";
import { VersionScrubber } from "./version-scrubber";
import { PaneToggle, WorkbenchPanes } from "./workbench-panes";

export interface CausalModelAssetViewProps {
  readonly workspaceId: string;
  readonly useSnapshot: SnapshotReader;
  readonly attempts: readonly TimelineRevision[];
  readonly dependencies: StudyJournal["dependencies"];
  readonly useActionPoll: UseActionPoll;
  readonly running: readonly RunningAction[];
  readonly navigation: TimelineNavigation;
}

/**
 * The viewer answers two questions about the action selected on the timeline. The action record
 * on the right shows its recorded input and execution log. The graph, with the details pane
 * below it, shows the state the action left: the graph one part at a glance, the details pane in
 * depth. The agent harness makes every change, so no pane offers writes.
 * A data_diff leaf records a comparison without changing that state: the graph and details
 * overlay its saved evidence on the parent version, and the record shows its input references.
 */
export function CausalModelAssetView(props: CausalModelAssetViewProps) {
  const { navigation } = props;
  const { selected, focusSeq, hasSelectedState, hasCurrentState } = useWorkbenchSnapshots(
    props.attempts,
    props.useSnapshot,
    navigation.seq,
  );
  if (selected.error) {
    return (
      <div role="alert" className="p-6">
        {selected.error.message}
      </div>
    );
  }
  if (!hasSelectedState || !hasCurrentState) {
    const tick =
      props.attempts.find((entry) => entry.record.seq === focusSeq) ?? props.attempts.at(-1);
    return (
      <div className="flex h-dvh flex-col gap-4 bg-muted/20">
        <header className="flex-none border-b bg-card p-4">
          <Link href="/" className="text-sm font-semibold">
            Grassmoth
          </Link>
          <p className="text-sm">{props.workspaceId}</p>
        </header>
        <VersionScrubber
          ticks={props.attempts}
          dependencies={props.dependencies}
          playhead={tick?.record.seq ?? 0}
          latest={0}
          navigation={navigation}
          onPlayhead={navigation.viewAt}
        />
        <section className="mx-4 mb-4 flex min-h-0 flex-1 flex-col overflow-hidden rounded-2xl border bg-card">
          <ActionRecord
            tick={tick}
            attempts={props.attempts}
            dependencies={props.dependencies}
            running={navigation.seq === null ? props.running : []}
            useActionPoll={props.useActionPoll}
            hrefAt={navigation.hrefAt}
            onSelectVersion={navigation.viewAt}
          />
        </section>
      </div>
    );
  }
  if (!selected.data) {
    return (
      <div role="status" className="p-6">
        Loading model version…
      </div>
    );
  }
  return (
    <ModelRevision
      {...props}
      result={selected.result}
      modelSnapshot={selected.data}
      loadingRevision={selected.isPlaceholderData === true}
      focusSeq={focusSeq}
    />
  );
}

function ModelRevision({
  useSnapshot,
  workspaceId,
  attempts,
  dependencies,
  useActionPoll,
  modelSnapshot,
  result,
  loadingRevision,
  navigation,
  focusSeq,
  running,
}: CausalModelAssetViewProps & {
  readonly modelSnapshot: ModelSnapshot;
  readonly result: ActionSuccess | undefined;
  readonly loadingRevision: boolean;
  readonly focusSeq: number;
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
    attempts,
    modelSnapshot,
    focusSeq,
    navigationKey: `${workspaceId}:${navigation.seq ?? "latest"}:${focusSeq}`,
    result,
    viewAt: navigation.viewAt,
  });
  const recordedPaths = useSimulationPaths(modelSnapshot);
  const tick = ticks.find((item) => item.record.seq === focusSeq);
  const dataReport =
    tick?.record.attempt.action === "data_diff" &&
    tick.record.attempt.outcome.status === "applied" &&
    result?.action === "data_diff"
      ? result.body.report
      : null;
  const dataComparison = useDataComparison(workspaceId, dataReport, attempts);
  const dataDiff = dataComparison.data;
  const context = { ...versionContext, dataDiff };
  const recordedComparison = result?.action === "model_diff" ? result.body : null;
  const comparisonFailure = compared.error ?? comparisonError;
  // Nodes chart what the viewed version's action produced.
  const step = tick?.record.attempt.action ?? null;

  return (
    <div
      className="flex h-dvh flex-col overflow-hidden bg-muted/20"
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
        navigation={navigation}
        onPlayhead={selectVersion}
        compare={{
          comparedSeq: activeComparison?.after.record.seq ?? null,
          onPreview: previewComparison,
          onEndPreview: endPreview,
          onRetainPreview: retainPreview,
          onKeep: (pair) => previewComparison(pair, true),
        }}
      />
      <main
        aria-busy={loadingRevision}
        className="relative flex min-h-0 flex-1 p-3 lg:px-5 lg:pb-4"
      >
        <WorkbenchPanes
          layoutId={`${workspaceId}:${tick?.commit_id ?? "unselected"}`}
          graph={
            <section
              aria-label="Causal graph"
              className="relative flex min-h-0 flex-1 flex-col overflow-hidden rounded-2xl border bg-card"
            >
              <div className="flex min-h-11 flex-none flex-wrap items-center gap-2 border-b px-4 py-1.5">
                <h1 className="mr-auto text-sm font-semibold">Causal model</h1>
                {activeComparison ? (
                  <ComparisonPreview
                    playhead={playhead}
                    comparison={activeComparison}
                    report={compared.data}
                    onRetainPreview={retainPreview}
                    onEndPreview={endPreview}
                    onToggle={toggleComparison}
                    onDismiss={dismissComparison}
                  />
                ) : null}
                <PaneToggle title="causal model" />
              </div>
              <div
                className="relative min-h-0 flex-1 overflow-hidden"
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
                    role={comparisonFailure ? "alert" : "status"}
                    className="absolute top-3 left-3 z-20 rounded border bg-card px-3 py-2 text-xs"
                  >
                    {comparisonFailure?.message ??
                      (compared.isLoading || compared.data
                        ? "Reading differences…"
                        : "No saved comparison for these model versions.")}
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
          }
          details={
            <DetailsPane
              selection={selection}
              context={context}
              loading={loadingRevision || dataComparison.isLoading}
              tick={tick}
            />
          }
          record={
            <aside
              aria-label="Action record"
              className="relative flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden rounded-2xl border bg-card"
            >
              <ActionRecord
                tick={tick}
                attempts={attempts}
                dependencies={dependencies}
                running={
                  // Work still running has no completed timeline node yet.
                  navigation.seq === null ? running : []
                }
                useActionPoll={useActionPoll}
                hrefAt={navigation.hrefAt}
                onSelectVersion={selectVersion}
              />
            </aside>
          }
        />
      </main>
    </div>
  );
}

/** The workbench reads semantic snapshots, committed history and recorded traces. */
export function CausalModelAsset({
  workspaceId,
  journal,
}: {
  readonly workspaceId: string;
  readonly journal: StudyJournal;
}) {
  const { selection, hrefAt, viewAt } = useTimelineNavigation(journal.attempts);
  const useSnapshot = useMemo(
    () =>
      function useWorkspaceSnapshot(commitId: string | undefined) {
        return useModelSnapshot(workspaceId, commitId);
      },
    [workspaceId],
  );
  const useWorkspaceActionPoll = useMemo<UseActionPoll>(
    () =>
      function useWorkspacePoll(action, callId) {
        return useActionPollFor(workspaceId, action, callId);
      },
    [workspaceId],
  );
  if (selection.status === "invalid") {
    return (
      <div role="alert" className="flex flex-col items-start gap-3 p-6">
        <h1 className="font-semibold">Timeline action unavailable</h1>
        <p>{selection.message}</p>
        <Link
          href={hrefAt(null)}
          prefetch={false}
          onNavigate={(event) => {
            event.preventDefault();
            viewAt(null);
          }}
          className="text-sm underline underline-offset-4"
        >
          Open latest action
        </Link>
      </div>
    );
  }
  return (
    <CausalModelAssetView
      key={workspaceId}
      workspaceId={workspaceId}
      useSnapshot={useSnapshot}
      attempts={journal.attempts}
      dependencies={journal.dependencies}
      useActionPoll={useWorkspaceActionPoll}
      running={journal.running}
      navigation={{ seq: selection.seq, hrefAt, viewAt }}
    />
  );
}
