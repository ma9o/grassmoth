"use client";

import { type ReactNode, useLayoutEffect } from "react";
import {
  Group,
  Panel,
  Separator,
  useDefaultLayout,
  useGroupRef,
  usePanelRef,
} from "react-resizable-panels";
import { cn } from "@/lib/utils";

/**
 * The gap between two panes: drag it, move it with the arrow keys once focused, or double-click
 * it to restore the default split.
 */
function Seam({ between }: { between: "columns" | "rows" }) {
  return (
    <Separator
      className={cn(
        "group/seam flex flex-none items-center justify-center outline-none",
        between === "columns" ? "w-3" : "h-3",
      )}
    >
      <span
        aria-hidden="true"
        className={cn(
          "rounded-full bg-border transition-colors group-data-[separator=hover]/seam:bg-muted-foreground/50 group-data-[separator=active]/seam:bg-ring group-data-[separator=focus]/seam:bg-ring",
          between === "columns" ? "h-8 w-1" : "h-1 w-8",
        )}
      />
    </Separator>
  );
}

/**
 * One group's split for one action. A split the user drags is saved under that action alone;
 * opening an action restores its saved split, or the default size of the panel that sets it.
 */
function useActionLayout(id: string, defaultSize: number | string) {
  const groupRef = useGroupRef();
  const panelRef = usePanelRef();
  const { defaultLayout, onLayoutChanged } = useDefaultLayout({
    id,
    onlySaveAfterUserInteractions: true,
  });
  // Runs whenever the action changes, even when neither action has a saved split.
  useLayoutEffect(() => {
    if (defaultLayout) groupRef.current?.setLayout(defaultLayout);
    else panelRef.current?.resize(defaultSize);
  }, [id, defaultLayout, defaultSize, groupRef, panelRef]);
  return { groupRef, panelRef, defaultLayout, onLayoutChanged };
}

/**
 * The workbench's three panes, resized by dragging the seams between them: the graph above the
 * details on the left and the action record on the right. Each action keeps its own split in
 * this browser, and the record keeps its width when the window resizes.
 */
export function WorkbenchPanes({
  layoutId,
  graph,
  details,
  record,
}: {
  /** The workspace and action whose split this is. */
  layoutId: string;
  graph: ReactNode;
  details: ReactNode;
  record: ReactNode;
}) {
  const columns = useActionLayout(`workbench-columns:${layoutId}`, 400);
  const rows = useActionLayout(`workbench-rows:${layoutId}`, "50%");
  return (
    <Group
      id="workbench-columns"
      orientation="horizontal"
      groupRef={columns.groupRef}
      defaultLayout={columns.defaultLayout}
      onLayoutChanged={columns.onLayoutChanged}
    >
      <Panel id="model" minSize="30%" className="flex min-h-0 min-w-0 flex-col">
        <Group
          id="workbench-rows"
          orientation="vertical"
          groupRef={rows.groupRef}
          defaultLayout={rows.defaultLayout}
          onLayoutChanged={rows.onLayoutChanged}
        >
          <Panel id="graph" minSize={160} className="flex min-h-0 min-w-0 flex-col">
            {graph}
          </Panel>
          <Seam between="rows" />
          <Panel
            id="details"
            panelRef={rows.panelRef}
            defaultSize="50%"
            minSize={120}
            className="flex min-h-0 min-w-0 flex-col"
          >
            {details}
          </Panel>
        </Group>
      </Panel>
      <Seam between="columns" />
      <Panel
        id="record"
        panelRef={columns.panelRef}
        defaultSize={400}
        minSize={280}
        maxSize="60%"
        groupResizeBehavior="preserve-pixel-size"
        className="flex min-h-0 min-w-0 flex-col"
      >
        {record}
      </Panel>
    </Group>
  );
}
