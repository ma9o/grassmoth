"use client";

import { ChevronDown, ChevronLeft, ChevronRight, ChevronUp, type LucideIcon } from "lucide-react";
import {
  createContext,
  type ReactNode,
  type RefObject,
  useCallback,
  useContext,
  useLayoutEffect,
  useMemo,
  useState,
} from "react";
import {
  Group,
  type GroupImperativeHandle,
  Panel,
  type PanelImperativeHandle,
  type PanelProps,
  Separator,
  useDefaultLayout,
  useGroupRef,
  usePanelRef,
} from "react-resizable-panels";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

/** A collapsed pane keeps its 44px title bar inside the card's 1px border. */
const TITLE_BAR = 46;

/** The edge a pane collapses toward. */
type Edge = "top" | "bottom" | "right";

interface PaneControl {
  readonly collapsed: boolean;
  readonly edge: Edge;
  readonly toggle: () => void;
}

const PaneContext = createContext<PaneControl | null>(null);

/** The collapse state of the workbench pane around a component; null outside the panes. */
export function usePane(): PaneControl | null {
  return useContext(PaneContext);
}

const TOGGLE_ICONS: Record<Edge, { collapse: LucideIcon; expand: LucideIcon }> = {
  top: { collapse: ChevronUp, expand: ChevronDown },
  bottom: { collapse: ChevronDown, expand: ChevronUp },
  right: { collapse: ChevronRight, expand: ChevronLeft },
};

/** Collapses the pane around it to its title bar, or restores it. */
export function PaneToggle({ title }: { title: string }) {
  const pane = usePane();
  if (pane === null) return null;
  const Icon = pane.collapsed ? TOGGLE_ICONS[pane.edge].expand : TOGGLE_ICONS[pane.edge].collapse;
  return (
    <Button
      type="button"
      size="icon-sm"
      variant="ghost"
      aria-label={`${pane.collapsed ? "Expand" : "Collapse"} ${title}`}
      aria-expanded={!pane.collapsed}
      onClick={pane.toggle}
      className="flex-none text-muted-foreground"
    >
      <Icon size={16} aria-hidden="true" />
    </Button>
  );
}

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

/** The record's default width; it keeps that width in pixels as the window resizes. */
const RECORD_WIDTH = 400;
/** The details pane's default share of the left column, in percent. */
const DETAILS_SHARE = 50;

/**
 * One group's split for one action. A split the user drags or collapses is saved under that
 * action alone; opening an action restores its saved split, or the group's default split.
 */
function useActionLayout(id: string, restoreDefault: (group: GroupImperativeHandle) => void) {
  const groupRef = useGroupRef();
  const saved = useDefaultLayout({ id, onlySaveAfterUserInteractions: true });
  // Runs whenever the action changes, even when neither action has a saved split.
  useLayoutEffect(() => {
    const group = groupRef.current;
    if (group === null) return;
    if (saved.defaultLayout) group.setLayout(saved.defaultLayout);
    else restoreDefault(group);
  }, [id, saved.defaultLayout, restoreDefault, groupRef]);
  const save = saved.onLayoutChanged;
  // The collapse toggles resize through the imperative API, which the library does not count as
  // user input, so each toggle saves the split it leaves.
  const saveNow = useCallback(() => {
    const layout = groupRef.current?.getLayout();
    if (layout) save(layout, { isUserInteraction: true });
  }, [groupRef, save]);
  return { groupRef, defaultLayout: saved.defaultLayout, onLayoutChanged: save, saveNow };
}

/**
 * A pane that collapses to its title bar, by its toggle or by dragging its seam past it.
 * Collapsing it while its sibling is collapsed opens the sibling, which takes the space.
 */
function CollapsiblePanel({
  edge,
  panelRef,
  sibling,
  onToggled,
  children,
  ...panel
}: Omit<
  PanelProps,
  "panelRef" | "collapsible" | "collapsedSize" | "onResize" | "className" | "children"
> & {
  edge: Edge;
  panelRef: RefObject<PanelImperativeHandle | null>;
  sibling: RefObject<PanelImperativeHandle | null> | null;
  onToggled: () => void;
  children: ReactNode;
}) {
  const [collapsed, setCollapsed] = useState(false);
  const control = useMemo(
    () => ({
      collapsed,
      edge,
      toggle: () => {
        const handle = panelRef.current;
        if (handle === null) return;
        if (handle.isCollapsed()) handle.expand();
        else {
          if (sibling?.current?.isCollapsed()) sibling.current.expand();
          handle.collapse();
        }
        onToggled();
      },
    }),
    [collapsed, edge, onToggled, panelRef, sibling],
  );
  return (
    <Panel
      {...panel}
      panelRef={panelRef}
      collapsible
      collapsedSize={TITLE_BAR}
      onResize={() => setCollapsed(panelRef.current?.isCollapsed() ?? false)}
      className="flex min-h-0 min-w-0 flex-col"
    >
      <PaneContext.Provider value={control}>{children}</PaneContext.Provider>
    </Panel>
  );
}

/**
 * The workbench's three panes, resized by dragging the seams between them and collapsed to their
 * title bars by the toggle in each: the graph above the details on the left and the action
 * record on the right. Each action keeps its own split in this browser, and the record keeps its
 * width when the window resizes.
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
  const graphRef = usePanelRef();
  const detailsRef = usePanelRef();
  const recordRef = usePanelRef();
  const restoreColumns = useCallback(() => recordRef.current?.resize(RECORD_WIDTH), [recordRef]);
  const restoreRows = useCallback(
    (group: GroupImperativeHandle) =>
      group.setLayout({ graph: 100 - DETAILS_SHARE, details: DETAILS_SHARE }),
    [],
  );
  const columns = useActionLayout(`workbench-columns:${layoutId}`, restoreColumns);
  const rows = useActionLayout(`workbench-rows:${layoutId}`, restoreRows);
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
          <CollapsiblePanel
            id="graph"
            edge="top"
            panelRef={graphRef}
            sibling={detailsRef}
            onToggled={rows.saveNow}
            minSize={160}
          >
            {graph}
          </CollapsiblePanel>
          <Seam between="rows" />
          <CollapsiblePanel
            id="details"
            edge="bottom"
            panelRef={detailsRef}
            sibling={graphRef}
            onToggled={rows.saveNow}
            defaultSize={`${DETAILS_SHARE}%`}
            minSize={120}
          >
            {details}
          </CollapsiblePanel>
        </Group>
      </Panel>
      <Seam between="columns" />
      <CollapsiblePanel
        id="record"
        edge="right"
        panelRef={recordRef}
        sibling={null}
        onToggled={columns.saveNow}
        defaultSize={RECORD_WIDTH}
        minSize={280}
        maxSize="60%"
        groupResizeBehavior="preserve-pixel-size"
      >
        {record}
      </CollapsiblePanel>
    </Group>
  );
}
