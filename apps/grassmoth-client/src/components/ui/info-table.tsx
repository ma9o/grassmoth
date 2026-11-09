"use client";

import { type ColumnDef, flexRender, type SortDirection } from "@tanstack/react-table";
import { ChevronDown, ChevronsUpDown, ChevronUp, Search } from "lucide-react";
import type { ReactNode } from "react";
import { StatTooltip } from "@/components/ui/stat-tooltip";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useInfoTable } from "@/lib/tables/use-info-table";
import { cn } from "@/lib/utils";

// ---------- Column meta typing ----------
declare module "@tanstack/react-table" {
  interface ColumnMeta<TData, TValue> {
    readonly align?: "left" | "center" | "right";
    readonly mono?: boolean;
    /** Thresholding function: "fail" → red bg, "warn" → orange bg, undefined → default. */
    readonly severity?: (value: TValue, row: TData) => "fail" | "warn" | undefined;
  }
}

function severityClass(level: "fail" | "warn" | undefined): string | undefined {
  if (level === "fail") return "bg-destructive/10";
  if (level === "warn") return "bg-warning/15";
  return undefined;
}

const ALIGN_CLASS = { left: "text-left", center: "text-center", right: "text-right" } as const;

const ARIA_SORT = { asc: "ascending", desc: "descending" } as const;

function SortIcon({ direction }: { readonly direction: false | SortDirection }) {
  if (direction === "asc") return <ChevronUp className="h-3.5 w-3.5" />;
  if (direction === "desc") return <ChevronDown className="h-3.5 w-3.5" />;
  return <ChevronsUpDown className="h-3 w-3 text-muted-foreground/50" />;
}

// ---------- HeaderWithTooltip helper ----------
export function HeaderWithTooltip({
  label,
  tooltip,
  className,
}: {
  readonly label: string;
  readonly tooltip: string;
  readonly className?: string;
}) {
  return (
    <span className={cn("inline-flex items-center gap-1", className)}>
      {label}
      <StatTooltip explanation={tooltip} />
    </span>
  );
}

// ---------- InfoTable ----------

interface InfoTableProps<TData> {
  readonly columns: readonly ColumnDef<TData>[];
  readonly data: readonly TData[];
  readonly sorting?: boolean;
  readonly filtering?: boolean;
  readonly compact?: boolean;
  readonly maxHeight?: string;
  readonly estimateRowHeight?: number;
  readonly groupBy?: (row: TData) => string;
  readonly renderGroupHeader?: (groupKey: string, rows: readonly TData[]) => ReactNode;
  readonly rowClassName?: (row: TData, index: number) => string | undefined;
  readonly isRowExpanded?: (row: TData) => boolean;
  readonly renderExpandedRow?: (row: TData) => ReactNode;
}

export function InfoTable<TData>({
  columns,
  data,
  sorting: enableSorting = true,
  filtering: enableFiltering = true,
  compact: isCompact = false,
  maxHeight = "max-h-[32rem]",
  estimateRowHeight = 40,
  groupBy,
  renderGroupHeader,
  rowClassName,
  isRowExpanded,
  renderExpandedRow,
}: InfoTableProps<TData>) {
  "use no memo"; // TODO: remove when TanStack Table supports React Compiler
  const {
    searchQuery,
    setSearchQuery,
    table,
    flatItems,
    parentRef,
    virtualizer,
    virtualItems,
    paddingTop,
    paddingBottom,
    isFiltered,
    focusedRowId,
    containerProps,
    filteredCount,
  } = useInfoTable({
    columns,
    data,
    enableSorting,
    estimateRowHeight,
    groupBy,
    hasGroupHeaders: renderGroupHeader !== undefined,
    isRowExpanded,
  });

  const visibleItems = new Map(virtualItems.map((item) => [item.index, item]));
  return (
    <div className="overflow-hidden rounded-md border">
      {enableFiltering && (
        <div className="flex items-center gap-2 border-b px-3 py-1.5">
          <Search className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
          <input
            type="text"
            placeholder={`Search ${data.length} rows\u2026`}
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="flex-1 bg-transparent text-sm outline-none placeholder:text-muted-foreground"
          />
          {isFiltered && (
            <span className="shrink-0 text-xs text-muted-foreground">
              {filteredCount} of {data.length}
            </span>
          )}
        </div>
      )}
      <div ref={parentRef} className={cn(maxHeight, "overflow-auto")} {...containerProps}>
        <Table>
          <TableHeader className="sticky top-0 bg-background z-10 shadow-[0_1px_0_var(--border)]">
            {table.getHeaderGroups().map((hg) => (
              <TableRow key={hg.id}>
                {hg.headers.map((header) => {
                  const meta = header.column.columnDef.meta;
                  const alignClass = ALIGN_CLASS[meta?.align ?? "left"];
                  const sorted = header.column.getIsSorted();

                  return (
                    <TableHead
                      key={header.id}
                      className={cn(alignClass)}
                      onClick={
                        enableSorting && header.column.getCanSort()
                          ? header.column.getToggleSortingHandler()
                          : undefined
                      }
                      style={
                        enableSorting && header.column.getCanSort()
                          ? { cursor: "pointer", userSelect: "none" }
                          : undefined
                      }
                      aria-sort={enableSorting && sorted !== false ? ARIA_SORT[sorted] : undefined}
                    >
                      <span className="inline-flex items-center gap-1">
                        {header.isPlaceholder
                          ? null
                          : flexRender(header.column.columnDef.header, header.getContext())}
                        {enableSorting && header.column.getCanSort() && (
                          <SortIcon direction={sorted} />
                        )}
                      </span>
                    </TableHead>
                  );
                })}
              </TableRow>
            ))}
          </TableHeader>
          <TableBody>
            {paddingTop > 0 && (
              <tr>
                <td style={{ height: paddingTop, padding: 0, border: "none" }} />
              </tr>
            )}
            {flatItems.flatMap((item, index) => {
              const vi = visibleItems.get(index);
              if (vi === undefined) return [];
              if (item.kind === "group-header") {
                return (
                  <TableRow
                    key={`group-${item.groupKey}`}
                    className="bg-muted/50 hover:bg-muted/50"
                    data-index={vi.index}
                    ref={virtualizer.measureElement}
                  >
                    <TableCell colSpan={columns.length} className="py-2">
                      {renderGroupHeader?.(item.groupKey, item.rows)}
                    </TableCell>
                  </TableRow>
                );
              }
              if (item.kind === "expanded-row") {
                return (
                  <TableRow
                    key={`expanded-${item.row.id}`}
                    className="bg-muted/20 hover:bg-muted/20"
                    data-index={vi.index}
                    ref={virtualizer.measureElement}
                  >
                    <TableCell colSpan={columns.length} className="p-0">
                      {renderExpandedRow?.(item.row.original)}
                    </TableCell>
                  </TableRow>
                );
              }
              const { row } = item;
              const nextItem = flatItems.at(vi.index + 1);
              const hasExpandedBelow = nextItem?.kind === "expanded-row";
              return (
                <TableRow
                  key={row.id}
                  className={cn(
                    hasExpandedBelow && "border-b-0",
                    focusedRowId === row.id && "ring-2 ring-ring ring-inset",
                    rowClassName?.(row.original, row.index),
                  )}
                  data-index={vi.index}
                  ref={virtualizer.measureElement}
                >
                  {row.getVisibleCells().map((cell) => {
                    const meta = cell.column.columnDef.meta;
                    const alignClass = ALIGN_CLASS[meta?.align ?? "left"];
                    const sev = meta?.severity?.(cell.getValue(), cell.row.original);

                    return (
                      <TableCell
                        key={cell.id}
                        className={cn(
                          alignClass,
                          meta?.mono && "font-mono",
                          severityClass(sev),
                          isCompact && "py-0.5",
                        )}
                      >
                        {flexRender(cell.column.columnDef.cell, cell.getContext())}
                      </TableCell>
                    );
                  })}
                </TableRow>
              );
            })}
            {paddingBottom > 0 && (
              <tr>
                <td style={{ height: paddingBottom, padding: 0, border: "none" }} />
              </tr>
            )}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
