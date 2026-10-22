"use client";

import { useMemo, useRef, useState } from "react";
import {
  type ColumnDef,
  type SortingState,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
} from "@tanstack/react-table";
import { useVirtualizer } from "@tanstack/react-virtual";
import { useTableKeyboardNav } from "./use-table-keyboard-nav";
import { tableItems } from "./rows";

interface InfoTableOptions<TData> {
  columns: ColumnDef<TData, unknown>[];
  data: TData[];
  enableSorting: boolean;
  estimateRowHeight: number;
  groupBy: ((row: TData) => string) | undefined;
  hasGroupHeaders: boolean;
  isRowExpanded: ((row: TData) => boolean) | undefined;
}

const GROUP_HEADER_HEIGHT = 36;

export function useInfoTable<TData>({
  columns,
  data,
  enableSorting,
  estimateRowHeight,
  groupBy,
  hasGroupHeaders,
  isRowExpanded,
}: InfoTableOptions<TData>) {
  "use no memo";
  const [sortingState, setSortingState] = useState<SortingState>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const parentRef = useRef<HTMLDivElement>(null);

  // Pre-filter data before passing to TanStack Table
  const filteredData = useMemo(() => {
    if (!searchQuery) return data;
    const search = searchQuery.toLowerCase();
    return data.filter((row) => JSON.stringify(row).toLowerCase().includes(search));
  }, [data, searchQuery]);

  // eslint-disable-next-line react-hooks/incompatible-library -- This hook and its renderers opt out with "use no memo".
  const table = useReactTable({
    data: filteredData,
    columns,
    ...(enableSorting
      ? { state: { sorting: sortingState }, onSortingChange: setSortingState }
      : {}),
    getCoreRowModel: getCoreRowModel(),
    ...(enableSorting && { getSortedRowModel: getSortedRowModel() }),
  });

  const rows = table.getRowModel().rows;

  // Build flat list: group headers interleaved with data rows
  const flatItems = useMemo(
    () => tableItems(rows, groupBy, hasGroupHeaders, isRowExpanded),
    [rows, groupBy, hasGroupHeaders, isRowExpanded],
  );

  // For small tables, render every row by setting overscan to the full count.
  // With variable-height cells (e.g. wrapping text) the browser's auto table-layout
  // recomputes column widths from visible cells, so virtualizing rows in/out can
  // oscillate column widths → row heights → measurements in an infinite feedback loop.
  // Keeping all rows mounted breaks the loop; virtualization still helps for large tables.
  const virtualizer = useVirtualizer({
    count: flatItems.length,
    getScrollElement: () => parentRef.current,
    estimateSize: (index) =>
      flatItems[index]?.kind === "group-header" ? GROUP_HEADER_HEIGHT : estimateRowHeight,
    overscan: flatItems.length <= 100 ? flatItems.length : 5,
  });

  const navigableRows = flatItems.flatMap((item) => (item.kind === "row" ? [item.row] : []));
  const { focusedRowIndex, containerProps } = useTableKeyboardNav(navigableRows.length);
  const focusedRowId = focusedRowIndex === null ? null : navigableRows[focusedRowIndex]?.id;

  const virtualItems = virtualizer.getVirtualItems();
  const totalSize = virtualizer.getTotalSize();

  // Spacer heights for the padding approach (keeps <table> semantics)
  const paddingTop = virtualItems.at(0)?.start ?? 0;
  const paddingBottom = totalSize - (virtualItems.at(-1)?.end ?? totalSize);

  const isFiltered = searchQuery.length > 0;

  return {
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
    filteredCount: filteredData.length,
  };
}
