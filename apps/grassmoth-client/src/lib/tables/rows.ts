import type { Row } from "@tanstack/react-table";

export type FlatItem<TData> =
  | { kind: "group-header"; groupKey: string; rows: TData[] }
  | { kind: "row"; row: Row<TData> }
  | { kind: "expanded-row"; row: Row<TData> };

/** Flatten the displayed order once for rendering, expansion and keyboard navigation. */
export function tableItems<TData>(
  rows: Row<TData>[],
  groupBy: ((row: TData) => string) | undefined,
  hasGroupHeaders: boolean,
  isRowExpanded: ((row: TData) => boolean) | undefined,
): FlatItem<TData>[] {
  if (!groupBy) {
    const items: FlatItem<TData>[] = [];
    for (const row of rows) {
      items.push({ kind: "row", row });
      if (isRowExpanded?.(row.original)) {
        items.push({ kind: "expanded-row", row });
      }
    }
    return items;
  }
  const map = new Map<string, typeof rows>();
  for (const row of rows) {
    const key = groupBy(row.original);
    const list = map.get(key) ?? [];
    list.push(row);
    map.set(key, list);
  }
  const items: FlatItem<TData>[] = [];
  for (const [groupKey, groupRows] of map) {
    if (hasGroupHeaders) {
      items.push({
        kind: "group-header",
        groupKey,
        rows: groupRows.map((r) => r.original),
      });
    }
    for (const row of groupRows) {
      items.push({ kind: "row", row });
      if (isRowExpanded?.(row.original)) {
        items.push({ kind: "expanded-row", row });
      }
    }
  }
  return items;
}
