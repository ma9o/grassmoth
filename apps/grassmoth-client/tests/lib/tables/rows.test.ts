import { createTable, getCoreRowModel, getSortedRowModel } from "@tanstack/react-table";
import { expect, it } from "vitest";
import { tableItems } from "@/lib/tables/rows";

it("keeps sorting, grouped rows and expanded details in one displayed order", () => {
  const data = [
    { group: "a", value: 3 },
    { group: "b", value: 2 },
    { group: "a", value: 1 },
  ];
  const table = createTable({
    data,
    columns: [{ accessorKey: "value" }],
    state: { sorting: [{ id: "value", desc: false }] },
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    onStateChange: () => {},
    renderFallbackValue: null,
  });
  const rows = table.getRowModel().rows;
  const items = tableItems(
    rows,
    (row) => row.group,
    true,
    (row) => row.value === 1,
  );
  expect(
    items.map((item) =>
      item.kind === "group-header" ? item.groupKey : `${item.kind}:${item.row.original.value}`,
    ),
  ).toEqual(["a", "row:1", "expanded-row:1", "row:3", "b", "row:2"]);
  expect(items.flatMap((item) => (item.kind === "row" ? [item.row.id] : []))).toEqual([
    "2",
    "0",
    "1",
  ]);
  expect(tableItems(rows, undefined, false, undefined).map((item) => item.kind)).toEqual([
    "row",
    "row",
    "row",
  ]);
  expect(data.map((row) => row.value)).toEqual([3, 2, 1]);
});
