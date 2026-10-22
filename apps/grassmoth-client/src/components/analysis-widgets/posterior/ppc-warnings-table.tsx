"use client";

import type {
  ObservationSpec,
  PPCOverlay,
  PPCTestStat,
  PosteriorPredictiveChecks,
} from "@grassmoth/api-types";
import { type ColumnDef, createColumnHelper } from "@tanstack/react-table";
import { DrawsChart } from "@/components/charts/draws-chart";
import { overlayChart } from "@/components/charts/series-adapters";
import { StatStrip } from "@/components/charts/stat-strip";
import { HeaderWithTooltip, InfoTable } from "@/components/ui/info-table";
import { StatusIcon } from "@/components/model/scope-primitives";
import { formatNumber } from "@/lib/utils/format";

// ── Row type (one per variable) ──────────────────────────

type PPCWarning = PosteriorPredictiveChecks["findings"][number];
type CheckType = "calibration" | "autocorrelation" | "variance";
type StatName = PPCTestStat["stat_name"];

interface PPCVariableRow {
  variable: string;
  checks: Partial<Record<string, PPCWarning>>;
  testStats: Partial<Record<StatName, PPCTestStat>>;
  overlay?: PPCOverlay;
}

function getOrCreate(map: Map<string, PPCVariableRow>, variable: string): PPCVariableRow {
  let row = map.get(variable);
  if (!row) {
    row = { variable, checks: {}, testStats: {} };
    map.set(variable, row);
  }
  return row;
}

function buildRows(
  warnings: readonly PPCWarning[],
  testStats: readonly PPCTestStat[],
  overlays: readonly PPCOverlay[],
  indicators: Pick<ObservationSpec<string | null>, "id" | "name">[],
): PPCVariableRow[] {
  const map = new Map<string, PPCVariableRow>();
  for (const w of warnings) {
    getOrCreate(map, w.subject.target.id).checks[w.code] = w;
  }
  for (const ts of testStats) {
    getOrCreate(map, ts.indicator_id).testStats[ts.stat_name] = ts;
  }
  for (const ov of overlays) {
    getOrCreate(map, ov.indicator_id).overlay = ov;
  }
  const definitions = new Map<string, Pick<ObservationSpec<string | null>, "id" | "name">>(
    indicators.map((indicator) => [indicator.id, indicator]),
  );
  return Array.from(map.values()).map((row) => ({
    ...row,
    variable: definitions.get(row.variable)?.name ?? row.variable,
  }));
}

// ── Overlay: observations over every retained replicate ─

function OverlayCell({ overlay, variable }: { overlay?: PPCOverlay; variable: string }) {
  if (!overlay) return <span className="text-xs text-muted-foreground">—</span>;
  return (
    <div className="w-60">
      <DrawsChart
        {...overlayChart(overlay, `${variable}: observations over every retained replicate`)}
        height={72}
        compact
      />
    </div>
  );
}

function StatCell({ stat }: { stat?: PPCTestStat }) {
  if (!stat) return <span className="text-xs text-muted-foreground">—</span>;
  return (
    <div className="w-28">
      <StatStrip stat={stat} height={26} />
    </div>
  );
}

// ── Table columns ────────────────────────────────────────

const col = createColumnHelper<PPCVariableRow>();

const CHECK_TYPES: CheckType[] = ["calibration", "autocorrelation", "variance"];
const STAT_NAMES: StatName[] = ["mean", "sd", "min", "max"];

const CHECK_TOOLTIPS: Record<CheckType, string> = {
  calibration:
    "Fraction of observed timepoints falling within the 95% predictive interval. Expected ~0.95.",
  autocorrelation:
    "Lag-1 autocorrelation of residuals (observed minus predicted mean). High values suggest missing dynamics.",
  variance: "Ratio of predictive std to observed std. Values far from 1 indicate scale misfit.",
};

const STAT_TOOLTIPS: Record<StatName, string> = {
  mean: "Compares observed mean to distribution of replicated means. Vertical line is T(y).",
  sd: "Compares observed std dev to distribution of replicated std devs.",
  min: "Compares observed minimum to distribution of replicated minima.",
  max: "Compares observed maximum to distribution of replicated maxima.",
};

const columns: ColumnDef<PPCVariableRow, unknown>[] = [
  col.display({
    id: "variable",
    header: "Variable",
    cell: ({ row }) => (
      <span className="font-medium font-mono text-xs">{row.original.variable}</span>
    ),
  }),
  col.display({
    id: "overlay",
    header: () => (
      <HeaderWithTooltip
        label="y vs y_rep"
        tooltip="Observations (dark dots) over every retained predictive replicate, on the time axis."
      />
    ),
    cell: ({ row }) => (
      <OverlayCell
        variable={row.original.variable}
        {...(row.original.overlay === undefined ? {} : { overlay: row.original.overlay })}
      />
    ),
  }),
  ...CHECK_TYPES.map((ct) =>
    col.display({
      id: ct,
      header: () => (
        <HeaderWithTooltip
          label={ct === "autocorrelation" ? "Autocorr" : ct.charAt(0).toUpperCase() + ct.slice(1)}
          tooltip={CHECK_TOOLTIPS[ct]}
        />
      ),
      cell: ({ row }) => {
        const warning = row.original.checks[ct];
        if (!warning) return <span className="text-xs text-muted-foreground">—</span>;
        return (
          <span
            className="font-mono text-xs"
            title={warning.kind === "evaluated" ? warning.evidence.note : warning.detail}
          >
            <StatusIcon status={warning.kind === "evaluated" ? warning.outcome : "not_evaluated"} />{" "}
            {warning.kind === "evaluated" ? formatNumber(warning.evidence.value) : "Not evaluated"}
          </span>
        );
      },
    }),
  ),
  ...STAT_NAMES.map((sn) =>
    col.display({
      id: `t_${sn}`,
      header: () => <HeaderWithTooltip label={`T(${sn})`} tooltip={STAT_TOOLTIPS[sn]} />,
      cell: ({ row }) => {
        const stat = row.original.testStats[sn];
        return <StatCell {...(stat === undefined ? {} : { stat })} />;
      },
    }),
  ),
];

// ── Exported component ───────────────────────────────────

export function PPCWarningsTable({
  warnings,
  testStats,
  overlays,
  indicators,
}: {
  warnings: readonly PPCWarning[];
  testStats: readonly PPCTestStat[];
  overlays: readonly PPCOverlay[];
  indicators: Pick<ObservationSpec<string | null>, "id" | "name">[];
}) {
  const rows = buildRows(warnings, testStats, overlays, indicators);
  if (rows.length === 0) return null;

  return <InfoTable columns={columns} data={rows} estimateRowHeight={80} sorting={false} />;
}
