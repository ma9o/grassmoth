"use client";

import type {
  ObservationSpec,
  PosteriorPredictiveChecks,
  PPCOverlay,
  PPCTestStat,
} from "@grassmoth/api-types";
import { type ColumnDef, createColumnHelper } from "@tanstack/react-table";
import { DrawsChart } from "@/components/charts/draws-chart";
import { overlayChart } from "@/components/charts/series-adapters";
import { StatStrip } from "@/components/charts/stat-strip";
import { StatusIcon } from "@/components/model/scope-primitives";
import { HeaderWithTooltip, InfoTable } from "@/components/ui/info-table";
import { formatNumber } from "@/lib/utils/format";

// ── Row type (one per variable) ──────────────────────────

type PPCWarning = PosteriorPredictiveChecks["findings"][number];
type CheckType = "calibration" | "autocorrelation" | "variance";
type StatName = PPCTestStat["stat_name"];

interface PPCVariableRow {
  readonly variable: string;
  readonly checks: Readonly<Partial<Record<string, PPCWarning>>>;
  readonly testStats: Readonly<Partial<Record<StatName, PPCTestStat>>>;
  readonly overlay?: PPCOverlay;
}

function buildRows(
  warnings: readonly PPCWarning[],
  testStats: readonly PPCTestStat[],
  overlays: readonly PPCOverlay[],
  indicators: readonly Pick<ObservationSpec<string | null>, "id" | "name">[],
): PPCVariableRow[] {
  const map = new Map<
    string,
    {
      variable: string;
      checks: Partial<Record<string, PPCWarning>>;
      testStats: Partial<Record<StatName, PPCTestStat>>;
      overlay?: PPCOverlay;
    }
  >();
  const getOrCreate = (variable: string) => {
    let row = map.get(variable);
    if (!row) {
      row = { variable, checks: {}, testStats: {} };
      map.set(variable, row);
    }
    return row;
  };
  for (const w of warnings) {
    getOrCreate(w.subject.target.id).checks[w.code] = w;
  }
  for (const ts of testStats) {
    getOrCreate(ts.indicator_id).testStats[ts.stat_name] = ts;
  }
  for (const ov of overlays) {
    getOrCreate(ov.indicator_id).overlay = ov;
  }
  const definitions = new Map<string, Pick<ObservationSpec<string | null>, "id" | "name">>(
    indicators.map((indicator) => [indicator.id, indicator]),
  );
  return [...map.values()].map((row) => ({
    ...row,
    variable: definitions.get(row.variable)?.name ?? row.variable,
  }));
}

// ── Overlay: observations over saved predictive replicates ─

function OverlayCell({
  overlay,
  variable,
}: {
  readonly overlay?: PPCOverlay;
  readonly variable: string;
}) {
  if (!overlay) return <span className="text-xs text-muted-foreground">—</span>;
  return (
    <div className="w-60">
      <DrawsChart
        {...overlayChart(overlay, `${variable}: observations over saved predictive replicates`)}
        height={72}
        compact
      />
    </div>
  );
}

function StatCell({ stat }: { readonly stat?: PPCTestStat }) {
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

const columns: ColumnDef<PPCVariableRow>[] = [
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
  readonly warnings: readonly PPCWarning[];
  readonly testStats: readonly PPCTestStat[];
  readonly overlays: readonly PPCOverlay[];
  readonly indicators: readonly Pick<ObservationSpec<string | null>, "id" | "name">[];
}) {
  const rows = buildRows(warnings, testStats, overlays, indicators);
  if (rows.length === 0) return null;

  return <InfoTable columns={columns} data={rows} estimateRowHeight={80} sorting={false} />;
}
