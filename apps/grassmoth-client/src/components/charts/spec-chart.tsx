import type { FlowPlot } from "@/lib/model-asset/flow-plots";
import type { ResponseAxis, ResponseSlice } from "@/lib/model-asset/spec-plots";
import type { LawPoint, Range, SpecLawPlot } from "@/lib/model-asset/spec-state-law";
import { formatSignificant } from "@/lib/utils/format";
import { CHART_COLORS, cssColor } from "./chart-tokens";
import { extentOf, jitter, linearScale, padDomain } from "./plot-geometry";
import { LawMarks, ResponseMarks } from "./spec-marks";

const ink = cssColor(CHART_COLORS.prior);
const text = cssColor(CHART_COLORS.axis);
const grid = cssColor(CHART_COLORS.grid);
const scale = (domain: Range, pixels: Range) =>
  linearScale(domain[0] === domain[1] ? padDomain(domain) : domain, pixels);
type Coordinate = (value: number) => number;

function AxisEnds({
  domain,
  pixels,
  y,
  size,
}: {
  readonly domain: Range;
  readonly pixels: Range;
  readonly y: number;
  readonly size: number;
}) {
  return (
    <g fill={text} fontSize={size}>
      {domain[0] === domain[1] ? (
        <text x={(pixels[0] + pixels[1]) / 2} y={y} textAnchor="middle">
          {formatSignificant(domain[0])}
        </text>
      ) : (
        <>
          <text x={pixels[0]} y={y}>
            {formatSignificant(domain[0])}
          </text>
          <text x={pixels[1]} y={y} textAnchor="end">
            {formatSignificant(domain[1])}
          </text>
        </>
      )}
    </g>
  );
}

export function SpecLawChart({
  law,
  width,
  height,
  detail = false,
  captions = true,
}: {
  readonly law: SpecLawPlot;
  readonly width: number;
  readonly height: number;
  readonly detail?: boolean;
  readonly captions?: boolean;
}) {
  const shape = law.shape;
  const label = `${law.label} · ${law.family}`;
  const size = detail ? 10 : 9;
  const plotTop = captions ? size * 2 + 4 : 1;
  const pixels: Range = [3, width - 3];
  return (
    <svg
      width="100%"
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={label}
    >
      <title>{label}</title>
      {captions && (
        <>
          <text x={3} y={size + 1} fontSize={size + 0.5} fontWeight={600} fill={text}>
            {law.label}
          </text>
          <text x={3} y={size * 2 + 2} fontSize={size} fill={text}>
            {law.family}
          </text>
        </>
      )}
      {shape.kind === "unavailable" ? (
        <text x={3} y={height - 2} fill={text} fontSize={size}>
          {shape.reason}
        </text>
      ) : (
        <>
          <g transform={`translate(0,${plotTop})`}>
            <LawMarks
              shape={shape.value}
              x={scale(shape.value.domain, pixels)}
              height={Math.max(3, height - plotTop - size - 2)}
            />
          </g>
          <AxisEnds
            domain={
              shape.value.binWidth === undefined
                ? shape.value.domain
                : [shape.value.domain[0] + 0.5, shape.value.domain[1] - 0.5]
            }
            pixels={pixels}
            y={height - 1}
            size={size}
          />
        </>
      )}
    </svg>
  );
}

function OutputAxis({
  limits,
  y,
  pixels,
  values,
  font,
}: {
  readonly limits: Range;
  readonly y: Coordinate;
  readonly pixels: Range;
  readonly values: readonly number[];
  readonly font: number;
}) {
  const [left, right] = pixels;
  return (
    <>
      <g aria-label="Output range" fontSize={font} fill={text} textAnchor="end">
        <text x={left - 3} y={y(limits[1]) + 3}>
          {formatSignificant(limits[1])}
        </text>
        {limits[0] !== limits[1] && (
          <text x={left - 3} y={y(limits[0])}>
            {formatSignificant(limits[0])}
          </text>
        )}
      </g>
      {values.map((v, i) => (
        <circle
          // eslint-disable-next-line react/no-array-index-key -- Draw indices identify the aligned display draws.
          key={i}
          cx={right + 5 + 22 * jitter(i)}
          cy={y(v)}
          r={1.3}
          fill={ink}
          fillOpacity={0.4}
        />
      ))}
    </>
  );
}

function InputAxis({
  axis,
  x,
  pixels,
  bottom,
}: {
  readonly axis: ResponseAxis;
  readonly x: Coordinate;
  readonly pixels: Range;
  readonly bottom: number;
}) {
  const [left, right] = pixels;
  return (
    <g transform={`translate(0,${bottom + 3})`}>
      <line x1={left} x2={right} y1={0} y2={0} stroke={grid} />
      {axis.law?.shape.kind === "plot" && (
        <LawMarks shape={axis.law.shape.value} x={x} height={19} />
      )}
      {axis.given.map((v) => (
        <line key={v} x1={x(v)} x2={x(v)} y1={2} y2={19} stroke={ink} strokeWidth={2} />
      ))}
    </g>
  );
}

/** Input-free terms span the row without inventing an input range. */
function functionSeries(axis: ResponseAxis | null, points: readonly LawPoint[]) {
  const first = points[0];
  return !axis && first ? [first, { ...first, x: 1 }] : points;
}

function firstDayValues(output: FlowPlot | undefined) {
  if (output?.kind !== "draws") return [];
  return output.rows.map((row) => row[0]).filter((value) => value !== undefined);
}

/** In details the input law shares the function's x axis and the day-one outputs share its y axis. */
export function ResponseChart({
  slice,
  width,
  height,
  detail = false,
  output,
  reading = false,
}: {
  readonly slice: ResponseSlice;
  readonly width: number;
  readonly height: number;
  readonly detail?: boolean;
  readonly output?: FlowPlot | undefined;
  readonly reading?: boolean;
}) {
  if (slice.plot.kind === "unavailable") {
    return (
      <svg width="100%" height={height} role="img" aria-label={slice.plot.reason}>
        <text x={2} y={detail ? 14 : 9} fill={text} fontSize={detail ? 11 : 7}>
          {slice.plot.reason}
        </text>
      </svg>
    );
  }
  const { axis, points } = slice.plot.value;
  const values = detail ? firstDayValues(output) : [];
  const font = detail ? 10 : 9;
  const shelf = detail && axis ? 23 : 0;
  const bottom = height - font - 3 - shelf;
  const limits = extentOf([
    ...(reading ? [] : [0]),
    ...points.flatMap((p) => [p.value, p.min, p.max]),
    ...values,
  ]) ?? [0, 0];
  const left = Math.max(...limits.map((v) => formatSignificant(v).length)) * font * 0.62 + 5;
  const right = width - (detail ? 32 : 3);
  const pixels: Range = [left, right];
  const domain = axis?.domain ?? [0, 1];
  const x = scale(domain, pixels);
  const y = scale(limits, [bottom, font / 2 + 1]);
  const series = functionSeries(axis, points);
  return (
    <svg
      width="100%"
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label={axis ? `Function of ${axis.label}` : "Constant term"}
    >
      {limits[0] <= 0 && limits[1] >= 0 && (
        <line x1={left} x2={right} y1={y(0)} y2={y(0)} stroke={grid} strokeWidth={1} />
      )}
      <ResponseMarks points={series} x={x} y={y} single={series.length === 1} />
      <OutputAxis limits={limits} y={y} pixels={pixels} values={values} font={font} />
      {detail && axis && <InputAxis axis={axis} x={x} pixels={pixels} bottom={bottom} />}
      {axis ? (
        <AxisEnds domain={domain} pixels={pixels} y={height - 1} size={font} />
      ) : (
        <text x={left} y={height - 1} fontSize={font} fill={text}>
          No state input · {formatSignificant(points[0]?.value ?? 0)}
        </text>
      )}
    </svg>
  );
}
