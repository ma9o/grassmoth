import { area, line } from "d3-shape";
import type { LawPoint, StateShape } from "@/lib/model-asset/spec-state-law";
import { CHART_COLORS, cssColor } from "./chart-tokens";
import { linearScale } from "./plot-geometry";

type Coordinate = (value: number) => number;
const ink = cssColor(CHART_COLORS.prior);
function curve<T extends LawPoint>(x: Coordinate, y: Coordinate, value: (p: T) => number) {
  return line<T>()
    .x((p) => x(p.x))
    .y((p) => y(value(p)));
}

function band<T extends LawPoint>(
  x: Coordinate,
  y: Coordinate,
  low: (p: T) => number,
  high: (p: T) => number,
) {
  return area<T>()
    .x((p) => x(p.x))
    .y0((p) => y(low(p)))
    .y1((p) => y(high(p)));
}

export function LawMarks({
  shape,
  x,
  height,
}: {
  readonly shape: StateShape;
  readonly x: Coordinate;
  readonly height: number;
}) {
  const y = linearScale([0, 1], [height, 1]);
  const points = shape.curve;
  if (shape.binWidth !== undefined) {
    const half = shape.binWidth / 2;
    return points.map((p) => {
      const left = x(p.x - half);
      const width = x(p.x + half) - left;
      const gap = Math.min(1.5, width * 0.15);
      return (
        <g key={p.x} transform={`translate(${left + gap / 2},0)`}>
          <rect
            width={width - gap}
            y={y(p.value)}
            height={height - y(p.value)}
            fill={ink}
            fillOpacity={0.55}
          />
          {p.max > p.min && (
            <rect
              width={width - gap}
              y={y(p.max)}
              height={y(p.min) - y(p.max)}
              fill="none"
              stroke={ink}
              strokeWidth={0.8}
              strokeDasharray="3 1.5"
            />
          )}
        </g>
      );
    });
  }
  const varied = points.some((p) => p.min !== p.max);
  return (
    <g>
      <path
        d={
          band<LawPoint>(
            x,
            y,
            () => 0,
            (p) => p.value,
          )(points) ?? ""
        }
        fill={ink}
        fillOpacity={0.38}
      />
      <path
        d={curve<LawPoint>(x, y, (p) => p.value)(points) ?? ""}
        fill="none"
        stroke={ink}
        strokeWidth={1.3}
      />
      {varied && (
        <path
          d={
            band<LawPoint>(
              x,
              y,
              (p) => p.min,
              (p) => p.max,
            )(points) ?? ""
          }
          fill="none"
          stroke={ink}
          strokeWidth={0.85}
          strokeDasharray="3 1.5"
        />
      )}
      {shape.atoms && shape.atoms[0] !== shape.atoms[1] && (
        <rect
          x={x(shape.atoms[0])}
          y={1}
          width={x(shape.atoms[1]) - x(shape.atoms[0])}
          height={height - 1}
          fill="none"
          stroke={ink}
          strokeDasharray="3 1.5"
        />
      )}
      {shape.referenceAtom !== null && (
        <line
          x1={x(shape.referenceAtom)}
          x2={x(shape.referenceAtom)}
          y1={height}
          y2={1}
          stroke={ink}
          strokeWidth={2}
        />
      )}
    </g>
  );
}

export function ResponseMarks({
  points,
  x,
  y,
  single,
}: {
  readonly points: readonly LawPoint[];
  readonly x: Coordinate;
  readonly y: Coordinate;
  readonly single: boolean;
}) {
  const first = points[0];
  if (!first) return null;
  if (single) {
    return (
      <g>
        <line
          x1={x(first.x)}
          x2={x(first.x)}
          y1={y(first.min)}
          y2={y(first.max)}
          stroke={ink}
          strokeDasharray="3 1.5"
        />
        <circle cx={x(first.x)} cy={y(first.value)} r={2} fill={ink} />
      </g>
    );
  }
  return (
    <g>
      {points.some((p) => p.min !== p.max) && (
        <path
          d={
            band<LawPoint>(
              x,
              y,
              (p) => p.min,
              (p) => p.max,
            )(points) ?? ""
          }
          fill="none"
          stroke={ink}
          strokeWidth={0.9}
          strokeDasharray="3 1.5"
        />
      )}
      <path
        d={curve<LawPoint>(x, y, (p) => p.value)(points) ?? ""}
        fill="none"
        stroke={ink}
        strokeWidth={1.5}
      />
    </g>
  );
}
