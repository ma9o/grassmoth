import type { ReasoningSpec } from "@grassmoth/api-types";
import { useState } from "react";
import { ReasoningText } from "@/components/model/scopes/reasoning";
import type { placeComparisonOverlay } from "@/lib/dag/comparison-overlay";
import { COMPARISON_COLORS } from "@/lib/dag/palette";
import { humanize } from "@/lib/model-asset/selection";
import { DagEdge } from "../core/dag-edge";
import { DagNodeShell } from "../core/dag-node";

const SYMBOLS = { added: "+", removed: "−", revised: "~" };

function MarkDetail({ detail }: { readonly detail: string | ReasoningSpec }) {
  if (typeof detail !== "string") return <ReasoningText value={detail} maxLength={25} />;
  return detail.length > 25 ? `${detail.slice(0, 24)}…` : detail;
}

export function LayeredComparisonOverlay({
  overlay,
  zoom,
}: {
  readonly overlay: ReturnType<typeof placeComparisonOverlay>;
  readonly zoom: number;
}) {
  const [hovered, setHovered] = useState<string | null>(null);
  return (
    <g aria-label="Model differences">
      {overlay.addedEdges.map((edge) => (
        <g key={edge.id} data-comparison-edge={edge.id} data-change="added">
          <DagEdge points={edge.points} color={COMPARISON_COLORS.added} width={3 / zoom} />
        </g>
      ))}
      {overlay.addedNodes.map(({ node, construct }) => (
        <g
          key={node.id}
          data-comparison-node={node.id}
          data-change="added"
          transform={`translate(${node.x},${node.y})`}
        >
          <DagNodeShell
            width={node.width}
            height={node.height}
            title={humanize(construct.name)}
            subtitle="Added in the compared revision"
            accent={COMPARISON_COLORS.added}
            highlighted
          />
        </g>
      ))}
      {overlay.marks.map((mark) => {
        const color = COMPARISON_COLORS[mark.change];
        const detail = typeof mark.detail === "string" ? mark.detail : mark.detail.text;
        const expanded = overlay.marks.length === 1 || hovered === mark.id;
        const left = mark.x > overlay.width / 2;
        const labelX = left ? -204 : 20;
        const labelY = mark.y * zoom > 64 ? -55 : 12;
        return (
          <g
            key={mark.id}
            transform={`translate(${mark.x},${mark.y}) scale(${1 / zoom})`}
            data-comparison-mark={mark.id}
            data-change={mark.change}
            role="img"
            aria-label={`${mark.title}: ${detail}`}
            tabIndex={0}
            onPointerEnter={() => setHovered(mark.id)}
            onPointerLeave={() => setHovered(null)}
            onFocus={() => setHovered(mark.id)}
            onBlur={() => setHovered(null)}
          >
            <title>{`${mark.title}: ${detail}`}</title>
            <circle r={8} fill={color} stroke="white" strokeWidth={2} />
            <text textAnchor="middle" y={3.5} fontSize={12} fontWeight={700} fill="white">
              {SYMBOLS[mark.change]}
            </text>
            {expanded && (
              <g>
                <path
                  d={`M${left ? -8 : 8},0 L${left ? -22 : 22},${labelY < 0 ? -18 : 18}`}
                  fill="none"
                  stroke={color}
                  strokeWidth={1.5}
                />
                <rect
                  x={labelX}
                  y={labelY}
                  width={184}
                  height={44}
                  rx={7}
                  fill="white"
                  stroke={color}
                />
                <text x={labelX + 10} y={labelY + 16} fontSize={10} fill={color}>
                  {mark.title}
                </text>
                <text x={labelX + 10} y={labelY + 32} fontSize={12} fontWeight={600} fill="#1e293b">
                  <MarkDetail detail={mark.detail} />
                </text>
              </g>
            )}
          </g>
        );
      })}
    </g>
  );
}
