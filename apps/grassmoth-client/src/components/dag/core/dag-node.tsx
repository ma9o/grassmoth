"use client";

import type { ReactNode } from "react";

interface DagNodeShellProps {
  readonly width: number;
  readonly height: number;
  readonly title?: string;
  readonly subtitle?: string;
  /** Card fill. Defaults to the theme card color. */
  readonly fill?: string;
  /** Border accent (e.g. sign color, status). Defaults to the theme border. */
  readonly accent?: string;
  readonly dashed?: boolean;
  readonly highlighted?: boolean;
  /** Outcome nodes get a heavier border. */
  readonly outcome?: boolean;
  readonly children?: ReactNode;
}

function shellStrokeWidth(highlighted: boolean | undefined, outcome: boolean | undefined): number {
  if (highlighted) return 3;
  if (outcome) return 2;
  return 1.4;
}

/**
 * The shared SVG card shell both DAG kinds build on: a rounded rect with an
 * optional title/subtitle. Variants render their body (badges, trajectory
 * charts, glyphs, do() controls) as children.
 */
export function DagNodeShell({
  width,
  height,
  title,
  subtitle,
  fill,
  accent,
  dashed,
  highlighted,
  outcome,
  children,
}: DagNodeShellProps) {
  const strokeWidth = shellStrokeWidth(highlighted, outcome);
  return (
    <>
      <rect
        width={width}
        height={height}
        rx={11}
        fill={fill ?? "var(--card)"}
        stroke={accent ?? "var(--border)"}
        strokeWidth={strokeWidth}
        strokeDasharray={dashed ? "6,4" : undefined}
        style={{ transition: "stroke 200ms" }}
      />
      {title ? (
        <text x={14} y={24} fontSize={13} fontWeight={650} fill="var(--foreground)">
          {title}
        </text>
      ) : null}
      {subtitle ? (
        <text x={14} y={40} fontSize={9.5} fill="var(--muted-foreground)">
          {subtitle}
        </text>
      ) : null}
      {children}
    </>
  );
}
