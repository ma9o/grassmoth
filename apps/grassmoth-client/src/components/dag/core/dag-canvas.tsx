import type { ComponentPropsWithoutRef, CSSProperties, ReactNode } from "react";
import { DAG_COLORS } from "@/lib/dag/palette";

interface DagSvgProps
  extends Omit<ComponentPropsWithoutRef<"svg">, "width" | "height" | "viewBox"> {
  contentWidth: number;
  contentHeight: number;
  zoom: number;
}

/** `fill` lets the frame take its container's height instead of the fixed workbench box. */
export function DagCanvasFrame({
  children,
  fill = false,
}: {
  children: ReactNode;
  fill?: boolean;
}) {
  return (
    <div style={fill ? { ...CANVAS_FRAME, ...CANVAS_FRAME_FILL } : CANVAS_FRAME}>{children}</div>
  );
}

export function DagSvg({
  contentWidth,
  contentHeight,
  zoom,
  children,
  style,
  ...svgProps
}: DagSvgProps) {
  return (
    <svg
      width={Math.ceil(contentWidth * zoom)}
      height={Math.ceil(contentHeight * zoom)}
      viewBox={`0 0 ${Math.ceil(contentWidth)} ${Math.ceil(contentHeight)}`}
      style={{ display: "block", ...style }}
      {...svgProps}
    >
      {children}
    </svg>
  );
}

const CANVAS_FRAME: CSSProperties = {
  background: "#fff",
  border: `1px solid ${DAG_COLORS.line}`,
  borderRadius: 14,
  padding: 6,
  minHeight: 560,
  maxHeight: "74vh",
  overflow: "auto",
  backgroundImage: `radial-gradient(${DAG_COLORS.line} .8px, transparent .8px)`,
  backgroundSize: "18px 18px",
};

const CANVAS_FRAME_FILL: CSSProperties = {
  minHeight: 0,
  maxHeight: "none",
  height: "100%",
  flex: "1 1 0%",
};
