import type { ComponentPropsWithoutRef, CSSProperties, ReactNode } from "react";
import { DAG_COLORS } from "@/lib/dag/palette";

interface DagSvgProps
  extends Omit<ComponentPropsWithoutRef<"svg">, "width" | "height" | "viewBox"> {
  contentWidth: number;
  contentHeight: number;
  zoom: number;
}

/** The scrolling frame that fills its pane. */
export function DagCanvasFrame({ children }: { children: ReactNode }) {
  return <div style={CANVAS_FRAME}>{children}</div>;
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
  minHeight: 0,
  height: "100%",
  flex: "1 1 0%",
  overflow: "auto",
  backgroundImage: `radial-gradient(${DAG_COLORS.line} .8px, transparent .8px)`,
  backgroundSize: "18px 18px",
};
