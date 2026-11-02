import type { CSSProperties } from "react";
import { DAG_COLORS } from "@/lib/dag/palette";

interface DagZoomControlsProps {
  zoom: number;
  onZoomChange: (zoom: number) => void;
  /** Return to the zoom that fits the whole graph in its pane. */
  onFit: () => void;
}

export function DagZoomControls({ zoom, onZoomChange, onFit }: DagZoomControlsProps) {
  return (
    <>
      <div style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
        <button
          type="button"
          onClick={() => onZoomChange(zoom / 1.2)}
          style={ZOOM_BUTTON}
          title="zoom out"
          aria-label="Zoom out"
        >
          −
        </button>
        <span style={ZOOM_VALUE}>{Math.round(zoom * 100)}%</span>
        <button
          type="button"
          onClick={() => onZoomChange(zoom * 1.2)}
          style={ZOOM_BUTTON}
          title="zoom in"
          aria-label="Zoom in"
        >
          +
        </button>
        <button
          type="button"
          onClick={onFit}
          style={ZOOM_BUTTON}
          title="fit to pane"
          aria-label="Fit to pane"
        >
          ⤢
        </button>
      </div>
    </>
  );
}

const ZOOM_VALUE: CSSProperties = {
  fontVariantNumeric: "tabular-nums",
  fontSize: 12,
  color: "#4a4f57",
  minWidth: 40,
  textAlign: "center",
};

const ZOOM_BUTTON: CSSProperties = {
  width: 26,
  height: 26,
  border: `1px solid ${DAG_COLORS.line2}`,
  background: "#fff",
  borderRadius: 7,
  cursor: "pointer",
  fontSize: 14,
  lineHeight: 1,
  display: "grid",
  placeItems: "center",
  color: "#4a4f57",
};
