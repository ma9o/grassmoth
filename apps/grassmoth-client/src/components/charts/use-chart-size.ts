"use client";

import { useEffect, useRef, useState } from "react";

export interface ChartSize {
  readonly width: number;
  readonly height: number;
}

/** The rendered box of a chart container, followed through every resize. */
export function useChartSize<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [size, setSize] = useState<ChartSize | null>(null);
  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const observer = new ResizeObserver((entries) => {
      const box = entries.at(-1)?.contentRect;
      if (!box) return;
      setSize((previous) =>
        previous && previous.width === box.width && previous.height === box.height
          ? previous
          : { width: box.width, height: box.height },
      );
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  return [ref, size] as const;
}

/** Size a canvas backing store for the screen and any zoom, then paint in CSS pixels. */
export function paintCanvas(
  canvas: HTMLCanvasElement,
  size: ChartSize,
  resolution: number,
  paint: (context: CanvasRenderingContext2D) => void,
) {
  const ratio = (window.devicePixelRatio || 1) * resolution;
  canvas.width = Math.max(1, Math.round(size.width * ratio));
  canvas.height = Math.max(1, Math.round(size.height * ratio));
  const context = canvas.getContext("2d");
  if (!context) throw new Error("This browser has no 2D canvas context");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, size.width, size.height);
  paint(context);
}
