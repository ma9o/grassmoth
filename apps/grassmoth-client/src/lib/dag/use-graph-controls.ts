"use client";

import { useEffect, useState } from "react";

/** Shared viewport and edge interaction state for the graph renderers. */
export function useGraphControls(initialZoom: number, minZoom: number, maxZoom: number) {
  const [zoom, updateZoom] = useState(initialZoom);
  const [hoveredEdge, setHoveredEdge] = useState<string | null>(null);
  const setZoom = (value: number) => updateZoom(Math.max(minZoom, Math.min(maxZoom, value)));
  return { zoom, setZoom, hoveredEdge, setHoveredEdge };
}

/** Play recorded frames; this never computes a simulation. */
export function usePlayback(frameCount: number, intervalMs: number, initialIndex = 0) {
  const [index, setIndex] = useState(initialIndex);
  const [playing, setPlaying] = useState(false);
  const currentIndex = Math.max(0, Math.min(Math.max(0, frameCount - 1), index));
  useEffect(() => {
    if (!playing || frameCount < 2) return;
    const timer = setInterval(
      () => setIndex((current) => (current >= frameCount - 1 ? 0 : current + 1)),
      intervalMs,
    );
    return () => clearInterval(timer);
  }, [playing, frameCount, intervalMs]);
  return { index: currentIndex, setIndex, playing, setPlaying };
}
