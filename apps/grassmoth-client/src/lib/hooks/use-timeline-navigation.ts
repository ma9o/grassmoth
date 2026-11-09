"use client";

import type { TimelineRevision } from "@grassmoth/api-types";
import { usePathname, useSearchParams } from "next/navigation";
import { timelineHref, timelineSelection } from "@/lib/model-asset/timeline-location";

/** The URL owns selection; journal polling never writes browser history. */
export function useTimelineNavigation(attempts: readonly TimelineRevision[]) {
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const params = new URLSearchParams(searchParams.toString());
  const selection = timelineSelection(params, attempts);
  const hrefAt = (seq: number | null) => timelineHref(pathname, params, seq);
  const viewAt = (seq: number | null) => {
    const current = new URL(window.location.href);
    const href = timelineHref(current.pathname, current.searchParams, seq) + current.hash;
    if (href !== current.pathname + current.search + current.hash) {
      window.history.pushState(null, "", href);
    }
  };
  return { selection, hrefAt, viewAt };
}
