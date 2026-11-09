"use client";

import type { TimelineRevision } from "@grassmoth/api-types";
import { Pin } from "lucide-react";
import type { KeyboardEvent } from "react";
import {
  NODE_HEIGHT,
  NODE_WIDTH,
  timelineTickLabel,
} from "@/lib/model-asset/timeline-presentation";
import { cn } from "@/lib/utils";

/** Pins a timeline action, or unpins it to follow the latest action, from the node's right end. */
export function PinToggle({
  tick,
  pinned,
  tabbable,
  x,
  y,
  onEnter,
  onToggle,
  onKeyDown,
}: {
  readonly tick: TimelineRevision;
  readonly pinned: boolean;
  readonly tabbable: boolean;
  /** Center of the action's node. */
  readonly x: number;
  readonly y: number;
  /** The pointer or keyboard focus reaches the toggle. */
  readonly onEnter: () => void;
  readonly onToggle: () => void;
  readonly onKeyDown: (event: KeyboardEvent<HTMLElement>) => void;
}) {
  return (
    <button
      type="button"
      aria-label={`Pin ${timelineTickLabel(tick)}`}
      aria-pressed={pinned}
      title={pinned ? "Unpin and follow latest" : "Pin this action"}
      tabIndex={tabbable ? 0 : -1}
      className={cn(
        "absolute flex cursor-pointer items-center justify-center rounded-r-md hover:bg-muted focus-visible:outline-2 focus-visible:outline-ring",
        pinned ? "text-slate-900" : "text-muted-foreground",
      )}
      style={{
        left: x + NODE_WIDTH / 2 - 28,
        top: y - NODE_HEIGHT / 2 + 1,
        width: 27,
        height: NODE_HEIGHT - 2,
      }}
      onPointerEnter={onEnter}
      onFocus={onEnter}
      onClick={onToggle}
      onKeyDown={onKeyDown}
    >
      <Pin size={13} aria-hidden="true" fill={pinned ? "currentColor" : "none"} />
    </button>
  );
}
