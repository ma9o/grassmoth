"use client";

import type { ModelDiffOutput } from "@grassmoth/api-types";
import { useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import type { TimelineComparison } from "@/lib/model-asset/timeline-comparison";

/** The graph header's spec comparison with the viewed version, which the user can keep or close. */
export function ComparisonPreview({
  playhead,
  comparison,
  report,
  onRetainPreview,
  onEndPreview,
  onToggle,
  onDismiss,
}: {
  readonly playhead: number;
  readonly comparison: TimelineComparison & { readonly pinned: boolean };
  readonly report: ModelDiffOutput | null | undefined;
  readonly onRetainPreview: () => void;
  readonly onEndPreview: () => void;
  readonly onToggle: () => void;
  readonly onDismiss: () => void;
}) {
  const pane = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (comparison.pinned) {
      pane.current?.focus({ preventScroll: true });
      pane.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  }, [comparison.pinned, comparison.before.record.seq, comparison.after.record.seq]);

  return (
    <div
      ref={pane}
      id="model-comparison-preview"
      role="region"
      aria-label="Model comparison preview"
      tabIndex={-1}
      className="flex min-w-0 flex-wrap items-center gap-2 text-[11px]"
      onPointerEnter={onRetainPreview}
      onPointerLeave={onEndPreview}
      onFocus={onRetainPreview}
    >
      <span className="whitespace-nowrap font-medium text-amber-700">
        Spec · {playhead} → {comparison.after.record.seq}
      </span>
      {report &&
        (Object.keys(report.changes).length === 0 ? (
          <span className="whitespace-nowrap text-muted-foreground">No spec changes</span>
        ) : (
          <span className="hidden items-center gap-2 text-[10px] text-muted-foreground sm:flex">
            <span className="text-amber-700">~ changed</span>
            <span className="text-emerald-700">+ added</span>
            <span className="text-rose-700">− removed</span>
          </span>
        ))}
      <Button
        type="button"
        size="sm"
        variant={comparison.pinned ? "secondary" : "ghost"}
        aria-label="Keep comparison"
        aria-pressed={comparison.pinned}
        onClick={onToggle}
      >
        Keep
      </Button>
      <Button
        type="button"
        size="icon-sm"
        variant="ghost"
        aria-label="Close comparison"
        onClick={onDismiss}
      >
        ×
      </Button>
    </div>
  );
}
