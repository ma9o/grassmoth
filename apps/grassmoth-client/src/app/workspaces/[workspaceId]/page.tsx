"use client";

import Link from "next/link";
import { use } from "react";
import { CausalModelAsset } from "@/components/model/causal-model-asset";
import { useStudyJournal } from "@/lib/hooks/use-study-journal";

/** Inspect the canonical study from its first, possibly empty, checkpoint. */
export default function ModelPage({ params }: { params: Promise<{ workspaceId: string }> }) {
  const { workspaceId } = use(params);
  const journalQuery = useStudyJournal(workspaceId);
  const journal = journalQuery.data;

  if (journalQuery.error && !journal) {
    return (
      <div className="flex min-h-screen items-center justify-center px-4 py-10 sm:px-6">
        <div className="max-w-md space-y-3 rounded-lg border bg-card p-6 text-center">
          <h1 className="text-lg font-semibold">Workspace unavailable</h1>
          <p className="text-sm text-muted-foreground">{journalQuery.error.message}</p>
          <Link
            href="/"
            className="inline-flex rounded-md border px-3 py-2 text-sm font-medium transition-colors hover:bg-secondary"
          >
            Return Home
          </Link>
        </div>
      </div>
    );
  }

  if (!journal) {
    return (
      <div className="flex min-h-screen items-center justify-center text-sm text-muted-foreground">
        Reading the journal…
      </div>
    );
  }

  return <CausalModelAsset workspaceId={workspaceId} question={undefined} journal={journal} />;
}
