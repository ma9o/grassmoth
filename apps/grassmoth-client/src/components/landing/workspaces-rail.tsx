"use client";

import type { WorkspaceList } from "@/lib/server/workspaces";
import { Skeleton } from "@/components/ui/skeleton";
import Link from "next/link";

export function WorkspacesRail({
  data,
  error,
  isLoading,
}: {
  data: WorkspaceList | undefined;
  error: string | null;
  isLoading: boolean;
}) {
  const workspaces = Object.entries(data ?? {});

  return (
    <div className="w-full max-w-2xl mx-auto space-y-3">
      {isLoading && (
        <>
          <SkeletonCard />
          <SkeletonCard />
          <SkeletonCard />
        </>
      )}

      {!isLoading && error && (
        <p className="text-sm text-muted-foreground text-center py-8">{error}</p>
      )}

      {!isLoading && !error && workspaces.length === 0 && (
        <p className="text-sm text-muted-foreground text-center py-8">
          No studies yet. Studies created by your agent will appear here.
        </p>
      )}

      {!isLoading &&
        !error &&
        workspaces.map(([workspaceId, question]) => (
          <WorkspaceCard key={workspaceId} workspaceId={workspaceId} question={question} />
        ))}
    </div>
  );
}

function WorkspaceCard({
  workspaceId,
  question,
}: {
  workspaceId: string;
  question: string | null;
}) {
  return (
    <Link
      href={`/v2/${workspaceId}`}
      className="block rounded-lg border bg-card px-4 py-3 shadow-sm transition-colors hover:bg-accent/50"
    >
      <p className="font-mono text-xs font-semibold tracking-wider text-muted-foreground">
        {workspaceId}
      </p>
      <p className="mt-1 text-sm leading-snug text-foreground">
        {question ?? "Question not available yet."}
      </p>
    </Link>
  );
}

function SkeletonCard() {
  return (
    <div className="rounded-lg border bg-card px-4 py-3 shadow-sm space-y-2">
      <Skeleton className="h-3 w-24" />
      <Skeleton className="h-4 w-full" />
    </div>
  );
}
