"use client";

import Link from "next/link";
import { Skeleton } from "@/components/ui/skeleton";

export function WorkspacesRail({
  data: workspaces = [],
  error,
  isLoading,
}: {
  readonly data: readonly string[] | undefined;
  readonly error: string | null;
  readonly isLoading: boolean;
}) {
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
        workspaces.map((workspaceId) => (
          <Link
            key={workspaceId}
            href={`/workspaces/${workspaceId}`}
            className="block rounded-lg border bg-card px-4 py-3 font-mono text-sm font-semibold shadow-sm transition-colors hover:bg-accent/50"
          >
            {workspaceId}
          </Link>
        ))}
    </div>
  );
}

function SkeletonCard() {
  return (
    <div className="rounded-lg border bg-card px-4 py-3 shadow-sm">
      <Skeleton className="h-5 w-32" />
    </div>
  );
}
