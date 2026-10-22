import type { WorkspaceList } from "@/lib/server/workspaces";
import { WorkspacesRail } from "./workspaces-rail";

export function LandingPageView({
  data,
  error,
  isLoading,
}: {
  data: WorkspaceList | undefined;
  error: string | null;
  isLoading: boolean;
}) {
  return (
    <div className="mx-auto flex min-h-screen w-full max-w-2xl flex-col justify-center gap-8 px-4 py-10 sm:px-6">
      <header className="space-y-3 text-center">
        <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">grassmoth</h1>
        <p className="text-sm text-muted-foreground">
          Follow your studies, inspect models, and review results as your agent works.
        </p>
      </header>
      <section aria-labelledby="workspaces-heading" className="space-y-3">
        <h2 id="workspaces-heading" className="text-sm font-semibold">
          Your studies
        </h2>
        <WorkspacesRail data={data} error={error} isLoading={isLoading} />
      </section>
    </div>
  );
}
