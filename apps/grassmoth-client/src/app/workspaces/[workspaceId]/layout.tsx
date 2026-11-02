import type { Metadata } from "next";
import type { ReactNode } from "react";

export async function generateMetadata({
  params,
}: {
  params: Promise<{ workspaceId: string }>;
}): Promise<Metadata> {
  const { workspaceId } = await params;
  return { title: workspaceId };
}

/** On desktop the workbench fills the viewport; its panes scroll independently. */
export default function ModelLayout({ children }: { children: ReactNode }) {
  return (
    <div className="md:flex md:h-dvh md:flex-col md:overflow-hidden">
      <div className="md:min-h-0 md:flex-1 [&>div]:md:h-full [&>div]:md:min-h-0">{children}</div>
    </div>
  );
}
