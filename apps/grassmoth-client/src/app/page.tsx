"use client";

import { useQuery } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { LandingPageView } from "@/components/landing/landing-page-view";
import { getMockFixture, isMockMode } from "@/lib/api/mock-provider";
import { getWorkspaces, getWorkspacesQueryKey } from "@/lib/api/workspaces";

export default function LandingPage() {
  const router = useRouter();
  const workspacesQuery = useQuery({
    queryKey: getWorkspacesQueryKey(),
    queryFn: getWorkspaces,
    staleTime: 30_000,
    refetchInterval: 30_000,
    retry: false,
  });

  useEffect(() => {
    if (isMockMode() && !sessionStorage.getItem("mock-landed")) {
      sessionStorage.setItem("mock-landed", "true");
      router.push(`/workspaces/${getMockFixture()}`);
    }
  }, [router]);

  return (
    <LandingPageView
      data={workspacesQuery.data}
      error={workspacesQuery.error?.message ?? null}
      isLoading={workspacesQuery.isLoading}
    />
  );
}
