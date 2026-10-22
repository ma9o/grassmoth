"use client";

import { LandingPageView } from "@/components/landing/landing-page-view";
import { getWorkspaces, getWorkspacesQueryKey } from "@/lib/api/workspaces";
import { getMockFixture, isMockMode } from "@/lib/api/mock-provider";
import { useQuery } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { useEffect } from "react";

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
      router.push(`/v2/${getMockFixture()}`);
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
