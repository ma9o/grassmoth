import type { NextConfig } from "next";
import { fileURLToPath } from "node:url";

const workspaceRoot = fileURLToPath(new URL("../..", import.meta.url));

const nextConfig: NextConfig = {
  outputFileTracingRoot: workspaceRoot,
  outputFileTracingExcludes: {
    "/*": [
      "../grassmoth-server/.pytest_cache/**/*",
      "../grassmoth-server/.ruff_cache/**/*",
      "../grassmoth-server/evals/**/*",
      "../grassmoth-server/logs/**/*",
      "../grassmoth-server/notebooks/**/*",
      "../grassmoth-server/tests/**/*",
      "../grassmoth-server/**/__pycache__/**/*",
      "../../data",
      "../../data/**/*",
      "../../scratchpad",
      "../../scratchpad/**/*",
    ],
  },
  turbopack: {
    root: workspaceRoot,
  },
};

export default nextConfig;
