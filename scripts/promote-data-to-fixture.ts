#!/usr/bin/env bun

import { randomUUID } from "node:crypto";
import { access, cp, mkdir, mkdtemp, rename, rm } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");

const WORKSPACE_ID = /^[A-Za-z0-9][A-Za-z0-9_-]*$/;

const DURABLE_ENTRIES = [
  "access.json",
  "input",
  "query.txt",
  "session.json",
  "sources",
  "store",
  "study",
] as const;

type ProjectedArtifactId = "model" | "identification_report" | "validation_report";
type ProjectedTraceId =
  | "raw_data"
  | "latent_structure"
  | "measurement_structure"
  | "measurements"
  | "statistical_model_spec";

interface PromotionOptions {
  sourceWorkspaceId: string;
  dataRoot?: string;
  fixtureWorkspaceId?: string;
}

export interface PromotionSummary {
  source: string;
  destination: string;
  artifacts: ProjectedArtifactId[];
  traces: ProjectedTraceId[];
}

async function exists(path: string): Promise<boolean> {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

function assertWorkspaceId(value: string, label: string): void {
  if (!WORKSPACE_ID.test(value)) {
    throw new Error(`${label} must match ${WORKSPACE_ID}; received ${JSON.stringify(value)}.`);
  }
}

async function projectStudy(
  sourceRoot: string,
  stagingRoot?: string,
): Promise<Pick<PromotionSummary, "artifacts" | "traces">> {
  const process = Bun.spawn(
    [
      "uv",
      "run",
      "--directory",
      join(repoRoot, "apps/data-pipeline"),
      "python",
      "-m",
      "scripts.fixtures.study",
      "project",
      sourceRoot,
      ...(stagingRoot ? [stagingRoot] : []),
    ],
    { stdout: "pipe", stderr: "pipe" },
  );
  const [output, error, code] = await Promise.all([
    new Response(process.stdout).text(),
    new Response(process.stderr).text(),
    process.exited,
  ]);
  if (code !== 0) throw new Error(error);
  return JSON.parse(output);
}

async function copyDurableWorkspace(sourceRoot: string, stagingRoot: string): Promise<void> {
  for (const entry of DURABLE_ENTRIES) {
    const source = join(sourceRoot, entry);
    if (!(await exists(source))) continue;
    await cp(source, join(stagingRoot, entry), { recursive: true });
  }
}

async function replaceFixture(stagingRoot: string, fixtureRoot: string): Promise<void> {
  const backupRoot = `${fixtureRoot}.backup-${randomUUID()}`;
  const hadFixture = await exists(fixtureRoot);

  if (hadFixture) await rename(fixtureRoot, backupRoot);
  try {
    await rename(stagingRoot, fixtureRoot);
  } catch (error) {
    if (hadFixture) await rename(backupRoot, fixtureRoot);
    throw error;
  }

  if (hadFixture) await rm(backupRoot, { recursive: true, force: true });
}

export async function promoteDataWorkspace({
  sourceWorkspaceId,
  dataRoot = join(repoRoot, "data"),
  fixtureWorkspaceId = "HEALTHDEMO",
}: PromotionOptions): Promise<PromotionSummary> {
  assertWorkspaceId(sourceWorkspaceId, "Source workspace id");
  assertWorkspaceId(fixtureWorkspaceId, "Fixture workspace id");
  if (sourceWorkspaceId === fixtureWorkspaceId) {
    throw new Error("Source workspace and fixture workspace must be different.");
  }

  const sourceRoot = join(dataRoot, sourceWorkspaceId);
  const fixtureRoot = join(dataRoot, fixtureWorkspaceId);
  if (!(await exists(sourceRoot))) {
    throw new Error(`Source workspace does not exist: ${sourceRoot}`);
  }

  await projectStudy(sourceRoot);
  await mkdir(dataRoot, { recursive: true });
  const stagingRoot = await mkdtemp(join(dataRoot, `.${fixtureWorkspaceId}-promotion-`));

  try {
    await copyDurableWorkspace(sourceRoot, stagingRoot);
    const { artifacts, traces } = await projectStudy(sourceRoot, stagingRoot);
    await replaceFixture(stagingRoot, fixtureRoot);
    return { source: sourceRoot, destination: fixtureRoot, artifacts, traces };
  } finally {
    if (await exists(stagingRoot)) {
      await rm(stagingRoot, { recursive: true, force: true });
    }
  }
}

function sourceWorkspaceFromArgs(args: string[]): string {
  if (args.length !== 2 || args[0] !== "--from") {
    throw new Error("Usage: bun run fixture:promote --from <workspace-id>");
  }
  return args[1];
}

if (import.meta.main) {
  try {
    const sourceWorkspaceId = sourceWorkspaceFromArgs(process.argv.slice(2));
    const result = await promoteDataWorkspace({ sourceWorkspaceId });
    console.log(`Promoted ${result.source} to ${result.destination}.`);
    console.log(
      `Copied ${result.artifacts.length} artifact projections and ${result.traces.length} trace projections.`,
    );
  } catch (error) {
    console.error(error instanceof Error ? error.message : error);
    process.exitCode = 1;
  }
}
