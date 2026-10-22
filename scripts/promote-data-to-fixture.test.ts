import { afterEach, describe, expect, it, setDefaultTimeout } from "bun:test";
import { access, mkdir, mkdtemp, readdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promoteDataWorkspace } from "./promote-data-to-fixture";

setDefaultTimeout(30_000);

const temporaryRoots: string[] = [];

async function pathExists(path: string): Promise<boolean> {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

async function writeJson(path: string, value: unknown): Promise<void> {
  await mkdir(join(path, ".."), { recursive: true });
  await writeFile(path, JSON.stringify(value));
}

async function seedCompleteWorkspace(
  dataRoot: string,
  workspaceId: string,
  options: { omit?: string; stalePanel?: boolean } = {},
): Promise<void> {
  const process = Bun.spawn(
    [
      "uv",
      "run",
      "--all-packages",
      "python",
      "-m",
      "tests.support.promotion",
      dataRoot,
      workspaceId,
      JSON.stringify(options),
    ],
    { stdout: "pipe", stderr: "pipe" },
  );
  const error = await new Response(process.stderr).text();
  if ((await process.exited) !== 0) throw new Error(error);
}

afterEach(async () => {
  await Promise.all(
    temporaryRoots.splice(0).map((root) => rm(root, { recursive: true, force: true })),
  );
});

describe("promoteDataWorkspace", () => {
  it("replaces HEALTHDEMO with one durable workspace and copy-only fixture projections", async () => {
    const root = await mkdtemp(join(tmpdir(), "grassmoth-fixture-promotion-"));
    temporaryRoots.push(root);
    const dataRoot = join(root, "data");
    await seedCompleteWorkspace(dataRoot, "CANDIDATE");

    await writeJson(join(dataRoot, "HEALTHDEMO", "fixture", "artifacts", "artificial.json"), {
      artificial: true,
    });
    await writeJson(join(dataRoot, "HEALTHDEMO", "scratch", "old.json"), { old: true });

    const summary = await promoteDataWorkspace({
      sourceWorkspaceId: "CANDIDATE",
      dataRoot,
    });

    expect(summary.artifacts).toEqual(["model"]);
    expect(await pathExists(join(dataRoot, "HEALTHDEMO", "study", "history.git"))).toBe(true);
    const restoredHistory = join(root, "restored.git");
    const restore = Bun.spawnSync([
      "git",
      "clone",
      "--mirror",
      join(dataRoot, "HEALTHDEMO", "study", "history.bundle"),
      restoredHistory,
    ]);
    expect(restore.exitCode).toBe(0);
    const refs = (repository: string) => {
      const result = Bun.spawnSync(["git", "--git-dir", repository, "show-ref"]);
      expect(result.exitCode).toBe(0);
      return result.stdout.toString();
    };
    expect(refs(restoredHistory)).toBe(refs(join(dataRoot, "HEALTHDEMO", "study", "history.git")));
    expect((await readdir(join(dataRoot, "HEALTHDEMO", "store", "blobs"))).length).toBeGreaterThan(0);
    expect(await pathExists(join(dataRoot, "HEALTHDEMO", "fixture", "artifacts", "raw_data.json"))).toBe(
      false,
    );
    expect(summary.traces).toHaveLength(5);
    expect(
      JSON.parse(await readFile(join(dataRoot, "HEALTHDEMO", "fixture", "inference.json"), "utf8")),
    ).toBeNull();
    expect(
      JSON.parse(
        await readFile(
          join(dataRoot, "HEALTHDEMO", "fixture", "traces", "statistical_model_spec.json"),
          "utf8",
        ),
      ),
    ).toMatchObject({
      model: "fixture",
      messages: [
        {
          role: "assistant",
          content: "statistical_model_spec: model-spec-sleep-attempt-001",
        },
      ],
    });
    expect(await pathExists(join(dataRoot, "HEALTHDEMO", "store", "model"))).toBe(false);
    expect(
      await pathExists(join(dataRoot, "HEALTHDEMO", "fixture", "artifacts", "artificial.json")),
    ).toBe(false);
    expect(await pathExists(join(dataRoot, "HEALTHDEMO", "scratch"))).toBe(false);
    expect(await pathExists(join(dataRoot, "HEALTHDEMO", "cache"))).toBe(false);
    expect(await pathExists(join(dataRoot, "HEALTHDEMO", "access.json"))).toBe(true);
  });

  it("leaves the existing fixture untouched when the source run is incomplete", async () => {
    const root = await mkdtemp(join(tmpdir(), "grassmoth-fixture-promotion-"));
    temporaryRoots.push(root);
    const dataRoot = join(root, "data");
    await seedCompleteWorkspace(dataRoot, "CANDIDATE", { omit: "panel" });
    await mkdir(join(dataRoot, "HEALTHDEMO"), { recursive: true });
    await writeFile(join(dataRoot, "HEALTHDEMO", "sentinel.txt"), "keep me");

    await expect(
      promoteDataWorkspace({ sourceWorkspaceId: "CANDIDATE", dataRoot }),
    ).rejects.toThrow("missing current artifacts: panel");

    expect(await readFile(join(dataRoot, "HEALTHDEMO", "sentinel.txt"), "utf8")).toBe("keep me");
    expect((await readdir(dataRoot)).some((entry) => entry.startsWith(".HEALTHDEMO-promotion-"))).toBe(
      false,
    );
  });

  it("rejects a complete-looking source whose current provenance chain is stale", async () => {
    const root = await mkdtemp(join(tmpdir(), "grassmoth-fixture-promotion-"));
    temporaryRoots.push(root);
    const dataRoot = join(root, "data");
    await seedCompleteWorkspace(dataRoot, "CANDIDATE", { stalePanel: true });
    await mkdir(join(dataRoot, "HEALTHDEMO"), { recursive: true });
    await writeFile(join(dataRoot, "HEALTHDEMO", "sentinel.txt"), "keep me");

    await expect(
      promoteDataWorkspace({ sourceWorkspaceId: "CANDIDATE", dataRoot }),
    ).rejects.toThrow("stale current artifacts: panel");
    expect(await readFile(join(dataRoot, "HEALTHDEMO", "sentinel.txt"), "utf8")).toBe("keep me");
  });
});
