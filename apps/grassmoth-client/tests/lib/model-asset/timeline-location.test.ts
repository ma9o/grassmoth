import type { TimelineRevision } from "@grassmoth/api-types";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { comparison, failedFit, revision, simulation } from "@/lib/__fixtures__/timeline";
import { timelineHref, timelineSelection } from "@/lib/model-asset/timeline-location";
import { useWorkbenchSnapshots } from "@/lib/model-asset/use-workbench";

const attempts = [failedFit, simulation, comparison];

function viewedSequence(query: string, journal: readonly TimelineRevision[]) {
  const selection = timelineSelection(new URLSearchParams(query), journal);
  if (selection.status !== "ready") throw new Error(selection.message);
  const { seq } = selection;
  function View() {
    const view = useWorkbenchSnapshots(journal, () => ({ data: undefined, error: null }), seq);
    return createElement("span", null, view.focusSeq);
  }
  return renderToStaticMarkup(createElement(View));
}

describe("timeline URL selection", () => {
  it("follows new actions only when the URL has no pinned sequence", () => {
    const advanced = [...attempts, revision(9, simulation.record.attempt)];
    expect(viewedSequence("", attempts)).toBe("<span>5</span>");
    expect(viewedSequence("", advanced)).toBe("<span>9</span>");
    expect(viewedSequence("seq=5", attempts)).toBe("<span>5</span>");
    expect(viewedSequence("seq=5", advanced)).toBe("<span>5</span>");
  });

  it("addresses failures and recorded comparisons independently of latest model state", () => {
    expect(viewedSequence("seq=4", attempts)).toBe("<span>4</span>");
    expect(viewedSequence("seq=6", attempts)).toBe("<span>6</span>");
    expect(timelineSelection(new URLSearchParams(), [])).toEqual({ status: "ready", seq: null });
  });

  it.each([
    "",
    "-1",
    "1.5",
    "1e0",
    "01",
    "NaN",
    "9007199254740992",
    "5&seq=6",
    "999",
  ])("rejects invalid or missing actions instead of silently showing latest: seq=%s", (query) => {
    expect(timelineSelection(new URLSearchParams(`seq=${query}`), attempts).status).toBe("invalid");
  });

  it("replaces or removes only the timeline parameter in shareable links", () => {
    const params = new URLSearchParams("filter=a&filter=b&seq=4");
    expect(timelineHref("/workspaces/example", params, 5)).toBe(
      "/workspaces/example?filter=a&filter=b&seq=5",
    );
    expect(timelineHref("/workspaces/example", params, null)).toBe(
      "/workspaces/example?filter=a&filter=b",
    );
    expect(timelineHref("/workspaces/example", new URLSearchParams("seq=5"), null)).toBe(
      "/workspaces/example",
    );
    expect(params.get("seq")).toBe("4");
  });
});
