import { fixtureValue } from "@/components/__fixtures__/fixture-value";
import { describe, expect, it } from "vitest";
import { outcome as construct } from "@/lib/__fixtures__/model";
import {
  authoredSnapshot as authored,
  fittedSnapshot as fitted,
} from "@/lib/__fixtures__/snapshot";
import { lawCurves, lawLabel, ownLawUses } from "@/lib/model-asset/laws";

describe("law curves", () => {
  it("follows a construct's own dynamics and noise laws by their authored roles", () => {
    expect(ownLawUses(construct).map((use) => use.role)).toEqual(["decay", "diffusion_scale"]);
  });

  it("draws authored laws as written until a fit supersedes them with posteriors", () => {
    const persistence = fixtureValue(lawCurves(authored, ownLawUses(construct)).at(0));
    expect(persistence).toMatchObject({ kind: "authored", family: "Beta", posteriors: [] });
    expect(persistence.prior.x.length).toBeGreaterThan(0);
    expect(lawLabel(persistence)).toBe("persistence");

    const decay = fixtureValue(lawCurves(fitted, ownLawUses(construct)).at(0));
    expect(decay).toMatchObject({ kind: "fitted", family: null });
    expect(decay.posteriors.map((marginal) => marginal.subject.parameter_id)).toEqual([
      persistence.parameter.id,
    ]);
    expect(lawLabel(decay)).toBe("decay");
  });

  it("keeps a model's recorded posterior when another data history is selected", () => {
    const revised = {
      ...fitted,
      state: { ...fitted.state, data: { revision: "9".repeat(40), replicate_index: 0 } },
    };
    expect(lawCurves(revised, ownLawUses(construct))).toEqual(
      lawCurves(fitted, ownLawUses(construct)),
    );
  });
});
