import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { expect, it } from "vitest";
import { LandingPageView } from "@/components/landing/landing-page-view";

it("offers existing study navigation without creation or upload controls", () => {
  const html = renderToStaticMarkup(
    createElement(LandingPageView, {
      data: { STUDY: "Does exercise improve sleep?" },
      error: null,
      isLoading: false,
    }),
  );
  expect(html).toContain('href="/v2/STUDY"');
  expect(html).toContain("Does exercise improve sleep?");
  expect(html).not.toMatch(/<(?:input|textarea|form|button)\b/);
  const empty = renderToStaticMarkup(
    createElement(LandingPageView, {
      data: {},
      error: null,
      isLoading: false,
    }),
  );
  expect(empty).toContain("Studies created by your agent will appear here.");
});
