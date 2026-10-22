/// <reference types="vite/client" />
import type { Decorator, Preview } from "@storybook/nextjs-vite";
import { type ReactNode, useCallback, useEffect, useState } from "react";
import {
  SVG_MATERIALIZER_ENDPOINT,
  SVG_MATERIALIZER_PARAMETER,
  type SvgMaterializerParameter,
} from "./constants.ts";

const INLINE_STYLE_PROPERTIES = [
  "background-color",
  "color",
  "dominant-baseline",
  "fill",
  "fill-opacity",
  "filter",
  "font-family",
  "font-size",
  "font-style",
  "font-weight",
  "letter-spacing",
  "opacity",
  "paint-order",
  "shape-rendering",
  "stroke",
  "stroke-dasharray",
  "stroke-dashoffset",
  "stroke-linecap",
  "stroke-linejoin",
  "stroke-miterlimit",
  "stroke-opacity",
  "stroke-width",
  "text-anchor",
  "text-decoration",
  "vector-effect",
  "visibility",
  "white-space",
] as const;

type MaterializationState =
  | { phase: "idle" }
  | { phase: "saving" }
  | { phase: "saved"; relativePath: string }
  | { phase: "error"; message: string };

function inlineComputedStyles(source: SVGSVGElement, clone: SVGSVGElement): void {
  const sourceElements = [source, ...source.querySelectorAll("*")];
  const cloneElements = [clone, ...clone.querySelectorAll("*")];
  if (sourceElements.length !== cloneElements.length) {
    throw new Error("The cloned SVG element tree differs from the rendered tree.");
  }

  sourceElements.forEach((sourceElement, index) => {
    const cloneElement = cloneElements[index];
    if (!(cloneElement instanceof SVGElement || cloneElement instanceof HTMLElement)) {
      return;
    }
    const computed = window.getComputedStyle(sourceElement);
    for (const property of INLINE_STYLE_PROPERTIES) {
      const value = computed.getPropertyValue(property);
      if (value) cloneElement.style.setProperty(property, value);
    }
  });
}

/** Serialize the full intrinsic DAG canvas, independent of Storybook's scroll viewport and zoom. */
export function serializeStandaloneSvg(source: SVGSVGElement, storyId: string): string {
  const viewBox = source.viewBox.baseVal;
  if (viewBox.width <= 0 || viewBox.height <= 0) {
    throw new Error("The rendered DAG SVG must have a positive viewBox before materialization.");
  }

  const clone = source.cloneNode(true) as SVGSVGElement; // eslint-disable-line @typescript-eslint/no-unsafe-type-assertion -- DOM cloneNode preserves the source SVG tag but returns Node.
  inlineComputedStyles(source, clone);
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  clone.setAttribute("xmlns:xlink", "http://www.w3.org/1999/xlink");
  clone.setAttribute("width", String(Math.ceil(viewBox.width)));
  clone.setAttribute("height", String(Math.ceil(viewBox.height)));
  clone.style.setProperty("background", "#fff");

  const metadata = document.createElementNS("http://www.w3.org/2000/svg", "metadata");
  metadata.textContent = `Storybook story: ${storyId}`;
  clone.prepend(metadata);
  return `${new XMLSerializer().serializeToString(clone)}\n`;
}

function SvgMaterializer({
  storyId,
  parameter,
  children,
}: {
  storyId: string;
  parameter: SvgMaterializerParameter;
  children: ReactNode;
}) {
  const [state, setState] = useState<MaterializationState>({ phase: "idle" });

  const materialize = useCallback(async () => {
    setState({ phase: "saving" });
    try {
      const matches = document.querySelectorAll(parameter.selector);
      const [match] = matches;
      if (!(match instanceof SVGSVGElement) || matches.length !== 1) {
        throw new Error(
          `Expected exactly one SVG matching ${JSON.stringify(parameter.selector)}; found ${matches.length}.`,
        );
      }
      const svg = serializeStandaloneSvg(match, storyId);
      const response = await fetch(SVG_MATERIALIZER_ENDPOINT, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ storyId, svg }),
      });
      const payload: unknown = await response.json();
      if (typeof payload !== "object" || payload === null)
        throw new Error("Invalid materializer response.");
      if ("error" in payload && typeof payload.error === "string") throw new Error(payload.error);
      if (
        !response.ok ||
        !("relativePath" in payload) ||
        typeof payload.relativePath !== "string"
      ) {
        throw new Error(`Request failed (${response.status}).`);
      }
      setState({ phase: "saved", relativePath: payload.relativePath });
    } catch (error) {
      setState({
        phase: "error",
        message: error instanceof Error ? error.message : String(error),
      });
    }
  }, [parameter.selector, storyId]);

  useEffect(() => {
    setState({ phase: "idle" });
    if (!parameter.auto) return;

    let cancelled = false;
    let attempts = 0;
    let timeoutId: number;
    const waitForSvg = () => {
      if (cancelled) return;
      attempts += 1;
      const matches = document.querySelectorAll(parameter.selector);
      if (matches.length === 1) {
        void materialize();
        return;
      }
      if (matches.length > 1 || attempts >= 40) {
        setState({
          phase: "error",
          message: `Expected one rendered SVG after layout; found ${matches.length}.`,
        });
        return;
      }
      timeoutId = window.setTimeout(waitForSvg, 250);
    };
    timeoutId = window.setTimeout(waitForSvg, 500);
    return () => {
      cancelled = true;
      window.clearTimeout(timeoutId);
    };
  }, [materialize, parameter.auto, parameter.selector]);

  const status =
    state.phase === "saved"
      ? state.relativePath
      : state.phase === "error"
        ? state.message
        : state.phase === "saving"
          ? "Serializing full SVG…"
          : "Writes the fully laid-out SVG to scratchpad";

  return (
    <>
      {children}
      <span hidden data-svg-materializer data-state={state.phase} data-status={status} />
    </>
  );
}

export const withSvgMaterializer: Decorator = (Story, context) => {
  const parameter: unknown = context.parameters[SVG_MATERIALIZER_PARAMETER];
  if (!import.meta.env.DEV || parameter === undefined) return <Story />;
  if (
    typeof parameter !== "object" ||
    parameter === null ||
    !("selector" in parameter) ||
    typeof parameter.selector !== "string"
  ) {
    throw new Error("SVG materializer needs a selector.");
  }
  const auto = "auto" in parameter ? parameter.auto : undefined;
  if (auto !== undefined && typeof auto !== "boolean")
    throw new Error("SVG materializer auto must be a boolean.");
  const options: SvgMaterializerParameter = {
    selector: parameter.selector,
    ...(auto === undefined ? {} : { auto }),
  };
  return (
    <SvgMaterializer storyId={context.id} parameter={options}>
      <Story />
    </SvgMaterializer>
  );
};

export default {
  decorators: [withSvgMaterializer],
} satisfies Preview;
