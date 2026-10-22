import katex from "katex";

/** Render a LaTeX string to an HTML string via KaTeX. */
function tex(latex: string, displayMode = true): string {
  return katex.renderToString(latex, {
    displayMode,
    throwOnError: false,
    strict: false,
  });
}

/** Inline KaTeX span. */
export function Katex({ latex }: { latex: string }) {
  // biome-ignore lint/security/noDangerouslySetInnerHtml: KaTeX renders sanitized math
  return <span dangerouslySetInnerHTML={{ __html: tex(latex, false) }} />;
}
