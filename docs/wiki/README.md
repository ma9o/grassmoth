<!-- cspell:words Docling chunker -->

# Local reference wiki

Generate a searchable Markdown wiki from the
[author-hosted Bayesian Workflow PDF](https://users.aalto.fi/~ave/Bayesian-Workflow.pdf):

```bash
bun run --cwd docs bayesian-workflow

# Download the current edition again and rebuild the wiki
bun run --cwd docs bayesian-workflow --refresh

# Convert a few pages into the cache before rebuilding the entire wiki
bun run --cwd docs bayesian-workflow --pages 40 41 44 45
```

The command requires `bun`, `uv`, `curl`, and `OPENROUTER_API_KEY` in the environment
or the repository's ignored `.env` file, which the script loads. Python
dependencies are pinned in the script and installed by `uv`.

Conversion uses the
[OpenRouter PDF parser with its Mistral OCR engine](https://openrouter.ai/docs/guides/overview/multimodal/pdfs).
OpenRouter currently lists OCR at $2 per 1,000 pages: about $1.10 for this
550-page edition, plus a small model input charge. The parser runs through a
chat request, so the script requests a one-token acknowledgment from Gemma and
uses the **file annotations** for the wiki. The generated chat reply is discarded.
OpenRouter chooses the Mistral OCR version behind its `mistral-ocr` engine.

The PDF is cached at `scratchpad/Bayesian-Workflow.pdf`. The generated index is
written to `docs/wiki/bayesian-workflow.md`. Chapter indexes sit directly under
`docs/wiki/bayesian-workflow/`; each chapter has a directory containing one file
per numbered section, including exercises. All generated files are gitignored.
Original API responses, including Markdown and images, are cached by PDF checksum
and request settings under `scratchpad/bayesian-workflow-ocr/`. Each PDF page is
converted separately, with eight concurrent requests by default (`--workers`
changes this). Rerunning the command reuses completed pages without API calls and
rebuilds the wiki, removing obsolete generated files. An interrupted conversion
resumes from the remaining pages. `--refresh` downloads the PDF again; changed
content starts a new conversion cache. Failed downloads or parses leave the
existing wiki intact. `--pages` fills only the requested page caches and leaves
the wiki untouched.

[Docling's hierarchical chunker](https://docling-project.github.io/docling/concepts/chunking/)
groups the cached Markdown blocks under chapter and section headings. This step
runs locally without OCR, model inference, or API calls. The wrapper retains the
original Markdown of each block so formulas, tables, code, and image references
are preserved through grouping.

The PDF outline supplies the hierarchy; matching printed headings locate exact
boundaries within pages. This handles multiple sections on one page and sections
continuing across pages. Chapter starts remain defined by the outline when OCR
omits the title. Unnumbered subheadings remain within their numbered section.
Section numbers use the chapter prefix and outline order to repair OCR omissions.
The wrapper removes repeated running headings, closes page-local code fences,
and joins clear prose continuations across pages. A separate front-matter file
contains the pages before the first bookmark.

Chapter indexes link to their section files, and every file has previous/next
navigation. Page references and printed page labels appear in a collapsible
source list at the end of each file. Chapter filenames start with the PDF page
number; section filenames start with their order within the chapter.
`structure.json` records the hierarchy and source ranges. Images are stored under
`docs/wiki/bayesian-workflow/assets/`. OCR can misread scientific notation and
table columns; check them against the linked source pages when precision matters.

Generated source is exempt from the repository's Markdown and spelling rules
via generated comments. The LaTeX documentation generator respects Git ignore
rules, so it leaves the wiki's equations in Markdown.
