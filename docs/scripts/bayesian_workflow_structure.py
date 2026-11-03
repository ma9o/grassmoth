"""Arrange cached Bayesian Workflow Markdown into a section-based wiki."""

import json
import posixpath
import re
from dataclasses import dataclass, field, replace
from html import unescape
from importlib.metadata import version
from itertools import pairwise
from pathlib import Path

from markdown_it import MarkdownIt

EDITION_NOTICE = "This electronic edition is for non-commercial purposes only."
GENERATED_HEADER = "<!-- Generated source extraction; do not edit. -->\n<!-- markdownlint-disable -->\n<!-- cspell:disable -->"
DOCLING_DOCS = "https://docling-project.github.io/docling/concepts/chunking/"
PARSER = MarkdownIt().enable("table")


@dataclass(frozen=True)
class Bookmark:
    title: str
    page: int
    depth: int


@dataclass(frozen=True)
class Block:
    markdown: str
    page: int
    line: int
    kind: str
    title: str = ""
    level: int = 0


@dataclass
class Entry:
    title: str
    path: Path
    start: int
    blocks: list[Block] = field(default_factory=list)

    @property
    def end(self) -> int:
        return max([self.start, *(block.page for block in self.blocks)])


@dataclass
class Chapter:
    index: Entry
    sections: list[Entry] = field(default_factory=list)
    source_end: int = 0


def heading_key(text: str) -> str:
    text = (
        unescape(re.sub(r"<[^>]+>", "", PARSER.renderInline(text))).casefold().strip()
    )
    text = re.sub(
        r"^(?:(?:chapter|part|appendix)\s+(?:\d+|[ivxlcdm]+|[ab])[:.]?"
        r"|\d+(?:\.\d+)*\.?|[a-z]\.\d+(?:\.\d+)*\.?)\s+",
        "",
        text,
    )
    return " ".join(re.sub(r"['`‘’“”\"$]", "", text).split())


def page_blocks(markdown: str, page: int) -> list[Block]:
    """Keep the source Markdown of complete blocks, including tables and code."""
    lines = markdown.splitlines()
    for token in PARSER.parse(markdown):
        if token.type == "inline" and token.content.partition("\n")[0] == "⬇":
            start = token.map[0]
            end = start + 1
            while end < len(lines) and lines[end].strip():
                end += 1
            lines[start] = "```text"
            lines[end - 1] += "\n```"
        elif (
            token.type == "inline"
            and token.map
            and any(child.type == "image" for child in token.children or [])
        ):
            # OCR sometimes puts an image and a running header in one paragraph.
            # Separate standalone images before identifying removable headings.
            for index in range(*token.map):
                children = PARSER.parseInline(lines[index].strip())[0].children
                if len(children) == 1 and children[0].type == "image":
                    lines[index] = "\n" + lines[index] + "\n"
    markdown = "\n".join(lines)
    # A page-local OCR fence must not swallow text and headings on later pages.
    for token in PARSER.parse(markdown):
        if token.type == "fence":
            last = markdown.splitlines()[token.map[1] - 1]
            closing = (
                r" {0,3}"
                + re.escape(token.markup[0])
                + "{"
                + str(len(token.markup))
                + r",}\s*"
            )
            if not re.fullmatch(closing, last):
                markdown += "\n" + token.markup
    lines = markdown.splitlines()
    tokens = PARSER.parse(markdown)
    blocks = []
    for index, token in enumerate(tokens):
        if token.level != 0 or token.map is None or token.nesting == -1:
            continue
        title = (
            tokens[index + 1].content
            if token.type in {"heading_open", "paragraph_open"}
            else ""
        )
        start, end = token.map
        blocks.append(
            Block(
                "\n".join(lines[start:end]),
                page,
                start,
                token.type,
                title,
                int(token.tag[1:]) if token.type == "heading_open" else 0,
            )
        )
    return blocks


def slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", heading_key(title)).strip("-")


def arrange(
    pages: dict[int, str], bookmarks: list[Bookmark]
) -> tuple[list[Chapter], dict]:
    """Use outline-aligned headings and Docling's chunker to assign source blocks."""
    from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker
    from docling_core.types.doc import DocItemLabel, DoclingDocument

    bookmark_keys = {heading_key(bookmark.title) for bookmark in bookmarks}
    blocks = []
    removed = 0
    for page, markdown in pages.items():
        lines = []
        for line in markdown.splitlines():
            title = re.sub(r"^#{1,6}\s+", "", line.strip())
            # Remove page furniture before parsing: OCR can attach it to images
            # or put it inside an unfinished code fence at a page boundary.
            if title == EDITION_NOTICE or (
                title.isupper() and heading_key(title) in bookmark_keys
            ):
                removed += 1
            else:
                lines.append(line)
        blocks.extend(page_blocks("\n".join(lines), page))
    boundaries = []
    for bookmark in bookmarks:
        # The outline is approximate: e.g. References points two pages before its
        # printed heading. Uppercase running headers are not section boundaries.
        candidates = [
            block
            for block in blocks
            if bookmark.page <= block.page <= bookmark.page + 2
            and block.title
            and not block.title.isupper()
            and heading_key(block.title) == heading_key(bookmark.title)
        ]
        if candidates:
            first_page = min(block.page for block in candidates)
            chosen = max(
                (block for block in candidates if block.page == first_page),
                key=lambda block: block.line,
            )
            boundaries.append((chosen.page, chosen.line, bookmark, chosen.title, True))
        elif bookmark.depth == 0:
            # Chapter starts remain defined by the PDF outline when OCR omitted
            # the printed title (as it does for "Building statistical models").
            boundaries.append((bookmark.page, 0, bookmark, bookmark.title, False))
        else:
            raise ValueError(
                f"Section heading missing near PDF page {bookmark.page}: {bookmark.title}"
            )
    if any(a[:2] >= b[:2] for a, b in pairwise(boundaries)):
        raise ValueError("Detected headings do not follow the PDF outline order.")

    doc = DoclingDocument(name="Bayesian Workflow")
    chapters = [Chapter(Entry("Front matter", Path("001-front-matter.md"), 1))]
    chapter = chapters[0]
    doc.add_heading(text=chapter.index.path.stem, level=1)
    doc.add_heading(text="overview", level=2)
    entries = {(chapter.index.path.stem, "overview"): chapter.index}
    source_blocks = {}
    boundary_index = 0
    for block in blocks:
        is_boundary = False
        while boundary_index < len(boundaries) and boundaries[boundary_index][:2] <= (
            block.page,
            block.line,
        ):
            page, line, bookmark, label, printed = boundaries[boundary_index]
            if bookmark.depth == 0:
                chapter = Chapter(
                    Entry(
                        bookmark.title,
                        Path(f"{page:03d}-{slug(bookmark.title)}.md"),
                        page,
                    )
                )
                chapters.append(chapter)
                doc.add_heading(text=chapter.index.path.stem, level=1)
                doc.add_heading(text="overview", level=2)
                entries[(chapter.index.path.stem, "overview")] = chapter.index
            else:
                entry = Entry(
                    label,
                    Path(chapter.index.path.stem)
                    / f"{len(chapter.sections) + 1:02d}-{slug(bookmark.title)}.md",
                    page,
                )
                chapter.sections.append(entry)
                doc.add_heading(text=entry.path.stem, level=2)
                entries[(chapter.index.path.stem, entry.path.stem)] = entry
            is_boundary = is_boundary or (
                printed and (page, line) == (block.page, block.line)
            )
            boundary_index += 1
        if is_boundary:
            continue
        if block.title and (
            block.title == EDITION_NOTICE
            or heading_key(block.title) in bookmark_keys
            or (
                len(chapters) > 1
                and re.fullmatch(
                    r"(?:Chapter|Part|Appendix)\s+(?:\d+|[IVXLCAB]+)",
                    block.title,
                    re.IGNORECASE,
                )
            )
        ):
            removed += 1
            continue
        if block.kind == "heading_open":
            level = max(3, block.level)
            doc.add_heading(text=block.title, level=level)
            block = replace(block, markdown="#" * (level - 1) + " " + block.title)
        # Feed intact Markdown blocks to the structure model. Keeping the source
        # block avoids re-serializing formulas, code, and image links as plain text.
        item = doc.add_text(label=DocItemLabel.TEXT, text=block.markdown)
        source_blocks[item.self_ref] = block
    if boundary_index != len(boundaries):
        raise ValueError("Some outline headings were not reached.")

    seen = set()
    for chunk in HierarchicalChunker().chunk(doc):
        entry = entries[tuple(chunk.meta.headings[:2])]
        for item in chunk.meta.doc_items:
            if item.self_ref in seen:
                raise ValueError("Docling returned the same source block twice.")
            seen.add(item.self_ref)
            entry.blocks.append(source_blocks[item.self_ref])
    if seen != source_blocks.keys():
        raise ValueError("Docling did not return every source block.")
    for index, chapter in enumerate(chapters):
        if chapter.sections:
            prefixes = {
                match[1]
                for entry in chapter.sections
                if (match := re.match(r"(\d+|[A-Z])\.\d+", entry.title))
            }
            if len(prefixes) != 1:
                raise ValueError(
                    f"Inconsistent section numbering: {chapter.index.title}"
                )
            prefix = prefixes.pop()
            for number, entry in enumerate(chapter.sections, 1):
                # OCR occasionally drops the chapter prefix or the whole number.
                title = re.sub(r"^(?:(?:\d+|[A-Z])\.)?\d+\.?\s+", "", entry.title)
                entry.title = f"{prefix}.{number} {title}"
        next_start = (
            chapters[index + 1].index.start
            if index + 1 < len(chapters)
            else len(pages) + 1
        )
        chapter.source_end = max(
            next_start - 1,
            chapter.index.end,
            *(entry.end for entry in chapter.sections),
        )
    return chapters, {
        "docling_core": version("docling-core"),
        "source_blocks": len(source_blocks),
        "running_headers_removed": removed,
        "printed_heading_boundaries": sum(boundary[4] for boundary in boundaries),
        "outline_boundaries": sum(not boundary[4] for boundary in boundaries),
    }


def body_markdown(entry: Entry) -> str:
    parts = []
    previous = None
    for block in entry.blocks:
        text = block.markdown
        if (
            previous is not None
            and previous.kind == block.kind == "paragraph_open"
            and block.page == previous.page + 1
            and parts
            and re.match(r"[a-z]", text)
            and not re.search(r"[.!?:;][\"'’”)]*$", parts[-1])
            and not parts[-1].lstrip().startswith(("$$", r"\[", "!["))
        ):
            parts[-1] += ("" if parts[-1].endswith("-") else " ") + text
        else:
            parts.append(text)
        previous = block
    markdown = "\n\n".join(parts).strip()
    if len(entry.path.parts) > 1:
        markdown = markdown.replace("](assets/", "](../assets/")
    return markdown


def link(source: Entry, target: Path) -> str:
    return posixpath.relpath(target.as_posix(), source.path.parent.as_posix())


def source_links(
    start: int, end: int, path: Path, labels: list[str], source_url: str
) -> str:
    pdf = "../" * (len(path.parts) + 2) + "scratchpad/Bayesian-Workflow.pdf"
    lines = ["<details>", f"<summary>Source pages: PDF {start}–{end}</summary>", ""]
    for page in range(start, end + 1):
        lines.append(
            f'- <a id="pdf-page-{page}"></a>[PDF {page}]({source_url}#page={page}) '
            f"· [Local PDF]({pdf}#page={page}) · printed page {labels[page - 1]}"
        )
    return "\n".join([*lines, "", "</details>"])


def render_wiki(
    pages: dict[int, str],
    bookmarks: list[Bookmark],
    labels: list[str],
    digest: str,
    source_url: str,
) -> dict[Path, bytes]:
    chapters, metadata = arrange(pages, bookmarks)
    documents = [
        entry for chapter in chapters for entry in [chapter.index, *chapter.sections]
    ]
    owners = {
        entry.path: chapter
        for chapter in chapters
        for entry in [chapter.index, *chapter.sections]
    }
    files = {}
    for index, entry in enumerate(documents):
        chapter = owners[entry.path]
        navigation = [f"[Book index]({link(entry, Path('../bayesian-workflow.md'))})"]
        if entry is not chapter.index:
            navigation.append(
                f"[Chapter: {chapter.index.title}]({link(entry, chapter.index.path)})"
            )
        if index:
            navigation.append(
                f"[Previous: {documents[index - 1].title}]({link(entry, documents[index - 1].path)})"
            )
        if index + 1 < len(documents):
            navigation.append(
                f"[Next: {documents[index + 1].title}]({link(entry, documents[index + 1].path)})"
            )
        end = chapter.source_end if entry is chapter.index else entry.end
        lines = [
            GENERATED_HEADER,
            f"# {entry.title}",
            "",
            " | ".join(navigation),
            "",
            f"[Source PDF pages {entry.start}–{end}]({source_url}#page={entry.start})",
            "",
            body_markdown(entry),
        ]
        if entry is chapter.index and chapter.sections:
            lines.extend(
                ["", "## Sections", "", "| Section | PDF pages |", "| --- | --- |"]
            )
            lines.extend(
                f"| [{section.title}]({link(entry, section.path)}) | {section.start}–{section.end} |"
                for section in chapter.sections
            )
        lines.extend(
            ["", source_links(entry.start, end, entry.path, labels, source_url), ""]
        )
        files[Path("bayesian-workflow") / entry.path] = "\n".join(lines).encode()
    index_lines = [
        GENERATED_HEADER,
        "# Bayesian Workflow",
        "",
        f"- Source: [Author-hosted PDF]({source_url}) · [Local PDF](../../scratchpad/Bayesian-Workflow.pdf).",
        f"- {len(pages)} PDF pages; {len(chapters)} chapter/index files and {len(documents) - len(chapters)} section files.",
        f"- SHA-256: `{digest}`.",
        "- OCR: Mistral through OpenRouter; original responses are cached locally.",
        f"- Structure: [Docling HierarchicalChunker]({DOCLING_DOCS}) (`docling-core {version('docling-core')}`).",
        "- Rebuild: `bun run --cwd docs bayesian-workflow` (add `--refresh` to download again).",
        "",
        EDITION_NOTICE,
        "",
        "Sections follow the book's outline and printed headings. Source-page links are collected at the end of each file.",
        "Check the linked PDF when precision matters: OCR can misread formulas and table columns.",
        "",
        "## Chapters",
        "",
        "| Chapter | Sections | PDF pages |",
        "| --- | --- | --- |",
    ]
    index_lines.extend(
        f"| [{chapter.index.title}](bayesian-workflow/{chapter.index.path}) | {len(chapter.sections)} | {chapter.index.start}–{chapter.source_end} |"
        for chapter in chapters
    )
    files[Path("bayesian-workflow.md")] = ("\n".join(index_lines) + "\n").encode()
    metadata["chapters"] = [
        {
            "path": str(chapter.index.path),
            "title": chapter.index.title,
            "pages": [chapter.index.start, chapter.source_end],
            "sections": [
                {
                    "path": str(entry.path),
                    "title": entry.title,
                    "pages": [entry.start, entry.end],
                }
                for entry in chapter.sections
            ],
        }
        for chapter in chapters
    ]
    files[Path("bayesian-workflow/structure.json")] = (
        json.dumps(metadata, indent=2) + "\n"
    ).encode()
    return files
