# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["pypdf>=6.19.0", "markdown-it-py>=4.2.0", "python-dotenv>=1.2.3", "docling-core[chunking]>=2.99.0"]
# ///
"""Download Bayesian Workflow and rebuild its local wiki indexes and section files."""

import argparse
import base64
import hashlib
import io
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen

from bayesian_workflow_structure import Bookmark, render_wiki
from dotenv import load_dotenv
from markdown_it import MarkdownIt
from pypdf import PdfReader, PdfWriter
from pypdf.generic import Destination

type Outline = list[Destination | Outline]

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_URL = "https://users.aalto.fi/~ave/Bayesian-Workflow.pdf"
PDF_PATH = REPO_ROOT / "scratchpad/Bayesian-Workflow.pdf"
WIKI_PATH = REPO_ROOT / "docs/wiki/bayesian-workflow.md"
CHAPTERS_PATH = WIKI_PATH.with_suffix("")
OCR_CACHE_PATH = REPO_ROOT / "scratchpad/bayesian-workflow-ocr"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OCR_OPTIONS = {
    "model": "google/gemma-3-4b-it",
    "plugins": [{"id": "file-parser", "pdf": {"engine": "mistral-ocr"}}],
    "max_tokens": 1,
    "temperature": 0,
    "stream": False,
}


def download_pdf() -> None:
    """Validate the download before replacing the cached PDF."""
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=".bayesian-workflow-", dir=PDF_PATH.parent
    ) as temporary:
        downloaded = Path(temporary) / PDF_PATH.name
        print(f"Downloading {SOURCE_URL}", flush=True)
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--connect-timeout",
                "30",
                "--max-time",
                "300",
                "--output",
                str(downloaded),
                SOURCE_URL,
            ],
            check=True,
        )
        with downloaded.open("rb") as source:
            pdf = PdfReader(source, strict=True)
            if not pdf.pdf_header.startswith("%PDF-") or len(pdf.pages) == 0:
                raise ValueError("The source did not return a nonempty PDF.")
        downloaded.replace(PDF_PATH)


def read_bookmarks(pdf: PdfReader, outline: Outline, depth: int = 0) -> list[Bookmark]:
    """Read PDF bookmarks in order, retaining their hierarchy and page numbers."""
    bookmarks = []
    for item in outline:
        if isinstance(item, list):
            bookmarks.extend(read_bookmarks(pdf, item, depth + 1))
        else:
            page_index = pdf.get_destination_page_number(item)
            if page_index is None or not 0 <= page_index < len(pdf.pages):
                raise ValueError(f"Bookmark has no valid page: {item.title}")
            title = re.sub(r"([\\`*_\[\]<>])", r"\\\1", " ".join(item.title.split()))
            bookmarks.append(Bookmark(title, page_index + 1, depth))
    return bookmarks


def conversion_cache(digest: str) -> Path:
    """Identify the source and the hosted parser settings used for each page."""
    identity = {
        "source_sha256": digest,
        "endpoint": OPENROUTER_URL,
        "options": OCR_OPTIONS,
        "pypdf": version("pypdf"),
        "pages_per_request": 1,
    }
    cache_key = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()
    cache_root = OCR_CACHE_PATH / cache_key
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / "source.json").write_text(
        json.dumps(identity, indent=2) + "\n", encoding="utf-8"
    )
    return cache_root


def extract_page(response: dict, page: int) -> tuple[str, dict[Path, bytes]]:
    """Read the parser's annotations, never the downstream model's generated reply."""
    annotations = response["choices"][0]["message"]["annotations"]
    files = [
        annotation["file"] for annotation in annotations if annotation["type"] == "file"
    ]
    if len(files) != 1 or files[0]["name"] != f"page-{page:03d}.pdf":
        raise ValueError(f"Missing OCR annotation for PDF page {page}.")
    parts = files[0]["content"]
    texts = [part["text"] for part in parts if part["type"] == "text"]
    if texts[0] != f'<file name="page-{page:03d}.pdf">' or texts[-1] != "</file>":
        raise ValueError(f"Unexpected OCR annotation format for PDF page {page}.")
    markdown = "\n\n".join(texts[1:-1]).strip()
    image_names = list(
        dict.fromkeys(
            child.attrGet("src")
            for token in MarkdownIt().parse(markdown)
            for child in token.children or []
            if child.type == "image"
        )
    )
    images = [part["image_url"]["url"] for part in parts if part["type"] == "image_url"]
    if len(image_names) != len(images):
        raise ValueError(f"OCR image count does not match Markdown on PDF page {page}.")
    assets = {}
    for name, image in zip(image_names, images, strict=True):
        if not re.fullmatch(r"img-\d+\.(?:jpeg|png)", name):
            raise ValueError(f"Unexpected OCR image name on PDF page {page}: {name}")
        media, encoded = image.split(",", 1)
        if media not in {"data:image/jpeg;base64", "data:image/png;base64"}:
            raise ValueError(f"Unexpected OCR image format on PDF page {page}.")
        path = Path("assets") / f"{page:03d}-{name}"
        assets[path] = base64.b64decode(encoded, validate=True)
        markdown = markdown.replace(f"]({name})", f"]({path.as_posix()})")
    return markdown, assets


def convert_page(page: int, document: bytes, cache: Path, key: str) -> None:
    encoded = base64.b64encode(document).decode("ascii")
    payload = {
        **OCR_OPTIONS,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Reply only OK."},
                    {
                        "type": "file",
                        "file": {
                            "filename": f"page-{page:03d}.pdf",
                            "file_data": f"data:application/pdf;base64,{encoded}",
                        },
                    },
                ],
            }
        ],
    }
    request = Request(
        OPENROUTER_URL,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urlopen(request, timeout=180) as response:
        result = json.load(response)
    # Persist the paid response before checking its structure, so it can be inspected
    # and repaired locally without paying to convert that page again.
    with TemporaryDirectory(prefix=".page-", dir=cache) as temporary:
        output = Path(temporary) / f"{page:03d}.json"
        output.write_text(json.dumps(result) + "\n", encoding="utf-8")
        output.replace(cache / output.name)
    extract_page(result, page)


def convert_pages(pdf: PdfReader, pages: list[int], cache: Path, workers: int) -> None:
    missing = [page for page in pages if not (cache / f"{page:03d}.json").exists()]
    print(
        f"Using {len(pages) - len(missing)} cached pages; converting {len(missing)} pages.",
        flush=True,
    )
    if not missing:
        return
    load_dotenv(REPO_ROOT / ".env")
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("Set OPENROUTER_API_KEY in the environment or root .env.")
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = []
        for page in missing:
            writer = PdfWriter()
            writer.add_page(pdf.pages[page - 1])
            document = io.BytesIO()
            writer.write(document)
            futures.append(
                pool.submit(convert_page, page, document.getvalue(), cache, key)
            )
        for count, future in enumerate(as_completed(futures), start=1):
            future.result()
            if count % 10 == 0 or count == len(missing):
                print(f"Converted {count}/{len(missing)} new pages.", flush=True)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def build_wiki(pdf: PdfReader, digest: str, cache: Path) -> dict[Path, bytes]:
    pages = {}
    files = {}
    for page in range(1, len(pdf.pages) + 1):
        response = json.loads((cache / f"{page:03d}.json").read_text(encoding="utf-8"))
        pages[page], assets = extract_page(response, page)
        files.update(
            {
                Path(CHAPTERS_PATH.name) / path: content
                for path, content in assets.items()
            }
        )
    files.update(
        render_wiki(
            pages, read_bookmarks(pdf, pdf.outline), pdf.page_labels, digest, SOURCE_URL
        )
    )
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh", action="store_true", help="Download the PDF again before parsing."
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="Concurrent OCR requests (default: 8)."
    )
    parser.add_argument(
        "--pages",
        type=int,
        nargs="+",
        help="Cache only these 1-based PDF pages; do not rebuild the wiki.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.refresh or not PDF_PATH.exists():
        download_pdf()
    else:
        print(f"Using cached {PDF_PATH.relative_to(REPO_ROOT)}", flush=True)

    with PDF_PATH.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
        pdf = PdfReader(source, strict=True)
        pages = (
            sorted(set(args.pages))
            if args.pages
            else list(range(1, len(pdf.pages) + 1))
        )
        if not all(1 <= page <= len(pdf.pages) for page in pages):
            parser.error(f"--pages must be between 1 and {len(pdf.pages)}")
        cache = conversion_cache(digest)
        convert_pages(pdf, pages, cache, args.workers)
        if args.pages:
            print(
                f"Page responses cached at {cache.relative_to(REPO_ROOT)}", flush=True
            )
            return
        files = build_wiki(pdf, digest, cache)

    WIKI_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=".bayesian-workflow-", dir=WIKI_PATH.parent
    ) as temporary:
        for relative_path, content in files.items():
            output = Path(temporary) / relative_path
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(content)
        CHAPTERS_PATH.mkdir(exist_ok=True)
        for relative_path in files:
            destination = WIKI_PATH.parent / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            (Path(temporary) / relative_path).replace(destination)
        for stale in CHAPTERS_PATH.rglob("*"):
            if stale.is_file() and stale.relative_to(WIKI_PATH.parent) not in files:
                stale.unlink()
    print(
        f"Wrote {WIKI_PATH.relative_to(REPO_ROOT)} and {len(files) - 1} section/index/asset files",
        flush=True,
    )


if __name__ == "__main__":
    main()
