"""Utility helpers shared across the Physics Book RAG project.

This file contains small, reusable functions for dealing with textbook PDFs:
cleaning up extracted text, splitting it into bite-sized pieces, and generating
stable identifiers. Think of it as the toolbox we reach for when preparing
documents before they become searchable.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterator, List, Tuple

import fitz  # type: ignore[import-not-found]


def normalize_text(text: str) -> str:
    """Clean raw PDF text so it is easier to chunk and embed.

    Layman translation: remove odd characters (like soft hyphens), squeeze
    repeated whitespace into single spaces, and trim stray spaces at the ends.
    This makes the downstream chunks tidier and helps the embedding model focus
    on the actual words.
    """

    text = text.replace("\u00ad", "")  # Soft hyphen: shows up in PDFs as word-break hints.
    text = re.sub(r"\s+", " ", text)  # Collapse every run of whitespace into a single space.
    return text.strip()


def chunk_text(
    text: str,
    target_tokens: int,
    overlap_tokens: int,
    min_chars: int,
) -> Iterator[str]:
    """Yield overlapping slices of a long string.

    In everyday terms: imagine moving a sliding window across the words so each
    chunk is digestible for the language model while still overlapping with its
    neighbours. Overlap keeps context like equations or sentence endings intact.
    """

    if not text:
        return  # Nothing to do if the page is empty or only images.
    tokens = text.split()
    start = 0
    length = len(tokens)
    while start < length:
        end = min(start + target_tokens, length)
        chunk = " ".join(tokens[start:end]).strip()
        if len(chunk) >= min_chars:
            yield chunk
        if end >= length:
            break
        start = max(end - overlap_tokens, start + 1)


def extract_pdf_content(
    pdf_path: Path,
    target_tokens: int,
    overlap_tokens: int,
    min_chars: int,
    image_output_root: Path,
    max_images_per_page: int,
) -> Tuple[List[dict[str, object]], List[dict[str, object]]]:
    """Return layout-aware text chunks and extracted images for a PDF.

    Args:
        pdf_path: The textbook PDF to process.
        target_tokens / overlap_tokens / min_chars: Same knobs used by
            :func:`chunk_text`, controlling how we split paragraphs into
            language-model-friendly segments.
        image_output_root: Directory under which we store cropped figures.
        max_images_per_page: Safety limit to avoid dumping dozens of small
            decorative icons from a single page.

    Returns:
        Two lists:
            1. Dictionaries describing text chunks (page, text, bbox, chunk_id).
            2. Dictionaries describing extracted figures (page, image_path,
               caption, etc.).
    """

    text_chunks: List[dict[str, object]] = []
    image_records: List[dict[str, object]] = []

    doc = fitz.open(pdf_path)
    image_subdir = image_output_root / pdf_path.stem

    for page_index, page in enumerate(doc, start=1):
        page_blocks = page.get_text("dict").get("blocks", [])
        images_on_page = 0

        for block in page_blocks:
            block_type = block.get("type")

            if block_type == 0:  # Text block
                block_text = _flatten_block_text(block)
                if not block_text:
                    continue
                for chunk in chunk_text(
                    block_text,
                    target_tokens=target_tokens,
                    overlap_tokens=overlap_tokens,
                    min_chars=min_chars,
                ):
                    text_chunks.append(
                        {
                            "page": page_index,
                            "text": chunk,
                            "chunk_id": hash_text(
                                f"{pdf_path.name}:{page_index}:{chunk[:48]}"
                            )[:12],
                            "bbox": block.get("bbox"),
                        }
                    )

            elif block_type == 1 and max_images_per_page > 0:  # Image block
                if images_on_page >= max_images_per_page:
                    continue
                xref = block.get("image")
                if xref is None:
                    continue

                try:
                    pix = fitz.Pixmap(doc, xref)
                except RuntimeError:
                    continue  # Skip embedded formats PyMuPDF cannot decode.

                if pix.n >= 5:  # Convert CMYK and similar to RGB.
                    pix = fitz.Pixmap(fitz.csRGB, pix)

                image_name = f"{pdf_path.stem}_p{page_index:03d}_img{images_on_page+1:02d}.png"
                image_path = image_subdir / image_name
                ensure_parent(image_path)
                pix.save(str(image_path))
                images_on_page += 1

                image_records.append(
                    {
                        "page": page_index,
                        "image_path": image_path,
                        "source": pdf_path.name,
                        "chunk_id": hash_text(
                            f"{pdf_path.name}:{page_index}:image:{xref}"
                        )[:12],
                        "bbox": block.get("bbox"),
                        "caption": f"Figure captured from page {page_index} of {pdf_path.name}",
                    }
                )

    return text_chunks, image_records


def hash_text(text: str) -> str:
    """Return a deterministic fingerprint for the provided text."""

    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def ensure_parent(path: Path) -> None:
    """Create the folder containing ``path`` if it does not already exist."""

    path.parent.mkdir(parents=True, exist_ok=True)


def _flatten_block_text(block: dict) -> str:
    lines = block.get("lines", [])
    parts = []
    for line in lines:
        for span in line.get("spans", []):
            parts.append(span.get("text", ""))
        parts.append("\n")
    return normalize_text(" ".join(parts))
