"""Data ingestion pipeline for the Physics Book RAG project.

This module turns raw textbook PDFs into a vector-searchable knowledge base. The
`ingest` entry point loads configuration, extracts and chunks text from each
document, creates dense embeddings, and persists both the FAISS index and the
chunk metadata so downstream components (retriever + chatbot) can cite the
original sources.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import List

import faiss
from PIL import Image
from rich.console import Console
from sentence_transformers import SentenceTransformer

from .config import load_config
from .utils import extract_pdf_content, ensure_parent

console = Console()


def ingest(config_path: str | Path) -> None:
    """Consume textbook PDFs, produce embeddings, and save the vector store.

    Args:
        config_path: Path to the YAML configuration that controls directories,
            chunk sizes, embedding parameters, and other runtime options.

    The function performs the following high-level steps:
        1. Load strongly-typed settings from disk and collect the list of PDF
           files awaiting ingestion.
        2. Iterate through each document, splitting it into overlapping chunks
           while gathering metadata required for later citation.
        3. Encode every chunk using the configured SentenceTransformer model and
           build an in-memory FAISS index (inner-product similarity).
        4. Persist the FAISS index and a JSONL metadata file to the configured
           output locations, enabling the retriever to reopen them on demand.
    """

    # Load configuration values and ensure the expected directory structure is
    # present (handled inside `load_config`).
    settings = load_config(config_path)

    # Discover every PDF in the raw-data directory. Sorting the list keeps runs
    # deterministic and simplifies comparisons when re-ingesting.
    pdf_paths = sorted(Path(settings.paths.raw_data).glob("*.pdf"))
    if not pdf_paths:
        console.print("[yellow]No PDFs found in data/raw. Add textbooks before ingesting.[/yellow]")
        return

    # Initialise the embedding model once, reusing it across all documents. The
    # `max_seq_length` guard ensures long chunks are truncated consistently.
    text_model = SentenceTransformer(settings.embedding.model_name)
    text_model.max_seq_length = settings.chunking.target_tokens + settings.chunking.overlap_tokens

    image_model: SentenceTransformer | None = None
    if settings.images.enabled:
        image_model = SentenceTransformer(settings.images.clip_model)

    metadata: List[dict[str, str | int | float]] = []
    texts: List[str] = []
    image_metadata: List[dict[str, str | int | float | list[float]]] = []
    image_paths: List[Path] = []

    # Process each PDF, chunking pages into overlapping windows of text and
    # recording enough metadata to reconstruct citations.
    for pdf_path in pdf_paths:
        console.print(f"Processing [cyan]{pdf_path.name}[/cyan]")
        if settings.images.enabled:
            pdf_image_dir = settings.paths.images_dir / pdf_path.stem
            if pdf_image_dir.exists():
                shutil.rmtree(pdf_image_dir)

        text_chunks, figure_records = extract_pdf_content(
            pdf_path,
            target_tokens=settings.chunking.target_tokens,
            overlap_tokens=settings.chunking.overlap_tokens,
            min_chars=settings.chunking.min_chars,
            image_output_root=settings.paths.images_dir,
            max_images_per_page=(
                settings.images.max_per_page if settings.images.enabled else 0
            ),
        )

        for chunk in text_chunks:
            record = {
                "source": pdf_path.name,
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
            }
            metadata.append(record)
            texts.append(str(chunk["text"]))

        if settings.images.enabled:
            for figure in figure_records:
                image_metadata.append(
                    {
                        "source": figure["source"],
                        "page": figure["page"],
                        "chunk_id": figure["chunk_id"],
                        "image_path": str(figure["image_path"]),
                        "caption": figure["caption"],
                        "bbox": figure.get("bbox"),
                    }
                )
                image_paths.append(Path(figure["image_path"]))

    # Bail out early if PDF parsing yielded no usable text (e.g. scanned images
    # without OCR). This prevents creating empty vector stores.
    if not texts:
        console.print("[yellow]No text extracted from PDFs. Check the files for selectable text.[/yellow]")
        return

    # Turn the aggregated chunks into dense vectors. The progress bar helps when
    # ingesting large corpora.
    console.print(f"Embedding {len(texts)} chunks with {settings.embedding.model_name}")
    vectors = text_model.encode(
        texts,
        batch_size=settings.embedding.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # Create a simple FAISS index using inner-product similarity and insert all
    # embeddings in one shot. Storing normalized vectors makes cosine == dot.
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    # Persist the FAISS index so the retriever can reopen it without recomputing
    # embeddings on every run.
    ensure_parent(settings.paths.vector_store)
    faiss.write_index(index, str(settings.paths.vector_store))

    # Write chunk metadata (source file, page, chunk id) in JSON Lines format so
    # the chatbot can cite its answers.
    ensure_parent(settings.paths.metadata_store)
    with settings.paths.metadata_store.open("w", encoding="utf-8") as fh:
        for record in metadata:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    console.print(
        f"[green]Stored {len(texts)} chunks -> {settings.paths.vector_store} and metadata -> {settings.paths.metadata_store}[/green]"
    )

    if settings.images.enabled and image_model is not None and image_paths:
        console.print(
            f"Embedding {len(image_paths)} images with {settings.images.clip_model}"
        )

        pil_images: List[Image.Image] = []
        for image_path in image_paths:
            with Image.open(image_path) as img:
                pil_images.append(img.convert("RGB"))

        image_vectors = image_model.encode(
            images=pil_images,
            batch_size=settings.images.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

        for img in pil_images:
            img.close()

        image_dim = image_vectors.shape[1]
        image_index = faiss.IndexFlatIP(image_dim)
        image_index.add(image_vectors)

        ensure_parent(settings.paths.image_vector_store)
        faiss.write_index(image_index, str(settings.paths.image_vector_store))

        ensure_parent(settings.paths.image_metadata)
        with settings.paths.image_metadata.open("w", encoding="utf-8") as fh:
            for record in image_metadata:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        console.print(
            f"[green]Stored {len(image_paths)} images -> {settings.paths.image_vector_store} and metadata -> {settings.paths.image_metadata}[/green]"
        )
    elif settings.images.enabled:
        console.print("[yellow]Image support enabled but no figures were extracted.[/yellow]")


def build_arg_parser() -> argparse.ArgumentParser:
    """Create a CLI argument parser for the ingestion script."""

    parser = argparse.ArgumentParser(description="Ingest textbook PDFs into the vector store")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to the YAML configuration file",
    )
    return parser


# Enable `python -m physics_rag.ingestion` execution for quick CLI usage.
if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    ingest(args.config)
