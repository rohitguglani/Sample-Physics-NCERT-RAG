"""Vector store access layer for the Physics Book RAG project.

In plain English, this module knows how to reopen the saved FAISS index and
metadata produced by ingestion, embed a user's question, and find the most
relevant textbook passages. The chatbot later uses these passages to answer with
citation-style references. When image support is enabled, we also maintain a
separate CLIP-powered index so figure-heavy questions can surface diagrams and
illustrations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import ImagesConfig, RetrievalConfig, Settings


@dataclass
class RetrievedChunk:
    text: str
    source: str
    page: int
    score: float
    chunk_id: str
    # Simple container describing one match from the vector store.


@dataclass
class RetrievedImage:
    image_path: str
    source: str
    page: int
    score: float
    chunk_id: str
    caption: str
    # Holds metadata about an extracted figure so the UI can display or link it.


class VectorStore:
    def __init__(
        self,
        index_path: Path,
        metadata_path: Path,
        embedding_model: SentenceTransformer,
        retrieval_config: RetrievalConfig,
        images_config: ImagesConfig,
        image_index_path: Path | None = None,
        image_metadata_path: Path | None = None,
        image_model: SentenceTransformer | None = None,
    ) -> None:
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.embedding_model = embedding_model
        self.retrieval_config = retrieval_config
        self.images_config = images_config
        self.image_index_path = image_index_path
        self.image_metadata_path = image_metadata_path
        self.image_model = image_model

        if not index_path.exists():
            raise FileNotFoundError(f"Vector index not found at {index_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")

        self.index = faiss.read_index(str(index_path))
        self.metadata = _load_metadata(metadata_path)

        self.image_index = None
        self.image_metadata: list[dict[str, object]] = []
        if (
            images_config.enabled
            and image_index_path is not None
            and image_metadata_path is not None
            and image_model is not None
            and image_index_path.exists()
            and image_metadata_path.exists()
        ):
            self.image_index = faiss.read_index(str(image_index_path))
            self.image_metadata = _load_metadata(image_metadata_path)

    def search(self, query: str) -> List[RetrievedChunk]:
        """Embed ``query`` and return the best-matching textbook chunks."""

        query_vec = self.embedding_model.encode(
            [query], normalize_embeddings=True
        )
        top_k = self.retrieval_config.top_k
        distances, indices = self.index.search(query_vec.astype(np.float32), top_k)
        hits: List[RetrievedChunk] = []
        for score, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            record = self.metadata[idx]
            if score < self.retrieval_config.score_threshold:
                continue
            hits.append(
                RetrievedChunk(
                    text=record["text"],
                    source=record["source"],
                    page=int(record["page"]),
                    chunk_id=record["chunk_id"],
                    score=float(score),
                )
            )
        return hits

    def search_images(self, query: str) -> List[RetrievedImage]:
        """Return figure matches for ``query`` if an image index is available."""

        if (
            not self.images_config.enabled
            or self.image_index is None
            or self.image_model is None
            or not self.image_metadata
            or self.retrieval_config.image_top_k <= 0
        ):
            return []

        query_vec = self.image_model.encode(
            [query], normalize_embeddings=True
        )
        top_k = self.retrieval_config.image_top_k
        distances, indices = self.image_index.search(query_vec.astype(np.float32), top_k)

        hits: List[RetrievedImage] = []
        for score, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            if score < self.retrieval_config.image_score_threshold:
                continue
            record = self.image_metadata[idx]
            hits.append(
                RetrievedImage(
                    image_path=str(record["image_path"]),
                    source=str(record["source"]),
                    page=int(record["page"]),
                    chunk_id=str(record["chunk_id"]),
                    score=float(score),
                    caption=str(record.get("caption", "")),
                )
            )
        return hits


def _load_metadata(path: Path) -> List[dict[str, object]]:
    """Read JSON Lines metadata produced during ingestion."""

    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def create_vector_store(settings: Settings) -> VectorStore:
    """Factory helper that wires configuration into a ready-to-use store."""

    embedding_model = SentenceTransformer(settings.embedding.model_name)
    image_model: SentenceTransformer | None = None
    if settings.images.enabled:
        image_model = SentenceTransformer(settings.images.clip_model)

    return VectorStore(
        index_path=settings.paths.vector_store,
        metadata_path=settings.paths.metadata_store,
        embedding_model=embedding_model,
        retrieval_config=settings.retrieval,
        images_config=settings.images,
        image_index_path=settings.paths.image_vector_store,
        image_metadata_path=settings.paths.image_metadata,
        image_model=image_model,
    )
