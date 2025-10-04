"""Configuration helpers for the Physics Book RAG project.

In plain terms, this file reads a YAML settings file and turns it into Python
objects with friendly attribute access (dataclasses). By doing the parsing once
up front we avoid sprinkling string lookups throughout the codebase, leading to
clearer error messages and automatic directory creation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class PathsConfig:
    raw_data: Path
    processed: Path
    vector_store: Path
    metadata_store: Path
    image_vector_store: Path
    image_metadata: Path
    images_dir: Path
    # Each attribute holds a filesystem path. Using Path objects instead of raw
    # strings gives us handy helpers (exists(), mkdir(), etc.).


@dataclass
class EmbeddingConfig:
    model_name: str
    batch_size: int
    # ``model_name`` is the identifier passed to SentenceTransformer.
    # ``batch_size`` controls how many text chunks we embed at a time.


@dataclass
class ChunkingConfig:
    target_tokens: int
    overlap_tokens: int
    min_chars: int
    # These numbers steer how the PDFs are sliced into manageable pieces.


@dataclass
class RetrievalConfig:
    top_k: int
    score_threshold: float
    image_top_k: int = 0
    image_score_threshold: float = 0.0
    # ``top_k`` limits how many chunks we show the language model.
    # ``score_threshold`` filters out low-confidence matches.


@dataclass
class LLMConfig:
    model: str
    base_url: str
    context_window: int
    max_tokens: int
    temperature: float
    top_p: float
    # Holds every knob we tune when talking to Ollama. In layman terms: choose
    # which model to use, where to send the request, and how creative the
    # responses should be.


@dataclass
class PromptingConfig:
    system_prompt: str
    # The system prompt acts like a personality script for the tutor bot.


@dataclass
class ImagesConfig:
    enabled: bool
    clip_model: str
    batch_size: int
    max_per_page: int
    query_triggers: list[str]
    # Knobs steering multimodal support: which model to use, throughput, and
    # simple heuristics for spotting figure-related questions.


@dataclass
class Settings:
    paths: PathsConfig
    embedding: EmbeddingConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    llm: LLMConfig
    prompting: PromptingConfig
    images: ImagesConfig
    # Wrapper that bundles all config sections together for easy passing.


def load_config(path: Path | str) -> Settings:
    config_path = Path(path)
    data = _load_yaml(config_path)

    paths_section = data["paths"]
    paths = PathsConfig(
        raw_data=Path(paths_section["raw_data"]),
        processed=Path(paths_section["processed"]),
        vector_store=Path(paths_section["vector_store"]),
        metadata_store=Path(paths_section["metadata_store"]),
        image_vector_store=Path(paths_section.get("image_vector_store", "data/vector_store/images.faiss")),
        image_metadata=Path(paths_section.get("image_metadata", "data/vector_store/image_metadata.jsonl")),
        images_dir=Path(paths_section.get("images_dir", "data/processed/images")),
    )

    embedding = EmbeddingConfig(**data["embedding"])
    chunking = ChunkingConfig(**data["chunking"])
    retrieval = RetrievalConfig(**data["retrieval"])
    llm = LLMConfig(**data["llm"])
    llm.base_url = llm.base_url.rstrip("/")
    prompting = PromptingConfig(**data["prompting"])
    images_section = data.get("images", {})
    images = ImagesConfig(
        enabled=images_section.get("enabled", False),
        clip_model=images_section.get("clip_model", "sentence-transformers/clip-ViT-B-32"),
        batch_size=images_section.get("batch_size", 8),
        max_per_page=images_section.get("max_per_page", 6),
        query_triggers=images_section.get("query_triggers", [
            "figure",
            "diagram",
            "image",
            "illustration",
            "graph",
            "chart",
        ]),
    )

    settings = Settings(
        paths=paths,
        embedding=embedding,
        chunking=chunking,
        retrieval=retrieval,
        llm=llm,
        prompting=prompting,
        images=images,
    )
    ensure_directories(settings)
    return settings


def ensure_directories(settings: Settings) -> None:
    """Create the folders referenced in the config if they are missing."""

    settings.paths.raw_data.mkdir(parents=True, exist_ok=True)
    settings.paths.processed.mkdir(parents=True, exist_ok=True)
    settings.paths.vector_store.parent.mkdir(parents=True, exist_ok=True)
    settings.paths.metadata_store.parent.mkdir(parents=True, exist_ok=True)
    settings.paths.image_vector_store.parent.mkdir(parents=True, exist_ok=True)
    settings.paths.image_metadata.parent.mkdir(parents=True, exist_ok=True)
    settings.paths.images_dir.mkdir(parents=True, exist_ok=True)


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Read the YAML file from disk and return a plain dictionary."""

    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
