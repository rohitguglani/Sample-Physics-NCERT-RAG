"""Command-line chat interface for the Physics Book RAG tutor.

When you run ``python -m physics_rag.chat`` this module wires together the
configuration, vector store, and local language model. It behaves like a simple
terminal app: you ask a question, it pulls supporting textbook passages, and it
prints a Markdown-formatted answer with citations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown

from .config import load_config
from .llm import ChatHistoryEntry, LocalLLM
from .retrieval import RetrievedImage, create_vector_store

console = Console()


EXIT_COMMANDS = {"/exit", "/quit", "exit", "quit"}


def chat(config_path: str | Path) -> None:
    """Run the interactive Q&A loop using the provided configuration file."""

    settings = load_config(config_path)  # Read all user-tuned knobs from YAML.
    vector_store = create_vector_store(settings)  # Reopen stored vector indexes.
    llm = LocalLLM(settings)  # Prepare the Ollama client with the same config.

    console.print("[bold green]Physics Book RAG[/bold green] - type /exit to quit")
    history: list[ChatHistoryEntry] = []

    while True:
        question = console.input("[bold blue]You:[/bold blue] ").strip()
        if not question:
            continue
        if question.lower() in EXIT_COMMANDS:
            console.print("[bold]Goodbye![/bold]")
            break

        retrieved = vector_store.search(question)  # Fetch the most relevant textbook slices.
        image_hits: list[RetrievedImage] = []
        if settings.images.enabled:
            lowered = question.lower()
            if any(trigger in lowered for trigger in settings.images.query_triggers):
                image_hits = vector_store.search_images(question)

        if not retrieved and not image_hits:
            console.print("[yellow]No relevant passages or figures found. Consider rephrasing the question.[/yellow]")
            continue

        answer = llm.generate(question, retrieved, history, image_hits=image_hits)
        history.append(ChatHistoryEntry(user=question, assistant=answer))  # Remember conversation flow.

        console.print("[bold green]Bot:[/bold green]")
        console.print(Markdown(answer))

        console.print("[dim]Context used:[/dim]")
        for idx, chunk in enumerate(retrieved, start=1):
            console.print(
                f"[dim][{idx}] {chunk.source} p.{chunk.page} (score: {chunk.score:.3f})[/dim]"
            )
        if image_hits:
            console.print("[dim]Images suggested:[/dim]")
            for idx, image in enumerate(image_hits, start=1):
                console.print(
                    f"[dim]<fig {idx}> {image.source} p.{image.page} (score: {image.score:.3f}) -> {image.image_path}[/dim]"
                )


def build_arg_parser() -> argparse.ArgumentParser:
    """Return a tiny argument parser so the script accepts ``--config``."""

    parser = argparse.ArgumentParser(description="Chat with the Physics RAG bot")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to the YAML configuration file",
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    chat(args.config)
