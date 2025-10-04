"""Ollama client wrapper for the Physics Book RAG chatbot.

This module takes the context gathered by the retriever, formats it into a chat
request, and asks a locally hosted Ollama model to craft an answer. The goal is
to hide HTTP details from the rest of the code so other modules can simply call
``LocalLLM.generate(...)`` and focus on their own responsibilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import requests
from requests import RequestException

from .config import Settings
from .retrieval import RetrievedChunk, RetrievedImage


@dataclass
class ChatHistoryEntry:
    user: str
    assistant: str
    # Stores one back-and-forth exchange so the model remembers prior context.


class LocalLLM:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.base_url = settings.llm.base_url
        self.model = settings.llm.model

    def generate(
        self,
        question: str,
        context_chunks: Sequence[RetrievedChunk],
        history: Sequence[ChatHistoryEntry] | None = None,
        image_hits: Sequence[RetrievedImage] | None = None,
    ) -> str:
        system_prompt = self.settings.prompting.system_prompt
        text_context = format_context(context_chunks)
        image_text = format_images(image_hits)

        context_segments = []
        if text_context:
            context_segments.append(text_context)
        if image_text:
            context_segments.append(f"Relevant figures:\n{image_text}")

        context_text = "\n\n".join(context_segments) if context_segments else "(context empty)"

        messages = [
            {"role": "system", "content": system_prompt.strip()},
        ]
        if history:
            for entry in history:
                messages.append({"role": "user", "content": entry.user})
                messages.append({"role": "assistant", "content": entry.assistant})

        question_with_context = (
            "Use the following context to answer the question with citations.\n\n"
            f"Context:\n{context_text}\n\nQuestion: {question}"
        )
        messages.append({"role": "user", "content": question_with_context})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.settings.llm.temperature,
                "top_p": self.settings.llm.top_p,
                "num_ctx": self.settings.llm.context_window,
                "num_predict": self.settings.llm.max_tokens,
            },
        }
        # The payload mirrors Ollama's chat API: specify which model to run,
        # provide the conversation history, and pass generation controls (similar
        # to dials on a radio) that shape the tone and length of the answer.

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
        except RequestException as exc:
            raise RuntimeError(f"Failed to call Ollama chat endpoint: {exc}") from exc
        # If we reach this point, the HTTP call succeeded. Now extract the
        # actual text from Ollama's JSON response.

        data = response.json()
        message = data.get("message", {})
        content = message.get("content", "").strip()
        if not content:
            raise RuntimeError("Ollama returned an empty response. Check the model and server logs.")
        return content


def format_context(chunks: Sequence[RetrievedChunk]) -> str:
    """Convert retrieved chunks into a citation-friendly block of text."""

    if not chunks:
        return ""
    formatted = []
    for idx, chunk in enumerate(chunks, start=1):
        citation = f"[{idx}] ({chunk.source} p.{chunk.page})"
        formatted.append(f"{citation} {chunk.text}")
    return "\n\n".join(formatted)


def format_images(images: Sequence[RetrievedImage] | None) -> str:
    if not images:
        return ""
    formatted = []
    for idx, image in enumerate(images, start=1):
        caption = image.caption or "Image extracted from the book."
        formatted.append(
            f"<fig {idx}> ({image.source} p.{image.page}) {caption}\nFile: {image.image_path}"
        )
    return "\n\n".join(formatted)
