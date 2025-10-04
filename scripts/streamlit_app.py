"""Streamlit front-end for the Physics Book RAG chatbot.

Run with:
    streamlit run scripts/streamlit_app.py -- --config configs/local.yaml
The extra ``--`` lets Streamlit ignore arguments meant for the app (such as
``--config``), keeping parity with the CLI workflow.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import streamlit as st

from physics_rag.config import Settings, load_config
from physics_rag.llm import ChatHistoryEntry, LocalLLM
from physics_rag.retrieval import RetrievedChunk, RetrievedImage, create_vector_store


@st.cache_resource(show_spinner="Loading RAG pipeline...")
def load_pipeline(config_path: Path) -> tuple[Settings, LocalLLM]:
    """Load configuration, vector stores, and the local LLM once per session."""

    settings = load_config(config_path)
    vector_store = create_vector_store(settings)
    llm = LocalLLM(settings)
    return settings, vector_store, llm


def render_context(chunks: Iterable[RetrievedChunk]) -> None:
    for idx, chunk in enumerate(chunks, start=1):
        st.write(f"[{idx}] {chunk.source} p.{chunk.page} (score {chunk.score:.3f})")
        with st.expander("Show text", expanded=False):
            st.write(chunk.text)


def render_images(images: Iterable[RetrievedImage]) -> None:
    for idx, image in enumerate(images, start=1):
        caption = image.caption or "Extracted figure"
        st.write(f"<fig {idx}> {image.source} p.{image.page} (score {image.score:.3f})")
        st.image(image.image_path, caption=caption, use_column_width=True)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to the YAML configuration file",
    )
    known_args, _ = parser.parse_known_args()
    config_path = Path(known_args.config)

    st.set_page_config(page_title="Physics Book RAG", layout="wide")
    settings, vector_store, llm = load_pipeline(config_path)
    st.title("Physics Book RAG")
    st.caption(
        "Ask questions about your ingested textbooks. The app retrieves relevant "
        "passages and optional figures, then feeds them to your locally hosted Ollama model."
    )

    history: List[ChatHistoryEntry] = st.session_state.setdefault("history", [])

    prompt = st.chat_input("Ask a question about your physics textbooks")
    if prompt:
        text_hits = vector_store.search(prompt)
        image_hits: List[RetrievedImage] = []

        if settings.images.enabled:
            lowered = prompt.lower()
            if any(trigger in lowered for trigger in settings.images.query_triggers):
                image_hits = vector_store.search_images(prompt)

        if not text_hits and not image_hits:
            st.warning(
                "No matching passages or figures found. Try rephrasing the question or ingesting more material."
            )
        else:
            answer = llm.generate(prompt, text_hits, history, image_hits=image_hits)
            history.append(ChatHistoryEntry(user=prompt, assistant=answer))
            st.session_state["latest_text_hits"] = text_hits
            st.session_state["latest_image_hits"] = image_hits

    # Display full conversation
    for entry in history:
        with st.chat_message("user"):
            st.markdown(entry.user)
        with st.chat_message("assistant"):
            st.markdown(entry.assistant)

    if history:
        st.divider()
        st.subheader("Context used")
        text_hits = st.session_state.get("latest_text_hits", [])
        image_hits = st.session_state.get("latest_image_hits", [])
        if text_hits:
            render_context(text_hits)
        if image_hits:
            st.subheader("Figures")
            render_images(image_hits)


if __name__ == "__main__":
    main()
