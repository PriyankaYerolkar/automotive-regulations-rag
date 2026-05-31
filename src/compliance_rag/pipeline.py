"""End-to-end compliance Q&A pipeline.

Pipeline stage (skills.md Skill 1):
    retrieve -> rerank -> generate -> cite

This module is the single entry point for answering a compliance question.
It wires retrieve_chunks() -> generate_answer() and owns nothing else.

Usage (CLI):
    python -m src.compliance_rag.pipeline \\
        "What is the HIC limit under FMVSS 571.208?"

Usage (module):
    from src.compliance_rag.pipeline import answer
    result = answer("What is the HIC limit under FMVSS 571.208?")
    print(result)

    # When you also need the retrieved chunks (e.g. to render a source panel
    # in the demo UI), use answer_with_sources() instead:
    from src.compliance_rag.pipeline import answer_with_sources
    res = answer_with_sources("What is the HIC limit under FMVSS 571.208?")
    print(res.answer)
    for c in res.chunks:
        ...
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import chromadb
from dotenv import load_dotenv

from .config import settings
from .generate import Chunk, generate_answer
from .retrieve import retrieve_chunks

# Load the project's .env explicitly so a stale parent-directory .env or a
# stale session env var cannot shadow the real keys. override=True makes the
# .env file authoritative over any existing process env vars.
#
# On Hugging Face Spaces there is no .env file (secrets are injected as plain
# env vars), so this call is a harmless no-op there and os.getenv still works.
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)

logger = logging.getLogger(__name__)

# -- defaults (override via CLI flags or by editing config.py) -----------------
_PERSIST = "data/chroma"
_COLLECTION = "fmvss_571_208"


@dataclass(frozen=True)
class QueryResult:
    """A generated answer plus the chunks that were retrieved to produce it.

    The UI uses `chunks` to render a "retrieved sources" panel. `answer` is the
    exact string a plain `answer()` call would return (cited + disclaimer).
    """

    answer: str
    chunks: list[Chunk]


def _get_embedder():  # type: ignore[return]
    """Lazy import so the module loads without OPENAI_API_KEY in test contexts."""
    # embed_index.py lives at src/embed_index.py (flat, Phase 2 layout).
    # When running as `python -m src.compliance_rag.pipeline` the project root
    # is on sys.path, so `src` is importable as a package.
    try:
        from src.embed_index import OpenAIEmbedder  # type: ignore[import]
    except ModuleNotFoundError:
        # Fallback for shells where src/ is on sys.path directly.
        sys.path.insert(0, "src")
        from embed_index import OpenAIEmbedder  # type: ignore[import]
    return OpenAIEmbedder()


def answer_with_sources(
    question: str,
    *,
    persist: str = _PERSIST,
    collection_name: str = _COLLECTION,
) -> QueryResult:
    """Run the full retrieve -> generate pipeline and return answer + chunks.

    Args:
        question:        Natural-language compliance question.
        persist:         Path to the Chroma persist directory.
        collection_name: Chroma collection name (default: fmvss_571_208).

    Returns:
        QueryResult with the cited answer string and the retrieved chunks.
    """
    if not question.strip():
        raise ValueError("question must not be empty")

    embedder = _get_embedder()

    client = chromadb.PersistentClient(path=persist)
    collection = client.get_collection(collection_name)

    chunks = retrieve_chunks(
        question,
        embedder,
        collection,
        top_k=settings.top_k,
        candidate_k=settings.top_k * 3,  # 3x oversample (Skill 1)
        mmr_lambda=settings.mmr_lambda,
    )

    logger.info("retrieved %d chunks for: %s", len(chunks), question[:80])

    text = generate_answer(question, chunks)
    return QueryResult(answer=text, chunks=chunks)


def answer(
    question: str,
    *,
    persist: str = _PERSIST,
    collection_name: str = _COLLECTION,
) -> str:
    """Run the full retrieve -> generate pipeline for one question.

    Thin wrapper over answer_with_sources() so both share one code path; the
    eval harness and CLI keep using this unchanged.

    Returns:
        Cited answer string with the Skill 3 disclaimer appended.
    """
    return answer_with_sources(question, persist=persist, collection_name=collection_name).answer


# -- CLI -----------------------------------------------------------------------


def _cli() -> None:
    load_dotenv()  # reads .env from cwd; no-op if not found
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Ask a compliance question against the FMVSS RAG pipeline."
    )
    parser.add_argument("question", help="compliance question in natural language")
    parser.add_argument("--persist", default=_PERSIST, help="Chroma persist directory")
    parser.add_argument("--collection", default=_COLLECTION, help="Chroma collection name")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set - see .env.example")
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set - see .env.example")

    result = answer(
        args.question,
        persist=args.persist,
        collection_name=args.collection,
    )

    print("\n" + "=" * 70)
    print(result)
    print("=" * 70)


if __name__ == "__main__":
    _cli()
