"""Embed FMVSS chunks with OpenAI and store them in a local Chroma index.

Pipeline stage (skills.md Skill 1):  embed -> store  (retrieve/rerank/generate
come in Phase 3). One embedding model per index, never mixed (Skill 1).

Requires the OPENAI_API_KEY environment variable (see .env.example). The key is
read by the OpenAI SDK automatically; it is never passed on the command line.

    export OPENAI_API_KEY=sk-...
    python embed_index.py data/processed/fmvss_571_208_chunks.json \\
        --persist data/chroma \\
        --query "What seat belt is required at rear seating positions?"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Protocol

import chromadb

logger = logging.getLogger("embed_index")

MODEL = "text-embedding-3-small"  # 1536-dim; needs no query/passage prefix
EMBED_BATCH = 256  # inputs per API call (limit is 2048 / 8191 tok)
COLLECTION = "fmvss_571_208"
TOP_K = 5

# Chroma metadata values must be str/int/float/bool (no None, no nesting).
# These are the chunk fields that travel as metadata; `text` becomes the
# document body and `chunk_id` becomes the Chroma id.
META_FIELDS = (
    "regulation",
    "section",
    "subsection",
    "page",
    "effective_date",
    "source_url",
    "parent_heading",
    "part_index",
)


class Embedder(Protocol):
    """Anything that turns a list of strings into a list of vectors.

    The real implementation wraps the OpenAI client; tests can pass a stub so
    the storage/search path runs without network access or an API key.
    """

    def __call__(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    """Batched embeddings via the OpenAI API."""

    def __init__(self, model: str = MODEL, batch_size: int = EMBED_BATCH) -> None:
        from openai import OpenAI  # imported lazily so tests need no key

        self._client = OpenAI()
        self._model = model
        self._batch = batch_size

    def __call__(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch):
            batch = texts[start : start + self._batch]
            resp = self._client.embeddings.create(model=self._model, input=batch)
            vectors.extend(item.embedding for item in resp.data)
            logger.info("embedded %d/%d", min(start + self._batch, len(texts)), len(texts))
        return vectors


# ---------------------------------------------------------------------------
def load_chunks(path: Path) -> list[dict]:
    chunks = json.loads(path.read_text())
    if not chunks:
        raise ValueError(f"no chunks found in {path}")
    logger.info("loaded %d chunks from %s", len(chunks), path)
    return chunks


def _metadata(chunk: dict) -> dict:
    """Project a chunk to Chroma-safe metadata (no None values)."""
    return {key: ("" if chunk.get(key) is None else chunk[key]) for key in META_FIELDS}


def build_index(
    chunks: list[dict],
    embed: Embedder,
    *,
    persist_dir: Path,
    collection_name: str = COLLECTION,
) -> chromadb.api.models.Collection.Collection:
    """Embed every chunk and (re)build a persistent cosine-distance index."""
    vectors = embed([c["text"] for c in chunks])
    if len(vectors) != len(chunks):
        raise RuntimeError(f"embedding count {len(vectors)} != chunk count {len(chunks)}")

    client = chromadb.PersistentClient(path=str(persist_dir))
    # Rebuild from scratch so re-runs are idempotent (last-write-wins).
    if collection_name in {c.name for c in client.list_collections()}:
        client.delete_collection(collection_name)
    collection = client.create_collection(
        collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=vectors,
        documents=[c["text"] for c in chunks],
        metadatas=[_metadata(c) for c in chunks],
    )
    logger.info(
        "indexed %d chunks into '%s' at %s", collection.count(), collection_name, persist_dir
    )
    return collection


def search(
    query: str,
    embed: Embedder,
    collection: chromadb.api.models.Collection.Collection,
    *,
    k: int = TOP_K,
) -> list[dict]:
    """Return the k nearest chunks to the query (plain similarity, no rerank yet)."""
    qvec = embed([query])[0]
    res = collection.query(
        query_embeddings=[qvec],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for doc, meta, dist in zip(
    res["documents"][0],
    res["metadatas"][0],
    res["distances"][0],
    strict=True,
):
        hits.append({"distance": dist, "metadata": meta, "text": doc})
    return hits


def _print_hits(query: str, hits: list[dict]) -> None:
    print(f"\nQUERY: {query}\n" + "-" * 70)
    for i, h in enumerate(hits, 1):
        m = h["metadata"]
        cite = f"FMVSS §{m['section']} {m['subsection']}, p.{m['page']}"
        print(f"[{i}] cos_dist={h['distance']:.3f}  {cite}")
        preview = h["text"][:200] + ("..." if len(h["text"]) > 200 else "")
        print(f"    {preview}")


# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set — see .env.example")
    chunks = load_chunks(Path(args.chunks))
    embed = OpenAIEmbedder()
    collection = build_index(chunks, embed, persist_dir=Path(args.persist))
    if args.query:
        _print_hits(args.query, search(args.query, embed, collection, k=args.k))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Embed FMVSS chunks into Chroma.")
    p.add_argument("chunks", help="path to the chunks JSON from parse_fmvss.py")
    p.add_argument("--persist", default="data/chroma", help="Chroma directory")
    p.add_argument("--query", default="", help="optional sanity-check question")
    p.add_argument("--k", type=int, default=TOP_K)
    return p


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(_build_parser().parse_args())
