"""Retrieve and rerank FMVSS chunks from Chroma for answer generation.

Pipeline stage (skills.md Skill 1):  retrieve -> rerank  (the next stage is
generate_answer() in generate.py).

Usage (module):
    from src.automotive_rag.retrieve import retrieve_chunks
    from src.automotive_rag.embed_index import OpenAIEmbedder
    import chromadb

    embed = OpenAIEmbedder()
    client = chromadb.PersistentClient(path="data/chroma")
    collection = client.get_collection("fmvss_571_208")

    chunks = retrieve_chunks("What HIC limit applies at the head restraint?",
                             embed, collection)

Usage (CLI sanity check):
    python retrieve.py "What seat belt is required at rear seating positions?"
"""

from __future__ import annotations

import argparse
import logging
import math
from typing import Protocol

import chromadb

from src.compliance_rag.generate import Chunk

logger = logging.getLogger(__name__)

# ── defaults ──────────────────────────────────────────────────────────────────
COLLECTION = "fmvss_571_208"
PERSIST_DIR = "data/chroma"
TOP_K = 5  # final output size (Skill 1)
CANDIDATE_K = 15  # oversample before MMR; ≥ 3× top_k is a safe floor
MMR_LAMBDA = 0.5  # Skill 1: relevance/diversity balance


# ── embedder protocol (mirrors embed_index.py) ───────────────────────────────
class Embedder(Protocol):
    """Anything that turns a list[str] into a list of float vectors.

    The production path uses OpenAIEmbedder; tests pass a stub so retrieval
    runs without network access or an API key.
    """

    def __call__(self, texts: list[str]) -> list[list[float]]: ...


# ── vector math ───────────────────────────────────────────────────────────────


def _dot(a: list[float], b: list[float]) -> float:
    """Dot product; cosine similarity when both vectors are L2-normalised."""
    return sum(x * y for x, y in zip(a, b, strict=True))


def _norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _unit(v: list[float]) -> list[float]:
    n = _norm(v)
    if n == 0.0:
        return v
    return [x / n for x in v]


# ── Chroma query (reuses the query_embeddings pattern from embed_index.py) ───


def _query_collection(
    query_vec: list[float],
    collection: chromadb.api.models.Collection.Collection,
    *,
    k: int,
) -> list[dict]:
    """Return k nearest neighbours from Chroma as plain dicts.

    Chroma was built without an embedding_function (Phase 2 decision);
    we must pass query_embeddings, never query_texts.

    Each returned dict: {"distance": float, "metadata": dict, "text": str,
                         "embedding": list[float]}
    The embedding is fetched so MMR can compute inter-candidate similarity
    without a second API call.
    """
    res = collection.query(
        query_embeddings=[query_vec],
        n_results=k,
        include=["documents", "metadatas", "distances", "embeddings"],
    )
    hits: list[dict] = []
    for doc, meta, dist, emb in zip(
        res["documents"][0],
        res["metadatas"][0],
        res["distances"][0],
        res["embeddings"][0],
        strict=True,
    ):
        hits.append(
            {
                "distance": dist,  # cosine distance; lower = more similar
                "metadata": meta,
                "text": doc,
                "embedding": emb,
            }
        )
    return hits


# ── MMR rerank ────────────────────────────────────────────────────────────────


def _mmr(
    query_vec: list[float],
    candidates: list[dict],
    *,
    k: int,
    lmbda: float,
) -> list[dict]:
    """Maximal Marginal Relevance (Carbonell & Goldstein 1998).

    Score = λ · sim(q, cᵢ) − (1−λ) · max_{s ∈ selected} sim(s, cᵢ)

    Chroma returns cosine *distance* (0 = identical, 2 = opposite for
    unit vectors).  Convert to similarity:  sim = 1 − distance.

    Args:
        query_vec:  unit-normalised query embedding.
        candidates: output of _query_collection — each has "distance" and
                    "embedding".
        k:          how many results to return.
        lmbda:      relevance weight; 0.5 = equal relevance/diversity.

    Returns:
        k candidates in selection order (most MMR-optimal first).
    """
    if not candidates:
        return []
    k = min(k, len(candidates))

    # Pre-normalise candidate embeddings once; avoids repeated norm calls.
    normed: list[list[float]] = [_unit(c["embedding"]) for c in candidates]

    # Relevance scores: sim = 1 − cosine_distance (Chroma space)
    relevance: list[float] = [1.0 - c["distance"] for c in candidates]

    selected_indices: list[int] = []
    remaining: list[int] = list(range(len(candidates)))

    while len(selected_indices) < k and remaining:
        best_idx: int | None = None
        best_score = float("-inf")

        for i in remaining:
            rel = relevance[i]
            if not selected_indices:
                # First pick: pure relevance, no redundancy penalty yet.
                redundancy = 0.0
            else:
                redundancy = max(_dot(normed[i], normed[j]) for j in selected_indices)
            score = lmbda * rel - (1.0 - lmbda) * redundancy
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx is None:
            break
        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    return [candidates[i] for i in selected_indices]


# ── hit → Chunk conversion ────────────────────────────────────────────────────


def _to_chunk(hit: dict) -> Chunk:
    """Convert a Chroma result dict to the Chunk dataclass expected by generate.py.

    Metadata keys guaranteed by Phase 2 schema (embed_index.py META_FIELDS):
        regulation, section, subsection, page, effective_date,
        source_url, parent_heading, part_index.

    Chunk only needs the subset that drives citation rendering.
    """
    m = hit["metadata"]
    return Chunk(
        text=hit["text"],
        regulation=str(m.get("regulation", "FMVSS")),
        section=str(m.get("section", "")),
        subsection=str(m.get("subsection", "")),
        page=int(m.get("page", 0)),
        effective_date=str(m.get("effective_date", "")),
    )


# ── public API ────────────────────────────────────────────────────────────────


def retrieve_chunks(
    query: str,
    embed: Embedder,
    collection: chromadb.api.models.Collection.Collection,
    *,
    top_k: int = TOP_K,
    candidate_k: int = CANDIDATE_K,
    mmr_lambda: float = MMR_LAMBDA,
) -> list[Chunk]:
    """Retrieve and rerank the top_k most relevant FMVSS chunks for a query.

    Steps:
        1. Embed the query with the supplied embedder.
        2. Fetch candidate_k neighbours from Chroma (cosine distance).
        3. MMR rerank to top_k, balancing relevance and diversity.
        4. Convert to list[Chunk] for generate_answer().

    Args:
        query:       Natural-language compliance question.
        embed:       Embedder instance (or stub for tests).
        collection:  Open Chroma collection (fmvss_571_208).
        top_k:       Final number of chunks returned (Skill 1 default: 5).
        candidate_k: Oversampling pool for MMR (Skill 1 default: 15).
        mmr_lambda:  Relevance/diversity balance (Skill 1 default: 0.5).

    Returns:
        list[Chunk] in MMR-ranked order; may be shorter than top_k if the
        collection is small.
    """
    if not query.strip():
        raise ValueError("query must not be empty")

    # 1. Embed query — unit-normalise once here so MMR math is clean.
    raw_qvec = embed([query])[0]
    query_vec = _unit(raw_qvec)
    logger.debug("query embedded (%d-dim)", len(query_vec))

    # 2. Chroma similarity search — oversample so MMR has room to diversify.
    effective_k = min(candidate_k, collection.count())
    if effective_k == 0:
        logger.warning("collection is empty — returning no chunks")
        return []

    candidates = _query_collection(query_vec, collection, k=effective_k)
    logger.info("retrieved %d candidates from Chroma", len(candidates))

    # 3. MMR rerank.
    reranked = _mmr(query_vec, candidates, k=top_k, lmbda=mmr_lambda)
    logger.info("MMR selected %d chunks (lambda=%.2f)", len(reranked), mmr_lambda)

    # 4. Convert to Chunk objects.
    chunks = [_to_chunk(h) for h in reranked]

    if logger.isEnabledFor(logging.DEBUG):
        for i, c in enumerate(chunks, 1):
            logger.debug(
                "[%d] §%s %s p.%d — %s…",
                i,
                c.section,
                c.subsection,
                c.page,
                c.text[:80],
            )

    return chunks


# ── CLI sanity check ──────────────────────────────────────────────────────────


def _cli() -> None:
    import os

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Retrieve + MMR rerank FMVSS chunks.")
    parser.add_argument("query", help="compliance question to look up")
    parser.add_argument("--persist", default=PERSIST_DIR)
    parser.add_argument("--collection", default=COLLECTION)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--candidate-k", type=int, default=CANDIDATE_K)
    parser.add_argument("--mmr-lambda", type=float, default=MMR_LAMBDA)
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set — see .env.example")

    # Import here so the module is usable without embed_index on the path.
    from embed_index import OpenAIEmbedder  # type: ignore[import]

    embedder = OpenAIEmbedder()
    client = chromadb.PersistentClient(path=args.persist)
    collection = client.get_collection(args.collection)

    chunks = retrieve_chunks(
        args.query,
        embedder,
        collection,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        mmr_lambda=args.mmr_lambda,
    )

    print(f"\nQUERY: {args.query}\n" + "-" * 70)
    for i, c in enumerate(chunks, 1):
        cite = f"FMVSS §{c.section} {c.subsection}, p.{c.page}"
        print(f"[{i}] {cite}")
        preview = c.text[:200] + ("..." if len(c.text) > 200 else "")
        print(f"    {preview}\n")


if __name__ == "__main__":
    _cli()
