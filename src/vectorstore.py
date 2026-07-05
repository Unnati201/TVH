"""Pinecone wrapper: index management, upsert, and semantic search.

Pinecone only ever stores an embedding + minimal metadata (ref_no,
catalogue, category). It is never treated as a source of truth for
product facts — that's Cloud SQL's job (see src/db.py). This keeps
the vector index cheap to rebuild and free of data-consistency
problems if the two stores drift.
"""
from __future__ import annotations

from pinecone import Pinecone, ServerlessSpec

from src.config import settings

EMBED_DIM = 3072  # gemini-embedding-001 output dimension

_pc = Pinecone(api_key=settings.pinecone_api_key)


def ensure_index() -> None:
    existing = {idx["name"] for idx in _pc.list_indexes()}
    if settings.pinecone_index not in existing:
        _pc.create_index(
            name=settings.pinecone_index,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud, region=settings.pinecone_region
            ),
        )


def get_index():
    return _pc.Index(settings.pinecone_index)


def upsert_products(items: list[dict]) -> None:
    """items: [{"ref_no": ..., "embedding": [...], "category": ..., "catalogue": ...}]"""
    index = get_index()
    vectors = [
        {
            "id": item["ref_no"],
            "values": item["embedding"],
            "metadata": {
                "category": item.get("category", ""),
                "catalogue": item.get("catalogue", ""),
            },
        }
        for item in items
    ]
    # Pinecone recommends batching upserts; fine at this scale (thousands),
    # would chunk into batches of ~100 for a full 50M-item catalogue.
    for i in range(0, len(vectors), 100):
        index.upsert(vectors=vectors[i : i + 100])


def semantic_search(
    query_embedding: list[float], top_k: int = 5, catalogue: str | None = None
) -> list[tuple[str, float]]:
    """Returns [(ref_no, similarity_score), ...] sorted by relevance."""
    index = get_index()
    filter_ = {"catalogue": catalogue} if catalogue else None
    result = index.query(
        vector=query_embedding, top_k=top_k, filter=filter_, include_metadata=False
    )
    return [(match["id"], match["score"]) for match in result["matches"]]
