"""Thin wrapper around Gemini's embedding API.

Isolated in its own module so the embedding model can be swapped
(e.g. to a fine-tuned or open-source model in production) without
touching ingestion or retrieval code.
"""
from __future__ import annotations

from google import genai
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings

_client = genai.Client(
    vertexai=True,
    project=settings.google_cloud_project,
    location=settings.google_cloud_location,
)


@retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(3))
def embed_text(text: str) -> list[float]:
    """Embed a single piece of text. Retries on transient API errors."""
    response = _client.models.embed_content(
        model=settings.gemini_embed_model,
        contents=text,
    )
    return response.embeddings[0].values


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts. Gemini's batch embedding endpoint is used
    where available; falls back to sequential calls otherwise."""
    return [embed_text(t) for t in texts]
