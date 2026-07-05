"""Central configuration, loaded once from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    google_cloud_project: str
    google_cloud_location: str
    gemini_chat_model: str
    gemini_embed_model: str

    pinecone_api_key: str
    pinecone_index: str
    pinecone_cloud: str
    pinecone_region: str

    supabase_url: str
    supabase_key: str


def load_settings() -> Settings:
    return Settings(
        google_cloud_project=os.environ.get("GOOGLE_CLOUD_PROJECT", ""),
        google_cloud_location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
        gemini_chat_model=os.environ.get("GEMINI_CHAT_MODEL", "gemini-2.5-flash"),
        gemini_embed_model=os.environ.get("GEMINI_EMBED_MODEL", "gemini-embedding-001"),
        pinecone_api_key=os.environ.get("PINECONE_API_KEY", ""),
        pinecone_index=os.environ.get("PINECONE_INDEX", "tvh-labels-signs"),
        pinecone_cloud=os.environ.get("PINECONE_CLOUD", "aws"),
        pinecone_region=os.environ.get("PINECONE_REGION", "us-east-1"),
        supabase_url=os.environ.get("SUPABASE_URL", ""),
        supabase_key=os.environ.get("SUPABASE_KEY", ""),
    )


settings = load_settings()
