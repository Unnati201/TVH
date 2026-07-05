"""Supabase data access layer — uses HTTPS REST API instead of direct Postgres.

All reads/writes go through PostgREST (supabase-py), so no TCP port 5432
or DNS resolution for the DB host is needed. The service_role key bypasses
Row Level Security for server-side operations.
"""
from __future__ import annotations

import json
from typing import Optional

from supabase import Client, create_client

from src.config import settings

_client: Client | None = None

_PRODUCT_COLS = (
    "ref_no, name, description, category, "
    "image_url, website_url, catalog_page_image_url, keywords"
)


def _get() -> Client:
    global _client
    if _client is None:
        _client = create_client(settings.supabase_url, settings.supabase_key)
    return _client


# ── Products ──────────────────────────────────────────────────────────────────

def get_products_by_refs(refs: list[str]) -> list[dict]:
    if not refs:
        return []
    result = _get().table("products").select(_PRODUCT_COLS).in_("ref_no", refs).execute()
    return result.data or []


def get_product(ref_no: str) -> Optional[dict]:
    result = (
        _get().table("products").select(_PRODUCT_COLS).eq("ref_no", ref_no).limit(1).execute()
    )
    return result.data[0] if result.data else None


def upsert_products_batch(products: list[dict]) -> None:
    if not products:
        return
    # Supabase upsert in chunks of 100 to stay well within request-size limits
    for i in range(0, len(products), 100):
        _get().table("products").upsert(products[i : i + 100]).execute()


# ── FBT ───────────────────────────────────────────────────────────────────────

def get_recommendations(ref_no: str, limit: int = 3) -> list[dict]:
    fbt = (
        _get()
        .table("frequently_bought_together")
        .select("related_ref, score")
        .eq("source_ref", ref_no)
        .order("score", desc=True)
        .limit(limit)
        .execute()
    )
    if not fbt.data:
        return []

    related_refs = [row["related_ref"] for row in fbt.data]
    products = get_products_by_refs(related_refs)

    score_map = {row["related_ref"]: float(row["score"]) for row in fbt.data}
    return [{**p, "score": score_map.get(p["ref_no"], 0.0)} for p in products]


def upsert_fbt_batch(rows: list[dict]) -> None:
    if not rows:
        return
    for i in range(0, len(rows), 100):
        _get().table("frequently_bought_together").upsert(rows[i : i + 100]).execute()


# ── Query logging (best-effort — never breaks a search) ───────────────────────

def log_query(
    raw_query: str,
    parsed_intent: dict,
    matched_refs: list[str],
    recommended_refs: list[str],
) -> Optional[int]:
    """Insert a query log row and return its auto-generated ID (or None on error)."""
    try:
        result = _get().table("query_logs").insert(
            {
                "raw_query": raw_query,
                "parsed_intent": parsed_intent,   # dict → JSONB
                "matched_refs": matched_refs,       # list → TEXT[]
                "recommended_refs": recommended_refs,
            }
        ).execute()
        if result.data:
            return result.data[0].get("id")
    except Exception:
        pass
    return None


def update_feedback(log_id: int, feedback: str) -> None:
    """Write user feedback ('correct' | 'incorrect') to a log row."""
    try:
        _get().table("query_logs").update(
            {"user_feedback": feedback}
        ).eq("id", log_id).execute()
    except Exception:
        pass
