"""Ingestion pipeline: catalogue CSV -> Supabase (facts) + Pinecone (embeddings).

Usage:
    python -m src.ingest data/sample_labels_signs.csv
    python -m src.ingest data/sample_handling_equipment.csv
    python -m src.ingest --fbt data/sample_fbt.csv
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import pandas as pd

from src.db import upsert_fbt_batch, upsert_products_batch
from src.embeddings import embed_batch
from src.vectorstore import ensure_index, upsert_products


def _safe_int(val) -> int | None:
    try:
        v = float(val)
        return int(v) if not math.isnan(v) else None
    except (TypeError, ValueError):
        return None


def _safe_str(val) -> str | None:
    if val is None:
        return None
    s = str(val).strip()
    return s if s and s.lower() != "nan" else None


def _parse_attrs(val) -> dict:
    if val is None:
        return {}
    try:
        return json.loads(val) if isinstance(val, str) else (val or {})
    except (json.JSONDecodeError, TypeError):
        return {}


def ingest_products(csv_path: str) -> None:
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    # 1. Upsert structured facts into Supabase via REST API.
    rows = [
        {
            "ref_no": str(row["ref_no"]),
            "name": _safe_str(row.get("name")),
            "description": str(row["description"]),
            "category": _safe_str(row.get("category")),
            "page_no": _safe_int(row.get("page_no")),
            "catalogue": str(row["catalogue"]),
            "attributes_json": _parse_attrs(
                row.get("attributes_json") if "attributes_json" in df.columns else None
            ),
            "image_url": _safe_str(row.get("image_url")),
            "website_url": _safe_str(row.get("website_url")),
            "catalog_page_image_url": _safe_str(row.get("catalog_page_image_url")),
            "keywords": _safe_str(row.get("keywords")),
        }
        for _, row in df.iterrows()
    ]
    upsert_products_batch(rows)
    print(f"Upserted {len(rows)} products into Supabase")

    # 2. Embed name + description + keywords, then upsert into Pinecone.
    ensure_index()
    texts = [
        " | ".join(
            filter(
                None,
                [
                    _safe_str(row.get("name")),
                    _safe_str(row.get("description")),
                    _safe_str(row.get("keywords")),
                ],
            )
        )
        for _, row in df.iterrows()
    ]
    embeddings = embed_batch(texts)
    items = [
        {
            "ref_no": str(row["ref_no"]),
            "embedding": emb,
            "category": _safe_str(row.get("category")) or "",
            "catalogue": str(row["catalogue"]),
        }
        for (_, row), emb in zip(df.iterrows(), embeddings)
    ]
    upsert_products(items)
    print(f"Upserted {len(items)} embeddings into Pinecone")


def ingest_fbt(csv_path: str) -> None:
    df = pd.read_csv(csv_path)
    rows = [
        {
            "source_ref": str(row["source_ref"]),
            "related_ref": str(row["related_ref"]),
            "score": float(row["score"]),
        }
        for _, row in df.iterrows()
    ]
    upsert_fbt_batch(rows)
    print(f"Upserted {len(rows)} frequently-bought-together rows into Supabase")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TVH findability ingestion")
    parser.add_argument("csv_path", nargs="?", help="Path to a product CSV")
    parser.add_argument("--fbt", dest="fbt_path", help="Path to a FBT CSV")
    args = parser.parse_args()

    if args.fbt_path:
        ingest_fbt(args.fbt_path)
    elif args.csv_path:
        ingest_products(args.csv_path)
    else:
        print("Nothing to do — pass a CSV path or --fbt path", file=sys.stderr)
        sys.exit(1)
