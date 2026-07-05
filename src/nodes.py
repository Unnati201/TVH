"""LangGraph node implementations. Each node reads/writes GraphState and does
exactly one job — easy to test or extend without touching the pipeline."""
from __future__ import annotations

from typing import Optional, TypedDict

from src import db, llm
from src.embeddings import embed_text
from src.vectorstore import semantic_search


class GraphState(TypedDict, total=False):
    query: str
    image_bytes: Optional[bytes]
    image_mime_type: Optional[str]
    image_description: Optional[str]
    parsed_intent: dict
    candidate_refs: list[str]
    candidates: list[dict]      # enriched with Supabase facts
    best_ref_no: Optional[str]
    best_product: Optional[dict]
    reasoning: str
    similar_products: list[dict]  # user-facing cards (never raw internal fields)
    answer: str
    log_id: Optional[int]       # query_logs PK — returned so UI can submit feedback


def describe_image_node(state: GraphState) -> GraphState:
    image_bytes = state.get("image_bytes")
    if not image_bytes:
        return {**state, "image_description": None}
    description = llm.describe_image(image_bytes, state.get("image_mime_type", "image/jpeg"))
    return {**state, "image_description": description}


def parse_query_node(state: GraphState) -> GraphState:
    effective_query = state["query"]
    if state.get("image_description"):
        effective_query = (
            f"{effective_query}\n\nPhoto shows: {state['image_description']}"
            if effective_query.strip()
            else f"Photo shows: {state['image_description']}"
        )
    intent = llm.parse_intent(effective_query)
    return {**state, "parsed_intent": intent}


def retrieve_node(state: GraphState, top_k: int = 8) -> GraphState:
    search_query = state["parsed_intent"].get("search_query", state["query"])
    query_embedding = embed_text(search_query)
    # No catalogue filter — search across all products
    matches = semantic_search(query_embedding, top_k=top_k)
    candidate_refs = [ref for ref, _score in matches]
    return {**state, "candidate_refs": candidate_refs}


def enrich_node(state: GraphState) -> GraphState:
    products = db.get_products_by_refs(state["candidate_refs"])
    candidates = [
        {
            "ref_no": p.get("ref_no"),
            "name": p.get("name"),
            "description": p.get("description"),
            "category": p.get("category"),
            "image_url": p.get("image_url"),
            "website_url": p.get("website_url"),
            "catalog_page_image_url": p.get("catalog_page_image_url"),
        }
        for p in products
    ]
    return {**state, "candidates": candidates}


def select_node(state: GraphState) -> GraphState:
    candidates = state.get("candidates") or []
    if not candidates:
        return {**state, "best_ref_no": None, "best_product": None, "reasoning": "No products found."}

    search_query = state["parsed_intent"].get("search_query", state["query"])
    result = llm.rerank(search_query, candidates)
    best_ref_no = result.get("best_ref_no")
    best_product = next((c for c in candidates if c["ref_no"] == best_ref_no), None)
    return {
        **state,
        "best_ref_no": best_ref_no,
        "best_product": best_product,
        "reasoning": result.get("reasoning", ""),
    }


def _to_display(p: dict) -> dict:
    return {
        "name": p.get("name"),
        "description": p.get("description"),
        "category": p.get("category"),
        "image_url": p.get("image_url"),
        "website_url": p.get("website_url"),
    }


def recommend_node(state: GraphState) -> GraphState:
    best_ref_no = state.get("best_ref_no")
    candidates = state.get("candidates") or []

    if not best_ref_no:
        # No best match — show top candidates as similar products
        similar = [_to_display(c) for c in candidates[:4]]
        return {**state, "similar_products": similar}

    # Try FBT recommendations first
    recs = db.get_recommendations(best_ref_no, limit=4)
    if recs:
        similar = [_to_display(r) for r in recs]
    else:
        # Fall back to other Pinecone candidates
        similar = [_to_display(c) for c in candidates if c.get("ref_no") != best_ref_no][:4]

    return {**state, "similar_products": similar}


def respond_node(state: GraphState) -> GraphState:
    product = state.get("best_product")
    if not product:
        answer = (
            "We couldn't find an exact match in our catalogue for that description. "
            "Here are some related products that may be what you're looking for."
        )
        return {**state, "answer": answer}

    answer = llm.synthesize_response(
        original_query=state["query"],
        product_name=product.get("name") or product.get("description", "")[:60],
        description=product.get("description", ""),
        category=product.get("category", ""),
        reasoning=state.get("reasoning", ""),
        similar_products=state.get("similar_products", []),
    )
    return {**state, "answer": answer}


def log_node(state: GraphState) -> GraphState:
    log_id = db.log_query(
        raw_query=state["query"],
        parsed_intent=state.get("parsed_intent", {}),
        matched_refs=[state["best_ref_no"]] if state.get("best_ref_no") else [],
        recommended_refs=[p.get("name", "") for p in state.get("similar_products", [])],
    )
    return {**state, "log_id": log_id}
