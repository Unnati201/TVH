"""Unit tests for graph nodes with external services mocked out.

These don't hit Gemini, Pinecone, or Cloud SQL — they check that each
node transforms GraphState the way the graph wiring expects, which is
what actually breaks when someone edits a node signature.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.nodes import (
    enrich_node,
    parse_query_node,
    recommend_node,
    respond_node,
    retrieve_node,
    select_node,
)


def test_parse_query_node_adds_intent():
    with patch("src.nodes.llm.parse_intent") as mock_parse:
        mock_parse.return_value = {"search_query": "hot surface warning label"}
        state = {"query": "sign for a hot surface"}
        result = parse_query_node(state)
        assert result["parsed_intent"]["search_query"] == "hot surface warning label"


def test_retrieve_node_returns_candidate_refs():
    with patch("src.nodes.embed_text") as mock_embed, patch(
        "src.nodes.semantic_search"
    ) as mock_search:
        mock_embed.return_value = [0.1, 0.2]
        mock_search.return_value = [("LS-1001", 0.9), ("LS-1002", 0.8)]
        state = {"query": "hot surface", "parsed_intent": {"search_query": "hot surface"}}
        result = retrieve_node(state)
        assert result["candidate_refs"] == ["LS-1001", "LS-1002"]


def test_select_node_handles_no_candidates():
    state = {"query": "q", "parsed_intent": {}, "candidates": []}
    result = select_node(state)
    assert result["best_ref_no"] is None
    assert "No candidates" in result["reasoning"]


def test_recommend_node_skips_when_no_match():
    state = {"best_ref_no": None}
    result = recommend_node(state)
    assert result["recommendations"] == []


def test_respond_node_handles_no_match_gracefully():
    state = {"query": "q", "best_product": None}
    result = respond_node(state)
    assert "couldn't find" in result["answer"].lower()
