"""Assembles the LangGraph state machine for the findability flow.

    parse_query -> retrieve -> enrich -> select -> recommend -> respond -> log

Kept as a single linear chain for the demo. The natural extension points
(discussed in the README) are a conditional edge after `select` that loops
back to a clarification node when confidence is low, and a conditional
edge after `retrieve` that widens the search (e.g. drops the catalogue
filter) if zero candidates come back.
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.nodes import (
    GraphState,
    describe_image_node,
    enrich_node,
    log_node,
    parse_query_node,
    recommend_node,
    respond_node,
    retrieve_node,
    select_node,
)


def _build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("describe_image", describe_image_node)
    graph.add_node("parse_query", parse_query_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("enrich", enrich_node)
    graph.add_node("select", select_node)
    graph.add_node("recommend", recommend_node)
    graph.add_node("respond", respond_node)
    graph.add_node("log", log_node)

    graph.set_entry_point("describe_image")
    graph.add_edge("describe_image", "parse_query")
    graph.add_edge("parse_query", "retrieve")
    graph.add_edge("retrieve", "enrich")
    graph.add_edge("enrich", "select")
    graph.add_edge("select", "recommend")
    graph.add_edge("recommend", "respond")
    graph.add_edge("respond", "log")
    graph.add_edge("log", END)

    return graph.compile()


# Compiled once at import time — not rebuilt on every request
_graph = _build_graph()


def run_query(
    query: str,
    image_bytes: bytes | None = None,
    image_mime_type: str | None = None,
) -> GraphState:
    return _graph.invoke(
        {
            "query": query,
            "image_bytes": image_bytes,
            "image_mime_type": image_mime_type,
        }
    )
