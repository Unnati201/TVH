"""All Gemini calls — prompts and response helpers — kept in one place."""
from __future__ import annotations

import json

from google import genai
from google.genai import types

from src.config import settings

_client = genai.Client(
    vertexai=True,
    project=settings.google_cloud_project,
    location=settings.google_cloud_location,
)


def _generate_json(prompt: str) -> dict:
    response = _client.models.generate_content(
        model=settings.gemini_chat_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1,
        ),
    )
    return json.loads(response.text)


# ── Intent parsing ─────────────────────────────────────────────────────────────

PARSE_INTENT_PROMPT = """You are helping match a customer's description to
industrial safety labels, signs, and equipment products at TVH. Given the
user's message, extract a JSON object with these fields:

- "equipment": type of equipment mentioned, if any (e.g. "forklift", "MEWP",
  "skid steer"), else null
- "sign_type": one of ["warning", "prohibition", "mandatory", "fire_safety",
  "emergency_first_aid", "capacity_load", "stock_control", "instructional",
  "reflective_tape", "application_tool", "mounting_hardware", "unknown"]
- "keywords": array of 3-6 short search keywords
- "search_query": a single rewritten sentence optimised for semantic search
  over a product catalogue (concise, descriptive, no fluff)

User message: {message}

Respond with ONLY the JSON object, no markdown fences."""


def parse_intent(message: str) -> dict:
    return _generate_json(PARSE_INTENT_PROMPT.format(message=message))


# ── Candidate reranking ────────────────────────────────────────────────────────

RERANK_PROMPT = """A customer is searching for: "{search_query}"

Here are candidate products from the TVH catalogue:
{candidates}

Choose the single best match. Return ONLY this JSON (no markdown):
{{"best_ref_no": "<ref_no of best match>", "reasoning": "<one sentence explaining why this product fits the customer's need, written in plain English>"}}"""


def rerank(search_query: str, candidates: list[dict]) -> dict:
    candidates_text = "\n".join(
        f"- ref:{c['ref_no']} | {c.get('name', '')} | {c.get('description', '')}"
        for c in candidates
    )
    return _generate_json(
        RERANK_PROMPT.format(search_query=search_query, candidates=candidates_text)
    )


# ── Natural language answer ────────────────────────────────────────────────────

RESPOND_PROMPT = """Write a short, professional answer for a TVH customer or
purchasing agent. They searched for: "{original_query}"

The best matching product is:
  Name: {product_name}
  Description: {description}
  Category: {category}

Why it matches: {reasoning}

Related products they may also need:
{similar_products}

Instructions:
- Write 2-4 natural sentences, like a knowledgeable parts specialist responding
- Refer to the product by its name only — NEVER mention internal codes,
  reference numbers (e.g. LS-1010), page numbers, or catalogue identifiers
- If related products are listed, mention one or two of them by name at the end
- Be concise, warm, and helpful"""


def synthesize_response(
    original_query: str,
    product_name: str,
    description: str,
    category: str,
    reasoning: str,
    similar_products: list[dict],
) -> str:
    sim_text = (
        "\n".join(
            f"- {p.get('name') or p.get('description', '')[:60]}"
            for p in similar_products
        )
        if similar_products
        else "(none)"
    )
    response = _client.models.generate_content(
        model=settings.gemini_chat_model,
        contents=RESPOND_PROMPT.format(
            original_query=original_query,
            product_name=product_name,
            description=description,
            category=category,
            reasoning=reasoning,
            similar_products=sim_text,
        ),
        config=types.GenerateContentConfig(temperature=0.3),
    )
    return response.text


# ── Image description ──────────────────────────────────────────────────────────

DESCRIBE_MEDIA_PROMPT = """You are analysing media (photo or video) submitted
by a warehouse worker or purchasing agent at TVH. Describe what you see:

- Any visible text, codes, or pictograms on safety signs, labels, or decals
  (e.g. "red circular prohibition sign with a black flame pictogram")
- The type of equipment the sign/label is mounted on or near, if visible
  (e.g. forklift mast, MEWP platform, warehouse wall)
- For video: describe the most relevant frame or the overall scene
- Visible condition (faded, torn, damaged) if relevant

Be concise (3-5 sentences). If the media doesn't clearly show a label, sign,
or piece of equipment, say so plainly."""


def describe_image(image_bytes: bytes, mime_type: str) -> str:
    """Accepts image or video bytes — Gemini handles both natively."""
    response = _client.models.generate_content(
        model=settings.gemini_chat_model,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            DESCRIBE_MEDIA_PROMPT,
        ],
        config=types.GenerateContentConfig(temperature=0.1),
    )
    return response.text
