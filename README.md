# TVH Findability POC

A demo answering: given a free-text description or scenario, point the
user to the right **labels & signs** product (reference number + catalogue
page) and suggest **handling equipment & accessories** frequently bought
alongside it.

## Architecture

```
 User query/scenario
        │
        ▼
 ┌────────────────────────────── LangGraph ──────────────────────────────┐
 │ parse_query → retrieve → enrich → select → recommend → respond → log  │
 └─────────────────────────────────────────────────────────────────────┘
```

| Stage | Tool | Job |
|---|---|---|
| `parse_query` | Gemini | Turn a free-text scenario into structured intent + a clean search query |
| `retrieve` | Pinecone | Semantic search over catalogue embeddings → top-k candidate ref numbers |
| `enrich` | Cloud SQL | Join candidate ref numbers to canonical facts (description, page, category) |
| `select` | Gemini | Rerank enriched candidates, pick the best match, explain why |
| `recommend` | Cloud SQL | Look up frequently-bought-together accessories for the selected product |
| `respond` | Gemini | Compose a natural-language answer citing ref number + page |
| `log` | Cloud SQL | Record query, match, and recommendations for later evaluation |

**Why two stores instead of one:** Pinecone is good at "these mean the same
thing" but bad at being a reliable source of facts (it can drift, and
similarity scores aren't the same as correctness). Cloud SQL is the
single source of truth for ref numbers, pages, and stock/category facts.
Splitting them means the vector index can be rebuilt/reindexed at any
time without risking data integrity, and structured filtering (category,
catalogue, stock status) stays fast and exact instead of approximate.

**Why LangGraph specifically:** the flow is naturally a small state
machine, and using LangGraph rather than a hand-rolled function chain
makes each stage independently testable (see `tests/test_nodes.py`),
makes the state explicit and inspectable (useful for debugging *why* a
wrong product was suggested), and gives an easy extension point for
branching logic (see "Not built, but the plan" below).

## Project layout

```
src/
  config.py       settings from env vars
  db.py           Cloud SQL models + queries (SQLAlchemy)
  embeddings.py   Gemini embedding calls
  vectorstore.py  Pinecone index management + search
  llm.py          Gemini chat calls (intent parsing, rerank, response)
  nodes.py        LangGraph node functions
  graph.py        graph assembly
  ingest.py       CSV → Cloud SQL + Pinecone loader
  api.py          FastAPI endpoint
  cli_demo.py     interactive terminal demo (good for live walkthroughs)
db/schema.sql     Cloud SQL DDL
data/*.csv        synthetic sample catalogue + frequently-bought-together data
tests/            unit tests for graph nodes (mocked external calls)
```

## Data note

The real labels & signs catalogue and the real frequently-bought-together
dataset weren't pulled into this demo — `data/*.csv` is **synthetic
sample data** shaped like the real thing (same columns TVH would need:
ref_no, description, category, page_no, catalogue). Swapping in the real
catalogue is a matter of writing an extraction step that produces this
same CSV shape from the actual PDF (see below) — the rest of the
pipeline doesn't change.

## Web UI (text, voice, photo)

`web/index.html` is a self-contained chat page served by the FastAPI app
at `/`. It supports three input modes:

- **Text** — typed straight into the composer.
- **Voice** — the mic button uses the browser's built-in
  `SpeechRecognition` API (client-side, no audio ever leaves the browser
  until it's transcribed to text); the transcript is sent as a normal
  text query. No server-side speech infrastructure needed for the demo.
- **Photo** — attach or take a picture of a part, an existing label, or
  the equipment itself. The image is sent to `/search` as multipart form
  data; a new `describe_image` graph node uses Gemini's vision capability
  to turn it into a text description ("red circular prohibition sign,
  black flame pictogram, mounted on a MEWP boom") which is folded into
  the query before `parse_query` runs. Everything downstream of that node
  stays text-only — the multimodal handling is isolated to one step.

Run it with `uvicorn src.api:app --reload` and open `http://localhost:8000`.

Design note: the UI intentionally borrows the visual language of the
subject matter itself — hazard-stripe accents, nameplate-style result
cards, monospace reference numbers — rather than a generic chat-app
look, since the tool exists specifically to find safety labels & signs.

**Production note on voice:** browser `SpeechRecognition` is Chrome/Edge
only and sends audio to Google's servers for transcription — fine for a
demo, but a production deployment for contact-center agents would likely
want a controlled STT path instead (e.g. Gemini's native audio input, or
Cloud Speech-to-Text) so audio handling is auditable and works in every
browser.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in GOOGLE_API_KEY, PINECONE_API_KEY, DATABASE_URL

python -m src.ingest --init-db data/sample_labels_signs.csv
python -m src.ingest data/sample_handling_equipment.csv
python -m src.ingest --fbt data/sample_fbt.csv

python -m src.cli_demo
# or: uvicorn src.api:app --reload
```

## Productionization plan

**1. Real catalogue ingestion.** The labels & signs catalogue is a PDF/flipbook,
not a CSV. Production ingestion needs an extraction step: PDF → per-page
text/table extraction (e.g. layout-aware parsing, or a Gemini vision call
per page for image-heavy pages) → structured rows matching the schema
above. This is the highest-risk, highest-effort part of going from demo
to production, because catalogue layout is inconsistent across TVH's
~50M-SKU range — I'd start with the labels & signs section only (as
scoped), validate extraction accuracy against a sample of known ref
numbers, then expand section by section.

**2. Scale.** 50M products means the current linear-scan-friendly design
needs: Pinecone namespaces or separate indexes per catalogue section,
batched embedding jobs (not per-request), and Cloud SQL read replicas or
caching (Redis) in front of the hot path for repeat queries.

**3. Evaluation loop.** `query_logs` already captures query → match →
recommendation. Add a feedback mechanism (thumbs up/down from the contact
center agent) and periodically review low-confidence or corrected
matches — this becomes both a regression test set and a source for
improving the intent-parsing prompt or fine-tuning retrieval.

**4. Confidence & fallback.** Right now `select` always picks a "best"
match even if it's a bad one. Production needs a confidence threshold:
below it, ask a clarifying question (a natural LangGraph branch — add a
conditional edge after `select`) instead of guessing.

**5. Observability & cost.** Add tracing (LangSmith or OpenTelemetry) across
the graph, latency/cost dashboards per node (Gemini + Pinecone calls are
the expensive ones), and alerting on error rates.

**6. Security & deployment.** Cloud SQL via the Cloud SQL Auth Proxy /
Python Connector with IAM auth rather than static passwords; API behind
Cloud Run or GKE with proper auth for internal contact-center use;
secrets in Secret Manager, not `.env` files.

**7. Multi-catalogue expansion.** Once labels & signs findability works,
the same graph pattern (parse → retrieve → enrich → select → recommend →
respond) generalizes to the other 49.9M products — the main new work per
catalogue section is the extraction step in (1), not the graph itself.

## What's not built (and why that's OK per the brief)

- No real catalogue/FBT data (access-gated; synthetic data used as a
  structural stand-in — see "Data note").
- No confidence-based clarification loop (noted above as the natural next
  LangGraph branch).
- No auth/deployment — out of scope for a 15-minute demo.

## Presenting this (15 min)

1. Problem framing (1 min): 50M SKUs, findability is the bottleneck.
2. Live demo via `cli_demo.py` — run 2-3 example queries showing the
   pipeline stages (parsed intent → candidates → match → recommendation).
3. Architecture walkthrough (this README's diagram) — explain the
   two-store split and why.
4. Productionization plan — pick 2-3 points above to go deep on.
5. Leave time for questions — know every file well enough to defend it.
