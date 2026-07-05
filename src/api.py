"""FastAPI wrapper around the LangGraph findability pipeline.

Run with: uvicorn src.api:app --reload
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src import db
from src.graph import run_query

app = FastAPI(title="TVH Parts Finder", version="1.0.0")

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


class SimilarProduct(BaseModel):
    product_name: str | None = None
    description: str | None = None
    category: str | None = None
    image_url: str | None = None
    website_url: str | None = None


class SearchResponse(BaseModel):
    answer: str
    product_name: str | None = None
    description: str | None = None
    category: str | None = None
    image_url: str | None = None
    website_url: str | None = None
    catalog_page_image_url: str | None = None
    similar_products: list[SimilarProduct] = []
    log_id: int | None = None   # links this result to query_logs for feedback


class FeedbackRequest(BaseModel):
    log_id: int
    feedback: str   # "correct" | "incorrect"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def index():
    return FileResponse(str(WEB_DIR / "index.html"))


@app.post("/search", response_model=SearchResponse)
async def search(
    query: str = Form(""),
    image: UploadFile | None = File(None),
) -> SearchResponse:
    image_bytes = await image.read() if image is not None else None
    image_mime_type = image.content_type if image is not None else None

    state = run_query(query, image_bytes=image_bytes, image_mime_type=image_mime_type)
    best = state.get("best_product") or {}

    return SearchResponse(
        answer=state.get("answer", ""),
        product_name=best.get("name"),
        description=best.get("description"),
        category=best.get("category"),
        image_url=best.get("image_url"),
        website_url=best.get("website_url"),
        catalog_page_image_url=best.get("catalog_page_image_url"),
        similar_products=[
            SimilarProduct(
                product_name=p.get("name"),
                description=p.get("description"),
                category=p.get("category"),
                image_url=p.get("image_url"),
                website_url=p.get("website_url"),
            )
            for p in state.get("similar_products", [])
        ],
        log_id=state.get("log_id"),
    )


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    if req.feedback not in ("correct", "incorrect"):
        raise HTTPException(status_code=400, detail="feedback must be 'correct' or 'incorrect'")
    db.update_feedback(req.log_id, req.feedback)
    return {"status": "ok"}
