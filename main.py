import json
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

PASSAGES_PATH = Path(__file__).parent / "passages.json"
MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
ABSTAIN_THRESHOLD = float(os.getenv("ABSTAIN_THRESHOLD", "0.34"))

with open(PASSAGES_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# Supports either a list of 50 strings or list of dicts with "text"/"passage"/"content"
PASSAGES: List[str] = [
    item if isinstance(item, str) else (item.get("text") or item.get("passage") or item.get("content") or "")
    for item in raw_data
]

app = FastAPI(title="RAG Colosseum Render Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = SentenceTransformer(MODEL_NAME)
PASSAGE_EMBEDDINGS = model.encode(
    PASSAGES,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=False,
)


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=3, ge=1, le=20)


class QueryResponse(BaseModel):
    answer: Optional[str]
    sources: List[int]
    scores: List[float]
    confidence: float


def sanitize_passage(text: str) -> str:
    """Remove obvious prompt-injection directives while keeping technical facts."""
    lines = text.splitlines()
    clean_lines = []
    injection_pattern = re.compile(
        r"(ignore (all |previous |above )?instructions|system prompt|you are now|override|disregard)",
        re.IGNORECASE,
    )
    for line in lines:
        if not injection_pattern.search(line):
            clean_lines.append(line)
    return " ".join(clean_lines).strip() or text.strip()


def build_answer(question: str, hits: List[Tuple[int, str, float]]) -> str:
    """
    Synthesize a concise factual answer strictly from top retrieved passages.
    If you plug in an LLM API, wrap each passage in <untrusted_passage> tags and
    instruct the model to answer ONLY from facts in those passages.
    """
    cleaned_snippets = [sanitize_passage(p_text) for _, p_text, _ in hits]
    # Concise grounded summary joining key sentences from top matching passages
    combined = " ".join(cleaned_snippets)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", combined) if s.strip()]
    return " ".join(sentences[:4]) if sentences else combined[:500]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "passages": len(PASSAGES)}


@app.post("/query", response_model=QueryResponse)
def query_kb(req: QueryRequest) -> QueryResponse:
    q = req.question.strip()
    if not q or not PASSAGES:
        return QueryResponse(answer=None, sources=[], scores=[], confidence=0.1)

    q_emb = model.encode(
        [q],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]

    # Map cosine similarity [-1, 1] -> [0.0, 1.0]
    sims = np.clip((np.dot(PASSAGE_EMBEDDINGS, q_emb) + 1.0) / 2.0, 0.0, 1.0)
    top_indices = np.argsort(-sims)[: req.top_k]

    top_score = float(sims[top_indices[0]])
    if top_score < ABSTAIN_THRESHOLD:
        return QueryResponse(
            answer=None,
            sources=[],
            scores=[],
            confidence=round(max(0.05, min(0.35, top_score)), 4),
        )

    sources: List[int] = []
    scores: List[float] = []
    hits: List[Tuple[int, str, float]] = []

    for idx in top_indices:
        score_val = round(float(sims[idx]), 4)
        if score_val >= ABSTAIN_THRESHOLD * 0.8:
            i = int(idx)
            sources.append(i)
            scores.append(score_val)
            hits.append((i, PASSAGES[i], score_val))

    if not sources:
        return QueryResponse(answer=None, sources=[], scores=[], confidence=0.15)

    answer_str = build_answer(q, hits)
    confidence_val = round(min(0.99, max(0.65, top_score)), 4)

    return QueryResponse(
        answer=answer_str,
        sources=sources,
        scores=scores,
        confidence=confidence_val,
    )
