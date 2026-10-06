"""
FastAPI server configuration and endpoints mapping for the Youtube-Scrapper
RAG interface.
"""

import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from groq import APIStatusError, RateLimitError
from pydantic import BaseModel, Field

from ytrag import config
from ytrag.answer import answer as answer_question
from ytrag.answer import search_only
from ytrag.index import stats as index_stats

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    from ytrag.embed import get_embedder

    print("Loading embedding model (first run downloads it)...", flush=True)
    embedder = get_embedder()
    print(f"Ready: {embedder.name} ({embedder.dim}-dim)", flush=True)
    yield


app = FastAPI(title="YT Lecture RAG", version="0.1.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=config.MAX_QUESTION_CHARS)
    top_k: int = Field(default=config.TOP_K, ge=1, le=20)


_HITS: dict[str, deque] = defaultdict(deque)


def _rate_limit(request: Request) -> None:
    client = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window = _HITS[client]

    while window and now - window[0] > config.RATE_LIMIT_WINDOW_SECONDS:
        window.popleft()

    if len(window) >= config.RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail=f"Slow down - max {config.RATE_LIMIT_REQUESTS} questions per "
            f"{config.RATE_LIMIT_WINDOW_SECONDS}s.",
        )
    window.append(now)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": config.LLM_MODEL,
        "embed_model": config.EMBED_MODEL,
    }


@app.get("/stats")
def stats():
    info = index_stats()
    info.pop("videos", None)
    return info


@app.post("/search")
def search(payload: AskRequest, request: Request):
    return search_only(payload.question, top_k=payload.top_k)


@app.post("/ask")
def ask(payload: AskRequest, request: Request):
    _rate_limit(request)
    try:
        return answer_question(payload.question, top_k=payload.top_k)
    except RateLimitError:
        raise HTTPException(
            status_code=429,
            detail="The language model's usage quota is exhausted. "
            "This is a provider limit, not a problem with your question - try again later.",
        )
    except APIStatusError as exc:
        raise HTTPException(
            status_code=502, detail=f"Language model error: {exc.status_code}"
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/meta")
def meta():
    from ytrag.transcribe import cached_video_ids, load_transcript

    ids = cached_video_ids()
    seconds = 0.0
    for vid in ids:
        data = load_transcript(vid)
        if data and data.get("segments"):
            seconds += data["segments"][-1]["end"]
    return {"lectures": len(ids), "hours": round(seconds / 3600, 1)}


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
