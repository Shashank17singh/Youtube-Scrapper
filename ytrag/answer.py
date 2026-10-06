"""Provides LLM integration (Groq/Gemini) to process transcript excerpts and
answer user queries in a grounded manner, with citations pointing back to the
original video segments."""

import re

from groq import Groq

from ytrag.config import (
    CONFIDENT_DISTANCE,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_BACKEND,
    LLM_MODEL,
    REFUSAL,
    TOP_K,
)
from ytrag.index import search, title_overlap
from ytrag.models import Chunk

_CLIENT: Groq | None = None

SYSTEM_PROMPT = f"""You are answering using ONLY the transcript excerpts below, which come from
the instructor's DSA lectures. The transcripts are auto-generated and may contain
minor errors - read past obvious mis-transcriptions of technical terms.

Rules:
- Answer only from the excerpts. If they don't cover it, say exactly:
  "{REFUSAL}"
- Cite with [1], [2] inline, using the excerpt numbers given below.
- Match the language of the question (Hinglish question -> Hinglish answer).
- 4-6 sentences max.
- Never invent a timestamp or a lecture name."""

_CITATION_RE = re.compile(r"\[(\d+)\]")


def get_client() -> Groq:
    global _CLIENT
    if _CLIENT is None:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set. Add it to the repo-root .env.")
        _CLIENT = Groq(api_key=GROQ_API_KEY)
    return _CLIENT


def _chat(system: str, user: str) -> str:
    """
    Routes the inference request to the configured backend (Groq or Gemini).
    Isolates the rest of the application from provider-specific SDK differences.
    """
    backend = LLM_BACKEND.lower()

    if backend == "none":
        raise RuntimeError("Explanations are disabled (YTRAG_LLM_BACKEND=none).")

    if backend == "gemini":
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=LLM_MODEL or GEMINI_MODEL,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system, temperature=0.2
            ),
        )
        return (response.text or "").strip()

    response = get_client().chat.completions.create(
        model=LLM_MODEL or GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return (response.choices[0].message.content or "").strip()


def build_context(chunks: list[Chunk]) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(f'[{i}] "{chunk.video_title}" @ {chunk.timestamp}\n{chunk.text}')
    return "\n\n".join(blocks)


def _citation(chunk: Chunk, distance: float) -> dict:
    return {
        "title": chunk.video_title,
        "timestamp": chunk.timestamp,
        "url": chunk.url,
        "start_sec": chunk.link_sec,
        "video_id": chunk.video_id,
        "distance": round(distance, 4),
    }


def _renumber(text: str, hits: list[tuple[Chunk, float]]) -> tuple[str, list[dict]]:
    order: list[int] = []
    for match in _CITATION_RE.finditer(text):
        idx = int(match.group(1))
        if 1 <= idx <= len(hits) and idx not in order:
            order.append(idx)

    if not order:
        return text, []

    remap = {old: new for new, old in enumerate(order, start=1)}
    rewritten = _CITATION_RE.sub(
        lambda m: f"[{remap[int(m.group(1))]}]" if int(m.group(1)) in remap else "",
        text,
    )
    citations = [_citation(*hits[old - 1]) for old in order]
    return rewritten, citations


def answer(
    question: str,
    top_k: int = TOP_K,
    video_id: str | None = None,
    max_distance: float | None = None,
) -> dict:
    """
    Full RAG pipeline: searches the vector database, formats the retrieved chunks,
    and asks the LLM to generate an answer with inline citations.
    """
    question = question.strip()
    if not question:
        return {"answer": REFUSAL, "citations": [], "grounded": False, "retrieved": 0}

    hits = search(question, top_k=top_k, video_id=video_id, max_distance=max_distance)
    if not hits:
        return {"answer": REFUSAL, "citations": [], "grounded": False, "retrieved": 0}

    chunks = [chunk for chunk, _ in hits]
    user_prompt = f"EXCERPTS\n{build_context(chunks)}\n\nQUESTION: {question}"

    text = _chat(SYSTEM_PROMPT, user_prompt)
    if REFUSAL.lower() in text.lower():
        return {
            "answer": REFUSAL,
            "citations": [],
            "grounded": False,
            "retrieved": len(hits),
        }

    text, citations = _renumber(text, hits)
    return {
        "answer": text,
        "citations": citations,
        "grounded": bool(citations),
        "retrieved": len(hits),
    }


def retrieve_only(
    question: str, top_k: int = TOP_K, filtered: bool = False
) -> list[tuple[Chunk, float]]:
    return search(question, top_k=top_k, max_distance=None if filtered else 2.0)


def _is_confident(question: str, hits: list[tuple[Chunk, float]]) -> bool:
    if not hits:
        return False
    chunk, distance = hits[0]
    return (
        title_overlap(question, chunk.video_title) > 0 or distance <= CONFIDENT_DISTANCE
    )


def search_only(question: str, top_k: int = TOP_K, video_id: str | None = None) -> dict:
    """Retrieval with no LLM at all - the timestamps, ranked.

    This is the main path. The timestamps *are* the product: a student wants
    to land on the moment the thing was explained, not read a paraphrase of
    it. Skipping the model makes this instant, free, unlimited, and incapable
    of hallucinating, since nothing is generated.

    `confident` reports whether the best match is close enough to be worth
    trusting. It is advisory, not a gate - a weak match still gets shown,
    because a ranked list the student can dismiss in one glance is far less
    harmful than a confident sentence that is wrong.
    """
    question = question.strip()
    if not question:
        return {"results": [], "confident": False, "query": question}

    hits = search(question, top_k=top_k, video_id=video_id)
    return {
        "query": question,
        "confident": _is_confident(question, hits),
        "results": [
            {
                "title": chunk.video_title,
                "timestamp": chunk.timestamp,
                "url": chunk.url,
                "start_sec": chunk.link_sec,
                "end_sec": chunk.end_sec,
                "video_id": chunk.video_id,
                "distance": round(distance, 4),
                "preview": chunk.text.split(chr(10) + chr(10), 1)[-1][:240].strip(),
            }
            for chunk, distance in hits
        ],
    }
