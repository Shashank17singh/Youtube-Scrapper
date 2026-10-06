"""
Qdrant vector database interface. Handles collection management, document
insertion (upsert), semantic searches using vector similarity (cosine), and
importing/exporting indexed vectors to disk (.npz).
"""

import atexit
import json
import re
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from ytrag.config import (
    COLLECTION,
    MAX_DISTANCE,
    QDRANT_API_KEY,
    QDRANT_PATH,
    QDRANT_URL,
    TITLE_BOOST,
    TOP_K,
    UPSERT_BATCH,
)
from ytrag.embed import get_embedder
from ytrag.models import _NAMESPACE, Chunk
from ytrag.util import with_retry

_CLIENT: QdrantClient | None = None


def _close_client() -> None:
    global _CLIENT
    if _CLIENT is not None:
        try:
            _CLIENT.close()
        except Exception:  # noqa: BLE001, S110
            pass
        _CLIENT = None


atexit.register(_close_client)


def get_client() -> QdrantClient:
    global _CLIENT
    if _CLIENT is None:
        if QDRANT_URL:
            _CLIENT = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)
        else:
            QDRANT_PATH.mkdir(parents=True, exist_ok=True)
            try:
                _CLIENT = QdrantClient(path=str(QDRANT_PATH))
            except RuntimeError as exc:
                if "already accessed" in str(exc) or "Storage folder" in str(exc):
                    raise RuntimeError(
                        "The local index is already open in another process - most likely "
                        "`ytrag serve` is running in another terminal. Stop it (Ctrl-C) and "
                        "try again, or set QDRANT_URL to use a hosted Qdrant which allows "
                        "many readers at once."
                    ) from exc
                raise
    return _CLIENT


def collection_name() -> str:
    return f"{COLLECTION}_{get_embedder().dim}"


def ensure_collection() -> str:
    client = get_client()
    name = collection_name()

    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=get_embedder().dim,
                distance=Distance.COSINE,
            ),
        )
        if QDRANT_URL:
            client.create_payload_index(
                collection_name=name,
                field_name="video_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
    return name


def upsert_chunks(chunks: list[Chunk], batch_size: int = UPSERT_BATCH) -> int:
    """
    Batches and inserts transcribed chunks into the Qdrant index.
    Embedding is the bottleneck, so we do it in configurable batches.
    """
    if not chunks:
        return 0

    name = ensure_collection()
    client = get_client()
    embedder = get_embedder()

    total = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors = embedder.embed_documents([c.text for c in batch])
        points = [
            PointStruct(id=c.point_id, vector=v, payload=c.to_payload())
            for c, v in zip(batch, vectors)
        ]
        with_retry(
            lambda points=points: client.upsert(
                collection_name=name, points=points, wait=True
            ),
            label=f"upsert {len(points)} points",
        )
        total += len(points)

    return total


def delete_video(video_id: str) -> None:
    client = get_client()
    name = ensure_collection()
    client.delete(
        collection_name=name,
        points_selector=Filter(
            must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))]
        ),
        wait=True,
    )


def indexed_video_ids() -> set[str]:
    client = get_client()
    name = collection_name()
    if not client.collection_exists(name):
        return set()

    found: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=name,
            limit=1000,
            offset=offset,
            with_payload=["video_id"],
            with_vectors=False,
        )
        for point in points:
            if point.payload and point.payload.get("video_id"):
                found.add(point.payload["video_id"])
        if offset is None:
            break
    return found


_STOP = {
    "kaise",
    "kya",
    "hai",
    "hain",
    "me",
    "ka",
    "ki",
    "ke",
    "aur",
    "kab",
    "karte",
    "karna",
    "hota",
    "nikale",
    "solve",
    "kare",
    "chahiye",
    "use",
    "kahan",
    "se",
    "ko",
    "pehchane",
    "difference",
    "farak",
    "the",
    "a",
    "is",
    "in",
    "what",
    "how",
    "do",
    "to",
    "of",
    "for",
    "video",
    "dsa",
    "patterns",
    "pattern",
    "episode",
    "leetcode",
    "interview",
    "questions",
    "question",
    "master",
    "best",
    "explained",
}


def _stem(word: str) -> str:
    """Crude plural stripping, enough to match 'hashmap' against 'HASHMAPS'."""
    for suffix in ("es", "s"):
        if len(word) > 4 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _terms(text: str) -> set[str]:
    return {
        _stem(w)
        for w in re.findall(r"[a-z0-9]+", text.lower())
        if w not in _STOP and len(w) > 2
    }


def title_overlap(query: str, title: str) -> int:
    """How many meaningful query words appear in the lecture's title."""
    return len(_terms(query) & _terms(title))


def search(
    query: str,
    top_k: int = TOP_K,
    video_id: str | None = None,
    max_distance: float | None = None,
) -> list[tuple[Chunk, float]]:
    """Return [(chunk, distance)] sorted best-first, already distance-filtered.

    Qdrant returns a cosine *similarity* score (higher is better); the rest of
    the system thinks in distance (lower is better), so convert once here.
    """
    name = ensure_collection()
    client = get_client()
    vector = get_embedder().embed_query(query)

    query_filter = None
    if video_id:
        query_filter = Filter(
            must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))]
        )
    results = client.query_points(
        collection_name=name,
        query=vector,
        limit=max(top_k * 4, 20),
        with_payload=True,
        query_filter=query_filter,
    ).points

    cutoff = MAX_DISTANCE if max_distance is None else max_distance
    scored: list[tuple[float, float, Chunk]] = []
    for point in results:
        distance = 1.0 - float(point.score)
        if distance > cutoff:
            continue
        chunk = Chunk.from_payload(point.payload)
        overlap = title_overlap(query, chunk.video_title)
        scored.append((distance - TITLE_BOOST * overlap, distance, chunk))

    scored.sort(key=lambda row: row[0])
    return [(chunk, distance) for _, distance, chunk in scored[:top_k]]


def stats() -> dict:
    """Collection size plus a per-video breakdown."""
    client = get_client()
    name = collection_name()

    if not client.collection_exists(name):
        return {"collection": name, "exists": False, "chunks": 0, "videos": {}}

    info = client.get_collection(name)
    videos: dict[str, dict] = {}

    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=name,
            limit=512,
            offset=offset,
            with_payload=["video_id", "video_title"],
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            vid = payload.get("video_id", "?")
            entry = videos.setdefault(
                vid, {"title": payload.get("video_title", "?"), "chunks": 0}
            )
            entry["chunks"] += 1
        if offset is None:
            break

    return {
        "collection": name,
        "exists": True,
        "chunks": info.points_count or 0,
        "dim": get_embedder().dim,
        "embed_model": get_embedder().name,
        "videos": videos,
    }


def export_vectors(path: Path) -> dict:
    """Dump every point's vector and payload to a compressed .npz."""
    import numpy as np

    client = get_client()
    name = collection_name()
    vectors, payloads = [], []

    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=name,
            limit=512,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        for point in points:
            vectors.append(point.vector)
            payloads.append(point.payload)
        if offset is None:
            break

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        vectors=np.array(vectors, dtype=np.float16),
        payloads=np.array(json.dumps(payloads)),
        model=np.array(get_embedder().name),
        dim=np.array(get_embedder().dim),
    )
    return {"count": len(vectors), "mb": path.stat().st_size / 1024 / 1024}


def import_vectors(path: Path, batch_size: int = UPSERT_BATCH) -> dict:
    """Load a .npz built by export_vectors straight into the collection.

    Refuses to load vectors built by a different embedding model - mixing them
    would silently wreck retrieval, since a query encoded by one model is
    meaningless against another's vectors.
    """
    import numpy as np

    data = np.load(path, allow_pickle=False)
    model = str(data["model"])
    if model != get_embedder().name:
        raise RuntimeError(
            f"These vectors were built with {model}, but YTRAG_EMBED_MODEL is "
            f"{get_embedder().name}. Set YTRAG_EMBED_MODEL={model}, or run "
            f"`ytrag reindex` to rebuild with your model."
        )

    vectors = data["vectors"].astype("float32")
    payloads = json.loads(str(data["payloads"]))
    name = ensure_collection()
    client = get_client()

    for start in range(0, len(vectors), batch_size):
        chunk_payloads = payloads[start : start + batch_size]
        points = [
            PointStruct(
                id=str(uuid.uuid5(_NAMESPACE, p["chunk_id"])),
                vector=v.tolist(),
                payload=p,
            )
            for v, p in zip(vectors[start : start + batch_size], chunk_payloads)
        ]
        client.upsert(collection_name=name, points=points, wait=True)

    return {"count": len(vectors), "model": model}
