"""
Provides integration layers for various embedding models (e.g., FastEmbed,
SentenceTransformers) to vectorize text segments for Qdrant index insertion.
"""

import contextlib
import io
import re
import sys
from typing import Protocol

from ytrag.config import EMBED_BATCH, EMBED_MODEL, EMBED_QUERY_PREFIX

_BENIGN = re.compile(r"unauthenticated requests to the HF Hub|Loading weights:|^\s*$")


@contextlib.contextmanager
def _quiet_load():
    captured = io.StringIO()
    try:
        with contextlib.redirect_stderr(captured):
            yield
    finally:
        for line in captured.getvalue().splitlines():
            if not _BENIGN.search(line):
                print(line, file=sys.stderr)


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = EMBED_MODEL, batch_size: int = EMBED_BATCH):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.batch_size = batch_size
        with _quiet_load():
            self.model = SentenceTransformer(model_name)
        get_dim = getattr(self.model, "get_embedding_dimension", None) or (
            self.model.get_sentence_embedding_dimension
        )
        self.dim = int(get_dim())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(
            EMBED_QUERY_PREFIX + text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector.tolist()


class FastEmbedder:
    def __init__(self, model_name: str = EMBED_MODEL, batch_size: int = EMBED_BATCH):
        from fastembed import TextEmbedding

        self.name = model_name
        self.batch_size = batch_size
        fastembed_model = model_name
        if model_name == "all-MiniLM-L6-v2":
            fastembed_model = "sentence-transformers/all-MiniLM-L6-v2"

        with _quiet_load():
            self.model = TextEmbedding(fastembed_model)
        dummy = next(iter(self.model.embed(["test"])))
        self.dim = len(dummy)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = list(self.model.embed(texts, batch_size=self.batch_size))
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = next(iter(self.model.embed([EMBED_QUERY_PREFIX + text])))
        return vector.tolist()


_EMBEDDER: Embedder | None = None


def get_embedder() -> Embedder:
    global _EMBEDDER
    if _EMBEDDER is None:
        try:
            import fastembed  # noqa: F401

            _EMBEDDER = FastEmbedder()
        except ImportError:
            _EMBEDDER = SentenceTransformerEmbedder()
    return _EMBEDDER
