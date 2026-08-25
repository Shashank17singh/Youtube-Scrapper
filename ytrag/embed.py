"""Pluggable embedder.

One backend today (sentence-transformers), but everything downstream talks to
the Protocol, so swapping the model is a config change plus `ytrag reindex`.
"""

import contextlib
import io
import re
import sys
from typing import Protocol

from ytrag.config import EMBED_BATCH, EMBED_MODEL, EMBED_QUERY_PREFIX

# Noise the model loader prints from a compiled extension, which no env var
# turns off. Filtered rather than suppressed wholesale: anything that is not
# one of these still reaches stderr, so real failures are never hidden.
_BENIGN = re.compile(
    r"unauthenticated requests to the HF Hub|Loading weights:|^\s*$"
)


@contextlib.contextmanager
def _quiet_load():
    """Swallow the known-benign loader chatter, re-emit everything else."""
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


class GoogleGenAIEmbedder:
    """Cloud embeddings using Google Gemini (text-embedding-004).
    
    This replaces the heavy local sentence-transformers model to keep memory 
    usage low enough for free cloud deployments (like Render 512MB tier).
    """

    def __init__(self, model_name: str = EMBED_MODEL, batch_size: int = EMBED_BATCH):
        from google import genai
        from ytrag.config import GEMINI_API_KEY
        
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY environment variable is missing. It is required for embeddings.")
            
        self.name = model_name
        self.batch_size = batch_size
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        # text-embedding-004 output dimension
        self.dim = 768

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
            
        vectors = []
        # Gemini allows batching, but we chunk it according to EMBED_BATCH
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = self.client.models.embed_content(
                model=self.name,
                contents=batch,
                config={"task_type": "RETRIEVAL_DOCUMENT"}
            )
            for emb in response.embeddings:
                vectors.append(emb.values)
                
        return vectors

    def embed_query(self, text: str) -> list[float]:
        response = self.client.models.embed_content(
            model=self.name,
            contents=text,
            config={"task_type": "RETRIEVAL_QUERY"}
        )
        return response.embeddings[0].values


_EMBEDDER: Embedder | None = None


def get_embedder() -> Embedder:
    """Load the embedder once per process."""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = GoogleGenAIEmbedder()
    return _EMBEDDER
