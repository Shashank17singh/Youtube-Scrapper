"""Limits on the question endpoint, which spends LLM quota on every call."""

import threading
import time
from collections import deque


class RateLimiter:
    """Sliding-window limits for one client and for everyone together.

    X-Forwarded-For can be forged, so the client limit is only a courtesy. The overall
    limit is what protects the LLM quota.
    """

    def __init__(self, per_client: int, overall: int, window_seconds: float) -> None:
        self.per_client = per_client
        self.overall = overall
        self.window_seconds = window_seconds
        self._hits: deque[tuple[float, str]] = deque()
        self._lock = threading.Lock()

    def allow(self, client: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self._lock:
            while self._hits and now - self._hits[0][0] >= self.window_seconds:
                self._hits.popleft()
            if len(self._hits) >= self.overall:
                return False
            if sum(hit == client for _, hit in self._hits) >= self.per_client:
                return False
            self._hits.append((now, client))
            return True
