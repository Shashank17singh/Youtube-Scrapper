"""
Utility functions including reliable network connectivity checks, automatic
pausing when offline, and exponential backoff retry mechanisms.
"""

import socket
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")
_PROBE_HOSTS = [("www.youtube.com", 443), ("1.1.1.1", 53), ("8.8.8.8", 53)]


def network_up(timeout: float = 4.0) -> bool:
    for host, port in _PROBE_HOSTS:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            continue
    return False


def wait_for_network(
    max_wait_seconds: float = 6 * 3600,
    poll_seconds: float = 30.0,
    on_wait: Callable[[str], None] | None = None,
) -> bool:
    if network_up():
        return True

    waited = 0.0
    say = on_wait or (lambda m: print(f"[network] {m}"))
    say("connection lost - pausing until it returns")

    while waited < max_wait_seconds:
        time.sleep(poll_seconds)
        waited += poll_seconds
        if network_up():
            say(f"connection restored after {waited / 60:.0f} min - resuming")
            return True
        if waited % 300 == 0:
            say(f"still offline after {waited / 60:.0f} min, still waiting")

    say(f"gave up after {max_wait_seconds / 3600:.1f}h offline")
    return False


def with_retry(
    fn: Callable[[], T],
    attempts: int = 5,
    base_delay: float = 3.0,
    max_delay: float = 60.0,
    label: str = "operation",
    on_retry: Callable[[str], None] | None = None,
    is_rate_limit: Callable[[Exception], bool] | None = None,
    rate_limit_delay: float = 300.0,
) -> T:
    delay = base_delay
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == attempts:
                raise
            if not network_up():
                wait_for_network(on_wait=on_retry)
            this_delay = delay
            if is_rate_limit and is_rate_limit(exc):
                this_delay = rate_limit_delay * attempt
                reason = f"rate limited - waiting {this_delay / 60:.0f} min"
            else:
                reason = f"retrying in {this_delay:.0f}s"

            message = (
                f"{label} failed (attempt {attempt}/{attempts}): "
                f"{type(exc).__name__}: {str(exc)[:90]} - {reason}"
            )
            if on_retry:
                on_retry(message)
            else:
                print(f"[retry] {message}")
            time.sleep(this_delay)
            delay = min(delay * 2, max_delay)
    raise RuntimeError("unreachable")
