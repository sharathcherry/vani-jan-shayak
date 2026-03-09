"""cache.py — In-memory LRU answer cache.

Keeps RAG answers in memory across warm Lambda invocations so repeated
questions are answered in ~0 ms instead of invoking the RAG Lambda (~1-8 s).
The cache is size-bounded (max 256 entries) with a 24-hour per-entry TTL.
"""

import threading
import time
from collections import OrderedDict

from config import CACHE_TTL_SECONDS


class LRUCache:
    """Thread-safe fixed-size LRU cache with per-entry TTL."""

    def __init__(self, maxsize: int = 128, ttl: int = CACHE_TTL_SECONDS):
        self._cache: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self._maxsize = maxsize
        self._ttl     = ttl
        self._lock    = threading.Lock()

    def get(self, key: str) -> str | None:
        with self._lock:
            if key not in self._cache:
                return None
            value, ts = self._cache[key]
            if time.time() - ts > self._ttl:
                del self._cache[key]
                return None
            self._cache.move_to_end(key)   # mark as recently used
            return value

    def set(self, key: str, value: str) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (value, time.time())
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)  # evict least-recently-used


# Shared singleton — survives across warm Lambda invocations
answer_cache = LRUCache(maxsize=256, ttl=CACHE_TTL_SECONDS)
