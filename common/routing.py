"""Deliberately simple request placement, shared by the live gateway and shadow tests.
(Pool by prompt length, least-in-flight within a pool. No prefix hashing — the smart part is the agent.)"""
from __future__ import annotations

import threading


def estimate_tokens(messages: list[dict]) -> int:
    return sum(len(str(m.get("content", "")).split()) for m in messages)


def pick_pool(split_threshold: int | None, prompt_tokens: int) -> str:
    if split_threshold is None:
        return "shared"
    return "long" if prompt_tokens >= split_threshold else "short"


class Balancer:
    """Least-in-flight across the replica URLs of each pool."""

    def __init__(self, endpoints: dict[str, list[str]]):
        self._lock = threading.Lock()
        self.endpoints = endpoints
        self.inflight = {u: 0 for urls in endpoints.values() for u in urls}

    def acquire(self, pool: str) -> str:
        with self._lock:
            urls = self.endpoints[pool]
            url = min(urls, key=lambda u: self.inflight.get(u, 0))
            self.inflight[url] = self.inflight.get(url, 0) + 1
            return url

    def release(self, url: str):
        with self._lock:
            if url in self.inflight:
                self.inflight[url] -= 1
