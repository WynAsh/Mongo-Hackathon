"""Text embeddings and deterministic lexical retrieval helpers.

The embedding client deliberately has no required dependency beyond the
project's existing httpx package. Without an OpenRouter key it returns no
vector; callers can use :func:`lexical_score` instead of persisting fabricated
vectors.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

import httpx

from common import config

EMBEDDING_DIMENSIONS = 1536
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/text-embedding-3-small"


def _words(text: str) -> list[str]:
    return re.findall(r"[\w'-]+", text.lower())


def lexical_score(query: str, text: str) -> float:
    """Return a stable [0, 1] BM25-like overlap score for offline retrieval."""
    q = Counter(_words(query))
    d = Counter(_words(text))
    if not q or not d:
        return 0.0
    overlap = sum(min(count, d[word]) for word, count in q.items())
    precision = overlap / max(sum(q.values()), 1)
    coverage = len(set(q) & set(d)) / max(len(set(q)), 1)
    return round(0.6 * precision + 0.4 * coverage, 6)


def approximate_tokens(text: str) -> int:
    """Count tokens with tiktoken when installed, otherwise use a stable estimate."""
    try:
        import tiktoken
        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        # tiktoken may need to download its vocabulary on first use. Context
        # budgeting must remain available in offline/sandboxed deployments.
        return max(1, math.ceil(len(_words(text)) * 1.25)) if text.strip() else 0


class OpenRouterEmbedder:
    """Small synchronous OpenRouter embedding client with an explicit offline path."""

    def __init__(self, api_key: str | None = None, model: str | None = None,
                 base_url: str | None = None, timeout: float = 30.0,
                 client: httpx.Client | None = None):
        self.api_key = config.OPENROUTER_API_KEY if api_key is None else api_key
        self.model = model or getattr(config, "EMBEDDING_MODEL", DEFAULT_MODEL)
        self.base_url = (base_url or getattr(config, "OPENROUTER_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.timeout = timeout
        self._client = client

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def embed(self, texts: Iterable[str]) -> list[list[float]] | None:
        values = list(texts)
        if not values:
            return []
        if not self.available:
            return None
        client = self._client or httpx.Client(timeout=self.timeout)
        owns_client = self._client is None
        try:
            vectors = []
            for start in range(0, len(values), 64):
                batch = values[start:start + 64]
                response = client.post(
                    f"{self.base_url}/embeddings",
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json={"model": self.model, "input": batch, "dimensions": EMBEDDING_DIMENSIONS},
                )
                response.raise_for_status()
                rows = response.json().get("data", [])
                rows.sort(key=lambda row: row.get("index", 0))
                vectors.extend(row["embedding"] for row in rows)
            if len(vectors) != len(values) or any(len(v) != EMBEDDING_DIMENSIONS for v in vectors):
                raise ValueError("OpenRouter returned embeddings with an unexpected count or dimension")
            return vectors
        finally:
            if owns_client:
                client.close()
