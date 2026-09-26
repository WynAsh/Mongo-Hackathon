"""Hash-addressed ingestion of local and curated web documentation."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import httpx

from memory.embeddings import OpenRouterEmbedder, approximate_tokens


def _html_sections(html: str) -> tuple[str, list[tuple[str, str]]]:
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise RuntimeError("beautifulsoup4 is required to ingest web documentation") from exc
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    root = soup.find("main") or soup.find("article") or soup.body or soup
    sections: list[tuple[str, str]] = []
    heading = title or "Introduction"
    pieces: list[str] = []
    for element in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre"]):
        if element.name.startswith("h"):
            if pieces:
                sections.append((heading, "\n".join(pieces)))
            heading = element.get_text(" ", strip=True) or heading
            pieces = []
        else:
            text = element.get_text(" ", strip=True)
            if text:
                pieces.append(text)
    if pieces:
        sections.append((heading, "\n".join(pieces)))
    if not sections:
        text = root.get_text("\n", strip=True)
        if text:
            sections.append((title or "Document", text))
    return title, sections


def _text_sections(text: str) -> tuple[str, list[tuple[str, str]]]:
    title, sections, heading, lines = "", [], "Introduction", []
    for line in text.splitlines():
        match = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if match:
            if lines:
                sections.append((heading, "\n".join(lines).strip()))
            heading = match.group(2).strip()
            if not title:
                title = heading
            lines = []
        elif line.strip():
            lines.append(line.strip())
    if lines:
        sections.append((heading, "\n".join(lines).strip()))
    if not sections and text.strip():
        sections = [(title or "Document", text.strip())]
    return title or "Document", sections


def _chunks(sections: list[tuple[str, str]], target_tokens: int, overlap_tokens: int) -> list[tuple[str, str]]:
    chunks = []
    for heading, body in sections:
        words = body.split()
        if not words:
            continue
        # Word windows approximate tokens; for actual tokenizer sizing, merge
        # adjacent windows until the target estimate is reached.
        step = max(1, target_tokens - overlap_tokens)
        for start in range(0, len(words), step):
            text = " ".join(words[start:start + target_tokens])
            if text:
                chunks.append((heading, text))
    return chunks


class DocumentIngestor:
    def __init__(self, collection, embedder: OpenRouterEmbedder | None = None,
                 target_tokens: int = 800, overlap_tokens: int = 100,
                 client: httpx.Client | None = None, timeout: float = 30.0):
        self.collection = collection
        self.embedder = embedder or OpenRouterEmbedder()
        self.target_tokens = max(1, target_tokens)
        self.overlap_tokens = max(0, min(overlap_tokens, self.target_tokens - 1))
        self.client = client
        self.timeout = timeout

    def ingest(self, sources: Iterable[str | Path]) -> list[dict]:
        records = []
        for source in sources:
            source_id = str(source)
            if source_id.startswith(("https://", "http://")):
                content, title, sections = self._fetch(source_id)
            else:
                path = Path(source)
                content = path.read_text(encoding="utf-8")
                source_id = str(path.resolve())
                if path.suffix.lower() in {".html", ".htm"}:
                    title, sections = _html_sections(content)
                else:
                    title, sections = _text_sections(content)
            source_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            # A changed source replaces only its own obsolete chunks. Other
            # sources and older experiment evidence remain untouched.
            self.collection.delete_many({"source": source_id, "source_hash": {"$ne": source_hash}})
            chunks = _chunks(sections, self.target_tokens, self.overlap_tokens)
            vectors = None
            try:
                vectors = self.embedder.embed([text for _, text in chunks]) if chunks else []
            except Exception:
                vectors = None
            for index, (heading, text) in enumerate(chunks):
                # Repeated prose can legitimately produce identical chunks.
                # Include the stable chunk position so each source location has
                # its own address while repeated ingestion remains idempotent.
                digest = hashlib.sha256(
                    f"{source_id}\0{index}\0{heading}\0{text}".encode("utf-8")
                ).hexdigest()
                record = {
                    "doc_id": digest,
                    "source": source_id,
                    "source_hash": source_hash,
                    "content_hash": digest,
                    "title": title,
                    "section": heading,
                    "chunk_index": index,
                    "text": text,
                    "token_estimate": approximate_tokens(text),
                }
                if vectors and index < len(vectors):
                    record["embedding"] = vectors[index]
                self.collection.update_one({"content_hash": digest}, {"$set": record}, upsert=True)
                records.append(record)
        return records

    def _fetch(self, url: str) -> tuple[str, str, list[tuple[str, str]]]:
        owns = self.client is None
        client = self.client or httpx.Client(timeout=self.timeout, follow_redirects=True,
                                             headers={"User-Agent": "long-horizon-agent-doc-ingest/1.0"})
        try:
            response = client.get(url)
            response.raise_for_status()
            content = response.text
            parsed = urlparse(response.url)
            if parsed.scheme not in {"http", "https"}:
                raise ValueError("documentation URL must use HTTP or HTTPS")
            title, sections = _html_sections(content)
            return content, title or url, sections
        finally:
            if owns:
                client.close()
