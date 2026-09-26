from mongomock import MongoClient

from docs.ingest import DocumentIngestor, _html_sections, _text_sections


class NoEmbedding:
    available = False

    def embed(self, texts):
        return None


def test_markdown_ingestion_chunks_and_upserts(tmp_path):
    source = tmp_path / "guide.md"
    source.write_text("# Guide\n\n" + "word " * 45 + "\n\n## Scaling\n\n" + "replica " * 25, encoding="utf-8")
    collection = MongoClient().db.docs
    ingestor = DocumentIngestor(collection, NoEmbedding(), target_tokens=20, overlap_tokens=5)

    first = ingestor.ingest([source])
    second = ingestor.ingest([source])

    assert len(first) > 1
    assert collection.count_documents({}) == len(first)
    assert len(second) == len(first)
    assert {record["section"] for record in first} == {"Guide", "Scaling"}


def test_html_normalization_preserves_headings():
    title, sections = _html_sections("<html><title>Docs</title><body><h1>Intro</h1><p>Hello <b>world</b></p><h2>Use</h2><p>Details</p></body></html>")
    assert title == "Docs"
    assert sections == [("Intro", "Hello world"), ("Use", "Details")]


def test_text_fallback_and_content_hash_identity(tmp_path):
    source = tmp_path / "notes.txt"
    source.write_text("Plain text without headings. " * 10, encoding="utf-8")
    docs = MongoClient().db.docs
    ingestor = DocumentIngestor(docs, NoEmbedding(), target_tokens=8, overlap_tokens=2)
    records = ingestor.ingest([source])
    assert records
    assert all(row["content_hash"] == row["doc_id"] for row in records)
    assert all(row["section"] == "Introduction" for row in records)

