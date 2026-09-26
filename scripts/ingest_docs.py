"""Ingest configured Markdown/text files and curated documentation URLs."""
from common import config, db
from docs.ingest import DocumentIngestor


def main() -> int:
    if not config.DOC_SOURCES:
        print("DOC_SOURCES is empty; nothing to ingest")
        return 0
    ingestor = DocumentIngestor(db.db().docs)
    records = ingestor.ingest(config.DOC_SOURCES)
    embedded = sum(1 for record in records if record.get("embedding"))
    print(f"ingested {len(records)} documentation chunks ({embedded} embedded)")
    for failure in ingestor.errors:
        print(f"source skipped: {failure['source']}: {failure['error']}")
    if ingestor.embedding_errors:
        print(f"embedding batches skipped: {len(ingestor.embedding_errors)}")
    return len(records)


if __name__ == "__main__":
    main()
