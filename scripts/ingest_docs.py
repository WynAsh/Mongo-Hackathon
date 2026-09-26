"""Ingest configured Markdown/text files and curated documentation URLs."""
from common import config, db
from docs.ingest import DocumentIngestor


def main() -> int:
    if not config.DOC_SOURCES:
        print("DOC_SOURCES is empty; nothing to ingest")
        return 0
    records = DocumentIngestor(db.db().docs).ingest(config.DOC_SOURCES)
    print(f"ingested {len(records)} documentation chunks")
    return len(records)


if __name__ == "__main__":
    main()
