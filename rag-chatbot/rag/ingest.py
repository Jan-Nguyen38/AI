"""Command line ingestion.

    python -m rag.ingest sample_docs/            # a folder (recursively)
    python -m rag.ingest report.pdf notes.docx   # files
    python -m rag.ingest https://example.com/faq # web pages
    python -m rag.ingest --list
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .loaders import SUPPORTED_EXTENSIONS, load_path, load_url
from .store import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Add documents to the RAG index.")
    parser.add_argument("inputs", nargs="*", help="Files, folders or URLs")
    parser.add_argument("--list", action="store_true", help="List indexed documents")
    args = parser.parse_args()

    store = VectorStore()
    if args.list or not args.inputs:
        for doc in store.list_documents():
            print(f"{doc['id']}  {doc['chunks']:>4} chunks  {doc['name']}")
        return

    for item in args.inputs:
        if item.startswith(("http://", "https://")):
            title, sections = load_url(item)
            info = store.add_document(title, sections, source=item)
            print(f"Indexed {info['name']} ({info['chunks']} chunks)")
            continue
        path = Path(item)
        files = sorted(p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS) if path.is_dir() else [path]
        for f in files:
            try:
                info = store.add_document(f.name, load_path(f), source=str(f))
                print(f"Indexed {info['name']} ({info['chunks']} chunks)")
            except ValueError as e:
                print(f"Skipped {f.name}: {e}")


if __name__ == "__main__":
    main()
