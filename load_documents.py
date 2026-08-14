#!/usr/bin/env python3
"""
Build the TF-IDF vector store from data/documents/.

Rebuilds from scratch every run. The previous version of this script called
add_chunks() against a store already loaded from disk, which appended rather
than replaced — so each run added another complete copy of every document.
The index it left behind held 641 chunks covering only 186 unique passages.

Usage:
    python load_documents.py            # rebuild the index from scratch
    python load_documents.py --append   # add only documents not yet indexed
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.config import settings
from app.tools.document_loader import DocumentLoader
from app.tools.vector_store import SimpleVectorStore

SUPPORTED = (".pdf", ".doc", ".docx")


def main() -> int:
    append = "--append" in sys.argv

    docs_dir = Path(settings.DOCUMENTS_DIR)
    docs_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in docs_dir.iterdir() if p.suffix.lower() in SUPPORTED)
    if not files:
        print(f"No {'/'.join(SUPPORTED)} files found in {docs_dir.absolute()}")
        return 1

    print(f"Source directory : {docs_dir.absolute()}")
    print(f"Mode             : {'append' if append else 'full rebuild'}")
    print(f"Documents found  : {len(files)}")
    for f in files:
        print(f"  - {f.name}")

    loader = DocumentLoader(chunk_size=500, chunk_overlap=50)
    store = SimpleVectorStore(storage_path=settings.VECTOR_STORE_PATH)

    print("\nExtracting and chunking...")
    all_chunks = []
    failed = []
    for path in files:
        try:
            chunks = loader.load_file(str(path))
            all_chunks.extend(chunks)
            print(f"  ok   {path.name}: {len(chunks)} chunks")
        except Exception as exc:
            failed.append((path.name, exc))
            print(f"  FAIL {path.name}: {exc}")

    if not all_chunks:
        print("\nNo text extracted; index left unchanged.")
        return 1

    print("\nIndexing...")
    added = store.add_chunks(all_chunks, replace=not append)

    stats = store.get_stats()
    print("\n" + "=" * 60)
    print(f"Chunks submitted : {len(all_chunks)}")
    print(f"Chunks indexed   : {added} (duplicates skipped: {len(all_chunks) - added})")
    print(f"Total in store   : {stats['total_chunks']}")
    print(f"Vocabulary       : {stats['vocabulary_size']} terms")
    print("\nPer source:")
    for source, count in sorted(stats["sources"].items(), key=lambda x: -x[1]):
        print(f"  {count:5d}  {source}")
    print(f"\nWritten to {Path(settings.VECTOR_STORE_PATH).absolute()}")

    if failed:
        print(f"\n{len(failed)} document(s) failed to load:")
        for name, exc in failed:
            print(f"  - {name}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
