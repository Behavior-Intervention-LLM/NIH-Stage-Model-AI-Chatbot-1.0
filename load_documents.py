#!/usr/bin/env python3
"""
： PDF  DOC 
"""
import sys
import os
from pathlib import Path

# 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.tools.document_loader import DocumentLoader
from app.tools.vector_store import SimpleVectorStore


def main():
    """"""
    print("=" * 60)
    print(" - NIH Stage Model AI Chatbot")
    print("=" * 60)
    
    # （）
    docs_dir = Path("data/documents")
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n: {docs_dir.absolute()}")
    print(" PDF  DOC ")
    
    # 
    pdf_files = list(docs_dir.glob("*.pdf"))
    doc_files = list(docs_dir.glob("*.doc"))
    docx_files = list(docs_dir.glob("*.docx"))
    
    all_files = pdf_files + doc_files + docx_files
    
    if not all_files:
        print("\n⚠️  ！")
        print(f" PDF、DOC  DOCX : {docs_dir}")
        return
    
    print(f"\n {len(all_files)} :")
    for f in all_files:
        print(f"  - {f.name}")
    
    # 
    print("\n...")
    loader = DocumentLoader(chunk_size=500, chunk_overlap=50)
    
    print("...")
    vector_store = SimpleVectorStore(storage_path="data/vector_store")
    
    # 
    print("\n...")
    all_chunks = []
    
    for file_path in all_files:
        try:
            print(f"\n: {file_path.name}...")
            chunks = loader.load_file(str(file_path))
            all_chunks.extend(chunks)
            print(f"  ✓  {len(chunks)} ")
        except Exception as e:
            print(f"  ✗ : {e}")
    
    if not all_chunks:
        print("\n⚠️  ！")
        return
    
    # 
    print(f"\n...")
    vector_store.add_chunks(all_chunks)
    
    # 
    stats = vector_store.get_stats()
    print("\n" + "=" * 60)
    print("!:")
    print("=" * 60)
    print(f": {stats['total_chunks']}")
    print(f": {stats['vocabulary_size']}")
    print("\n:")
    for source, count in stats['sources'].items():
        print(f"  - {source}: {count} ")
    
    print(f"\n: data/vector_store/")
    print("\n✓  RAG !")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
