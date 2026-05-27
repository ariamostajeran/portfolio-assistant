"""
Ingest markdown knowledge files into the knowledge_base Chroma collection.

Usage (from portfolio-assistant/ root):
    python -m ingestion.ingest_knowledge                                        # all files
    python -m ingestion.ingest_knowledge --file knowledge/projects/cv.md       # one file
"""

import re
import hashlib
import argparse
from pathlib import Path
from typing import List, Dict

from sentence_transformers import SentenceTransformer
import chromadb

from config import (
    CHROMA_PATH, KNOWLEDGE_COLLECTION, EMBEDDING_MODEL,
    KNOWLEDGE_DIR, SKIP_SECTIONS, MIN_CHUNK_LENGTH
)
from ingestion.persistence import save_backup, restore_backup


# ── Collection setup ──────────────────────────────────────────────────────────

def get_collection():
    restore_backup()
    client     = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=KNOWLEDGE_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_markdown(text: str, source: str) -> List[Dict]:
    """
    Split a markdown file into chunks by ## heading.
    Each ## section = one coherent idea = one chunk.

    Why ## and not #?
    - # is the document title — too broad, dilutes meaning
    - ## sections are the right granularity: one topic each

    Skips:
    - Sections shorter than MIN_CHUNK_LENGTH (noise like Links, Contact)
    - Utility sections defined in SKIP_SECTIONS
    """
    chunks   = []
    sections = re.split(r'\n(?=## )', text.strip())

    for section in sections:
        if not section.strip():
            continue

        lines   = section.strip().split('\n')
        heading = lines[0].replace('#', '').strip() if lines[0].startswith('#') else "intro"
        content = '\n'.join(lines).strip()

        if len(content) < MIN_CHUNK_LENGTH:
            continue
        if heading.lower() in SKIP_SECTIONS:
            continue

        # Stable ID: same file + heading always produces same ID
        # Means re-running is safe — upsert updates instead of duplicating
        chunk_id = hashlib.md5(f"{source}::{heading}".encode()).hexdigest()

        chunks.append({
            "id":      chunk_id,
            "text":    content,
            "source":  source,
            "section": heading,
            "type":    "project" if "projects/" in source else "cv"
        })

    return chunks


# ── Embed + store ─────────────────────────────────────────────────────────────

def embed_and_store(chunks: List[Dict], collection, model: SentenceTransformer):
    if not chunks:
        return

    texts   = [c["text"] for c in chunks]

    # Batch encoding — much faster than encoding one by one
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    collection.upsert(
        ids        = [c["id"]    for c in chunks],
        embeddings = [v.tolist() for v in vectors],
        documents  = [c["text"]  for c in chunks],
        metadatas  = [{
            "source":  c["source"],
            "section": c["section"],
            "type":    c["type"]
        } for c in chunks]
    )


# ── Public API ────────────────────────────────────────────────────────────────

def ingest_file(filepath: str, collection=None, model=None) -> int:
    """
    Ingest a single markdown file.
    Creates collection and model if not provided — safe to call standalone.
    Returns number of chunks ingested.
    """
    if collection is None:
        collection = get_collection()
    if model is None:
        model = SentenceTransformer(EMBEDDING_MODEL)

    filepath = Path(filepath)
    text     = filepath.read_text(encoding="utf-8")
    source   = filepath.name
    chunks   = chunk_markdown(text, source)

    embed_and_store(chunks, collection, model)
    print(f"✓ {filepath.name} — {len(chunks)} chunks ingested")

    save_backup()
    return len(chunks)


def ingest_all(collection=None, model=None):
    """
    Ingest all markdown files found recursively in KNOWLEDGE_DIR.
    Shares one model instance across all files for efficiency.
    """
    if collection is None:
        collection = get_collection()
    if model is None:
        model = SentenceTransformer(EMBEDDING_MODEL)

    files = list(KNOWLEDGE_DIR.rglob("*.md"))
    print(f"Found {len(files)} markdown files")

    total = 0
    for f in files:
        total += ingest_file(str(f), collection, model)

    print(f"\nDone — {total} chunks ingested, {collection.count()} total in store")
    save_backup()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest markdown knowledge files")
    parser.add_argument("--file", type=str, help="Path to a single markdown file to ingest")
    args = parser.parse_args()

    if args.file:
        ingest_file(args.file)
    else:
        ingest_all()