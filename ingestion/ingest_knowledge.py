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

from openai import OpenAI
import chromadb

from config import (
    CHROMA_PATH, KNOWLEDGE_COLLECTION, CODE_COLLECTION,
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

def embed_and_store(chunks: List[Dict], collection, client: OpenAI):
    if not chunks:
        return

    texts   = [c["text"] for c in chunks]
    resp    = client.embeddings.create(model="text-embedding-3-small", input=texts)
    vectors = [item.embedding for item in resp.data]

    collection.upsert(
        ids        = [c["id"]   for c in chunks],
        embeddings = vectors,
        documents  = [c["text"] for c in chunks],
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
        model = OpenAI()

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
    Shares one OpenAI client across all files for efficiency.
    """
    if collection is None:
        collection = get_collection()
    if model is None:
        model = OpenAI()

    files = list(KNOWLEDGE_DIR.rglob("*.md"))
    print(f"Found {len(files)} markdown files")

    total = 0
    for f in files:
        total += ingest_file(str(f), collection, model)

    print(f"\nDone — {total} chunks ingested, {collection.count()} total in store")
    save_backup()


# ── Full pipeline ─────────────────────────────────────────────────────────────

def extract_github_urls() -> List[str]:
    """Find all unique GitHub repo URLs across all knowledge markdown files."""
    pattern = re.compile(r'github\.com/([\w.-]+/[\w.-]+)')
    urls    = set()
    for f in KNOWLEDGE_DIR.rglob("*.md"):
        text = f.read_text(encoding="utf-8")
        for match in pattern.finditer(text):
            repo = match.group(1).rstrip('/')
            urls.add(f"https://github.com/{repo}")
    return sorted(urls)


def ingest_all_with_code():
    """
    Full ingestion pipeline: knowledge files + all GitHub repos found in
    knowledge files. Saves backup once at the end.
    """
    from ingestion.ingest_code import ingest_repo, get_collection as get_code_collection

    client = OpenAI()

    print("=== Step 1: Knowledge files ===")
    ingest_all(collection=None, model=client)

    urls = extract_github_urls()
    print(f"\n=== Step 2: Source code from {len(urls)} repos ===")
    for url in urls:
        print(f"\n── {url}")
        code_collection = get_code_collection()
        ingest_repo(url, collection=code_collection, model=client)

    save_backup()
    print("\n✓ Full ingestion complete — backup saved")


# ── Status check ──────────────────────────────────────────────────────────────

def ingestion_status():
    """Print a summary of what's been ingested in both collections."""
    restore_backup()
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    print("── Knowledge base ───────────────────────────────")
    try:
        knowledge = client.get_collection(name=KNOWLEDGE_COLLECTION)
        data      = knowledge.get(include=["metadatas"])
        sources   = sorted(set(m["source"] for m in data["metadatas"]))
        print(f"{knowledge.count()} chunks from {len(sources)} files:")
        for s in sources:
            count = sum(1 for m in data["metadatas"] if m["source"] == s)
            print(f"  {s} ({count} chunks)")
    except Exception:
        print("  empty or not found")

    print("\n── Code base ────────────────────────────────────")
    try:
        code  = client.get_collection(name=CODE_COLLECTION)
        data  = code.get(include=["metadatas"])
        files = sorted(set(m["filename"] for m in data["metadatas"]))
        print(f"{code.count()} chunks from {len(files)} files:")
        for f in files:
            count = sum(1 for m in data["metadatas"] if m["filename"] == f)
            print(f"  {f} ({count} chunks)")
    except Exception:
        print("  empty or not found")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest markdown knowledge files")
    parser.add_argument("--file", type=str, help="Path to a single markdown file to ingest")
    args = parser.parse_args()

    if args.file:
        ingest_file(args.file)
    else:
        ingest_all()