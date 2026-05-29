"""
Ingest source code from a git repo into the code_base Chroma collection.
Uses AST parsing for Python (exact function/class boundaries),
regex for JavaScript (common function patterns).

Usage (from portfolio-assistant/ root):
    python -m ingestion.ingest_code --repo /local/path/to/repo
    python -m ingestion.ingest_code --repo https://github.com/ariamostajeran/aria-portfolio
"""

import re
import ast
import json
import hashlib
import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Optional

from openai import OpenAI
import chromadb

from config import (
    CHROMA_PATH, CODE_COLLECTION,
    SKIP_DIRS, SKIP_FILES, CODE_EXTENSIONS, MIN_CODE_LINES
)
from ingestion.persistence import save_backup, restore_backup


# ── Collection setup ──────────────────────────────────────────────────────────

def get_collection():
    restore_backup()
    client     = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=CODE_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


# ── Python chunking (AST-based) ───────────────────────────────────────────────

def chunk_python_file(filepath: Path) -> List[Dict]:
    """
    Parse a Python file with AST and extract each top-level
    function and class as a separate chunk.

    Why AST over regex for Python?
    - Understands actual syntax — not fooled by code inside strings or comments
    - Gives exact start/end line numbers for clean extraction
    - Handles decorators, async functions, nested classes correctly
    - ast.iter_child_nodes() gives only top-level definitions — no duplicates
      from nested functions being extracted separately from their parent
    """
    try:
        source = filepath.read_text(encoding="utf-8")
        tree   = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"  Skipping {filepath.name}: {e}")
        return []

    lines  = source.splitlines()
    chunks = []

    # iter_child_nodes gives only direct children of the module
    # (top-level definitions) — not nested functions inside classes
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue

        start_line = node.lineno - 1      # ast is 1-indexed, lists are 0-indexed
        end_line   = node.end_lineno

        # Pull in decorators — they're part of the function definition
        if node.decorator_list:
            start_line = node.decorator_list[0].lineno - 1

        code = "\n".join(lines[start_line:end_line])

        if len(code.strip().splitlines()) < MIN_CODE_LINES:
            continue

        node_type = "class" if isinstance(node, ast.ClassDef) else "function"
        chunk_id  = hashlib.md5(f"{filepath}::{node.name}".encode()).hexdigest()

        # Prepend file + function name to the embedded text
        # "File: data_fetcher.py / Function: get_latest_signals" is far more
        # searchable than raw code alone — gives the model context
        embed_text = (
            f"File: {filepath.name}\n"
            f"{node_type.capitalize()}: {node.name}\n\n"
            f"{code}"
        )

        chunks.append({
            "id":       chunk_id,
            "text":     embed_text,
            "source":   str(filepath),
            "filename": filepath.name,
            "name":     node.name,
            "type":     node_type,
            "line":     node.lineno,
            "language": "python"
        })

    return chunks


# ── JavaScript chunking (regex-based) ────────────────────────────────────────

def chunk_js_file(filepath: Path) -> List[Dict]:
    """
    Extract JavaScript functions using regex.
    Less precise than AST but covers the common patterns in portfolio code:
    - function declarations:  function name(...) { ... }
    - arrow functions:        const name = (...) => { ... }
    - class methods:          name(...) { ... }

    Why not a proper JS AST?
    - No built-in JS parser in Python stdlib
    - esprima/acorn require extra dependencies
    - Regex is good enough for a portfolio-sized codebase
    """
    try:
        source = filepath.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []

    chunks  = []
    pattern = re.compile(
        r'(?:^|\n)'
        r'(?:export\s+default\s+|export\s+)?'
        r'(?:async\s+)?'
        r'(?:'
        r'function\s+(\w+)\s*\([^)]*\)\s*\{|'
        r'(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)|\w+)\s*=>\s*\{|'
        r'(\w+)\s*\([^)]*\)\s*\{'
        r')',
        re.MULTILINE
    )

    for match in pattern.finditer(source):
        name = match.group(1) or match.group(2) or match.group(3)
        if not name or name in {"if", "for", "while", "switch", "catch"}:
            continue

        # Extract function body by counting opening/closing braces
        start       = match.start()
        brace_count = 0
        end         = start

        for i, char in enumerate(source[start:], start):
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    end = i + 1
                    break

        code = source[start:end].strip()
        if len(code.splitlines()) < MIN_CODE_LINES:
            continue

        chunk_id   = hashlib.md5(f"{filepath}::{name}".encode()).hexdigest()
        embed_text = f"File: {filepath.name}\nFunction: {name}\n\n{code}"
        line_num   = source[:start].count("\n") + 1

        chunks.append({
            "id":       chunk_id,
            "text":     embed_text,
            "source":   str(filepath),
            "filename": filepath.name,
            "name":     name,
            "type":     "function",
            "line":     line_num,
            "language": "javascript"
        })

    return chunks


def chunk_notebook_file(filepath: Path) -> List[Dict]:
    """Extract code cells from a Jupyter notebook, one chunk per cell."""
    try:
        nb = json.loads(filepath.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []

    chunks = []
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue

        source = cell.get("source", [])
        code   = "".join(source) if isinstance(source, list) else source
        code   = code.strip()

        if len(code.splitlines()) < MIN_CODE_LINES:
            continue

        chunk_id   = hashlib.md5(f"{filepath}::cell_{i}".encode()).hexdigest()
        embed_text = f"File: {filepath.name}\nNotebook cell {i + 1}:\n\n{code}"

        chunks.append({
            "id":       chunk_id,
            "text":     embed_text,
            "source":   str(filepath),
            "filename": filepath.name,
            "name":     f"cell_{i + 1}",
            "type":     "notebook_cell",
            "line":     i + 1,
            "language": "python"
        })

    return chunks


CHUNKERS = {
    ".py":   chunk_python_file,
    ".js":   chunk_js_file,
    ".ipynb": chunk_notebook_file,
}


# ── Repo ingestion ────────────────────────────────────────────────────────────

def ingest_repo(repo_path: str, collection=None, model=None) -> int:
    """
    Walk a repo, chunk all .py and .js files, embed and store.
    Accepts a local path or a GitHub URL (clones to a temp dir automatically).
    """
    if collection is None:
        collection = get_collection()
    if model is None:
        model = OpenAI()

    # Clone if GitHub URL provided
    tmp_dir: Optional[tempfile.TemporaryDirectory] = None
    if repo_path.startswith("http"):
        print(f"Cloning {repo_path}...")
        tmp_dir = tempfile.TemporaryDirectory()
        result  = subprocess.run(
            ["git", "clone", repo_path, tmp_dir.name],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"Clone failed: {result.stderr}")
            return 0
        repo_path = tmp_dir.name
        print("✓ Cloned")

    repo_path  = Path(repo_path)
    all_chunks = []

    for filepath in sorted(repo_path.rglob("*")):
        if any(part in SKIP_DIRS for part in filepath.parts):
            continue
        if filepath.name in SKIP_FILES:
            continue
        if filepath.suffix not in CHUNKERS:
            continue

        chunks = CHUNKERS[filepath.suffix](filepath)
        if chunks:
            rel = filepath.relative_to(repo_path)
            print(f"  {rel} — {len(chunks)} chunks")
            all_chunks.extend(chunks)

    if tmp_dir:
        tmp_dir.cleanup()

    if not all_chunks:
        print("No code chunks found — check the path")
        return 0

    print(f"\nEmbedding {len(all_chunks)} chunks...")
    texts   = [c["text"] for c in all_chunks]
    resp    = model.embeddings.create(model="text-embedding-3-small", input=texts)
    vectors = [item.embedding for item in resp.data]

    collection.upsert(
        ids        = [c["id"]   for c in all_chunks],
        embeddings = vectors,
        documents  = [c["text"] for c in all_chunks],
        metadatas  = [{
            "source":   c["source"],
            "filename": c["filename"],
            "name":     c["name"],
            "type":     c["type"],
            "line":     c["line"],
            "language": c["language"]
        } for c in all_chunks]
    )

    print(f"✓ {len(all_chunks)} code chunks stored")
    save_backup()
    return len(all_chunks)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest source code from a repo")
    parser.add_argument(
        "--repo", type=str, required=True,
        help="Local repo path or GitHub URL"
    )
    args = parser.parse_args()
    ingest_repo(args.repo)