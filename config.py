import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
KNOWLEDGE_DIR = BASE_DIR / "knowledge"

# Chroma lives in /tmp (writable on Databricks), backed up to Workspace
CHROMA_PATH = "/tmp/portfolio_assistant/chroma_store"
BACKUP_ZIP  = str(BASE_DIR / "chroma_backup.zip")

# ── Collections ───────────────────────────────────────────────────────────────
KNOWLEDGE_COLLECTION = "knowledge_base"
CODE_COLLECTION      = "code_base"

# ── Embedding model ───────────────────────────────────────────────────────────
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM   = 1536

# ── LLM ───────────────────────────────────────────────────────────────────────
# Load from environment variable - set this in your .env file or shell
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it in your .env file or environment.")

LLM_MODEL       = "gpt-3.5-turbo"
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS  = 500

# ── Chunking ──────────────────────────────────────────────────────────────────
SKIP_SECTIONS    = {"links", "contact", "languages", "interests"}
MIN_CHUNK_LENGTH = 150
SKIP_DIRS        = {".git", "__pycache__", "node_modules", ".venv", "venv", "dist", "build"}
SKIP_FILES       = {"setup.py", "conftest.py"}
CODE_EXTENSIONS  = {".py", ".js"}
MIN_CODE_LINES   = 3

# ── Retrieval ─────────────────────────────────────────────────────────────────
# How many chunks to retrieve per query
# Higher = more context for the LLM but more tokens = higher cost
KNOWLEDGE_TOP_K = 5   # chunks from CV + project docs
CODE_TOP_K      = 5   # chunks from source code