"""
Retriever — the single interface to both Chroma collections.

The agent never queries Chroma directly — it calls these methods.
This separation means you can swap the vector store later without
touching the agent code at all.

Two collections:
  knowledge_base  — markdown docs (CV + project write-ups)
  code_base       — source code (functions/classes from repos)
"""

from typing import List, Dict
from openai import OpenAI
import chromadb

from config import (
    CHROMA_PATH, KNOWLEDGE_COLLECTION, CODE_COLLECTION,
    EMBEDDING_MODEL, KNOWLEDGE_TOP_K, CODE_TOP_K
)
from ingestion.persistence import restore_backup


class Retriever:

    def __init__(self):
        restore_backup()

        self._openai = OpenAI()
        client       = chromadb.PersistentClient(path=CHROMA_PATH)

        self.knowledge = client.get_or_create_collection(
            name=KNOWLEDGE_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )
        self.code = client.get_or_create_collection(
            name=CODE_COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )

    def _embed(self, text: str) -> List[List[float]]:
        """Embed a single string. Returns [[float...]] as Chroma expects a list of query embeddings."""
        resp = self._openai.embeddings.create(model="text-embedding-3-small", input=text)
        return [resp.data[0].embedding]

    # ── Tools (called by agent) ───────────────────────────────────────────────

    def search_knowledge(self, query: str) -> str:
        """
        Semantic search over CV + project docs.
        Returns top KNOWLEDGE_TOP_K chunks formatted for LLM injection.

        The min() guard prevents Chroma errors when the collection has
        fewer chunks than TOP_K — common during development.
        """
        n = min(KNOWLEDGE_TOP_K, self.knowledge.count() or 1)

        results = self.knowledge.query(
            query_embeddings = self._embed(query),
            n_results        = n,
            include          = ["documents", "metadatas", "distances"]
        )

        if not results["documents"][0]:
            return "No relevant information found."

        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            similarity = round((1 - dist) * 100, 1)
            output.append(
                f"[{meta['source']} / {meta['section']} — {similarity}% match]\n{doc}"
            )

        return "\n\n---\n\n".join(output)

    def get_cv_section(self, section_name: str) -> str:
        """
        Retrieve a specific CV section by exact metadata match.
        More precise than semantic search for known section names like
        'Skills', 'Experience', 'Education', 'Availability'.

        Why not just use search_knowledge for this?
        Semantic search might return a project doc instead of the CV section
        if the query is ambiguous. Metadata filtering guarantees the CV.
        """
        results = self.knowledge.get(
            where   = {"source": "cv.md"},
            include = ["documents", "metadatas"]
        )

        for doc, meta in zip(results["documents"], results["metadatas"]):
            if section_name.lower() in meta["section"].lower():
                return f"[cv.md / {meta['section']}]\n{doc}"

        return f"Section '{section_name}' not found in CV."

    def search_code(self, query: str) -> str:
        """
        Semantic search over source code chunks.
        Returns top CODE_TOP_K functions/classes with file + line metadata.

        Returns a clear message if code hasn't been ingested yet —
        avoids a confusing Chroma error reaching the agent.
        """
        if self.code.count() == 0:
            return (
                "Code base not indexed yet. "
                "Run: python -m ingestion.ingest_code --repo <path>"
            )

        n = min(CODE_TOP_K, self.code.count())

        results = self.code.query(
            query_embeddings = self._embed(query),
            n_results        = n,
            include          = ["documents", "metadatas", "distances"]
        )

        if not results["documents"][0]:
            return "No relevant code found."

        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            similarity = round((1 - dist) * 100, 1)
            output.append(
                f"[{meta['filename']} / {meta['name']} ({meta['type']}) "
                f"line {meta['line']} — {similarity}% match]\n{doc}"
            )

        return "\n\n---\n\n".join(output)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def status(self) -> Dict:
        """Quick health check — how many chunks are in each collection."""
        return {
            "knowledge_chunks": self.knowledge.count(),
            "code_chunks":      self.code.count(),
            "embedding_model":  EMBEDDING_MODEL,
            "knowledge_top_k":  KNOWLEDGE_TOP_K,
            "code_top_k":       CODE_TOP_K
        }