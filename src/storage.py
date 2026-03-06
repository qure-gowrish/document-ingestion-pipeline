"""
Chroma vector storage wrapper for the document ingestion POC.

Architecture rules applied:
  - Memory must be abstracted behind an interface.
    → ChromaStorage hides all Chroma internals; callers only use add/query.
  - All tools must be idempotent.
    → add_chunks() uses upsert semantics; re-running with same IDs is safe.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.chunker import Chunk
from src.enricher import ChunkEnrichment


@dataclass
class StoredChunk:
    chunk_id: str
    text: str
    # Free-text enrichments
    summary: str
    hypothetical_questions: str
    # Pillar 1 — Contextual
    document_title: str
    section_header: str
    creation_date: str
    # Pillar 2 — Positional
    chunk_position: str
    position_label: str
    # Pillar 3 — Content tags
    content_type: str
    key_phrases: str
    document_type: str
    complexity_level: str
    language_tone: str
    entities: str
    # Provenance
    source_file: str
    chunk_index: int
    strategy: str
    provider: str
    model_name: str
    distance: float | None = None


class ChromaStorage:
    """
    Local persistent Chroma wrapper.

    Uses a single collection for the POC.  All metadata — including the
    chunking strategy, LLM provider, and model version — is stored alongside
    each chunk so enrichment provenance is always traceable.
    """

    def __init__(self, persist_dir: str = "./chroma_db", collection_name: str = "poc_documents") -> None:
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _make_id(source_file: str, chunk_index: int, strategy: str) -> str:
        """Deterministic, idempotent chunk ID."""
        raw = f"{source_file}::{strategy}::{chunk_index}"
        return hashlib.md5(raw.encode()).hexdigest()

    # ── Write ─────────────────────────────────────────────────────────────────

    def add_chunks(
        self,
        chunks: list[Chunk],
        enrichments: list[ChunkEnrichment],
        embeddings: list[list[float]],
        source_file: str,
        provider_info: dict[str, str],
    ) -> None:
        """
        Upsert enriched chunks into Chroma.

        Stores all enrichment signals (summary, keywords, hypothetical_questions)
        as metadata alongside the raw text and embedding.
        Idempotent: calling this twice with the same chunks will overwrite, not
        duplicate, because IDs are derived deterministically from the content key.
        """
        if not chunks:
            return

        ids = [self._make_id(source_file, c.index, c.strategy) for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "source_file": source_file,
                "chunk_index": c.index,
                "char_count": c.char_count,
                "strategy": c.strategy,
                # Free-text
                "summary": enrichments[i].summary if i < len(enrichments) else "",
                "hypothetical_questions": enrichments[i].hypothetical_questions if i < len(enrichments) else "",
                # Pillar 1 — Contextual
                "document_title": enrichments[i].document_title if i < len(enrichments) else "",
                "section_header": enrichments[i].section_header if i < len(enrichments) else "",
                "creation_date": enrichments[i].creation_date if i < len(enrichments) else "",
                # Pillar 2 — Positional
                "chunk_position": enrichments[i].chunk_position if i < len(enrichments) else "",
                "position_label": enrichments[i].position_label if i < len(enrichments) else "",
                # Pillar 3 — Content tags
                "content_type": enrichments[i].content_type if i < len(enrichments) else "",
                "key_phrases": enrichments[i].key_phrases if i < len(enrichments) else "",
                "document_type": enrichments[i].document_type if i < len(enrichments) else "",
                "complexity_level": enrichments[i].complexity_level if i < len(enrichments) else "",
                "language_tone": enrichments[i].language_tone if i < len(enrichments) else "",
                "entities": enrichments[i].entities if i < len(enrichments) else "",
                "provider": provider_info.get("provider", ""),
                "model_name": provider_info.get("model_name", ""),
                "ingested_at": str(int(time.time())),
            }
            for i, c in enumerate(chunks)
        ]

        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    # ── Read ──────────────────────────────────────────────────────────────────

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        query_embedding: list[float] | None = None,
    ) -> list[StoredChunk]:
        """Semantic similarity search.  Returns up to n_results chunks.

        Pass query_embedding when chunks were stored with an external embedding
        model (e.g. OpenAI) so Chroma doesn't use its own default model and
        produce a dimension mismatch.
        """
        kwargs: dict = dict(
            n_results=min(n_results, self._collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )
        if query_embedding is not None:
            kwargs["query_embeddings"] = [query_embedding]
        else:
            kwargs["query_texts"] = [query_text]

        results = self._collection.query(**kwargs)

        output: list[StoredChunk] = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, distances):
            output.append(
                StoredChunk(
                    chunk_id=self._make_id(
                        meta.get("source_file", ""),
                        meta.get("chunk_index", 0),
                        meta.get("strategy", ""),
                    ),
                    text=doc,
                    summary=meta.get("summary", ""),
                    hypothetical_questions=meta.get("hypothetical_questions", ""),
                    document_title=meta.get("document_title", ""),
                    section_header=meta.get("section_header", ""),
                    creation_date=meta.get("creation_date", ""),
                    chunk_position=meta.get("chunk_position", ""),
                    position_label=meta.get("position_label", ""),
                    content_type=meta.get("content_type", ""),
                    key_phrases=meta.get("key_phrases", ""),
                    document_type=meta.get("document_type", ""),
                    complexity_level=meta.get("complexity_level", ""),
                    language_tone=meta.get("language_tone", ""),
                    entities=meta.get("entities", ""),
                    source_file=meta.get("source_file", ""),
                    chunk_index=meta.get("chunk_index", 0),
                    strategy=meta.get("strategy", ""),
                    provider=meta.get("provider", ""),
                    model_name=meta.get("model_name", ""),
                    distance=dist,
                )
            )

        return output

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        """Delete all documents — useful for re-running the POC from scratch."""
        self._collection.delete(where={"chunk_index": {"$gte": 0}})
