"""
LLM enrichment for document chunks — three metadata augmentation pillars:

  1. CONTEXTUAL DATA  — document title, section header, creation date, source file
  2. POSITIONAL DATA  — chunk position in document, position label (beginning/middle/end)
  3. CONTENT TAGS     — content type (FAQ/policy/technical_spec/etc.), key phrases,
                        document type, complexity level, language tone, entities

LLM enrichments run concurrently via RunnableParallel:
  - summary               : 2-3 sentence factual summary         (StrOutputParser)
  - hypothetical_questions: 3 questions this chunk answers       (StrOutputParser)
  - metadata_tags         : all structured tags in one call      (with_structured_output)

Positional and file-level contextual metadata are derived without LLM calls,
keeping cost low while maximising filtering/search utility.

References:
  - RunnableParallel:  https://python.langchain.com/docs/expression_language/primitives/parallel/
  - Structured output: https://python.langchain.com/docs/concepts/structured_outputs/
  - HyDE:              https://arxiv.org/abs/2212.10496
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel

from src.chunker import Chunk


# ── Structured metadata schema (LLM-extracted via with_structured_output) ─────

class MetadataTags(BaseModel):
    """Schema-enforced metadata extracted by the LLM.

    Covers all three augmentation pillars:
    - Contextual : document_title, section_header
    - Content    : content_type, key_phrases, document_type
    - Audience   : complexity_level, language_tone, entities
    """

    # ── Contextual ───────────────────────────────────────────────────────────
    document_title: str = Field(
        description="Inferred title of the source document this chunk belongs to"
    )
    section_header: str = Field(
        description="Most likely section or chapter heading this chunk falls under"
    )

    # ── Content tags ─────────────────────────────────────────────────────────
    content_type: Literal[
        "overview", "tutorial", "technical_spec", "FAQ",
        "policy", "definition", "example", "narrative"
    ] = Field(
        description="Type of content: FAQ for Q&A, policy for rules/guidelines, "
                    "technical_spec for specs/APIs, tutorial for step-by-step guides, etc."
    )
    key_phrases: list[str] = Field(
        description="3 to 5 specific key phrases that best represent this chunk's content"
    )
    document_type: Literal[
        "educational", "technical", "narrative", "reference", "procedural"
    ] = Field(
        description="Broad genre of the document"
    )
    entities: list[str] = Field(
        description="Named entities present: people, organizations, technologies, "
                    "products, or places (max 5)"
    )

    # ── Audience ─────────────────────────────────────────────────────────────
    complexity_level: Literal["beginner", "intermediate", "advanced"] = Field(
        description="Knowledge level required to understand this chunk"
    )
    language_tone: Literal["formal", "technical", "conversational", "instructional"] = Field(
        description="Overall tone and register of the writing"
    )


# ── Enrichment container ──────────────────────────────────────────────────────

@dataclass
class ChunkEnrichment:
    """All enrichment signals for a single chunk across the three pillars.

    All fields are plain strings for direct Chroma metadata storage
    (Chroma requires str | int | float | bool values).

    Pillar 1 — Contextual Data (LLM-extracted + file-level):
        document_title, section_header, source_file, creation_date

    Pillar 2 — Positional Metadata (derived, no LLM):
        chunk_position  e.g. "3 of 14"
        position_label  beginning | middle | end

    Pillar 3 — Content Tags (LLM structured output):
        content_type, key_phrases, document_type,
        complexity_level, language_tone, entities

    Free-text enrichments:
        summary, hypothetical_questions
    """

    # Pillar 1 — Contextual
    document_title: str = ""
    section_header: str = ""
    source_file: str = ""
    creation_date: str = ""

    # Pillar 2 — Positional (derived)
    chunk_position: str = ""   # "3 of 14"
    position_label: str = ""   # beginning | middle | end

    # Pillar 3 — Content tags
    content_type: str = ""
    key_phrases: str = ""       # comma-separated
    document_type: str = ""
    complexity_level: str = ""
    language_tone: str = ""
    entities: str = ""          # comma-separated

    # Free-text
    summary: str = ""
    hypothetical_questions: str = ""

    @property
    def key_phrases_list(self) -> list[str]:
        return [p.strip() for p in self.key_phrases.split(",") if p.strip()]

    @property
    def entities_list(self) -> list[str]:
        return [e.strip() for e in self.entities.split(",") if e.strip()]

    @property
    def questions_list(self) -> list[str]:
        return [q.strip() for q in self.hypothetical_questions.splitlines() if q.strip()]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_prompt(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def _positional_label(index: int, total: int) -> str:
    if total <= 1:
        return "beginning"
    ratio = index / (total - 1)
    if ratio < 0.33:
        return "beginning"
    elif ratio < 0.67:
        return "middle"
    return "end"


# ── Public API ────────────────────────────────────────────────────────────────

def enrich_chunk(
    chunk: Chunk,
    llm: BaseChatModel,
    prompts_dir: Path,
    total_chunks: int = 1,
    source_file: str = "",
    creation_date: str = "",
) -> ChunkEnrichment:
    """Run all LLM enrichments concurrently and derive positional metadata.

    LLM calls (via RunnableParallel — all dispatched simultaneously):
      - summary               : StrOutputParser chain
      - hypothetical_questions: StrOutputParser chain
      - metadata_tags         : with_structured_output(MetadataTags) chain

    Positional metadata is derived instantly from chunk.index + total_chunks.
    File-level contextual metadata (source_file, creation_date) is passed in
    from the pipeline — no extra LLM call needed.

    Args:
        chunk:         The chunk to enrich.
        llm:           LangChain BaseChatModel (any provider).
        prompts_dir:   Directory containing versioned prompt .txt files.
        total_chunks:  Total number of chunks in this document (for position).
        source_file:   Original filename for contextual metadata.
        creation_date: File creation/modification date (ISO string).

    Returns:
        ChunkEnrichment with all three metadata pillars populated.
    """
    # ── Pillar 2: Positional (no LLM) ────────────────────────────────────────
    chunk_position = f"{chunk.index + 1} of {total_chunks}"
    position_label = _positional_label(chunk.index, total_chunks)

    # ── Pillars 1 & 3: LLM enrichments (parallel) ────────────────────────────
    summary_chain = (
        ChatPromptTemplate.from_template(_load_prompt(prompts_dir / "summarization_v1.txt"))
        | llm.with_retry(stop_after_attempt=2)
        | StrOutputParser()
    )
    questions_chain = (
        ChatPromptTemplate.from_template(_load_prompt(prompts_dir / "questions_v1.txt"))
        | llm.with_retry(stop_after_attempt=2)
        | StrOutputParser()
    )
    metadata_chain = (
        ChatPromptTemplate.from_template(_load_prompt(prompts_dir / "metadata_v1.txt"))
        | llm.with_structured_output(MetadataTags).with_retry(stop_after_attempt=2)
    )

    result = RunnableParallel(
        summary=summary_chain,
        hypothetical_questions=questions_chain,
        metadata_tags=metadata_chain,
    ).invoke({"chunk_text": chunk.text})

    tags: MetadataTags = result["metadata_tags"]

    return ChunkEnrichment(
        # Pillar 1 — Contextual
        document_title=tags.document_title,
        section_header=tags.section_header,
        source_file=source_file,
        creation_date=creation_date,
        # Pillar 2 — Positional
        chunk_position=chunk_position,
        position_label=position_label,
        # Pillar 3 — Content tags
        content_type=tags.content_type,
        key_phrases=", ".join(tags.key_phrases),
        document_type=tags.document_type,
        complexity_level=tags.complexity_level,
        language_tone=tags.language_tone,
        entities=", ".join(tags.entities),
        # Free-text
        summary=result["summary"].strip(),
        hypothetical_questions=result["hypothetical_questions"].strip(),
    )


def empty_enrichment(
    source_file: str = "",
    creation_date: str = "",
    chunk_index: int = 0,
    total_chunks: int = 1,
) -> ChunkEnrichment:
    """Blank enrichment for chunks skipped due to cost cap, with positional data."""
    return ChunkEnrichment(
        source_file=source_file,
        creation_date=creation_date,
        chunk_position=f"{chunk_index + 1} of {total_chunks}",
        position_label=_positional_label(chunk_index, total_chunks),
    )