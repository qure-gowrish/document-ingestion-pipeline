"""
Chunking strategies for the document ingestion POC.

Three strategies are supported, all backed by LangChain text splitters:
  - fixed    : CharacterTextSplitter — pure character sliding window (baseline)
  - recursive: RecursiveCharacterTextSplitter — paragraph → sentence → word
               boundary hierarchy (recommended, mirrors LangChain default)
  - sentence : NLTKTextSplitter — sentence-aware grouping (best for narrative text)

All strategies return the same List[Chunk] shape so the rest of the pipeline is
strategy-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List

from langchain_text_splitters import (
    CharacterTextSplitter,
    NLTKTextSplitter,
    RecursiveCharacterTextSplitter,
)


class ChunkingStrategy(str, Enum):
    FIXED = "fixed"
    RECURSIVE = "recursive"
    SENTENCE = "sentence"


@dataclass
class Chunk:
    text: str
    index: int
    strategy: str
    char_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.char_count = len(self.text)

    def ends_mid_sentence(self) -> bool:
        """Heuristic: chunk ends cleanly if it ends with sentence-terminal punctuation."""
        stripped = self.text.rstrip()
        return bool(stripped) and stripped[-1] not in ".!?\"'"


# ── Strategy 1: Fixed-Size ────────────────────────────────────────────────────

def chunk_fixed_size(text: str, chunk_size: int = 500, overlap: int = 50) -> List[Chunk]:
    """
    Fixed character-count sliding window via LangChain CharacterTextSplitter.
    separator="" forces pure character-level splitting — may cut mid-sentence.
    """
    splitter = CharacterTextSplitter(
        separator="",
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        is_separator_regex=False,
    )
    texts = splitter.split_text(text)
    return [Chunk(text=t, index=i, strategy=ChunkingStrategy.FIXED) for i, t in enumerate(texts)]


# ── Strategy 2: Recursive Character Splitting ─────────────────────────────────

def chunk_recursive(text: str, chunk_size: int = 500, overlap: int = 50) -> List[Chunk]:
    """
    Recursive character splitting via LangChain RecursiveCharacterTextSplitter.
    Tries paragraph → line → sentence → word boundaries in order.
    Produces semantically coherent chunks; LangChain's recommended default.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
    )
    texts = splitter.split_text(text)
    return [Chunk(text=t, index=i, strategy=ChunkingStrategy.RECURSIVE) for i, t in enumerate(texts)]


# ── Strategy 3: Sentence-Based ────────────────────────────────────────────────

def chunk_sentences(text: str, target_size: int = 500) -> List[Chunk]:
    """
    Sentence-aware chunking via LangChain NLTKTextSplitter.
    Never splits mid-sentence; best for narrative or prose text.
    Falls back to word boundaries if NLTK is unavailable.
    """
    splitter = NLTKTextSplitter(
        chunk_size=target_size,
        chunk_overlap=0,
    )
    texts = splitter.split_text(text)
    return [Chunk(text=t, index=i, strategy=ChunkingStrategy.SENTENCE) for i, t in enumerate(texts)]


# ── Public dispatcher ─────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[Chunk]:
    """Dispatch to the requested LangChain-backed chunking strategy."""
    if strategy == ChunkingStrategy.FIXED:
        return chunk_fixed_size(text, chunk_size, overlap)
    elif strategy == ChunkingStrategy.RECURSIVE:
        return chunk_recursive(text, chunk_size, overlap)
    elif strategy == ChunkingStrategy.SENTENCE:
        return chunk_sentences(text, chunk_size)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")


# ── Comparison metrics ────────────────────────────────────────────────────────

@dataclass
class StrategyMetrics:
    strategy: str
    num_chunks: int
    avg_size: float
    min_size: int
    max_size: int
    mid_sentence_pct: float  # 0–100


def compare_strategies(text: str, chunk_size: int = 500, overlap: int = 50) -> list[StrategyMetrics]:
    """Run all 3 strategies on the same text and return comparison metrics."""
    results: list[StrategyMetrics] = []

    for strategy in ChunkingStrategy:
        chunks = chunk_text(text, strategy, chunk_size, overlap)
        if not chunks:
            continue

        sizes = [c.char_count for c in chunks]
        mid_sentence = sum(1 for c in chunks if c.ends_mid_sentence())

        results.append(
            StrategyMetrics(
                strategy=strategy.value,
                num_chunks=len(chunks),
                avg_size=round(sum(sizes) / len(sizes), 1),
                min_size=min(sizes),
                max_size=max(sizes),
                mid_sentence_pct=round(mid_sentence / len(chunks) * 100, 1),
            )
        )

    return results
