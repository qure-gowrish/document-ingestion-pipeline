"""
Document Ingestion Pipeline — POC

Demonstrates the full pipeline end-to-end:
  1. Parse .txt and .pdf files from sample_data/
  2. Run all 3 chunking strategies and display a comparison table
  3. Enrich the first N chunks via LLM summarization
  4. Embed chunks via OpenAI embeddings
  5. Store enriched chunks in local Chroma
  6. Run a test query and print the retrieved results

Usage:
    uv run python poc_pipeline.py              # upsert into existing DB
    uv run python poc_pipeline.py --reset      # wipe DB and re-ingest fresh
    LLM_PROVIDER=anthropic uv run python poc_pipeline.py --reset
    CHUNKING_STRATEGY=sentence uv run python poc_pipeline.py --reset

Environment:
    Copy .env.example to .env and fill in your API keys before running.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# ── Bootstrap: make src importable without installing the package ─────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_settings
from src.chunker import ChunkingStrategy, Chunk, chunk_text, compare_strategies
from src.enricher import ChunkEnrichment, enrich_chunk, empty_enrichment
from src.llm_provider import get_provider
from src.storage import ChromaStorage

# ── Constants ─────────────────────────────────────────────────────────────────

SAMPLE_DATA_DIR = Path("sample_data")
PROMPTS_DIR = Path("prompts")
ENRICH_MAX_CHUNKS = 5          # cap LLM calls during POC to control cost
QUERY_N_RESULTS = 3


# ── Parsing tools (stateless, idempotent) ─────────────────────────────────────

def parse_txt(path: Path) -> str:
    """Extract text from a plain-text file."""
    return path.read_text(encoding="utf-8")


def parse_pdf(path: Path) -> str:
    """Extract text from a PDF using pypdf."""
    from pypdf import PdfReader  # noqa: PLC0415

    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(p.strip() for p in pages if p.strip())


def parse_file(path: Path) -> str:
    """Route to the correct parser based on file extension."""
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return parse_txt(path)
    elif suffix == ".pdf":
        return parse_pdf(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


# ── Embedding tool (stateless) ────────────────────────────────────────────────

def embed_texts(
    texts: list[str],
    model: str,
    api_key: str,
    provider_type: str = "openai",
    openai_api_key: str = "",
) -> list[list[float]]:
    """Generate embeddings via OpenAI directly or proxied through Portkey.

    When provider_type='portkey', Portkey is used as the gateway with
    provider='openai' (sets x-portkey-provider header) and the OpenAI key
    as Authorization — no virtual key required.
    """
    if provider_type == "portkey":
        from portkey_ai import Portkey  # noqa: PLC0415
        client = Portkey(
            api_key=api_key,
            provider="openai",
            Authorization=openai_api_key,
        )
    else:
        from openai import OpenAI  # noqa: PLC0415
        client = OpenAI(api_key=api_key)

    response = client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]




# ── Display helpers ───────────────────────────────────────────────────────────

def _sep(widths: list[int], char: str = "─", corner_l: str = "├", corner_r: str = "┤", cross: str = "┼") -> str:
    return corner_l + cross.join(char * (w + 2) for w in widths) + corner_r


def print_comparison_table(metrics_list: list) -> None:
    headers = ["Strategy", "Chunks", "Avg Size", "Min", "Max", "Mid-Sentence %"]
    rows = [
        [
            m.strategy,
            str(m.num_chunks),
            str(m.avg_size),
            str(m.min_size),
            str(m.max_size),
            f"{m.mid_sentence_pct}%",
        ]
        for m in metrics_list
    ]
    col_widths = [
        max(len(headers[i]), max(len(r[i]) for r in rows))
        for i in range(len(headers))
    ]

    def row_line(cells: list[str]) -> str:
        return "│ " + " │ ".join(c.ljust(col_widths[i]) for i, c in enumerate(cells)) + " │"

    top = "┌" + "┬".join("─" * (w + 2) for w in col_widths) + "┐"
    mid = _sep(col_widths, "─", "├", "┤", "┼")
    bot = "└" + "┴".join("─" * (w + 2) for w in col_widths) + "┘"

    print(top)
    print(row_line(headers))
    print(mid)
    for row in rows:
        print(row_line(row))
    print(bot)


def print_section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def print_ok(msg: str) -> None:
    print(f"  ✓  {msg}")


def print_info(msg: str) -> None:
    print(f"  •  {msg}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline() -> None:
    settings = get_settings()

    print("\n" + "═" * 60)
    print("  Document Ingestion Pipeline — POC")
    print("═" * 60)
    print_info(f"LLM provider  : {settings.llm_provider} / {settings.llm_model_name}")
    print_info(f"Embedding     : {settings.embedding_model}")
    print_info(f"Chunk strategy: {settings.chunking_strategy}")
    print_info(f"Chunk size    : {settings.chunk_size}  overlap: {settings.chunk_overlap}")

    # ── Step 1: Parse files ───────────────────────────────────────────────────
    print_section("Step 1 · Parsing files")

    files = list(SAMPLE_DATA_DIR.glob("*.txt")) + list(SAMPLE_DATA_DIR.glob("*.pdf"))
    if not files:
        print("  ✗  No files found in sample_data/. Aborting.")
        sys.exit(1)

    parsed_docs: list[tuple[str, str]] = []   # (filename, text)
    for f in sorted(files):
        try:
            text = parse_file(f)
            parsed_docs.append((f.name, text))
            print_ok(f"{f.name:30s} → {len(text):>6,} characters")
        except Exception as exc:
            print(f"  ✗  {f.name}: {exc}")

    if not parsed_docs:
        print("  ✗  No documents parsed successfully. Aborting.")
        sys.exit(1)

    # ── Step 2: Compare chunking strategies ──────────────────────────────────
    print_section("Step 2 · Chunking strategy comparison")

    all_chunks_by_strategy: dict[str, list[tuple[str, list[Chunk]]]] = {
        s.value: [] for s in ChunkingStrategy
    }

    for filename, text in parsed_docs:
        print_info(f"File: {filename}")
        metrics = compare_strategies(text, settings.chunk_size, settings.chunk_overlap)
        print_comparison_table(metrics)

        # Collect chunks per strategy for later use
        for strategy in ChunkingStrategy:
            chunks = chunk_text(text, strategy, settings.chunk_size, settings.chunk_overlap)
            all_chunks_by_strategy[strategy.value].append((filename, chunks))

    # ── Step 3: Select strategy and enrich ───────────────────────────────────
    selected_strategy = settings.chunking_strategy
    print_section(f"Step 3 · LLM enrichment  (strategy='{selected_strategy}')")

    # Verify all versioned prompts exist before starting LLM calls
    required_prompts = ["summarization_v1.txt", "questions_v1.txt", "metadata_v1.txt"]
    for prompt_file in required_prompts:
        if not (PROMPTS_DIR / prompt_file).exists():
            print(f"  ✗  Prompt file not found: {PROMPTS_DIR / prompt_file}")
            sys.exit(1)
    print_ok(f"Loaded prompts: {', '.join(required_prompts)}")

    # Determine API key based on provider
    if settings.llm_provider == "anthropic":
        api_key = settings.anthropic_api_key
    elif settings.llm_provider == "portkey":
        api_key = settings.portkey_api_key
    else:
        api_key = settings.llm_api_key

    if not api_key or api_key.startswith("sk-proj-xxxx") or api_key.startswith("sk-ant-xxxx"):
        print(f"api_key: {api_key}")
        print("\n  ✗  No valid API key found in .env.")
        print("     Copy .env.example to .env and set your real API key.")
        print("     Skipping enrichment and embedding — storing raw chunks only.\n")
        _store_without_enrichment(
            all_chunks_by_strategy[selected_strategy],
            settings,
        )
        return

    provider = get_provider(
        provider_type=settings.llm_provider,
        api_key=api_key,
        model_name=settings.llm_model_name,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
        timeout_seconds=settings.llm_timeout,
    )
    print_ok(f"Provider ready: {settings.llm_provider} / {settings.llm_model_name}")

    # ── Step 4: Embed + enrich + store ────────────────────────────────────────
    print_section("Step 4 · Embedding, storage")

    storage = ChromaStorage(settings.chroma_persist_dir, settings.chroma_collection_name)

    for filename, chunks in all_chunks_by_strategy[selected_strategy]:
        if not chunks:
            continue

        enrichable = chunks[:ENRICH_MAX_CHUNKS]
        non_enrichable = chunks[ENRICH_MAX_CHUNKS:]

        print_info(f"{filename}: {len(chunks)} chunks total, enriching first {len(enrichable)}")

        # File-level contextual metadata — derived without LLM
        import datetime
        file_path = SAMPLE_DATA_DIR / filename
        creation_date = (
            datetime.datetime.fromtimestamp(file_path.stat().st_mtime).strftime("%Y-%m-%d")
            if file_path.exists() else ""
        )
        total_chunks = len(chunks)

        enrichments: list[ChunkEnrichment] = []
        for i, chunk in enumerate(enrichable):
            try:
                enrichment = enrich_chunk(
                    chunk, provider, PROMPTS_DIR,
                    total_chunks=total_chunks,
                    source_file=filename,
                    creation_date=creation_date,
                )
                enrichments.append(enrichment)
                print_ok(
                    f"  Chunk {i} [{enrichment.chunk_position} · {enrichment.position_label}] "
                    f"§ {enrichment.section_header or 'unknown'} · "
                    f"{enrichment.content_type} · {enrichment.complexity_level}"
                )
            except Exception as exc:
                print(f"  ✗  Chunk {i} enrichment failed: {exc}")
                enrichments.append(empty_enrichment(
                    source_file=filename,
                    creation_date=creation_date,
                    chunk_index=i,
                    total_chunks=total_chunks,
                ))

        # Pad enrichments for non-enriched chunks (cost cap)
        enrichments.extend([
            empty_enrichment(
                source_file=filename,
                creation_date=creation_date,
                chunk_index=ENRICH_MAX_CHUNKS + j,
                total_chunks=total_chunks,
            )
            for j, _ in enumerate(non_enrichable)
        ])
        all_chunks_for_file = enrichable + non_enrichable

        # Embed all chunks — uses embedding_provider, decoupled from llm_provider
        texts_to_embed = [c.text for c in all_chunks_for_file]
        embedding_key = settings.portkey_api_key if settings.embedding_provider == "portkey" else settings.llm_api_key
        try:
            embeddings = embed_texts(
                texts_to_embed,
                settings.embedding_model,
                embedding_key,
                provider_type=settings.embedding_provider,
                openai_api_key=settings.llm_api_key,
            )
        except Exception as exc:
            print(f"  ✗  Embedding failed for {filename}: {exc}")
            continue

        provider_info = {
            "provider": settings.llm_provider,
            "model_name": settings.llm_model_name,
        }
        storage.add_chunks(all_chunks_for_file, enrichments, embeddings, filename, provider_info)
        print_ok(f"Stored {len(all_chunks_for_file)} chunks for '{filename}'")

    total = storage.count()
    print_ok(f"Total chunks in Chroma: {total}")

    # ── Step 5: Test retrieval ────────────────────────────────────────────────
    print_section("Step 5 · Test retrieval")

    test_queries = [
        "What are the different types of machine learning?",
        "How does chunking affect RAG performance?",
    ]

    for query in test_queries:
        print_info(f"Query: \"{query}\"")
        try:
            query_embedding = embed_texts(
                [query],
                settings.embedding_model,
                settings.llm_api_key,
                provider_type=settings.embedding_provider,
                openai_api_key=settings.llm_api_key,
            )[0]
        except Exception as exc:
            print(f"  ✗  Query embedding failed: {exc}")
            print()
            continue
        results = storage.query(query, n_results=QUERY_N_RESULTS, query_embedding=query_embedding)
        for rank, result in enumerate(results, 1):
            sim = round(1 - (result.distance or 0), 3)
            print(f"      [{rank}] (sim={sim}) [{result.source_file}] {result.text[:120].strip()}...")
            if result.summary:
                print(f"           Summary    : {result.summary[:120].strip()}")
            # Pillar 1 — Contextual
            if result.document_title or result.section_header:
                print(f"           Context    : doc='{result.document_title}' § '{result.section_header}' | created={result.creation_date}")
            # Pillar 2 — Positional
            if result.chunk_position:
                print(f"           Position   : {result.chunk_position} ({result.position_label})")
            # Pillar 3 — Content tags
            if result.content_type:
                print(f"           Tags       : [{result.content_type}] {result.document_type} · {result.complexity_level} · {result.language_tone}")
            if result.key_phrases:
                print(f"           Key phrases: {result.key_phrases}")
            if result.entities:
                print(f"           Entities   : {result.entities}")
            if result.hypothetical_questions:
                print(f"           Q sample   : {result.hypothetical_questions.splitlines()[0].strip()}")
        print()

    print("═" * 60)
    print("  POC Complete!")
    print("═" * 60 + "\n")


def _store_without_enrichment(
    docs: list[tuple[str, list[Chunk]]],
    settings,
) -> None:
    """Fallback: store raw chunks with no LLM enrichment (no API key needed)."""
    storage = ChromaStorage(settings.chroma_persist_dir, settings.chroma_collection_name)

    # Use dummy embeddings for demo when no API key is available
    import random
    random.seed(42)

    for filename, chunks in docs:
        if not chunks:
            continue
        dummy_embeddings = [[random.uniform(-1, 1) for _ in range(8)] for _ in chunks]
        storage._collection.upsert(
            ids=[f"{filename}::{i}" for i in range(len(chunks))],
            documents=[c.text for c in chunks],
            embeddings=dummy_embeddings,
            metadatas=[
                {
                    "source_file": filename,
                    "chunk_index": c.index,
                    "char_count": c.char_count,
                    "strategy": c.strategy,
                    "summary": "", "hypothetical_questions": "",
                    "document_title": "", "section_header": "", "creation_date": "",
                    "chunk_position": f"{c.index + 1} of {len(chunks)}",
                    "position_label": "",
                    "content_type": "", "key_phrases": "", "document_type": "",
                    "complexity_level": "", "language_tone": "", "entities": "",
                    "provider": "none",
                    "model_name": "none",
                    "ingested_at": "0",
                }
                for c in chunks
            ],
        )
        print_ok(f"Stored {len(chunks)} raw chunks for '{filename}' (no enrichment)")
    print_ok(f"Total chunks in Chroma: {storage.count()}")
    print("\n  Set OPENAI_API_KEY in .env to enable LLM enrichment.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document Ingestion Pipeline")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe the Chroma DB before ingesting (fresh start)",
    )
    args = parser.parse_args()

    if args.reset:
        settings = get_settings()
        db_path = Path(settings.chroma_persist_dir)
        if db_path.exists():
            shutil.rmtree(db_path)
            print(f"  ✓  Wiped existing DB at {db_path}")
        else:
            print(f"  •  No existing DB at {db_path} — nothing to wipe")

    run_pipeline()
