# Document Ingestion Pipeline — POC

A minimal proof-of-concept that validates the full document ingestion pipeline:

```
Local Files → Parser → Chunker (3 strategies) → LLM Enricher → Chroma Vector DB
```

---

## Quick Start (3 commands)

```bash
# 1. Install dependencies
uv sync

# 2. Set up your API key
cp .env.example .env
# Edit .env → set OPENAI_API_KEY (or ANTHROPIC_API_KEY)

# 3. Run the POC
uv run python poc_pipeline.py
```

---

## What the POC Validates

| Goal | Status |
|------|--------|
| Parse `.txt` and `.pdf` files | ✓ |
| Compare 3 chunking strategies | ✓ |
| LLM summarization via prompt file | ✓ (requires API key) |
| Embedding + Chroma storage | ✓ (requires API key) |
| Semantic retrieval test | ✓ (requires API key) |
| Switch LLM provider via env var | ✓ |

---

## Chunking Strategies

The POC runs **all 3 strategies** on each file and prints a comparison table:

| Strategy | How it works | Best for |
|----------|-------------|----------|
| `fixed` | Sliding window by character count | Highly structured data |
| `recursive` | Splits at paragraph → sentence → word boundaries (default) | General documents |
| `sentence` | NLTK sentence grouping — never splits mid-sentence | Narrative text |

**Example output:**

```
┌───────────┬────────┬──────────┬─────┬─────┬────────────────┐
│ Strategy  │ Chunks │ Avg Size │ Min │ Max │ Mid-Sentence % │
├───────────┼────────┼──────────┼─────┼─────┼────────────────┤
│ fixed     │ 13     │ 479.5    │ 237 │ 500 │ 92.3%          │
│ recursive │ 14     │ 448.1    │ 279 │ 529 │ 35.7%          │
│ sentence  │ 13     │ 431.7    │ 211 │ 499 │  0.0%          │
└───────────┴────────┴──────────┴─────┴─────┴────────────────┘
```

A lower **Mid-Sentence %** means cleaner chunk boundaries.

---

## Switching Providers

Change `LLM_PROVIDER` in `.env` — **zero code changes required**:

```bash
# Use OpenAI (default)
LLM_PROVIDER=openai
LLM_MODEL_NAME=gpt-4o-mini
OPENAI_API_KEY=sk-proj-xxxx

# Use Anthropic Claude
LLM_PROVIDER=anthropic
LLM_MODEL_NAME=claude-3-haiku-20240307
ANTHROPIC_API_KEY=sk-ant-xxxx
```

Or pass inline:

```bash
LLM_PROVIDER=anthropic CHUNKING_STRATEGY=sentence uv run python poc_pipeline.py
```

---

## Project Structure

```
doccumentPipelineIngestionFlow/
├── pyproject.toml              # UV project — Python 3.11+
├── uv.lock                     # Reproducible lockfile
├── .env.example                # Template — copy to .env
├── poc_pipeline.py             # Main POC script
├── src/
│   ├── __init__.py
│   ├── config.py               # Pydantic Settings — all config from env
│   ├── chunker.py              # 3 strategies + comparison metrics
│   ├── llm_provider.py         # BaseLLMProvider + OpenAI + Anthropic
│   └── storage.py              # Chroma wrapper (upsert / query)
├── prompts/
│   └── summarization_v1.txt    # Versioned prompt — never inline in code
└── sample_data/
    ├── sample.txt              # Machine Learning article (~5,600 chars)
    └── sample.pdf              # RAG overview document (~2,500 chars)
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | `openai` or `anthropic` |
| `LLM_MODEL_NAME` | `gpt-4o-mini` | Model name passed to the provider |
| `OPENAI_API_KEY` | — | Required for OpenAI provider and embeddings |
| `ANTHROPIC_API_KEY` | — | Required for Anthropic provider |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `CHUNKING_STRATEGY` | `recursive` | `fixed`, `recursive`, or `sentence` |
| `CHUNK_SIZE` | `500` | Target chunk size in characters |
| `CHUNK_OVERLAP` | `50` | Overlap between consecutive chunks |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Local Chroma storage path |
| `LLM_MAX_TOKENS` | `500` | Max tokens per LLM call |
| `LLM_TEMPERATURE` | `0.3` | LLM temperature |
| `LLM_TIMEOUT` | `30` | API call timeout in seconds |

---

## Architecture Rules Followed

This POC respects the project's `.cursor/rules/`:

- **Prompts are external** — stored in `/prompts/summarization_v1.txt`, never in code
- **No hardcoded model names** — all come from environment variables
- **Provider abstraction** — `BaseLLMProvider` interface; enricher doesn't know which provider
- **Stateless chains** — each function is pure; no shared state between calls
- **Idempotent tools** — `add_chunks()` uses upsert; safe to re-run
- **Safety guards** — prompt injection detection, timeout, max tokens enforced
- **Structured config** — Pydantic `BaseSettings` with strict types

---

## Path to Production

After validating the POC approach, the production system adds:

**Phase 2 — Production Components**
1. AWS SQS consumer with long-polling and DLQ support
2. All 4 LLM providers: OpenAI, Anthropic, Azure, Local (Ollama)
3. Parent-child chunking pattern for better retrieval context
4. All enrichment types: keywords, metadata, Q&A pair generation
5. Retry logic with exponential backoff

**Phase 3 — Production Hardening**
6. Docker + docker-compose with LocalStack for SQS simulation
7. Structured logging with correlation IDs
8. Unit and integration test suite
9. Observability hooks (metrics, tracing)
10. Security hardening and secrets management
