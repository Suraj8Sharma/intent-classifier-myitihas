# Intent Classifier

Routes natural language queries to the correct backend pipeline using LLM-based intent classification.

**Supported pipelines:**
- `rag` — Retrieval-Augmented Generation (answers factual questions from a knowledge base)
- `creative` — Open-ended creative content generation
- `fallback` — Graceful degradation for ambiguous or off-topic queries

---

## Quick Start

```bash
# 1. Clone & activate venv
python -m venv venv && source venv/bin/activate

# 2. Install dependencies
make dev-install

# 3. Configure environment
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY (and OPENAI_API_KEY for embeddings)

# 4. Run the API server
make run
# → http://localhost:8000

# 5. Classify a query
curl -X POST http://localhost:8000/api/v1/classify \
  -H "Content-Type: application/json" \
  -d '{"query": "What is our refund policy?"}'
```

---

## Project Structure

```
src/
├── classifier/     # Intent classification + routing logic
├── pipelines/      # RAG, creative, and fallback pipeline handlers
├── models/         # Pydantic schemas + LLM client wrapper
├── prompts/        # LLM prompt templates (.txt files)
├── utils/          # Config loader, logger, helpers
└── api/            # FastAPI app, routes, auth middleware

tests/
├── unit/           # Fast isolated tests (mocked LLM)
└── integration/    # API-level tests

configs/            # YAML configuration per environment
scripts/            # Data ingestion, embedding, and eval scripts
data/               # Raw docs, processed chunks, embeddings, intent taxonomy
```

---

## Common Commands

| Command | Description |
|---|---|
| `make run` | Start API server with hot reload |
| `make test` | Run full test suite with coverage |
| `make test-unit` | Unit tests only |
| `make lint` | Lint with ruff |
| `make format` | Auto-format with ruff |
| `make typecheck` | Type check with mypy |
| `make ingest` | Ingest raw docs into vector store |
| `make evaluate` | Evaluate classifier against fixtures |

---

## Adding a New Pipeline

1. Add a new value to `Intent` in `src/models/schemas.py`
2. Create `src/pipelines/<new>/` with `handler.py`
3. Add a new match arm in `src/classifier/router.py`
4. Add a prompt template in `src/prompts/`
5. Update `data/intents/intent_labels.json` with examples
