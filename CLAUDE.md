# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
make dev-install      # install all dependencies (runtime + dev/test)
make run              # start API server at http://localhost:8000 (hot-reload)
make test             # full test suite with coverage
make test-unit        # unit tests only (fast, mocked LLM)
make test-integration # API-level tests
make lint             # ruff check
make format           # ruff format
make typecheck        # mypy

# Data pipeline (run in order on first setup)
make ingest           # raw docs → processed chunks
make embed            # processed chunks → ChromaDB vector store
make evaluate         # offline classifier accuracy evaluation

# Run a single test file
pytest tests/unit/test_classifier.py -v
```

**Prerequisites before `make run`:** Ollama must be running locally (`ollama serve`) with the required models pulled:
```bash
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

## Environment

Copy `.env` and fill in any overrides (all defaults work for local development):
- `OLLAMA_BASE_URL` — defaults to `http://localhost:11434/v1`
- `CLASSIFIER_MODEL` / `RAG_MODEL` / `CREATIVE_MODEL` — default `qwen3:8b`
- `EMBEDDING_MODEL` — defaults to `nomic-embed-text`
- `API_KEY` — if set, all `/api/v1/*` routes require `X-API-Key` header; leave blank to disable auth (tests always bypass auth via `conftest.py`)

## Architecture

### Request Flow

```
POST /api/v1/classify
  → Auth middleware (verify_api_key)
  → IntentRouter.route()
  → LangGraph compiled_graph.ainvoke()
      → classify_intent_node   (instructor + Ollama → ClassifiedIntent)
      → _route_by_intent()     (conditional edge)
          ├─ confidence < 0.65 or intent = fallback → fallback_pipeline_node
          ├─ intent = "rag"     → rag_pipeline_node
          └─ intent = "creative" → creative_pipeline_node
  → PipelineResponse (JSON)
```

The evaluation UI is served at `http://localhost:8000/ui` from `src/api/static/index.html`.

### Key files and their roles

| File | Role |
|---|---|
| `src/models/schemas.py` | **Source of truth** — `Intent` enum, `GraphState`, all request/response Pydantic models |
| `src/models/llm.py` | **Single LLM provider boundary** — wraps Ollama OpenAI-compatible API; `get_structured_client()` returns an instructor-patched client |
| `src/classifier/router.py` | **LangGraph graph definition** — defines all four nodes, `CONFIDENCE_THRESHOLD = 0.65`, compiles the graph; also contains `IntentRouter` async façade |
| `src/classifier/intent_classifier.py` | Legacy classifier used for direct calls (not the LangGraph path) |
| `src/classifier/fallback.py` | `should_fallback()` helper — uses YAML threshold (0.75, different from graph threshold) |
| `src/utils/config.py` | `Settings` (pydantic-settings, reads `.env`) + `load_yaml_config()` (deep-merges YAML configs) |
| `src/prompts/` | All prompts as `.txt` files loaded at startup; prompt changes are file edits, not code changes |
| `src/pipelines/rag/vector_store.py` | **Swappable seam** — wraps ChromaDB; replace only this file to switch vector store |

### Two confidence thresholds

There are two separate confidence thresholds in play:
- **0.65** — hardcoded in `src/classifier/router.py::CONFIDENCE_THRESHOLD`, enforced by the LangGraph conditional edge (primary safety net)
- **0.75** — from `configs/default.yaml classifier.confidence_threshold`, used only by `src/classifier/fallback.py::should_fallback()` (legacy helper)

### Config layering

`configs/default.yaml` (base with `${VAR}` placeholders) is deep-merged with `configs/development.yaml` or `configs/production.yaml`. Secrets live only in `.env`, never in YAML.

## Adding a New Pipeline

1. Add the new value to `Intent` enum in `src/models/schemas.py`
2. Add a `case` arm in `src/classifier/router.py::_route_by_intent()` and wire a new node
3. Create `src/pipelines/<intent>/` with a single async entrypoint
4. Add `src/prompts/<intent>_prompt.txt` and register it in `src/prompts/system_prompts.py`
5. Update `data/intents/intent_labels.json` and `src/prompts/classifier_prompt.txt`
6. Add a `pipelines.<intent>` block in `configs/default.yaml`
7. Add tests in `tests/unit/test_pipelines.py` and labeled examples in `tests/fixtures/sample_queries.json`

## Testing Notes

- `asyncio_mode = "auto"` is set in `pyproject.toml` — no `@pytest.mark.asyncio` needed
- `tests/conftest.py` auto-bypasses API key auth for the full test session via `app.dependency_overrides`
- Unit tests mock `src.classifier.intent_classifier.completion` directly; integration tests hit the full FastAPI app
