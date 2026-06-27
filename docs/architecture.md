# Architecture Overview

## Request Flow

```
User Query (HTTP POST /api/v1/classify)
        │
        ▼
  Auth Middleware  ── invalid key ──▶  401
        │
        ▼
  IntentRouter
        │
        ▼
  IntentClassifier  (LLM call → JSON)
        │
        ├── confidence < threshold ──▶ FallbackHandler
        │
        ├── intent = "rag"     ──▶ RAGRetriever
        │                              ├── embed_one(query)
        │                              ├── vector_store.query(embedding)
        │                              └── LLM synthesis
        │
        ├── intent = "creative" ──▶ CreativeGenerator
        │                              └── LLM generation
        │
        └── intent = "fallback" ──▶ FallbackHandler (static message)
```

## Key Design Decisions

- **Single LLM call for classification**: The classifier uses `temperature=0.0` and strict JSON output format to make routing deterministic.
- **Confidence threshold as safety net**: Even if the LLM returns a non-fallback intent, low confidence forces the fallback pipeline (`classifier.confidence_threshold` in YAML).
- **Pipeline isolation**: Each pipeline (`rag`, `creative`, `fallback`) is its own module. Adding a new pipeline requires only: a new `Intent` enum value, a new pipeline module, and a new match arm in `router.py`.
- **Prompt versioning**: All prompts live in `src/prompts/` as `.txt` files loaded at startup. Changing a prompt is a file edit, not a code change.
- **Vector store abstraction**: `vector_store.py` wraps ChromaDB. To switch to Pinecone or Qdrant, only this file changes.
