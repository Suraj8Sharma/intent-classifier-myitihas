# Architecture

This document is the **map of the system**: what the Intent Classifier is, how the
code is organized, what belongs where, and how to extend it. For the runtime
*request flow* see [`docs/architecture.md`](docs/architecture.md); for setup and
day-to-day commands see [`README.md`](README.md).

---

## 1. Overview

The **Intent Classifier** is a routing layer for LLM applications. It takes a
natural-language query, decides *what the user is actually trying to do*, and
dispatches the query to the pipeline best suited to handle it.

Today it routes to three pipelines:

| Intent     | Pipeline           | Purpose                                                     |
| ---------- | ------------------ | ---------------------------------------------------------- |
| `rag`      | `RAGRetriever`     | Factual questions answered from a knowledge base (retrieval + synthesis). |
| `creative` | `CreativeGenerator`| Open-ended generation of original content.                 |
| `fallback` | `FallbackHandler`  | Ambiguous, off-topic, harmful, or low-confidence queries.  |

The classifier itself is a single, deterministic LLM call (`temperature=0.0`,
strict JSON output). A **confidence threshold** acts as a safety net: any
classification below the configured threshold is forced into the fallback
pipeline regardless of the predicted intent.

```
POST /api/v1/classify  →  IntentRouter  →  IntentClassifier  →  {rag | creative | fallback}  →  PipelineResponse
```

---

## 2. Directory Structure

```
intent-classifier-myitihas/
├── ARCHITECTURE.md              # ← this file (system map)
├── README.md                   # quick start + commands
├── Makefile                    # install / run / test / lint / data tasks
├── pyproject.toml              # project metadata + tool config (ruff, mypy, pytest)
├── requirements.txt            # runtime dependencies
├── requirements-dev.txt        # dev/test dependencies
├── .env                        # local secrets (gitignored)
│
├── configs/                    # declarative, environment-aware config (YAML)
│   ├── default.yaml            # base config; ${VARS} resolved from .env
│   ├── development.yaml        # dev overrides (deep-merged onto default)
│   └── production.yaml         # prod overrides
│
├── data/                       # data lifecycle (mostly gitignored)
│   ├── raw/                    # source documents, untouched
│   ├── processed/              # cleaned / chunked documents
│   ├── embeddings/             # persisted vector store (Chroma)
│   └── intents/
│       └── intent_labels.json  # canonical intent taxonomy: name, description, examples
│
├── docs/
│   └── architecture.md         # runtime request-flow diagram + design decisions
│
├── notebooks/                  # exploratory analysis & evaluation (not in prod path)
│   ├── 01_exploration.ipynb
│   ├── 02_embedding_analysis.ipynb
│   └── 03_classifier_evaluation.ipynb
│
├── scripts/                    # one-off operational entrypoints (run via `make`)
│   ├── ingest_data.py          # raw → processed
│   ├── generate_embeddings.py  # processed → embeddings/vector store
│   └── evaluate_classifier.py  # offline accuracy evaluation
│
├── src/                        # the application (importable package)
│   ├── api/                    # HTTP transport layer (FastAPI)
│   │   ├── main.py             # app factory, router wiring, startup
│   │   ├── middleware/
│   │   │   └── auth.py         # API-key auth
│   │   └── routes/
│   │       ├── classify.py     # POST /api/v1/classify
│   │       └── health.py       # GET  /health
│   │
│   ├── classifier/             # the routing brain
│   │   ├── intent_classifier.py# query → ClassifiedIntent (LLM call + JSON parse)
│   │   ├── router.py           # ClassifiedIntent → pipeline dispatch
│   │   └── fallback.py         # confidence-threshold logic (should_fallback)
│   │
│   ├── pipelines/              # one self-contained module per intent
│   │   ├── rag/
│   │   │   ├── retriever.py     # orchestrates retrieval + synthesis
│   │   │   ├── embeddings.py    # query/document embedding
│   │   │   └── vector_store.py  # Chroma wrapper (the swappable seam)
│   │   ├── creative/
│   │   │   └── generator.py
│   │   └── fallback/
│   │       └── handler.py
│   │
│   ├── prompts/                # all prompt text, versioned as files
│   │   ├── classifier_prompt.txt
│   │   ├── rag_prompt.txt
│   │   ├── creative_prompt.txt
│   │   └── system_prompts.py   # loads .txt templates + system strings
│   │
│   ├── models/                 # data contracts + model access
│   │   ├── schemas.py          # Pydantic models + Intent enum (source of truth)
│   │   └── llm.py              # thin LLM client wrapper (provider boundary)
│   │
│   └── utils/                  # cross-cutting helpers
│       ├── config.py           # Settings (.env) + YAML loader/deep-merge
│       ├── logger.py           # structured logging
│       └── helpers.py
│
└── tests/
    ├── unit/                   # test_classifier / test_router / test_pipelines
    ├── integration/            # test_api (full request lifecycle)
    └── fixtures/
        └── sample_queries.json
```

---

## 3. What Belongs in Each Folder

### `src/` — the application
The only code that ships. It is an importable package (`src.*`) with one job per
subpackage. The dependency direction flows **inward**: `api → classifier →
pipelines → models/utils`. Nothing in `models/` or `utils/` imports upward.

- **`src/api/`** — Transport only. Validates requests, enforces auth, serializes
  responses. Contains no business logic; it calls into `classifier/`.
- **`src/classifier/`** — The decision layer. `intent_classifier.py` produces a
  `ClassifiedIntent`; `router.py` maps that intent to a pipeline; `fallback.py`
  decides when confidence is too low to trust.
- **`src/pipelines/`** — The "doers." Each intent has its own subpackage with a
  single public entrypoint (`.answer()`, `.generate()`, `.handle()`). Pipelines
  do not know about each other or about routing.
- **`src/models/`** — Contracts and external access. `schemas.py` is the **source
  of truth** for the `Intent` enum and all request/response shapes. `llm.py` is
  the single boundary to the LLM provider.
- **`src/prompts/`** — Every prompt the system uses, stored as `.txt` so prompt
  changes are content edits, not code changes. `system_prompts.py` loads them.
- **`src/utils/`** — Config loading, logging, and small shared helpers.

### `prompts/` (inside `src/`)
Prompts are treated as **versioned assets, not string literals**. They live in
`.txt` files loaded at startup. This keeps prompt engineering out of Python
diffs, makes A/B testing and review straightforward, and lets non-code-owners
iterate on wording safely.

### `configs/`
Declarative, layered configuration. `default.yaml` holds the baseline with
`${ENV_VAR}` placeholders; `development.yaml` / `production.yaml` are deep-merged
on top (`utils/config.py::load_yaml_config`). Secrets never live here — they come
from `.env` via the `Settings` object.

### `data/`
The data lifecycle, staged by directory: `raw/` (source) → `processed/`
(cleaned/chunked) → `embeddings/` (vector store). `data/intents/intent_labels.json`
is the human-readable **intent taxonomy** used for documentation, evaluation, and
prompt grounding. Most of `data/` is gitignored.

### `scripts/`
Operational entrypoints run on demand (via `make ingest`, `make embed`,
`make evaluate`), not part of the request path. Keep orchestration here; keep
reusable logic in `src/`.

### `notebooks/`
Exploration and evaluation. Never imported by `src/`. Anything that earns a
permanent place graduates into `src/` or `scripts/`.

### `tests/`
- `tests/unit/` — fast, isolated tests of classifier, router, and pipelines.
- `tests/integration/` — full API request/response lifecycle.
- `tests/fixtures/` — shared test data such as labeled `sample_queries.json`.

---

## 4. How to Add a New Intent Category

Adding an intent is a small, mechanical change because the seams are explicit.
Example: adding a `summarize` intent.

1. **Declare the contract.** Add the value to the `Intent` enum in
   `src/models/schemas.py`:
   ```python
   class Intent(StrEnum):
       RAG = "rag"
       CREATIVE = "creative"
       SUMMARIZE = "summarize"   # new
       FALLBACK = "fallback"
   ```

2. **Teach the classifier.** Add the intent (with a description and a couple of
   examples) to `src/prompts/classifier_prompt.txt` so the LLM can predict it,
   and mirror it in `data/intents/intent_labels.json` to keep the taxonomy in
   sync.

3. **Build the pipeline.** Create `src/pipelines/summarize/` with `__init__.py`
   and a module exposing a single async entrypoint, e.g.
   `Summarizer.summarize(query) -> str`. Add its prompt as
   `src/prompts/summarize_prompt.txt` and wire it into `system_prompts.py`.

4. **Route to it.** In `src/classifier/router.py`, construct the pipeline in
   `__init__` and add a `case` arm:
   ```python
   case Intent.SUMMARIZE:
       answer = await self._summarizer.summarize(query)
   ```

5. **Configure it.** Add a `pipelines.summarize` block to `configs/default.yaml`
   (model, params) and any new env vars to `Settings` in `src/utils/config.py`.

6. **Test it.** Add unit tests in `tests/unit/test_pipelines.py` and labeled
   examples in `tests/fixtures/sample_queries.json`; extend
   `tests/unit/test_router.py` to assert the new routing.

7. **Verify.** `make test && make lint && make typecheck`, then evaluate
   classification accuracy with `make evaluate`.

> Note the fallback safety net is automatic: any prediction below
> `classifier.confidence_threshold` is rerouted to `fallback` regardless of the
> predicted intent, so a new intent can't silently mis-route low-confidence
> queries.

---

## 5. Swapping the Model / Router

The design isolates the two things most likely to change behind narrow seams:

- **Change the LLM provider/model** → edit `src/models/llm.py` (the only file
  that talks to the provider SDK) and/or the `*_model` fields in
  `src/utils/config.py`. Nothing else imports the SDK.
- **Change retrieval backend** (Chroma → Pinecone/Qdrant) → edit
  `src/pipelines/rag/vector_store.py` only.
- **Change the classification strategy** (LLM-based → semantic/vector-based
  router) → provide a new implementation that returns the same
  `ClassifiedIntent` contract, and have `router.py` use it. Because the router
  depends on the *contract* in `schemas.py`, not on how the intent was derived,
  swapping the classifier requires no change to pipelines or the API.

---

## 6. Keeping the Docs in Sync

Three docs, three jobs — don't duplicate:

| File                   | Owns                                                     |
| ---------------------- | ------------------------------------------------------- |
| `README.md`            | Setup, environment, run/test commands (the "how to run").|
| `ARCHITECTURE.md`      | Structure, responsibilities, extension points (this file).|
| `docs/architecture.md` | Runtime request flow + specific design decisions.        |

When you add an intent or move a seam, update the `Intent` enum first (the source
of truth), then this file's tree/extension steps, then the request-flow diagram.
