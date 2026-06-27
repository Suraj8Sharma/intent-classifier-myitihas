"""LangGraph orchestration for the intent-classification routing pipeline.

Workflow
--------
  classify_intent_node  ─── conditional edge ──►  rag_pipeline_node
                                                   creative_pipeline_node
                                                   fallback_pipeline_node

The conditional edge forces the *fallback* node whenever the classifier
confidence falls below CONFIDENCE_THRESHOLD (0.65), regardless of the
predicted intent.  This is a tighter guard than the YAML config threshold
(0.75) used by the legacy should_fallback helper; the graph-level routing
is the primary safety net.
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, StateGraph

from src.models.llm import get_structured_client
from src.models.schemas import ClassifiedIntent, GraphState, Intent, PipelineResponse
from src.pipelines.creative.generator import CreativeGenerator
from src.pipelines.fallback.handler import FallbackHandler
from src.pipelines.rag.retriever import RAGRetriever
from src.prompts.system_prompts import CLASSIFIER_PROMPT_TEMPLATE, CLASSIFIER_SYSTEM
from src.utils.config import get_settings
from src.utils.logger import get_logger

log = get_logger(__name__)

# Queries with confidence below this are routed to fallback unconditionally.
CONFIDENCE_THRESHOLD: float = 0.65

# ---------------------------------------------------------------------------
# Pipeline singletons — one instance shared across all graph invocations
# ---------------------------------------------------------------------------
_rag = RAGRetriever()
_creative = CreativeGenerator()
_fallback = FallbackHandler()


# ---------------------------------------------------------------------------
# Node: classify intent with instructor + qwen3:8b
# ---------------------------------------------------------------------------


def classify_intent_node(state: GraphState) -> GraphState:
    """Classify the user query via instructor-enforced structured output.

    Uses the instructor-patched OpenAI client targeting the local Ollama
    endpoint (http://localhost:11434/v1).  Temperature is fixed at 0.0 for
    deterministic, reproducible classifications.  Instructor handles JSON
    schema validation and retries automatically.

    Node is intentionally synchronous: the underlying OpenAI SDK call is
    blocking, and for local single-user deployments this is acceptable.
    Wrap in asyncio.to_thread if running under concurrent load.
    """
    settings = get_settings()
    client = get_structured_client()

    # Build the user prompt from the classifier template (handles literal-brace
    # JSON example in the prompt without str.format KeyError risk).
    user_prompt = CLASSIFIER_PROMPT_TEMPLATE.replace("{query}", state["query"])

    # System message extended with multilingual scope so the model does not
    # default to English-only classification behaviour.
    system = (
        f"{CLASSIFIER_SYSTEM}\n"
        "You must classify queries written in any language including English, "
        "Hindi (Devanagari), Marathi, and other scripts.  "
        "Do NOT output <think> blocks, markdown fences, or any text other than "
        "the required JSON object."
    )

    classified: ClassifiedIntent = client.chat.completions.create(
        model=settings.classifier_model,
        response_model=ClassifiedIntent,
        temperature=0.0,
        max_retries=2,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
    )

    log.info(
        "intent_classified",
        intent=classified.intent.value,
        confidence=classified.confidence,
        reasoning=classified.reasoning,
    )

    return {
        **state,
        "intent": classified.intent.value,
        "confidence": classified.confidence,
        "reasoning": classified.reasoning,
    }


# ---------------------------------------------------------------------------
# Node: RAG pipeline  (simulates GPU 0 / GPU 2 vector retrieval)
# ---------------------------------------------------------------------------


async def rag_pipeline_node(state: GraphState) -> GraphState:
    """Factual retrieval-augmented generation via ChromaDB + qwen3:8b.

    Simulates GPU 0 / GPU 2 vector-store operations in the 4x GPU cluster.
    Delegates to RAGRetriever which embeds the query, fetches top-k context
    chunks from ChromaDB, and synthesises an answer with the LLM.
    """
    log.info("rag_pipeline_invoked", query=state["query"])
    answer = await _rag.answer(state["query"])
    return {**state, "final_response": answer}


# ---------------------------------------------------------------------------
# Node: creative pipeline  (simulates GPU 3 open-ended generation)
# ---------------------------------------------------------------------------


async def creative_pipeline_node(state: GraphState) -> GraphState:
    """Open-ended creative generation via qwen3:8b.

    Simulates GPU 3 storytelling workload in the 4x GPU cluster.  Delegates
    to CreativeGenerator which applies a higher temperature for creative
    diversity.
    """
    log.info("creative_pipeline_invoked", query=state["query"])
    answer = await _creative.generate(state["query"])
    return {**state, "final_response": answer}


# ---------------------------------------------------------------------------
# Node: fallback pipeline  (safe rejection / low-confidence catch-all)
# ---------------------------------------------------------------------------


async def fallback_pipeline_node(state: GraphState) -> GraphState:
    """Safe rejection for ambiguous, out-of-domain, or low-confidence queries.

    Invoked either because the classifier predicted 'fallback' directly or
    because the confidence score fell below CONFIDENCE_THRESHOLD.
    """
    log.info(
        "fallback_pipeline_invoked",
        query=state["query"],
        confidence=state.get("confidence"),
        intent=state.get("intent"),
    )
    answer = await _fallback.handle(state["query"])
    return {**state, "intent": Intent.FALLBACK.value, "final_response": answer}


# ---------------------------------------------------------------------------
# Conditional routing edge
# ---------------------------------------------------------------------------


def _route_by_intent(
    state: GraphState,
) -> Literal["rag", "creative", "fallback"]:
    """Return the next node name after the classify node.

    Forces 'fallback' when:
    - confidence is below CONFIDENCE_THRESHOLD (0.65), or
    - the classifier itself predicted 'fallback'.
    """
    confidence: float = state.get("confidence") or 0.0
    intent: str = state.get("intent") or Intent.FALLBACK.value

    if confidence < CONFIDENCE_THRESHOLD or intent == Intent.FALLBACK.value:
        return "fallback"

    # intent is either "rag" or "creative" at this point
    return intent  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Graph assembly & compilation
# ---------------------------------------------------------------------------


def _build_graph():
    workflow: StateGraph = StateGraph(GraphState)

    workflow.add_node("classify", classify_intent_node)
    workflow.add_node("rag", rag_pipeline_node)
    workflow.add_node("creative", creative_pipeline_node)
    workflow.add_node("fallback", fallback_pipeline_node)

    workflow.set_entry_point("classify")
    workflow.add_conditional_edges(
        "classify",
        _route_by_intent,
        {
            "rag": "rag",
            "creative": "creative",
            "fallback": "fallback",
        },
    )
    workflow.add_edge("rag", END)
    workflow.add_edge("creative", END)
    workflow.add_edge("fallback", END)

    return workflow.compile()


# Exported for direct use by src/api/routes/classify.py and tests.
compiled_graph = _build_graph()


# ---------------------------------------------------------------------------
# IntentRouter — thin async wrapper around compiled_graph
# ---------------------------------------------------------------------------


class IntentRouter:
    """Async façade over the compiled LangGraph.

    Holds a reference to the module-level `compiled_graph` as `self._graph`
    so that tests can patch `router._graph.ainvoke` without touching the
    module global directly.
    """

    def __init__(self) -> None:
        self._graph = compiled_graph

    async def route(self, query: str, session_id: str | None = None) -> PipelineResponse:
        initial: GraphState = {
            "query": query,
            "intent": None,
            "confidence": None,
            "reasoning": None,
            "final_response": None,
        }
        result: GraphState = await self._graph.ainvoke(initial)

        return PipelineResponse(
            query=query,
            intent=Intent(result.get("intent") or Intent.FALLBACK.value),
            confidence=result.get("confidence") or 0.0,
            answer=result.get("final_response") or "",
            session_id=session_id,
        )
