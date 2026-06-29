"""Unit tests for the LangGraph-based IntentRouter.

The router's `route()` method delegates entirely to the compiled LangGraph
`compiled_graph.ainvoke()`.  Tests patch `router._graph.ainvoke` with a
pre-built GraphState so no LLM call or ChromaDB connection is needed.
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.classifier.router import CONFIDENCE_THRESHOLD, IntentRouter, _route_by_intent
from src.models.schemas import GraphState, Intent


@pytest.fixture
def router() -> IntentRouter:
    return IntentRouter()


# ---------------------------------------------------------------------------
# IntentRouter.route() — patching the compiled graph
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_router_routes_to_rag(router: IntentRouter) -> None:
    mock_state: GraphState = {
        "query": "What is the refund policy?",
        "intent": "rag",
        "confidence": 0.9,
        "reasoning": "Factual query about policy.",
        "final_response": "The refund window is 30 days.",
    }
    with patch.object(router._graph, "ainvoke", new=AsyncMock(return_value=mock_state)):
        response = await router.route("What is the refund policy?")

    assert response.intent == Intent.RAG
    assert "30 days" in response.answer
    assert response.confidence == 0.9


@pytest.mark.asyncio
async def test_router_routes_to_creative(router: IntentRouter) -> None:
    mock_state: GraphState = {
        "query": "Write a poem about the Maratha forts.",
        "intent": "creative",
        "confidence": 0.88,
        "reasoning": "Creative generation request.",
        "final_response": "Atop Sinhagad the winds howl free…",
    }
    with patch.object(router._graph, "ainvoke", new=AsyncMock(return_value=mock_state)):
        response = await router.route("Write a poem about the Maratha forts.")

    assert response.intent == Intent.CREATIVE
    assert "Sinhagad" in response.answer


@pytest.mark.asyncio
async def test_router_returns_fallback_when_graph_routes_there(router: IntentRouter) -> None:
    """Graph already decided fallback (low confidence handled inside the graph)."""
    mock_state: GraphState = {
        "query": "asdfghjkl",
        "intent": "fallback",
        "confidence": 0.1,
        "reasoning": "Unintelligible input.",
        "final_response": "I'm not sure how to help with that. Could you rephrase?",
    }
    with patch.object(router._graph, "ainvoke", new=AsyncMock(return_value=mock_state)):
        response = await router.route("asdfghjkl")

    assert response.intent == Intent.FALLBACK


@pytest.mark.asyncio
async def test_router_sets_session_id(router: IntentRouter) -> None:
    mock_state: GraphState = {
        "query": "Hello",
        "intent": "fallback",
        "confidence": 0.5,
        "reasoning": None,
        "final_response": "Could you rephrase?",
    }
    with patch.object(router._graph, "ainvoke", new=AsyncMock(return_value=mock_state)):
        response = await router.route("Hello", session_id="sess-123")

    assert response.session_id == "sess-123"


@pytest.mark.asyncio
async def test_router_handles_none_intent_gracefully(router: IntentRouter) -> None:
    """If the graph somehow returns None intent, route() should default to FALLBACK."""
    mock_state: GraphState = {
        "query": "edge case",
        "intent": None,
        "confidence": None,
        "reasoning": None,
        "final_response": None,
    }
    with patch.object(router._graph, "ainvoke", new=AsyncMock(return_value=mock_state)):
        response = await router.route("edge case")

    assert response.intent == Intent.FALLBACK
    assert response.confidence == 0.0
    assert response.answer == ""


# ---------------------------------------------------------------------------
# _route_by_intent — conditional edge logic unit tests
# ---------------------------------------------------------------------------


def _state(intent: str, confidence: float) -> GraphState:
    return {
        "query": "test",
        "intent": intent,
        "confidence": confidence,
        "reasoning": None,
        "final_response": None,
    }


def test_route_rag_above_threshold() -> None:
    assert _route_by_intent(_state("rag", CONFIDENCE_THRESHOLD + 0.01)) == "rag"


def test_route_creative_above_threshold() -> None:
    assert _route_by_intent(_state("creative", 0.9)) == "creative"


def test_route_forces_fallback_below_threshold() -> None:
    # Classified as 'rag' but confidence is too low → must be re-routed
    result = _route_by_intent(_state("rag", CONFIDENCE_THRESHOLD - 0.01))
    assert result == "fallback"


def test_route_forces_fallback_for_fallback_intent() -> None:
    # Even if confidence is high, explicit fallback stays fallback
    assert _route_by_intent(_state("fallback", 0.99)) == "fallback"


def test_route_exact_threshold_passes_through() -> None:
    # Boundary: the guard is `< CONFIDENCE_THRESHOLD` (strict), so exactly
    # at threshold the query is NOT forced to fallback.
    result = _route_by_intent(_state("rag", CONFIDENCE_THRESHOLD))
    assert result == "rag"
