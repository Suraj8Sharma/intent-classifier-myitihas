from unittest.mock import AsyncMock, patch

import pytest

from src.classifier.router import IntentRouter
from src.models.schemas import ClassifiedIntent, Intent


@pytest.fixture
def router() -> IntentRouter:
    return IntentRouter()


@pytest.mark.asyncio
async def test_router_routes_to_rag(router: IntentRouter) -> None:
    classified = ClassifiedIntent(intent=Intent.RAG, confidence=0.9)
    with (
        patch.object(router._classifier, "classify", new=AsyncMock(return_value=classified)),
        patch.object(
            router._rag, "answer", new=AsyncMock(return_value="The refund window is 30 days.")
        ),
    ):
        response = await router.route("What is the refund policy?")

    assert response.intent == Intent.RAG
    assert "30 days" in response.answer


@pytest.mark.asyncio
async def test_router_routes_to_fallback_on_low_confidence(router: IntentRouter) -> None:
    classified = ClassifiedIntent(intent=Intent.RAG, confidence=0.3)
    with (
        patch.object(router._classifier, "classify", new=AsyncMock(return_value=classified)),
        patch.object(router._fallback, "handle", new=AsyncMock(return_value="Could you rephrase?")),
    ):
        response = await router.route("?")

    assert response.intent == Intent.FALLBACK
