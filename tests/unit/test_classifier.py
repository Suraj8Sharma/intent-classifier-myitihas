import json
from unittest.mock import AsyncMock, patch

import pytest

from src.classifier.intent_classifier import IntentClassifier
from src.models.schemas import Intent


@pytest.fixture
def classifier() -> IntentClassifier:
    return IntentClassifier()


@pytest.mark.asyncio
async def test_classify_rag_intent(classifier: IntentClassifier) -> None:
    mock_response = json.dumps({"intent": "rag", "confidence": 0.95, "reasoning": "Factual query"})
    with patch("src.classifier.intent_classifier.completion", new=AsyncMock(return_value=mock_response)):
        result = await classifier.classify("What is our refund policy?")
    assert result.intent == Intent.RAG
    assert result.confidence == 0.95


@pytest.mark.asyncio
async def test_classify_creative_intent(classifier: IntentClassifier) -> None:
    mock_response = json.dumps({"intent": "creative", "confidence": 0.92, "reasoning": "Creative request"})
    with patch("src.classifier.intent_classifier.completion", new=AsyncMock(return_value=mock_response)):
        result = await classifier.classify("Write a poem about the sea")
    assert result.intent == Intent.CREATIVE


@pytest.mark.asyncio
async def test_classify_falls_back_on_bad_json(classifier: IntentClassifier) -> None:
    with patch("src.classifier.intent_classifier.completion", new=AsyncMock(return_value="not json")):
        result = await classifier.classify("some query")
    assert result.intent == Intent.FALLBACK
    assert result.confidence == 0.0
