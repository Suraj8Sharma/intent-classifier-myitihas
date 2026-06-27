from unittest.mock import AsyncMock, patch

import pytest

from src.pipelines.creative.generator import CreativeGenerator
from src.pipelines.fallback.handler import FallbackHandler
from src.pipelines.rag.retriever import RAGRetriever


@pytest.mark.asyncio
async def test_rag_returns_answer_when_chunks_found() -> None:
    retriever = RAGRetriever()
    with (
        patch("src.pipelines.rag.retriever.embed_one", return_value=[0.1] * 1536),
        patch(
            "src.pipelines.rag.retriever.vector_query",
            return_value=["Refunds are processed in 5 days."],
        ),
        patch(
            "src.pipelines.rag.retriever.completion",
            new=AsyncMock(return_value="Refunds take 5 days."),
        ),
    ):
        answer = await retriever.answer("How long do refunds take?")
    assert "5 days" in answer


@pytest.mark.asyncio
async def test_rag_returns_no_info_when_no_chunks() -> None:
    retriever = RAGRetriever()
    with (
        patch("src.pipelines.rag.retriever.embed_one", return_value=[0.0] * 1536),
        patch("src.pipelines.rag.retriever.vector_query", return_value=[]),
    ):
        answer = await retriever.answer("Unknown topic")
    assert "don't have enough" in answer


@pytest.mark.asyncio
async def test_creative_generator_calls_llm() -> None:
    generator = CreativeGenerator()
    with patch(
        "src.pipelines.creative.generator.completion",
        new=AsyncMock(return_value="Once upon a time..."),
    ):
        result = await generator.generate("Write a story")
    assert result == "Once upon a time..."


@pytest.mark.asyncio
async def test_fallback_handler_returns_default_message() -> None:
    handler = FallbackHandler()
    result = await handler.handle("???")
    assert isinstance(result, str)
    assert len(result) > 0
