from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.models.schemas import Intent, PipelineResponse


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_classify_endpoint_returns_pipeline_response(client: TestClient) -> None:
    mock_response = PipelineResponse(
        query="What is the refund policy?",
        intent=Intent.RAG,
        confidence=0.95,
        answer="Refunds are processed within 30 days.",
    )
    with patch("src.api.routes.classify._router.route", new=AsyncMock(return_value=mock_response)):
        response = client.post("/api/v1/classify", json={"query": "What is the refund policy?"})

    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "rag"
    assert data["confidence"] == 0.95


def test_classify_rejects_empty_query(client: TestClient) -> None:
    response = client.post("/api/v1/classify", json={"query": ""})
    assert response.status_code == 422
