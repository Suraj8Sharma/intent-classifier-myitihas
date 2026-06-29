from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class Intent(str, Enum):
    RAG = "rag"
    CREATIVE = "creative"
    FALLBACK = "fallback"


class GraphState(TypedDict):
    """Shared state threaded through every LangGraph node in the pipeline."""

    query: str
    intent: Optional[str]
    confidence: Optional[float]
    reasoning: Optional[str]
    final_response: Optional[str]


class ClassifyRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096)
    session_id: str | None = None


class ClassifiedIntent(BaseModel):
    intent: Intent
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str | None = None


class PipelineResponse(BaseModel):
    query: str
    intent: Intent
    confidence: float
    answer: str
    session_id: str | None = None
