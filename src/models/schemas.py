from enum import Enum

from pydantic import BaseModel, Field


class Intent(str, Enum):
    RAG = "rag"
    CREATIVE = "creative"
    FALLBACK = "fallback"


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
