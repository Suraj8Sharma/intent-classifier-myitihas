"""LLM client for the local stack.

Talks to a locally running Ollama daemon through its OpenAI-compatible API
(http://localhost:11434/v1). `instructor` patches the client so callers can
request structured, schema-validated outputs when they need them.
"""

import instructor
from openai import OpenAI

from src.utils.config import get_settings

_client: OpenAI | None = None
_structured_client: instructor.Instructor | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        settings = get_settings()
        _client = OpenAI(
            base_url=settings.ollama_base_url,
            api_key=settings.ollama_api_key,
        )
    return _client


def get_structured_client() -> instructor.Instructor:
    """OpenAI client patched by Instructor for `response_model=` structured calls."""
    global _structured_client
    if _structured_client is None:
        _structured_client = instructor.from_openai(get_client(), mode=instructor.Mode.JSON)
    return _structured_client


async def completion(
    *,
    model: str,
    system: str,
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> str:
    client = get_client()
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content or ""
