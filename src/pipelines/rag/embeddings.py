from openai import OpenAI

from src.utils.config import get_settings

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        settings = get_settings()
        _client = OpenAI(
            base_url=settings.ollama_base_url,
            api_key=settings.ollama_api_key,
        )
    return _client


def embed(texts: list[str]) -> list[list[float]]:
    settings = get_settings()
    response = _get_client().embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )
    return [item.embedding for item in response.data]


def embed_one(text: str) -> list[float]:
    return embed([text])[0]
