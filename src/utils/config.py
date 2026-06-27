from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    anthropic_api_key: str = Field(..., alias="ANTHROPIC_API_KEY")
    classifier_model: str = "claude-sonnet-4-6"
    rag_model: str = "claude-sonnet-4-6"
    creative_model: str = "claude-sonnet-4-6"

    # Vector store
    vector_store_provider: str = "chroma"
    chroma_persist_dir: str = "./data/embeddings"

    # Embeddings
    embedding_model: str = "text-embedding-3-small"
    embedding_provider: str = "openai"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: str = ""

    # App
    env: str = "development"
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]


def load_yaml_config(env: str = "development") -> dict:
    base = Path("configs/default.yaml")
    override = Path(f"configs/{env}.yaml")
    cfg: dict = {}
    if base.exists():
        cfg = yaml.safe_load(base.read_text()) or {}
    if override.exists():
        import copy
        overrides = yaml.safe_load(override.read_text()) or {}
        cfg = _deep_merge(copy.deepcopy(cfg), overrides)
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_merge(base[k], v)
        else:
            base[k] = v
    return base
