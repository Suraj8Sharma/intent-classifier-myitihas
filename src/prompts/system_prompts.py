from pathlib import Path

_PROMPT_DIR = Path(__file__).parent


def _load(filename: str) -> str:
    return (_PROMPT_DIR / filename).read_text(encoding="utf-8")


CLASSIFIER_SYSTEM = "You are a precise intent classification engine. Output only valid JSON."
RAG_SYSTEM = "You are a knowledgeable assistant. Answer only from provided context."
CREATIVE_SYSTEM = "You are a creative writing assistant. Be original and engaging."

CLASSIFIER_PROMPT_TEMPLATE = _load("classifier_prompt.txt")
RAG_PROMPT_TEMPLATE = _load("rag_prompt.txt")
CREATIVE_PROMPT_TEMPLATE = _load("creative_prompt.txt")
