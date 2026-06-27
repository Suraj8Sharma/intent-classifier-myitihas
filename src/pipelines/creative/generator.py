from src.models.llm import completion
from src.prompts.system_prompts import CREATIVE_PROMPT_TEMPLATE, CREATIVE_SYSTEM
from src.utils.config import get_settings, load_yaml_config

_cfg = load_yaml_config()
_creative_cfg = _cfg.get("pipelines", {}).get("creative", {})


class CreativeGenerator:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._temperature: float = _creative_cfg.get("temperature", 0.9)
        self._max_tokens: int = _creative_cfg.get("max_tokens", 1024)

    async def generate(self, query: str) -> str:
        prompt = CREATIVE_PROMPT_TEMPLATE.format(query=query)
        return await completion(
            model=self._settings.creative_model,
            system=CREATIVE_SYSTEM,
            prompt=prompt,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
