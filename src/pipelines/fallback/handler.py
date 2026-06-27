from src.utils.config import load_yaml_config

_cfg = load_yaml_config()
_DEFAULT_MSG: str = (
    _cfg.get("pipelines", {})
    .get("fallback", {})
    .get("message", "I'm not sure how to help with that. Could you rephrase?")
)


class FallbackHandler:
    async def handle(self, query: str) -> str:  # noqa: ARG002
        return _DEFAULT_MSG
