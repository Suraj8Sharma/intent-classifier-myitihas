import json

from src.models.llm import completion
from src.models.schemas import ClassifiedIntent, Intent
from src.prompts.system_prompts import CLASSIFIER_PROMPT_TEMPLATE, CLASSIFIER_SYSTEM
from src.utils.config import get_settings
from src.utils.logger import get_logger

log = get_logger(__name__)


class IntentClassifier:
    def __init__(self) -> None:
        self._settings = get_settings()

    async def classify(self, query: str) -> ClassifiedIntent:
        # The classifier prompt embeds a literal JSON example, so str.format()
        # would choke on its braces — substitute the placeholder directly.
        prompt = CLASSIFIER_PROMPT_TEMPLATE.replace("{query}", query)
        raw = await completion(
            model=self._settings.classifier_model,
            system=CLASSIFIER_SYSTEM,
            prompt=prompt,
            max_tokens=256,
            temperature=0.0,
        )

        try:
            data = json.loads(raw.strip())
            return ClassifiedIntent(
                intent=Intent(data["intent"]),
                confidence=float(data["confidence"]),
                reasoning=data.get("reasoning"),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            log.warning("classifier_parse_error", raw=raw, error=str(exc))
            return ClassifiedIntent(intent=Intent.FALLBACK, confidence=0.0)
