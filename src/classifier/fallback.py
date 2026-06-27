from src.models.schemas import ClassifiedIntent, Intent
from src.utils.config import load_yaml_config

_cfg = load_yaml_config()
_THRESHOLD: float = _cfg.get("classifier", {}).get("confidence_threshold", 0.75)


def should_fallback(classified: ClassifiedIntent) -> bool:
    """Return True when the classifier result is too uncertain to trust."""
    if classified.intent == Intent.FALLBACK:
        return True
    return classified.confidence < _THRESHOLD
