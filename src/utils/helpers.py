import re
import time
from collections.abc import Callable
from typing import Any


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"[\s_-]+", "-", text)


def truncate(text: str, max_chars: int = 200) -> str:
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + "…"


def timer(fn: Callable) -> Callable:
    """Decorator that logs wall-clock time of any sync function."""
    from functools import wraps
    from src.utils.logger import get_logger

    log = get_logger(__name__)

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        log.info("timed", fn=fn.__name__, elapsed_ms=round((time.perf_counter() - start) * 1000, 2))
        return result

    return wrapper
