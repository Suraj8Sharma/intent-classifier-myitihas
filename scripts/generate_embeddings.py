"""
Batch-generate and persist embeddings for all processed documents.
Usage: python scripts/generate_embeddings.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipelines.rag.embeddings import embed
from src.utils.logger import configure_logging, get_logger

configure_logging("INFO")
log = get_logger(__name__)

PROCESSED_DIR = Path("data/processed")
EMBEDDINGS_DIR = Path("data/embeddings")


def main() -> None:
    import json

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    files = list(PROCESSED_DIR.glob("**/*.txt"))

    if not files:
        log.warning("no_processed_files", dir=str(PROCESSED_DIR))
        return

    texts = [f.read_text(encoding="utf-8") for f in files]
    vectors = embed(texts)

    output = [{"file": f.name, "embedding": v} for f, v in zip(files, vectors, strict=True)]
    out_path = EMBEDDINGS_DIR / "embeddings.json"
    out_path.write_text(json.dumps(output, indent=2))
    log.info("embeddings_saved", count=len(output), path=str(out_path))


if __name__ == "__main__":
    main()
