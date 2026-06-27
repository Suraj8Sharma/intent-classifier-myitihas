"""
Ingest raw documents from data/raw/ into the vector store.
Usage: python scripts/ingest_data.py
"""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipelines.rag.embeddings import embed
from src.pipelines.rag.vector_store import upsert_documents
from src.utils.config import load_yaml_config
from src.utils.logger import configure_logging, get_logger

configure_logging("INFO")
log = get_logger(__name__)

_cfg = load_yaml_config()
_CHUNK_SIZE: int = _cfg.get("pipelines", {}).get("rag", {}).get("chunk_size", 512)
_CHUNK_OVERLAP: int = _cfg.get("pipelines", {}).get("rag", {}).get("chunk_overlap", 64)
RAW_DIR = Path("data/raw")


def chunk_text(text: str, size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap
    return chunks


def ingest_file(path: Path) -> None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    chunks = chunk_text(text)
    ids = [hashlib.md5(f"{path.name}:{i}".encode()).hexdigest() for i in range(len(chunks))]
    embeddings = embed(chunks)
    metadatas = [{"source": path.name, "chunk": i} for i in range(len(chunks))]
    upsert_documents(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
    log.info("ingested", file=path.name, chunks=len(chunks))


def main() -> None:
    files = list(RAW_DIR.glob("**/*.txt")) + list(RAW_DIR.glob("**/*.md"))
    if not files:
        log.warning("no_files_found", dir=str(RAW_DIR))
        return
    for f in files:
        ingest_file(f)
    log.info("ingest_complete", total_files=len(files))


if __name__ == "__main__":
    main()
