import chromadb
from chromadb import Collection

from src.utils.config import get_settings

_collection: Collection | None = None


def get_collection(name: str | None = None) -> Collection:
    global _collection
    if _collection is None:
        settings = get_settings()
        client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        collection_name = name or "intent_docs"
        _collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def upsert_documents(
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict] | None = None,
) -> None:
    col = get_collection()
    col.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas or [{} for _ in ids],
    )


def query(embedding: list[float], top_k: int = 5) -> list[str]:
    col = get_collection()
    results = col.query(query_embeddings=[embedding], n_results=top_k)
    docs: list[str] = results.get("documents", [[]])[0]
    return docs
