from src.models.llm import completion
from src.pipelines.rag.embeddings import embed_one
from src.pipelines.rag.vector_store import query as vector_query
from src.prompts.system_prompts import RAG_PROMPT_TEMPLATE, RAG_SYSTEM
from src.utils.config import get_settings, load_yaml_config
from src.utils.logger import get_logger

log = get_logger(__name__)

_cfg = load_yaml_config()
_TOP_K: int = _cfg.get("pipelines", {}).get("rag", {}).get("top_k", 5)


class RAGRetriever:
    def __init__(self) -> None:
        self._settings = get_settings()

    async def answer(self, query: str) -> str:
        query_embedding = embed_one(query)
        chunks = vector_query(query_embedding, top_k=_TOP_K)

        if not chunks:
            log.warning("rag_no_chunks_found", query=query)
            return "I don't have enough information to answer that."

        context = "\n\n---\n\n".join(chunks)
        prompt = RAG_PROMPT_TEMPLATE.format(context=context, query=query)

        return await completion(
            model=self._settings.rag_model,
            system=RAG_SYSTEM,
            prompt=prompt,
            max_tokens=1024,
        )
