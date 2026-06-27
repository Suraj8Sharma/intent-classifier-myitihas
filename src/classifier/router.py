from src.classifier.fallback import should_fallback
from src.classifier.intent_classifier import IntentClassifier
from src.models.schemas import ClassifiedIntent, Intent, PipelineResponse
from src.pipelines.creative.generator import CreativeGenerator
from src.pipelines.fallback.handler import FallbackHandler
from src.pipelines.rag.retriever import RAGRetriever
from src.utils.logger import get_logger

log = get_logger(__name__)


class IntentRouter:
    def __init__(self) -> None:
        self._classifier = IntentClassifier()
        self._rag = RAGRetriever()
        self._creative = CreativeGenerator()
        self._fallback = FallbackHandler()

    async def route(self, query: str, session_id: str | None = None) -> PipelineResponse:
        classified: ClassifiedIntent = await self._classifier.classify(query)
        log.info("intent_classified", intent=classified.intent, confidence=classified.confidence)

        if should_fallback(classified):
            classified.intent = Intent.FALLBACK

        match classified.intent:
            case Intent.RAG:
                answer = await self._rag.answer(query)
            case Intent.CREATIVE:
                answer = await self._creative.generate(query)
            case _:
                answer = await self._fallback.handle(query)

        return PipelineResponse(
            query=query,
            intent=classified.intent,
            confidence=classified.confidence,
            answer=answer,
            session_id=session_id,
        )
