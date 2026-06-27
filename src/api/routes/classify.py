from fastapi import APIRouter, Depends

from src.api.middleware.auth import verify_api_key
from src.classifier.router import IntentRouter
from src.models.schemas import ClassifyRequest, PipelineResponse

router = APIRouter(tags=["classify"], dependencies=[Depends(verify_api_key)])

_router = IntentRouter()


@router.post("/classify", response_model=PipelineResponse)
async def classify(request: ClassifyRequest) -> PipelineResponse:
    return await _router.route(query=request.query, session_id=request.session_id)
