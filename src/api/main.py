from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.routes import classify, health
from src.utils.config import get_settings
from src.utils.logger import configure_logging, get_logger

log = get_logger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    log.info(
        "startup",
        env=settings.env,
        model=settings.classifier_model,
        ollama_url=settings.ollama_base_url,
    )
    yield
    log.info("shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Intent Classifier API",
        description="Routes natural language queries to the correct backend pipeline.",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS — permissive for local cluster / browser-based evaluation UI.
    # Tighten allow_origins in production.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(classify.router, prefix="/api/v1")

    # Serve the evaluation UI at /ui  (index.html is the root page).
    # Mount after API routes so /api/v1/* is never shadowed.
    if _STATIC_DIR.exists():
        app.mount("/ui", StaticFiles(directory=str(_STATIC_DIR), html=True), name="ui")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
