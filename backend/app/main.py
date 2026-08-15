from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.embedding_routes import router as embedding_router
from app.api.chat_routes import router as chat_router
from app.api.pinecone_routes import router as pinecone_router
from app.api.repository_routes import router as repository_router
from app.api.rag_routes import router as rag_router
from app.core.config import Settings, get_settings
from app.core.lifespan import Lifespan, application_lifespan


def create_app(
    *,
    lifespan: Lifespan = application_lifespan,
    settings: Settings | None = None,
) -> FastAPI:
    application_settings = settings or get_settings()
    application = FastAPI(
        title="Codebase Intelligence Assistant API",
        description=(
            "Backend API for analyzing and chatting with GitHub repositories."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    if application_settings.allowed_cors_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=application_settings.allowed_cors_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["*"],
        )

    application.include_router(
        repository_router,
        prefix="/api",
    )
    application.include_router(
        embedding_router,
        prefix="/api",
    )
    application.include_router(
        pinecone_router,
        prefix="/api",
    )
    application.include_router(
        rag_router,
        prefix="/api",
    )
    application.include_router(
        chat_router,
        prefix="/api",
    )

    @application.get("/health", tags=["System"])
    async def health_check() -> dict[str, str]:
        return {"status": "ok"}

    return application


app = create_app()
