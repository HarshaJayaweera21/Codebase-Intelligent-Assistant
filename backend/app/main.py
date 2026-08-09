from fastapi import FastAPI

from app.api.embedding_routes import router as embedding_router
from app.api.pinecone_routes import router as pinecone_router
from app.api.repository_routes import router as repository_router
from app.core.lifespan import Lifespan, application_lifespan


def create_app(*, lifespan: Lifespan = application_lifespan) -> FastAPI:
    application = FastAPI(
        title="Codebase Intelligence Assistant API",
        description=(
            "Backend API for analyzing and chatting with GitHub repositories."
        ),
        version="0.1.0",
        lifespan=lifespan,
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

    @application.get("/health", tags=["System"])
    async def health_check() -> dict[str, str]:
        return {"status": "ok"}

    return application


app = create_app()
