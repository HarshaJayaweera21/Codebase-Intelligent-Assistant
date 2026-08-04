from fastapi import FastAPI

from app.api.repository_routes import router as repository_router


app = FastAPI(
    title="Codebase Intelligence Assistant API",
    description="Backend API for analyzing and chatting with GitHub repositories.",
    version="0.1.0",
)


app.include_router(
    repository_router,
    prefix="/api",
)


@app.get("/health", tags=["System"])
async def health_check() -> dict[str, str]:
    return {"status": "ok"}