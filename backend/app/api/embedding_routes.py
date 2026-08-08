from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, Field

from app.embeddings.embedding_service import (
    EmbeddingSmokeTestResult,
    run_embedding_smoke_test,
)


router = APIRouter(
    prefix="/development/embeddings",
    tags=["Development"],
)


class EmbeddingTestRequest(BaseModel):
    query: str = "Where is repository authentication implemented?"
    documents: list[str] = Field(
        default_factory=lambda: [
            "Authentication is implemented in AuthService.",
            "The application configuration is stored in config.yaml.",
        ],
        min_length=2,
        max_length=8,
    )


def get_embedding_service(request: Request) -> Embeddings:
    return request.app.state.embedding_service


@router.post("/test", response_model=EmbeddingSmokeTestResult)
async def test_embeddings(
    payload: EmbeddingTestRequest,
    request: Request,
) -> EmbeddingSmokeTestResult:
    embeddings = get_embedding_service(request)
    expected_dimension = request.app.state.settings.embedding_dimension
    return await run_in_threadpool(
        run_embedding_smoke_test,
        embeddings,
        expected_dimension=expected_dimension,
        query=payload.query,
        documents=payload.documents,
    )
