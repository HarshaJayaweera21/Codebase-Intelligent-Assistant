from pathlib import Path
import re
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app.rag.chunking_router import chunk_repository_files
from app.rag.document_processor import create_langchain_documents
from app.services.repository_scanner import scan_repository
from app.vectorstores.pinecone_vector_store import PineconeVectorStore


router = APIRouter(
    prefix="/development/pinecone",
    tags=["Development - Pinecone"],
)
REPOSITORY_ID_PATTERN = re.compile(r"^repo_[a-f0-9]{8}$")
REPOSITORY_STORAGE_PATH = Path("storage/repositories")


class RepositoryIndexRequest(BaseModel):
    repository_id: str
    replace_namespace: bool = True


class RepositoryIndexResponse(BaseModel):
    repository_id: str
    namespace: str
    indexed_documents: int


class RepositorySearchRequest(BaseModel):
    repository_id: str
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    metadata_filter: dict[str, Any] | None = None


class SearchResultResponse(BaseModel):
    vector_id: str
    score: float
    vector_score: float
    lexical_score: float
    page_content: str
    metadata: dict[str, Any]


class RepositorySearchResponse(BaseModel):
    repository_id: str
    namespace: str
    results: list[SearchResultResponse]


def get_vector_store(request: Request) -> PineconeVectorStore:
    vector_store = getattr(request.app.state, "vector_store", None)
    if vector_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Pinecone is disabled. Set PINECONE_ENABLED=true and add "
                "PINECONE_API_KEY in backend/.env, then restart FastAPI."
            ),
        )
    return vector_store


VectorStoreDependency = Annotated[PineconeVectorStore, Depends(get_vector_store)]


@router.get("/status")
async def pinecone_status(request: Request) -> dict[str, Any]:
    settings = request.app.state.settings
    return {
        "enabled": settings.pinecone_enabled,
        "connected": getattr(request.app.state, "vector_store", None) is not None,
        "index_name": settings.pinecone_index_name,
        "dimension": settings.pinecone_index_dimension,
        "metric": settings.pinecone_metric,
        "cloud": settings.pinecone_cloud,
        "region": settings.pinecone_region,
    }


@router.post("/index", response_model=RepositoryIndexResponse)
async def index_repository(
    payload: RepositoryIndexRequest,
    vector_store: VectorStoreDependency,
) -> RepositoryIndexResponse:
    repository_path = _resolve_repository_path(payload.repository_id)

    def scan_chunk_and_index() -> int:
        scan_result = scan_repository(repository_path)
        chunks = chunk_repository_files(payload.repository_id, scan_result.files)
        documents = create_langchain_documents(chunks)
        return vector_store.index_documents(
            payload.repository_id,
            documents,
            replace_namespace=payload.replace_namespace,
        )

    indexed_documents = await run_in_threadpool(scan_chunk_and_index)
    return RepositoryIndexResponse(
        repository_id=payload.repository_id,
        namespace=payload.repository_id,
        indexed_documents=indexed_documents,
    )


@router.post("/search", response_model=RepositorySearchResponse)
async def search_repository(
    payload: RepositorySearchRequest,
    vector_store: VectorStoreDependency,
) -> RepositorySearchResponse:
    _resolve_repository_path(payload.repository_id)
    results = await run_in_threadpool(
        vector_store.search,
        payload.repository_id,
        payload.query,
        top_k=payload.top_k,
        metadata_filter=payload.metadata_filter,
    )
    return RepositorySearchResponse(
        repository_id=payload.repository_id,
        namespace=payload.repository_id,
        results=[
            SearchResultResponse(
                vector_id=result.vector_id,
                score=result.score,
                vector_score=result.vector_score,
                lexical_score=result.lexical_score,
                page_content=result.document.page_content,
                metadata=result.document.metadata,
            )
            for result in results
        ],
    )


@router.delete("/repositories/{repository_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_repository_vectors(
    repository_id: str,
    vector_store: VectorStoreDependency,
) -> None:
    _validate_repository_id(repository_id)
    await run_in_threadpool(vector_store.delete_repository, repository_id)


def _resolve_repository_path(repository_id: str) -> Path:
    _validate_repository_id(repository_id)
    repository_path = REPOSITORY_STORAGE_PATH / repository_id
    if not repository_path.is_dir():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Repository was not found in local storage.",
        )
    return repository_path


def _validate_repository_id(repository_id: str) -> None:
    if REPOSITORY_ID_PATTERN.fullmatch(repository_id) is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid repository_id.",
        )
