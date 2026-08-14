from fastapi import (
    APIRouter,
    HTTPException,
    Request,
    status,
)
from fastapi.concurrency import run_in_threadpool
from pathlib import Path
import re

from app.models.repository import (
    RepositoryCreateRequest,
    RepositoryCreateResponse,
    RepositoryDeleteResponse,
    RepositoryProcessingStatus,
    RepositoryStatusResponse,
)
from app.services.repository_lifecycle import (
    RepositoryLifecycleConflict,
    RepositoryLifecycleService,
)
from app.services.repository_processing import (
    RepositoryJobCoordinator,
    RepositoryProcessingLimits,
    RepositoryProcessingRecord,
    RepositoryProcessingStore,
    process_repository_pipeline,
    reindex_repository_pipeline,
    repository_details_from_record,
)
from app.services.repository_service import prepare_repository
from app.vectorstores.pinecone_vector_store import PineconeVectorStore
from app.models.repository_file import RepositoryFilePreview
from app.services.repository_scanner import create_file_previews
from app.rag.document_processor import create_langchain_documents
from app.services.repository_scanner import scan_repository
from app.rag.chunking_router import (
    chunk_repository_file,
    chunk_repository_files,
    supports_structural_chunking,
)


router = APIRouter(
    prefix="/repositories",
    tags=["Repositories"],
)
REPOSITORY_ID_PATTERN = re.compile(r"^repo_[a-f0-9]{8}$")


@router.get("", response_model=list[RepositoryStatusResponse])
async def list_repositories(request: Request) -> list[RepositoryStatusResponse]:
    records = await run_in_threadpool(_get_processing_store(request).list)
    return [_record_to_response(record) for record in records]


@router.post(
    "",
    response_model=RepositoryCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_repository_endpoint(
    payload: RepositoryCreateRequest,
    request: Request,
) -> RepositoryCreateResponse:
    vector_store = _get_vector_store(request)
    processing_store = _get_processing_store(request)
    existing = await run_in_threadpool(
        processing_store.find_by_url,
        str(payload.repository_url),
    )
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "This GitHub repository has already been submitted.",
                "repository_id": existing.repository_id,
                "chat_id": existing.chat_id,
                "status": existing.status.value,
            },
        )
    repository = prepare_repository(repository_url=str(payload.repository_url))
    record = await run_in_threadpool(processing_store.create, repository)
    coordinator = _get_job_coordinator(request)
    coordinator.submit(
        repository.repository_id,
        process_repository_pipeline,
        repository,
        processing_store,
        vector_store,
        _get_processing_limits(request),
        coordinator,
    )

    return RepositoryCreateResponse(
        repository_id=repository.repository_id,
        chat_id=repository.chat_id,
        repository_name=repository.repository_name,
        repository_owner=repository.repository_owner,
        repository_url=repository.repository_url,
        local_path=repository.local_path,
        status=record.status,
        status_url=f"/api/repositories/{repository.repository_id}/status",
    )


@router.get(
    "/{repository_id}/status",
    response_model=RepositoryStatusResponse,
)
async def get_repository_processing_status(
    repository_id: str,
    request: Request,
) -> RepositoryStatusResponse:
    record = await run_in_threadpool(
        _get_processing_store(request).get,
        repository_id,
    )
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Repository processing job was not found.",
        )
    return _record_to_response(record)


@router.post(
    "/{repository_id}/retry",
    response_model=RepositoryStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def retry_repository_processing(
    repository_id: str,
    request: Request,
) -> RepositoryStatusResponse:
    record = await _require_record(request, repository_id)
    if record.status is not RepositoryProcessingStatus.FAILED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Only failed repository processing jobs can be retried.",
        )
    vector_store = _get_vector_store(request)
    processing_store = _get_processing_store(request)
    queued = await run_in_threadpool(processing_store.reset, repository_id)
    coordinator = _get_job_coordinator(request)
    started = coordinator.submit(
        repository_id,
        process_repository_pipeline,
        repository_details_from_record(queued),
        processing_store,
        vector_store,
        _get_processing_limits(request),
        coordinator,
    )
    if not started:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Repository processing is already running.",
        )
    return _record_to_response(queued)


@router.post(
    "/{repository_id}/reindex",
    response_model=RepositoryStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def reindex_repository(
    repository_id: str,
    request: Request,
) -> RepositoryStatusResponse:
    record = await _require_record(request, repository_id)
    if record.status is not RepositoryProcessingStatus.READY:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Only ready repositories can be reindexed.",
        )
    vector_store = _get_vector_store(request)
    processing_store = _get_processing_store(request)
    queued = await run_in_threadpool(processing_store.reset, repository_id)
    coordinator = _get_job_coordinator(request)
    started = coordinator.submit(
        repository_id,
        reindex_repository_pipeline,
        repository_details_from_record(queued),
        processing_store,
        vector_store,
        _get_processing_limits(request),
        coordinator,
    )
    if not started:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Repository processing is already running.",
        )
    return _record_to_response(queued)


@router.delete(
    "/{repository_id}",
    response_model=RepositoryDeleteResponse,
)
async def delete_repository(
    repository_id: str,
    request: Request,
) -> RepositoryDeleteResponse:
    _validate_repository_id(repository_id)
    try:
        result = await run_in_threadpool(
            _get_lifecycle_service(request).delete_repository,
            repository_id,
        )
    except RepositoryLifecycleConflict as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Repository was not found.",
        )
    return RepositoryDeleteResponse(**result.__dict__)


def _get_vector_store(request: Request) -> PineconeVectorStore:
    vector_store = getattr(request.app.state, "vector_store", None)
    if vector_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Repository processing requires Pinecone. Set "
                "PINECONE_ENABLED=true and restart FastAPI."
            ),
        )
    return vector_store


def _get_processing_store(request: Request) -> RepositoryProcessingStore:
    return request.app.state.repository_processing_store


def _get_processing_limits(request: Request) -> RepositoryProcessingLimits:
    return request.app.state.repository_processing_limits


def _get_job_coordinator(request: Request) -> RepositoryJobCoordinator:
    return request.app.state.repository_job_coordinator


def _get_lifecycle_service(request: Request) -> RepositoryLifecycleService:
    service = getattr(request.app.state, "repository_lifecycle_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Repository lifecycle management requires Pinecone.",
        )
    return service


async def _require_record(
    request: Request,
    repository_id: str,
) -> RepositoryProcessingRecord:
    _validate_repository_id(repository_id)
    record = await run_in_threadpool(
        _get_processing_store(request).get,
        repository_id,
    )
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Repository was not found.",
        )
    return record


def _validate_repository_id(repository_id: str) -> None:
    if REPOSITORY_ID_PATTERN.fullmatch(repository_id) is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Invalid repository_id.",
        )


def _record_to_response(
    record: RepositoryProcessingRecord,
) -> RepositoryStatusResponse:
    return RepositoryStatusResponse(
        repository_id=record.repository_id,
        chat_id=record.chat_id,
        repository_name=record.repository_name,
        repository_owner=record.repository_owner,
        repository_url=record.repository_url,
        status=record.status,
        progress_percent=record.progress_percent,
        status_message=record.status_message,
        created_at=record.created_at,
        updated_at=record.updated_at,
        scan_summary=record.scan_summary,
        chunk_count=record.chunk_count,
        indexed_document_count=record.indexed_document_count,
        error=record.error,
    )

@router.get(
    "/preview-files",
    response_model=list[RepositoryFilePreview],
)
async def preview_repository_files(
    repository_path: str,
) -> list[RepositoryFilePreview]:
    path = Path(repository_path)

    previews = create_file_previews(path)

    return [
        RepositoryFilePreview(**preview)
        for preview in previews
    ]


@router.get("/preview-documents")
async def preview_langchain_documents(
    repository_id: str,
) -> list[dict]:
    repository_path = (
        Path("storage/repositories") / repository_id
    )

    scan_result = scan_repository(repository_path)

    chunks = chunk_repository_files(repository_id, scan_result.files)
    documents = create_langchain_documents(chunks)

    return [
        {
            "page_content_preview": document.page_content[:200],
            "metadata": document.metadata,
        }
        for document in documents
    ]


@router.get("/preview-code-chunks")
async def preview_code_chunks(
    repository_id: str,
    file_path: str,
) -> list[dict]:
    repository_path = (
        Path("storage/repositories")
        / repository_id
    )

    scan_result = scan_repository(
        repository_path
    )

    repository_file = next(
        (
            file
            for file in scan_result.files
            if file.relative_path == file_path
        ),
        None,
    )

    if repository_file is None:
        raise HTTPException(
            status_code=404,
            detail="Repository file not found.",
        )

    
    if not supports_structural_chunking(repository_file.language):
        raise HTTPException(
            status_code=400,
            detail=(
                "Structural chunking is not "
                "supported for this file type."
            ),
        )

    chunks = chunk_repository_file(
        repository_id=repository_id,
        repository_file=repository_file,
    )

    return [
        {
            "chunk_type": chunk.chunk_type,
            "symbol_name": chunk.symbol_name,

            "symbol_start_line": (
                chunk.symbol_start_line
            ),
            "symbol_end_line": (
                chunk.symbol_end_line
            ),

            "source_ranges": [
                {
                    "start_line": source_range.start_line,
                    "end_line": source_range.end_line,
                }
                for source_range in chunk.source_ranges
            ],

            "content_preview": chunk.content[:300],
        }
        for chunk in chunks
    ]


@router.get("/{repository_id}", response_model=RepositoryStatusResponse)
async def get_repository(
    repository_id: str,
    request: Request,
) -> RepositoryStatusResponse:
    return _record_to_response(
        await _require_record(request, repository_id)
    )
