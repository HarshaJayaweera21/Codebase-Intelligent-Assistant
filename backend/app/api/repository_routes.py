from fastapi import (
    APIRouter,
    HTTPException,
    status,
)
from pathlib import Path

from app.core.exceptions import (
    RepositoryCloneError,
    RepositoryScanError,
)
from app.models.repository import (
    RepositoryCreateRequest,
    RepositoryCreateResponse,
)
from app.services.repository_service import create_repository
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


@router.post(
    "",
    response_model=RepositoryCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_repository_endpoint(
    request: RepositoryCreateRequest,
) -> RepositoryCreateResponse:
    try:
        repository = create_repository(
            repository_url=str(request.repository_url)
        )

    except RepositoryCloneError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=error.message,
        ) from error

    except RepositoryScanError as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error.message,
        ) from error

    return RepositoryCreateResponse(
        repository_id=repository.repository_id,
        chat_id=repository.chat_id,
        repository_name=repository.repository_name,
        repository_owner=repository.repository_owner,
        repository_url=repository.repository_url,
        local_path=repository.local_path,
        status="scanned",
        scan_summary=repository.scan_summary,
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
