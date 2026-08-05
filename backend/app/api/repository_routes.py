from fastapi import (
    APIRouter,
    HTTPException,
    status,
)

from app.core.exceptions import (
    RepositoryCloneError,
    RepositoryScanError,
)
from app.models.repository import (
    RepositoryCreateRequest,
    RepositoryCreateResponse,
)
from app.services.repository_service import create_repository


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