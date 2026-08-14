from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field, HttpUrl, field_validator
from app.models.repository_scan import RepositoryScanSummary

class RepositoryCreateRequest(BaseModel):
    repository_url: HttpUrl = Field(
        description="Public GitHub repository URL",
        examples=["https://github.com/tiangolo/fastapi"],
    )

    @field_validator("repository_url")
    @classmethod
    def validate_github_url(cls, url: HttpUrl) -> HttpUrl:
        host = url.host.lower() if url.host else ""

        if host not in {"github.com", "www.github.com"}:
            raise ValueError("Only GitHub repository URLs are supported")

        path_parts = [part for part in url.path.split("/") if part]

        if len(path_parts) != 2:
            raise ValueError(
                "URL must follow the format "
                "https://github.com/owner/repository"
            )

        return url


class RepositoryProcessingStatus(StrEnum):
    QUEUED = "queued"
    CLONING = "cloning"
    SCANNING = "scanning"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"


class RepositoryCreateResponse(BaseModel):
    repository_id: str
    chat_id: str
    repository_name: str
    repository_owner: str
    repository_url: str
    local_path: str
    status: RepositoryProcessingStatus
    status_url: str


class RepositoryStatusResponse(BaseModel):
    repository_id: str
    chat_id: str
    repository_name: str
    repository_owner: str
    repository_url: str
    status: RepositoryProcessingStatus
    progress_percent: int = Field(ge=0, le=100)
    status_message: str
    created_at: datetime
    updated_at: datetime
    scan_summary: RepositoryScanSummary | None = None
    chunk_count: int | None = None
    indexed_document_count: int | None = None
    error: str | None = None


class RepositoryDeleteResponse(BaseModel):
    repository_id: str
    chat_id: str
    deleted: bool = True

