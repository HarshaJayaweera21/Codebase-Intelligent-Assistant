from typing import Literal

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


class RepositoryCreateResponse(BaseModel):
    repository_id: str
    chat_id: str
    repository_name: str
    repository_owner: str
    repository_url: str
    local_path: str
    status: Literal["scanned"]
    scan_summary: RepositoryScanSummary

