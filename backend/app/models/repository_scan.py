from dataclasses import dataclass

from pydantic import BaseModel

from app.models.repository_file import RepositoryFile


class RepositoryScanSummary(BaseModel):
    total_files: int
    supported_files: int
    ignored_files: int
    languages: dict[str, int]


@dataclass(frozen=True)
class RepositoryScanResult:
    summary: RepositoryScanSummary
    files: list[RepositoryFile]