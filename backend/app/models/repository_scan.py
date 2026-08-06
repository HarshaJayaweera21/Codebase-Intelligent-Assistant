from pydantic import BaseModel


class LanguageSummary(BaseModel):
    language: str
    file_count: int


class RepositoryScanSummary(BaseModel):
    total_files: int
    supported_files: int
    ignored_files: int
    languages: dict[str, int]