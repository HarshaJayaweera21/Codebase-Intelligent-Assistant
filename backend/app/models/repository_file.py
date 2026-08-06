from dataclasses import dataclass
from pydantic import BaseModel

@dataclass(frozen=True)
class RepositoryFile:
    relative_path: str
    language: str
    size_bytes: int
    content: str

class RepositoryFilePreview(BaseModel):
    relative_path: str
    language: str
    size_bytes: int
    content_preview: str

