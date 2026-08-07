from collections.abc import Callable

from app.models.code_chunk import CodeChunk
from app.models.repository_file import RepositoryFile
from app.rag.json_chunker import chunk_json_file
from app.rag.markdown_chunker import chunk_markdown_file
from app.rag.tree_sitter_chunker import chunk_code_file
from app.rag.tree_sitter_profiles import LANGUAGE_PROFILES


Chunker = Callable[[str, RepositoryFile], list[CodeChunk]]

STRUCTURED_FILE_CHUNKERS: dict[str, Chunker] = {
    "JSON": chunk_json_file,
    "Markdown": chunk_markdown_file,
    "Node Package Configuration": chunk_json_file,
    "TypeScript Configuration": chunk_json_file,
}


def supports_structural_chunking(language: str) -> bool:
    return language in LANGUAGE_PROFILES or language in STRUCTURED_FILE_CHUNKERS


def chunk_repository_file(
    repository_id: str,
    repository_file: RepositoryFile,
) -> list[CodeChunk]:
    if repository_file.language in LANGUAGE_PROFILES:
        return chunk_code_file(repository_id, repository_file)

    chunker = STRUCTURED_FILE_CHUNKERS.get(repository_file.language)
    if chunker is None:
        return []

    return chunker(repository_id, repository_file)


def chunk_repository_files(
    repository_id: str,
    repository_files: list[RepositoryFile],
) -> list[CodeChunk]:
    return [
        chunk
        for repository_file in repository_files
        for chunk in chunk_repository_file(repository_id, repository_file)
    ]
