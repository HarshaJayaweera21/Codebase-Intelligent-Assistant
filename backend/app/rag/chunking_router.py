from collections.abc import Callable

from app.models.code_chunk import CodeChunk
from app.models.repository_file import RepositoryFile
from app.rag.chunk_size_validator import (
    DEFAULT_CHUNK_OVERLAP_CHARS,
    DEFAULT_MAX_CHUNK_CHARS,
    enforce_chunk_size,
)
from app.rag.cohesive_file_chunker import chunk_cohesive_file
from app.rag.json_chunker import chunk_json_file
from app.rag.html_chunker import chunk_html_file
from app.rag.markdown_chunker import chunk_markdown_file
from app.rag.sql_chunker import chunk_sql_file
from app.rag.stylesheet_chunker import chunk_stylesheet_file
from app.rag.toml_chunker import chunk_toml_file
from app.rag.tree_sitter_chunker import chunk_code_file
from app.rag.tree_sitter_profiles import LANGUAGE_PROFILES
from app.rag.xml_chunker import chunk_xml_file
from app.rag.yaml_chunker import chunk_yaml_file


Chunker = Callable[[str, RepositoryFile], list[CodeChunk]]

STRUCTURED_FILE_CHUNKERS: dict[str, Chunker] = {
    "Docker Compose": chunk_yaml_file,
    "Dockerfile": chunk_cohesive_file,
    "Gradle Configuration": chunk_cohesive_file,
    "Gradle Kotlin Configuration": chunk_cohesive_file,
    "HTML": chunk_html_file,
    "JSON": chunk_json_file,
    "Makefile": chunk_cohesive_file,
    "Markdown": chunk_markdown_file,
    "Maven Configuration": chunk_xml_file,
    "Node Package Configuration": chunk_json_file,
    "Python Pipfile": chunk_toml_file,
    "Python Project Configuration": chunk_toml_file,
    "Python Requirements": chunk_cohesive_file,
    "SCSS": chunk_stylesheet_file,
    "CSS": chunk_stylesheet_file,
    "SQL": chunk_sql_file,
    "TOML": chunk_toml_file,
    "TypeScript Configuration": chunk_json_file,
    "XML": chunk_xml_file,
    "YAML": chunk_yaml_file,
}


def supports_structural_chunking(language: str) -> bool:
    return language in LANGUAGE_PROFILES or language in STRUCTURED_FILE_CHUNKERS


def chunk_repository_file(
    repository_id: str,
    repository_file: RepositoryFile,
    *,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    overlap_chars: int = DEFAULT_CHUNK_OVERLAP_CHARS,
) -> list[CodeChunk]:
    if repository_file.language in LANGUAGE_PROFILES:
        chunks = chunk_code_file(repository_id, repository_file)
    else:
        chunker = STRUCTURED_FILE_CHUNKERS.get(repository_file.language)
        if chunker is None:
            return []
        chunks = chunker(repository_id, repository_file)

    return enforce_chunk_size(
        chunks,
        max_chunk_chars=max_chunk_chars,
        overlap_chars=overlap_chars,
    )


def chunk_repository_files(
    repository_id: str,
    repository_files: list[RepositoryFile],
    *,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    overlap_chars: int = DEFAULT_CHUNK_OVERLAP_CHARS,
) -> list[CodeChunk]:
    return [
        chunk
        for repository_file in repository_files
        for chunk in chunk_repository_file(
            repository_id,
            repository_file,
            max_chunk_chars=max_chunk_chars,
            overlap_chars=overlap_chars,
        )
    ]
