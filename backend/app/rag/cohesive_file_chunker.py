from app.models.code_chunk import CodeChunk
from app.models.repository_file import RepositoryFile
from app.rag.structured_chunk_utils import create_whole_file_chunk


COHESIVE_CHUNK_TYPES = {
    "Dockerfile": "dockerfile",
    "Gradle Configuration": "gradle_configuration",
    "Gradle Kotlin Configuration": "gradle_kotlin_configuration",
    "Makefile": "makefile",
    "Python Requirements": "python_requirements",
}


def chunk_cohesive_file(
    repository_id: str,
    repository_file: RepositoryFile,
) -> list[CodeChunk]:
    chunk = create_whole_file_chunk(
        repository_id=repository_id,
        repository_file=repository_file,
        chunk_type=COHESIVE_CHUNK_TYPES.get(
            repository_file.language, "cohesive_document"
        ),
    )
    return [chunk] if chunk is not None else []
