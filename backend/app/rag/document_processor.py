from langchain_core.documents import Document

from app.models.repository_file import RepositoryFile


def create_langchain_documents(
    repository_id: str,
    repository_files: list[RepositoryFile],
) -> list[Document]:
    documents: list[Document] = []

    for repository_file in repository_files:
        document = Document(
            page_content=repository_file.content,
            metadata={
                "repository_id": repository_id,
                "file_path": repository_file.relative_path,
                "language": repository_file.language,
                "size_bytes": repository_file.size_bytes,
            },
        )

        documents.append(document)

    return documents