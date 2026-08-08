from langchain_core.documents import Document

from app.models.code_chunk import CodeChunk


def code_chunk_to_document(chunk: CodeChunk) -> Document:
    """Convert one final structural chunk into a LangChain document."""
    return Document(
        page_content=chunk.content,
        metadata={
            "repository_id": chunk.repository_id,
            "file_path": chunk.file_path,
            "language": chunk.language,
            "chunk_type": chunk.chunk_type,
            "symbol_name": chunk.symbol_name,
            "symbol_start_line": chunk.symbol_start_line,
            "symbol_end_line": chunk.symbol_end_line,
            "source_ranges": [
                {
                    "start_line": source_range.start_line,
                    "end_line": source_range.end_line,
                }
                for source_range in chunk.source_ranges
            ],
        },
    )


def create_langchain_documents(chunks: list[CodeChunk]) -> list[Document]:
    """Convert final chunks to documents while preserving source order."""
    return [code_chunk_to_document(chunk) for chunk in chunks]
