import unittest

from app.models.code_chunk import CodeChunk, SourceRange
from app.rag.document_processor import (
    code_chunk_to_document,
    create_langchain_documents,
)


def code_chunk(
    *,
    symbol_name: str | None = "CheckoutService",
    content: str = "class CheckoutService {}",
) -> CodeChunk:
    return CodeChunk(
        repository_id="repo_documents",
        file_path="src/checkout.py",
        language="Python",
        chunk_type="class_context",
        symbol_name=symbol_name,
        symbol_start_line=10,
        symbol_end_line=30,
        source_ranges=(SourceRange(10, 12), SourceRange(29, 30)),
        content=content,
    )


class DocumentProcessorTests(unittest.TestCase):
    def test_converts_chunk_content_and_all_metadata(self):
        document = code_chunk_to_document(code_chunk())

        self.assertEqual(document.page_content, "class CheckoutService {}")
        self.assertEqual(
            document.metadata,
            {
                "repository_id": "repo_documents",
                "file_path": "src/checkout.py",
                "language": "Python",
                "chunk_type": "class_context",
                "symbol_name": "CheckoutService",
                "symbol_start_line": 10,
                "symbol_end_line": 30,
                "source_ranges": [
                    {"start_line": 10, "end_line": 12},
                    {"start_line": 29, "end_line": 30},
                ],
            },
        )

    def test_preserves_none_symbol_name(self):
        document = code_chunk_to_document(code_chunk(symbol_name=None))
        self.assertIsNone(document.metadata["symbol_name"])

    def test_converts_multiple_chunks_in_input_order(self):
        chunks = [
            code_chunk(content="first"),
            code_chunk(content="second"),
        ]

        documents = create_langchain_documents(chunks)

        self.assertEqual(
            [document.page_content for document in documents],
            ["first", "second"],
        )

    def test_empty_chunk_list_produces_no_documents(self):
        self.assertEqual(create_langchain_documents([]), [])


if __name__ == "__main__":
    unittest.main()
