import unittest

from app.models.code_chunk import SourceRange
from app.models.repository_file import RepositoryFile
from app.rag.json_chunker import chunk_json_file


def chunk_json(source: str, language: str = "JSON"):
    return chunk_json_file(
        repository_id="repo_json",
        repository_file=RepositoryFile(
            relative_path="config.json",
            language=language,
            size_bytes=len(source.encode()),
            content=source,
        ),
    )


class JsonChunkerTests(unittest.TestCase):
    def test_root_object_is_split_by_top_level_property(self):
        source = """{
  "name": "demo",
  "scripts": {
    "start": "python app.py"
  },
  "dependencies": {
    "fastapi": "latest"
  }
}
"""
        chunks = chunk_json(source)
        self.assertEqual(
            [(chunk.chunk_type, chunk.symbol_name) for chunk in chunks],
            [
                ("json_property", "name"),
                ("json_property", "scripts"),
                ("json_property", "dependencies"),
            ],
        )
        self.assertEqual(chunks[0].source_ranges, (SourceRange(2, 2),))
        self.assertEqual(chunks[1].source_ranges, (SourceRange(3, 5),))
        self.assertIn('"start": "python app.py"', chunks[1].content)

    def test_root_array_is_split_by_item(self):
        chunks = chunk_json('[\n  {"id": 1},\n  {"id": 2}\n]')
        self.assertEqual(
            [chunk.symbol_name for chunk in chunks],
            ["$[0]", "$[1]"],
        )
        self.assertEqual(
            [chunk.chunk_type for chunk in chunks],
            ["json_array_item", "json_array_item"],
        )

    def test_escaped_property_name_is_decoded(self):
        chunks = chunk_json('{"display\\u0020name": "Assistant"}')
        self.assertEqual(chunks[0].symbol_name, "display name")

    def test_empty_container_and_invalid_json_fall_back_to_document(self):
        for source in ("{}", "[]", '{"incomplete":'):
            with self.subTest(source=source):
                chunks = chunk_json(source)
                self.assertEqual(len(chunks), 1)
                self.assertEqual(chunks[0].chunk_type, "json_document")
                self.assertEqual(chunks[0].content, source)

    def test_empty_file_produces_no_chunks(self):
        self.assertEqual(chunk_json("  \n"), [])


if __name__ == "__main__":
    unittest.main()
