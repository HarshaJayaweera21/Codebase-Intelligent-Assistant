import unittest
from pathlib import Path

from app.models.code_chunk import SourceRange
from app.models.repository_file import RepositoryFile
from app.rag.chunking_router import chunk_repository_file
from app.rag.cohesive_file_chunker import chunk_cohesive_file
from app.rag.toml_chunker import chunk_toml_file
from app.rag.xml_chunker import chunk_xml_file
from app.rag.yaml_chunker import chunk_yaml_file
from app.services.repository_scanner import detect_language


def repository_file(
    language: str,
    content: str,
    path: str = "config.txt",
) -> RepositoryFile:
    return RepositoryFile(
        relative_path=path,
        language=language,
        size_bytes=len(content.encode()),
        content=content,
    )


class YamlChunkerTests(unittest.TestCase):
    def test_mapping_is_split_by_top_level_key(self):
        source = """name: app
services:
  api:
    image: demo
  db:
    image: postgres
"""
        chunks = chunk_yaml_file(
            "repo_yaml", repository_file("YAML", source, "compose.yaml")
        )
        self.assertEqual(
            [(chunk.chunk_type, chunk.symbol_name) for chunk in chunks],
            [
                ("yaml_property", "name"),
                ("yaml_property", "services"),
            ],
        )
        self.assertEqual(chunks[1].source_ranges, (SourceRange(2, 6),))
        self.assertIn("image: postgres", chunks[1].content)

    def test_multiple_documents_have_distinct_names(self):
        source = "name: first\n---\nname: second\nenabled: true\n"
        chunks = chunk_yaml_file(
            "repo_yaml", repository_file("YAML", source, "multi.yaml")
        )
        self.assertEqual(
            [chunk.symbol_name for chunk in chunks],
            ["doc[1].name", "doc[2].name", "doc[2].enabled"],
        )

    def test_root_sequence_is_split_by_item(self):
        source = "- name: api\n  port: 8000\n- name: worker\n  port: 9000\n"
        chunks = chunk_yaml_file(
            "repo_yaml", repository_file("YAML", source, "items.yaml")
        )
        self.assertEqual(
            [chunk.symbol_name for chunk in chunks], ["$[0]", "$[1]"]
        )
        self.assertTrue(
            all(chunk.chunk_type == "yaml_sequence_item" for chunk in chunks)
        )

    def test_invalid_yaml_falls_back_to_whole_document(self):
        source = "key: [unterminated"
        chunks = chunk_yaml_file(
            "repo_yaml", repository_file("YAML", source, "broken.yaml")
        )
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_type, "yaml_document")


class XmlChunkerTests(unittest.TestCase):
    def test_root_children_are_element_chunks_with_root_context(self):
        source = """<project xmlns="https://example.test">
  <groupId>demo</groupId>
  <dependencies>
    <dependency><name>api</name></dependency>
  </dependencies>
</project>
"""
        chunks = chunk_xml_file(
            "repo_xml", repository_file("XML", source, "pom.xml")
        )
        self.assertEqual(
            [chunk.symbol_name for chunk in chunks],
            ["project > groupId", "project > dependencies"],
        )
        self.assertEqual(
            chunks[1].source_ranges,
            (SourceRange(1, 1), SourceRange(3, 5)),
        )
        self.assertIn('<project xmlns="https://example.test">', chunks[1].content)
        self.assertIn("<dependencies>", chunks[1].content)

    def test_leaf_root_and_invalid_xml_fall_back_to_document(self):
        for source in ("<name>demo</name>", "<project>"):
            with self.subTest(source=source):
                chunks = chunk_xml_file(
                    "repo_xml", repository_file("XML", source, "data.xml")
                )
                self.assertEqual(len(chunks), 1)
                self.assertEqual(chunks[0].chunk_type, "xml_document")


class TomlChunkerTests(unittest.TestCase):
    def test_properties_tables_and_table_arrays_are_separate(self):
        source = """name = "demo"
[project]
version = "1.0"
[project.dependencies]
fastapi = "latest"
[[servers]]
host = "localhost"
"""
        chunks = chunk_toml_file(
            "repo_toml", repository_file("TOML", source, "pyproject.toml")
        )
        self.assertEqual(
            [(chunk.chunk_type, chunk.symbol_name) for chunk in chunks],
            [
                ("toml_property", "name"),
                ("toml_table", "project"),
                ("toml_table", "project.dependencies"),
                ("toml_table_array", "servers"),
            ],
        )
        self.assertEqual(chunks[2].source_ranges, (SourceRange(4, 5),))

    def test_invalid_toml_falls_back_to_document(self):
        chunks = chunk_toml_file(
            "repo_toml",
            repository_file("TOML", '[project\nname = "demo"', "broken.toml"),
        )
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_type, "toml_document")


class CohesiveAndRouterTests(unittest.TestCase):
    def test_small_cohesive_file_stays_whole(self):
        source = "fastapi==1.0\nuvicorn==1.0\n"
        chunks = chunk_cohesive_file(
            "repo_text",
            repository_file(
                "Python Requirements", source, "requirements.txt"
            ),
        )
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_type, "python_requirements")
        self.assertEqual(chunks[0].source_ranges, (SourceRange(1, 2),))

    def test_router_aliases_reach_expected_chunkers(self):
        cases = (
            ("Docker Compose", "services:\n  api:\n    image: demo\n", "yaml_property"),
            ("Maven Configuration", "<project><name>demo</name></project>", "xml_element"),
            ("Python Project Configuration", '[project]\nname = "demo"\n', "toml_table"),
            ("Dockerfile", "FROM python:3.13\n", "dockerfile"),
            ("Makefile", "run:\n\tpython app.py\n", "makefile"),
            ("Gradle Configuration", "rootProject.name = 'demo'\n", "gradle_configuration"),
        )
        for language, source, expected_type in cases:
            with self.subTest(language=language):
                chunks = chunk_repository_file(
                    "repo_router", repository_file(language, source)
                )
                self.assertTrue(chunks)
                self.assertEqual(chunks[0].chunk_type, expected_type)

    def test_generic_toml_extension_and_specific_filename_detection(self):
        self.assertEqual(detect_language(Path("Cargo.toml")), "TOML")
        self.assertEqual(
            detect_language(Path("pyproject.toml")),
            "Python Project Configuration",
        )


if __name__ == "__main__":
    unittest.main()
