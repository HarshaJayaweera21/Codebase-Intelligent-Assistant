import unittest
from pathlib import Path

from app.models.repository_file import RepositoryFile
from app.rag.chunking_router import (
    chunk_repository_file,
    chunk_repository_files,
    supports_structural_chunking,
)
from app.services.repository_scanner import detect_language


def repository_file(language: str, content: str, path: str) -> RepositoryFile:
    return RepositoryFile(
        relative_path=path,
        language=language,
        size_bytes=len(content.encode()),
        content=content,
    )


class ChunkingRouterTests(unittest.TestCase):
    def test_routes_programming_language_to_tree_sitter(self):
        chunks = chunk_repository_file(
            "repo_router",
            repository_file("Python", "def run():\n    pass\n", "app.py"),
        )
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_type, "function")
        self.assertEqual(chunks[0].symbol_name, "run")

    def test_routes_markdown_to_markdown_chunker(self):
        chunks = chunk_repository_file(
            "repo_router",
            repository_file("Markdown", "# Guide\nText.\n", "README.md"),
        )
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_type, "markdown_section")
        self.assertEqual(chunks[0].symbol_name, "Guide")

    def test_routes_json_and_filename_specific_json_formats(self):
        for language, path in (
            ("JSON", "config.json"),
            ("Node Package Configuration", "package.json"),
            ("TypeScript Configuration", "tsconfig.json"),
        ):
            with self.subTest(language=language):
                json_file = repository_file(
                    language, '{"enabled": true}', path
                )
                self.assertTrue(supports_structural_chunking(language))
                chunks = chunk_repository_file("repo_router", json_file)
                self.assertEqual(len(chunks), 1)
                self.assertEqual(chunks[0].chunk_type, "json_property")
                self.assertEqual(chunks[0].symbol_name, "enabled")

    def test_unsupported_format_waits_for_its_own_chunker(self):
        text_file = repository_file("Plain Text", "Notes", "notes.txt")
        self.assertFalse(supports_structural_chunking("Plain Text"))
        self.assertEqual(chunk_repository_file("repo_router", text_file), [])

    def test_jsp_is_detected_and_routed_through_html_chunking(self):
        source = """<%@ page contentType="text/html" %>
<html><body><form id="loginForm"><input name="password"></form></body></html>
"""

        self.assertEqual(detect_language(Path("login.jsp")), "JSP")
        self.assertTrue(supports_structural_chunking("JSP"))
        chunks = chunk_repository_file(
            "repo_router",
            repository_file("JSP", source, "login.jsp"),
        )
        self.assertTrue(chunks)
        self.assertEqual(chunks[0].file_path, "login.jsp")

    def test_chunks_multiple_files_in_input_order(self):
        files = [
            repository_file("Markdown", "# Docs\nText.\n", "README.md"),
            repository_file("Shell", "run() { echo ok; }\n", "run.sh"),
        ]
        chunks = chunk_repository_files("repo_router", files)
        self.assertEqual(
            [(chunk.file_path, chunk.symbol_name) for chunk in chunks],
            [("README.md", "Docs"), ("run.sh", "run")],
        )


if __name__ == "__main__":
    unittest.main()
