import unittest

from app.models.code_chunk import SourceRange
from app.models.repository_file import RepositoryFile
from app.rag.markdown_chunker import chunk_markdown_file


def chunk_markdown(source: str):
    return chunk_markdown_file(
        repository_id="repo_docs",
        repository_file=RepositoryFile(
            relative_path="README.md",
            language="Markdown",
            size_bytes=len(source.encode()),
            content=source,
        ),
    )


class MarkdownChunkerTests(unittest.TestCase):
    def test_heading_hierarchy_front_matter_and_code_fences(self):
        source = """---
title: Guide
---
Introduction.
# Guide
Overview.

## Setup
Install it.
### Windows
Run this:
```bash
# This is not a heading
```
## Usage
Use it.
"""
        chunks = chunk_markdown(source)

        self.assertEqual(
            [chunk.chunk_type for chunk in chunks],
            [
                "markdown_preamble",
                "markdown_section",
                "markdown_section",
                "markdown_section",
                "markdown_section",
            ],
        )
        self.assertEqual(
            [chunk.symbol_name for chunk in chunks],
            [
                None,
                "Guide",
                "Guide > Setup",
                "Guide > Setup > Windows",
                "Guide > Usage",
            ],
        )

        windows = chunks[3]
        self.assertEqual(windows.symbol_start_line, 10)
        self.assertEqual(windows.symbol_end_line, 14)
        self.assertEqual(
            windows.source_ranges,
            (
                SourceRange(5, 5),
                SourceRange(8, 8),
                SourceRange(10, 14),
            ),
        )
        self.assertIn("# Guide", windows.content)
        self.assertIn("## Setup", windows.content)
        self.assertIn("# This is not a heading", windows.content)

    def test_setext_headings_and_literal_hash_in_title(self):
        source = """Project Guide
=============
Overview.
Installation
------------
Steps.
## C#
Examples.
"""
        chunks = chunk_markdown(source)
        self.assertEqual(
            [chunk.symbol_name for chunk in chunks],
            [
                "Project Guide",
                "Project Guide > Installation",
                "Project Guide > C#",
            ],
        )
        self.assertEqual(chunks[0].source_ranges, (SourceRange(1, 3),))
        self.assertEqual(chunks[1].source_ranges[-1], SourceRange(4, 6))

    def test_document_without_headings_remains_whole(self):
        chunks = chunk_markdown("A short document.\n\nStill one section.")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_type, "markdown_document")
        self.assertIsNone(chunks[0].symbol_name)
        self.assertEqual(chunks[0].source_ranges, (SourceRange(1, 3),))

    def test_empty_document_produces_no_chunks(self):
        self.assertEqual(chunk_markdown(""), [])


if __name__ == "__main__":
    unittest.main()
