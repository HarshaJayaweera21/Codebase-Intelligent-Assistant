import unittest

from app.models.code_chunk import CodeChunk, SourceRange
from app.models.repository_file import RepositoryFile
from app.rag.chunk_size_validator import enforce_chunk_size
from app.rag.chunking_router import chunk_repository_file


def code_chunk(
    content: str,
    source_ranges: tuple[SourceRange, ...],
) -> CodeChunk:
    return CodeChunk(
        repository_id="repo_size",
        file_path="large.py",
        language="Python",
        chunk_type="function",
        symbol_name="large_function",
        symbol_start_line=source_ranges[0].start_line,
        symbol_end_line=source_ranges[-1].end_line,
        source_ranges=source_ranges,
        content=content,
    )


class ChunkSizeValidatorTests(unittest.TestCase):
    def test_contiguous_chunk_is_split_with_narrowed_ranges(self):
        content = "\n".join(f"line {index}: " + "x" * 20 for index in range(10))
        original = code_chunk(content, (SourceRange(10, 19),))
        parts = enforce_chunk_size(
            [original], max_chunk_chars=80, overlap_chars=15
        )

        self.assertGreater(len(parts), 1)
        self.assertTrue(all(len(part.content) <= 80 for part in parts))
        self.assertTrue(
            all(
                10 <= source_range.start_line <= source_range.end_line <= 19
                for part in parts
                for source_range in part.source_ranges
            )
        )
        self.assertTrue(
            all(part.symbol_start_line == 10 for part in parts)
        )
        self.assertTrue(all(part.symbol_end_line == 19 for part in parts))

    def test_non_contiguous_context_ranges_remain_non_contiguous(self):
        content = "# Parent\n\n## Child\n" + "\n".join(
            f"detail {index} " + "x" * 15 for index in range(8)
        )
        original = code_chunk(
            content,
            (SourceRange(1, 1), SourceRange(5, 13)),
        )
        parts = enforce_chunk_size(
            [original], max_chunk_chars=70, overlap_chars=10
        )

        self.assertGreater(len(parts), 1)
        valid_lines = {1, *range(5, 14)}
        self.assertTrue(
            all(
                line in valid_lines
                for part in parts
                for source_range in part.source_ranges
                for line in range(source_range.start_line, source_range.end_line + 1)
            )
        )

    def test_long_single_line_keeps_its_one_line_source_range(self):
        original = code_chunk("x" * 240, (SourceRange(7, 7),))
        parts = enforce_chunk_size(
            [original], max_chunk_chars=80, overlap_chars=10
        )
        self.assertGreater(len(parts), 1)
        self.assertTrue(
            all(part.source_ranges == (SourceRange(7, 7),) for part in parts)
        )

    def test_overlap_starts_at_complete_line_boundary(self):
        lines = [
            f"foodItemService.operation{index}(); // " + "x" * 25
            for index in range(12)
        ]
        original = code_chunk(
            "\n".join(lines),
            (SourceRange(20, 31),),
        )

        parts = enforce_chunk_size(
            [original],
            max_chunk_chars=160,
            overlap_chars=45,
        )

        self.assertGreater(len(parts), 1)
        self.assertTrue(
            all(part.content.startswith("foodItemService") for part in parts)
        )
        self.assertTrue(all(len(part.content) <= 160 for part in parts))

    def test_router_applies_default_size_limit(self):
        source = "FROM python:3.13\n" + "RUN echo value\n" * 180
        repository_file = RepositoryFile(
            relative_path="Dockerfile",
            language="Dockerfile",
            size_bytes=len(source.encode()),
            content=source,
        )
        chunks = chunk_repository_file("repo_size", repository_file)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.content) <= 1_500 for chunk in chunks))

    def test_invalid_size_settings_are_rejected(self):
        original = code_chunk("text", (SourceRange(1, 1),))
        for max_chars, overlap in ((0, 0), (10, -1), (10, 10)):
            with self.subTest(max_chars=max_chars, overlap=overlap):
                with self.assertRaises(ValueError):
                    enforce_chunk_size(
                        [original],
                        max_chunk_chars=max_chars,
                        overlap_chars=overlap,
                    )


if __name__ == "__main__":
    unittest.main()
