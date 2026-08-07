from dataclasses import dataclass


@dataclass(frozen=True)
class SourceRange:
    start_line: int
    end_line: int


@dataclass(frozen=True)
class CodeChunk:
    repository_id: str
    file_path: str
    language: str

    chunk_type: str
    symbol_name: str | None

    # Location of the complete symbol in the original source file.
    symbol_start_line: int
    symbol_end_line: int

    # Exact lines that contributed content to this chunk.
    source_ranges: tuple[SourceRange, ...]

    content: str