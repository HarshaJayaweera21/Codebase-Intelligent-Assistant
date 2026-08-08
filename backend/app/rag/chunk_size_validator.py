from bisect import bisect_right

from app.models.code_chunk import CodeChunk, SourceRange


DEFAULT_MAX_CHUNK_CHARS = 1_500
DEFAULT_CHUNK_OVERLAP_CHARS = 200
SPLIT_SEPARATORS = ("\n\n", "\n", "; ", ";", " ")


def enforce_chunk_size(
    chunks: list[CodeChunk],
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    overlap_chars: int = DEFAULT_CHUNK_OVERLAP_CHARS,
) -> list[CodeChunk]:
    validate_size_settings(max_chunk_chars, overlap_chars)
    return [
        part
        for chunk in chunks
        for part in split_oversized_chunk(
            chunk,
            max_chunk_chars=max_chunk_chars,
            overlap_chars=overlap_chars,
        )
    ]


def split_oversized_chunk(
    chunk: CodeChunk,
    max_chunk_chars: int,
    overlap_chars: int,
) -> list[CodeChunk]:
    if len(chunk.content) <= max_chunk_chars:
        return [chunk]

    content_line_map = build_content_line_map(chunk)
    line_start_offsets = find_line_start_offsets(chunk.content)
    parts: list[CodeChunk] = []

    for start_offset, end_offset in split_text_offsets(
        chunk.content, max_chunk_chars, overlap_chars
    ):
        content = chunk.content[start_offset:end_offset].strip("\r\n")
        if not content.strip():
            continue

        source_ranges = ranges_for_offsets(
            start_offset,
            end_offset,
            line_start_offsets,
            content_line_map,
        )
        parts.append(
            CodeChunk(
                repository_id=chunk.repository_id,
                file_path=chunk.file_path,
                language=chunk.language,
                chunk_type=chunk.chunk_type,
                symbol_name=chunk.symbol_name,
                symbol_start_line=chunk.symbol_start_line,
                symbol_end_line=chunk.symbol_end_line,
                source_ranges=source_ranges or chunk.source_ranges,
                content=content,
            )
        )

    return parts or [chunk]


def split_text_offsets(
    content: str,
    max_chunk_chars: int,
    overlap_chars: int,
) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    start = 0

    while start < len(content):
        target_end = min(start + max_chunk_chars, len(content))
        end = target_end

        if target_end < len(content):
            search_start = start + max_chunk_chars // 2
            for separator in SPLIT_SEPARATORS:
                boundary = content.rfind(separator, search_start, target_end)
                if boundary >= search_start:
                    end = boundary + len(separator)
                    break

        if end <= start:
            end = target_end

        offsets.append((start, end))
        if end >= len(content):
            break

        next_start = max(start + 1, end - overlap_chars)
        start = next_start

    return offsets


def build_content_line_map(chunk: CodeChunk) -> list[int | None]:
    source_lines: list[int] = []
    range_boundaries: set[int] = set()
    for index, source_range in enumerate(chunk.source_ranges):
        if index:
            range_boundaries.add(len(source_lines))
        source_lines.extend(
            range(source_range.start_line, source_range.end_line + 1)
        )

    content_lines = chunk.content.splitlines(keepends=True) or [chunk.content]
    line_map: list[int | None] = []
    source_index = 0

    for content_line in content_lines:
        if not content_line.strip() and source_index in range_boundaries:
            line_map.append(None)
            continue

        if source_index < len(source_lines):
            line_map.append(source_lines[source_index])
            source_index += 1
        elif source_lines:
            line_map.append(source_lines[-1])
        else:
            line_map.append(None)

    return line_map


def find_line_start_offsets(content: str) -> list[int]:
    offsets = [0]
    offsets.extend(
        index + 1 for index, character in enumerate(content) if character == "\n"
    )
    return offsets


def ranges_for_offsets(
    start_offset: int,
    end_offset: int,
    line_start_offsets: list[int],
    content_line_map: list[int | None],
) -> tuple[SourceRange, ...]:
    start_line_index = bisect_right(line_start_offsets, start_offset) - 1
    end_character = max(start_offset, end_offset - 1)
    end_line_index = bisect_right(line_start_offsets, end_character) - 1
    mapped_lines = [
        line
        for line in content_line_map[start_line_index : end_line_index + 1]
        if line is not None
    ]
    return group_source_lines(mapped_lines)


def group_source_lines(lines: list[int]) -> tuple[SourceRange, ...]:
    if not lines:
        return ()

    ranges: list[SourceRange] = []
    start_line = lines[0]
    end_line = lines[0]
    for line in lines[1:]:
        if line == end_line:
            continue
        if line == end_line + 1:
            end_line = line
            continue

        ranges.append(SourceRange(start_line, end_line))
        start_line = line
        end_line = line

    ranges.append(SourceRange(start_line, end_line))
    return tuple(ranges)


def validate_size_settings(max_chunk_chars: int, overlap_chars: int) -> None:
    if max_chunk_chars <= 0:
        raise ValueError("max_chunk_chars must be greater than zero")
    if overlap_chars < 0:
        raise ValueError("overlap_chars cannot be negative")
    if overlap_chars >= max_chunk_chars:
        raise ValueError("overlap_chars must be smaller than max_chunk_chars")
