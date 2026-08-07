import re
from dataclasses import dataclass

from app.models.code_chunk import CodeChunk, SourceRange
from app.models.repository_file import RepositoryFile


ATX_HEADING_PATTERN = re.compile(
    r"^[ \t]{0,3}(#{1,6})(?:[ \t]+(.*?)|[ \t]*)$"
)
SETEXT_UNDERLINE_PATTERN = re.compile(
    r"^[ \t]{0,3}(=+|-+)[ \t]*$"
)
FENCE_PATTERN = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")
TRAILING_HEADING_MARKERS_PATTERN = re.compile(r"[ \t]+#+[ \t]*$")


@dataclass(frozen=True)
class MarkdownHeading:
    level: int
    text: str
    start_index: int
    end_index: int


def chunk_markdown_file(
    repository_id: str,
    repository_file: RepositoryFile,
) -> list[CodeChunk]:
    lines = repository_file.content.splitlines()
    if not lines:
        return []

    headings = find_markdown_headings(lines)
    if not headings:
        return [
            create_markdown_chunk(
                repository_id=repository_id,
                repository_file=repository_file,
                chunk_type="markdown_document",
                symbol_name=None,
                symbol_start_index=0,
                symbol_end_index=len(lines) - 1,
                content_parts=[lines],
                source_ranges=(line_range(0, len(lines) - 1),),
            )
        ]

    chunks: list[CodeChunk] = []
    first_heading = headings[0]
    preamble_lines = lines[: first_heading.start_index]
    if any(line.strip() for line in preamble_lines):
        preamble_end = first_heading.start_index - 1
        chunks.append(
            create_markdown_chunk(
                repository_id=repository_id,
                repository_file=repository_file,
                chunk_type="markdown_preamble",
                symbol_name=None,
                symbol_start_index=0,
                symbol_end_index=preamble_end,
                content_parts=[preamble_lines],
                source_ranges=(line_range(0, preamble_end),),
            )
        )

    heading_stack: list[MarkdownHeading] = []
    for index, heading in enumerate(headings):
        while heading_stack and heading_stack[-1].level >= heading.level:
            heading_stack.pop()

        section_end = (
            headings[index + 1].start_index - 1
            if index + 1 < len(headings)
            else len(lines) - 1
        )
        ancestors = tuple(heading_stack)
        path = " > ".join(
            ancestor.text for ancestor in (*ancestors, heading)
        )

        content_parts = [
            lines[ancestor.start_index : ancestor.end_index + 1]
            for ancestor in ancestors
        ]
        content_parts.append(lines[heading.start_index : section_end + 1])

        source_ranges = tuple(
            line_range(ancestor.start_index, ancestor.end_index)
            for ancestor in ancestors
        ) + (line_range(heading.start_index, section_end),)

        chunks.append(
            create_markdown_chunk(
                repository_id=repository_id,
                repository_file=repository_file,
                chunk_type="markdown_section",
                symbol_name=path,
                symbol_start_index=heading.start_index,
                symbol_end_index=section_end,
                content_parts=content_parts,
                source_ranges=source_ranges,
            )
        )
        heading_stack.append(heading)

    return chunks


def find_markdown_headings(lines: list[str]) -> list[MarkdownHeading]:
    fenced_lines = find_fenced_lines(lines) | find_front_matter_lines(lines)
    headings: list[MarkdownHeading] = []
    index = 0

    while index < len(lines):
        if index in fenced_lines:
            index += 1
            continue

        atx_match = ATX_HEADING_PATTERN.match(lines[index])
        if atx_match is not None:
            heading_text = atx_match.group(2) or ""
            heading_text = TRAILING_HEADING_MARKERS_PATTERN.sub(
                "", heading_text
            ).strip()
            headings.append(
                MarkdownHeading(
                    level=len(atx_match.group(1)),
                    text=heading_text,
                    start_index=index,
                    end_index=index,
                )
            )
            index += 1
            continue

        if index + 1 < len(lines) and index + 1 not in fenced_lines:
            setext_match = SETEXT_UNDERLINE_PATTERN.match(lines[index + 1])
            if setext_match is not None and lines[index].strip():
                headings.append(
                    MarkdownHeading(
                        level=1 if setext_match.group(1).startswith("=") else 2,
                        text=lines[index].strip(),
                        start_index=index,
                        end_index=index + 1,
                    )
                )
                index += 2
                continue

        index += 1

    return headings


def find_fenced_lines(lines: list[str]) -> set[int]:
    fenced_lines: set[int] = set()
    fence_character: str | None = None
    fence_length = 0

    for index, line in enumerate(lines):
        if fence_character is None:
            fence_match = FENCE_PATTERN.match(line)
            if fence_match is not None:
                marker = fence_match.group(1)
                fence_character = marker[0]
                fence_length = len(marker)
                fenced_lines.add(index)
            continue

        fenced_lines.add(index)
        stripped_line = line.lstrip(" \t")
        marker_length = len(stripped_line) - len(
            stripped_line.lstrip(fence_character)
        )
        marker_suffix = stripped_line[marker_length:]
        if (
            len(line) - len(stripped_line) <= 3
            and marker_length >= fence_length
            and not marker_suffix.strip()
        ):
            fence_character = None
            fence_length = 0

    return fenced_lines


def find_front_matter_lines(lines: list[str]) -> set[int]:
    if not lines or lines[0].strip() != "---":
        return set()

    for index in range(1, len(lines)):
        if lines[index].strip() in {"---", "..."}:
            return set(range(index + 1))

    return set()


def create_markdown_chunk(
    repository_id: str,
    repository_file: RepositoryFile,
    chunk_type: str,
    symbol_name: str | None,
    symbol_start_index: int,
    symbol_end_index: int,
    content_parts: list[list[str]],
    source_ranges: tuple[SourceRange, ...],
) -> CodeChunk:
    content = "\n\n".join(
        "\n".join(part).strip("\n") for part in content_parts if part
    )
    return CodeChunk(
        repository_id=repository_id,
        file_path=repository_file.relative_path,
        language=repository_file.language,
        chunk_type=chunk_type,
        symbol_name=symbol_name,
        symbol_start_line=symbol_start_index + 1,
        symbol_end_line=symbol_end_index + 1,
        source_ranges=source_ranges,
        content=content,
    )


def line_range(start_index: int, end_index: int) -> SourceRange:
    return SourceRange(start_line=start_index + 1, end_line=end_index + 1)
