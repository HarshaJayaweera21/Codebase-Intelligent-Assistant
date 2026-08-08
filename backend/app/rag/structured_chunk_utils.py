from tree_sitter import Node

from app.models.code_chunk import CodeChunk, SourceRange
from app.models.repository_file import RepositoryFile


def node_text(node: Node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte : node.end_byte].decode("utf-8")


def node_source_range(node: Node) -> SourceRange:
    return SourceRange(
        start_line=node.start_point.row + 1,
        end_line=node_end_line(node),
    )


def node_end_line(node: Node) -> int:
    # An end point at column zero means the node stops before contributing
    # content to that line (commonly after consuming a trailing newline).
    end_line = node.end_point.row + (1 if node.end_point.column else 0)
    return max(node.start_point.row + 1, end_line)


def create_node_chunk(
    node: Node,
    repository_id: str,
    repository_file: RepositoryFile,
    source_bytes: bytes,
    chunk_type: str,
    symbol_name: str | None,
    context_nodes: tuple[Node, ...] = (),
) -> CodeChunk:
    content_nodes = (*context_nodes, node)
    content_parts: list[str] = []
    source_ranges: list[SourceRange] = []
    for content_node in content_nodes:
        content_part = node_text(content_node, source_bytes).strip()
        if not content_part:
            continue

        content_parts.append(content_part)
        source_range = node_source_range(content_node)
        if source_range not in source_ranges:
            source_ranges.append(source_range)

    return CodeChunk(
        repository_id=repository_id,
        file_path=repository_file.relative_path,
        language=repository_file.language,
        chunk_type=chunk_type,
        symbol_name=symbol_name,
        symbol_start_line=node.start_point.row + 1,
        symbol_end_line=node_end_line(node),
        source_ranges=tuple(source_ranges),
        content="\n\n".join(content_parts),
    )


def create_whole_file_chunk(
    repository_id: str,
    repository_file: RepositoryFile,
    chunk_type: str,
) -> CodeChunk | None:
    if not repository_file.content.strip():
        return None

    line_count = max(1, len(repository_file.content.splitlines()))
    return CodeChunk(
        repository_id=repository_id,
        file_path=repository_file.relative_path,
        language=repository_file.language,
        chunk_type=chunk_type,
        symbol_name=None,
        symbol_start_line=1,
        symbol_end_line=line_count,
        source_ranges=(SourceRange(1, line_count),),
        content=repository_file.content,
    )
