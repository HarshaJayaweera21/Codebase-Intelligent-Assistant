import json

from tree_sitter import Node
from tree_sitter_language_pack import Error as LanguagePackError
from tree_sitter_language_pack import get_parser

from app.models.code_chunk import CodeChunk, SourceRange
from app.models.repository_file import RepositoryFile


def chunk_json_file(
    repository_id: str,
    repository_file: RepositoryFile,
) -> list[CodeChunk]:
    if not repository_file.content.strip():
        return []

    source_bytes = repository_file.content.encode("utf-8")
    try:
        tree = get_parser("json").parse(source_bytes)
    except LanguagePackError:
        return []

    root_value = first_named_child(tree.root_node)
    if root_value is None or tree.root_node.has_error:
        return [create_document_chunk(repository_id, repository_file)]

    if root_value.type == "object":
        chunks = create_object_property_chunks(
            root_value,
            repository_id,
            repository_file,
            source_bytes,
        )
        return chunks or [create_document_chunk(repository_id, repository_file)]

    if root_value.type == "array":
        chunks = create_array_item_chunks(
            root_value,
            repository_id,
            repository_file,
            source_bytes,
        )
        return chunks or [create_document_chunk(repository_id, repository_file)]

    return [create_document_chunk(repository_id, repository_file)]


def create_object_property_chunks(
    object_node: Node,
    repository_id: str,
    repository_file: RepositoryFile,
    source_bytes: bytes,
) -> list[CodeChunk]:
    chunks: list[CodeChunk] = []

    for pair_node in object_node.named_children:
        if pair_node.type != "pair":
            continue

        key_node = pair_node.child_by_field_name("key")
        if key_node is None:
            continue

        key_source = node_text(key_node, source_bytes)
        try:
            key = json.loads(key_source)
        except (TypeError, json.JSONDecodeError):
            key = key_source.strip('"')

        chunks.append(
            create_node_chunk(
                node=pair_node,
                repository_id=repository_id,
                repository_file=repository_file,
                source_bytes=source_bytes,
                chunk_type="json_property",
                symbol_name=str(key),
            )
        )

    return chunks


def create_array_item_chunks(
    array_node: Node,
    repository_id: str,
    repository_file: RepositoryFile,
    source_bytes: bytes,
) -> list[CodeChunk]:
    return [
        create_node_chunk(
            node=item_node,
            repository_id=repository_id,
            repository_file=repository_file,
            source_bytes=source_bytes,
            chunk_type="json_array_item",
            symbol_name=f"$[{index}]",
        )
        for index, item_node in enumerate(array_node.named_children)
    ]


def create_node_chunk(
    node: Node,
    repository_id: str,
    repository_file: RepositoryFile,
    source_bytes: bytes,
    chunk_type: str,
    symbol_name: str,
) -> CodeChunk:
    start_line = node.start_point.row + 1
    end_line = node.end_point.row + (1 if node.end_point.column else 0)
    end_line = max(start_line, end_line)
    return CodeChunk(
        repository_id=repository_id,
        file_path=repository_file.relative_path,
        language=repository_file.language,
        chunk_type=chunk_type,
        symbol_name=symbol_name,
        symbol_start_line=start_line,
        symbol_end_line=end_line,
        source_ranges=(SourceRange(start_line, end_line),),
        content=node_text(node, source_bytes),
    )


def create_document_chunk(
    repository_id: str,
    repository_file: RepositoryFile,
) -> CodeChunk:
    line_count = max(1, len(repository_file.content.splitlines()))
    return CodeChunk(
        repository_id=repository_id,
        file_path=repository_file.relative_path,
        language=repository_file.language,
        chunk_type="json_document",
        symbol_name=None,
        symbol_start_line=1,
        symbol_end_line=line_count,
        source_ranges=(SourceRange(1, line_count),),
        content=repository_file.content,
    )


def first_named_child(node: Node) -> Node | None:
    return node.named_child(0) if node.named_child_count else None


def node_text(node: Node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte : node.end_byte].decode("utf-8")
