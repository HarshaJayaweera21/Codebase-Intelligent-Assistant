from tree_sitter import Node
from tree_sitter_language_pack import Error as LanguagePackError
from tree_sitter_language_pack import get_parser

from app.models.code_chunk import CodeChunk
from app.models.repository_file import RepositoryFile
from app.rag.structured_chunk_utils import (
    create_node_chunk,
    create_whole_file_chunk,
    node_text,
)


def chunk_toml_file(
    repository_id: str,
    repository_file: RepositoryFile,
) -> list[CodeChunk]:
    if not repository_file.content.strip():
        return []

    source_bytes = repository_file.content.encode("utf-8")
    try:
        tree = get_parser("toml").parse(source_bytes)
    except LanguagePackError:
        return []

    if tree.root_node.has_error:
        return whole_toml_document(repository_id, repository_file)

    chunks: list[CodeChunk] = []
    for child in tree.root_node.named_children:
        if child.type == "pair":
            key_node = first_named_child(child)
            chunks.append(
                create_node_chunk(
                    node=child,
                    repository_id=repository_id,
                    repository_file=repository_file,
                    source_bytes=source_bytes,
                    chunk_type="toml_property",
                    symbol_name=(
                        node_text(key_node, source_bytes)
                        if key_node is not None
                        else None
                    ),
                )
            )
        elif child.type in {"table", "table_array_element"}:
            name_node = first_named_child(child)
            chunks.append(
                create_node_chunk(
                    node=child,
                    repository_id=repository_id,
                    repository_file=repository_file,
                    source_bytes=source_bytes,
                    chunk_type=(
                        "toml_table"
                        if child.type == "table"
                        else "toml_table_array"
                    ),
                    symbol_name=(
                        node_text(name_node, source_bytes)
                        if name_node is not None
                        else None
                    ),
                )
            )

    return chunks or whole_toml_document(repository_id, repository_file)


def first_named_child(node: Node) -> Node | None:
    return node.named_child(0) if node.named_child_count else None


def whole_toml_document(
    repository_id: str,
    repository_file: RepositoryFile,
) -> list[CodeChunk]:
    chunk = create_whole_file_chunk(
        repository_id, repository_file, "toml_document"
    )
    return [chunk] if chunk is not None else []
