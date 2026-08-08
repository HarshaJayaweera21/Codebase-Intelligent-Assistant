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


YAML_MAPPING_TYPES = {"block_mapping", "flow_mapping"}
YAML_SEQUENCE_TYPES = {"block_sequence", "flow_sequence"}


def chunk_yaml_file(
    repository_id: str,
    repository_file: RepositoryFile,
) -> list[CodeChunk]:
    if not repository_file.content.strip():
        return []

    source_bytes = repository_file.content.encode("utf-8")
    try:
        tree = get_parser("yaml").parse(source_bytes)
    except LanguagePackError:
        return []

    if tree.root_node.has_error:
        return whole_yaml_document(repository_id, repository_file)

    documents = [
        child
        for child in tree.root_node.named_children
        if child.type == "document"
    ]
    chunks: list[CodeChunk] = []

    for document_index, document_node in enumerate(documents):
        root_value = unwrap_yaml_root(document_node)
        if root_value is None:
            continue

        prefix = f"doc[{document_index + 1}]." if len(documents) > 1 else ""

        if root_value.type in YAML_MAPPING_TYPES:
            chunks.extend(
                create_yaml_mapping_chunks(
                    root_value,
                    prefix,
                    repository_id,
                    repository_file,
                    source_bytes,
                )
            )
        elif root_value.type in YAML_SEQUENCE_TYPES:
            chunks.extend(
                create_yaml_sequence_chunks(
                    root_value,
                    prefix,
                    repository_id,
                    repository_file,
                    source_bytes,
                )
            )
        else:
            chunks.append(
                create_node_chunk(
                    node=root_value,
                    repository_id=repository_id,
                    repository_file=repository_file,
                    source_bytes=source_bytes,
                    chunk_type="yaml_document",
                    symbol_name=(
                        f"doc[{document_index + 1}]"
                        if len(documents) > 1
                        else None
                    ),
                )
            )

    return chunks or whole_yaml_document(repository_id, repository_file)


def create_yaml_mapping_chunks(
    mapping_node: Node,
    prefix: str,
    repository_id: str,
    repository_file: RepositoryFile,
    source_bytes: bytes,
) -> list[CodeChunk]:
    chunks: list[CodeChunk] = []
    for pair_node in mapping_node.named_children:
        if pair_node.type not in {"block_mapping_pair", "flow_pair"}:
            continue

        key_node = pair_node.child_by_field_name("key")
        if key_node is None:
            continue

        key = normalize_yaml_key(node_text(key_node, source_bytes))
        chunks.append(
            create_node_chunk(
                node=pair_node,
                repository_id=repository_id,
                repository_file=repository_file,
                source_bytes=source_bytes,
                chunk_type="yaml_property",
                symbol_name=f"{prefix}{key}",
            )
        )

    return chunks


def create_yaml_sequence_chunks(
    sequence_node: Node,
    prefix: str,
    repository_id: str,
    repository_file: RepositoryFile,
    source_bytes: bytes,
) -> list[CodeChunk]:
    item_nodes = [
        child
        for child in sequence_node.named_children
        if child.type in {"block_sequence_item", "flow_node"}
    ]
    return [
        create_node_chunk(
            node=item_node,
            repository_id=repository_id,
            repository_file=repository_file,
            source_bytes=source_bytes,
            chunk_type="yaml_sequence_item",
            symbol_name=f"{prefix}$[{index}]",
        )
        for index, item_node in enumerate(item_nodes)
    ]


def unwrap_yaml_root(node: Node) -> Node | None:
    current = node
    while current.type not in YAML_MAPPING_TYPES | YAML_SEQUENCE_TYPES:
        children = [
            child for child in current.named_children if child.type != "comment"
        ]
        if len(children) != 1:
            return current if current is not node else None
        current = children[0]
    return current


def normalize_yaml_key(key_source: str) -> str:
    key = key_source.strip()
    if len(key) >= 2 and key[0] == key[-1] and key[0] in {'"', "'"}:
        return key[1:-1]
    return key


def whole_yaml_document(
    repository_id: str,
    repository_file: RepositoryFile,
) -> list[CodeChunk]:
    chunk = create_whole_file_chunk(
        repository_id, repository_file, "yaml_document"
    )
    return [chunk] if chunk is not None else []
