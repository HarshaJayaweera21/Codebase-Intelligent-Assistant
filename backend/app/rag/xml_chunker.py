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


def chunk_xml_file(
    repository_id: str,
    repository_file: RepositoryFile,
) -> list[CodeChunk]:
    if not repository_file.content.strip():
        return []

    source_bytes = repository_file.content.encode("utf-8")
    try:
        tree = get_parser("xml").parse(source_bytes)
    except LanguagePackError:
        return []

    root_element = tree.root_node.child_by_field_name("root")
    if root_element is None or tree.root_node.has_error:
        return whole_xml_document(repository_id, repository_file)

    root_name = get_element_name(root_element, source_bytes)
    root_start_tag = next(
        (
            child
            for child in root_element.named_children
            if child.type in {"STag", "EmptyElemTag"}
        ),
        None,
    )
    content_node = next(
        (child for child in root_element.named_children if child.type == "content"),
        None,
    )
    if content_node is None:
        return whole_xml_document(repository_id, repository_file)

    child_elements = [
        child for child in content_node.named_children if child.type == "element"
    ]
    if not child_elements:
        return whole_xml_document(repository_id, repository_file)

    context_nodes = (root_start_tag,) if root_start_tag is not None else ()
    return [
        create_node_chunk(
            node=element_node,
            repository_id=repository_id,
            repository_file=repository_file,
            source_bytes=source_bytes,
            chunk_type="xml_element",
            symbol_name=" > ".join(
                name
                for name in (
                    root_name,
                    get_element_name(element_node, source_bytes),
                )
                if name
            ),
            context_nodes=context_nodes,
        )
        for element_node in child_elements
    ]


def get_element_name(node: Node, source_bytes: bytes) -> str | None:
    tag_node = next(
        (
            child
            for child in node.named_children
            if child.type in {"STag", "EmptyElemTag"}
        ),
        None,
    )
    if tag_node is None:
        return None

    name_node = next(
        (child for child in tag_node.named_children if child.type == "Name"),
        None,
    )
    return node_text(name_node, source_bytes) if name_node is not None else None


def whole_xml_document(
    repository_id: str,
    repository_file: RepositoryFile,
) -> list[CodeChunk]:
    chunk = create_whole_file_chunk(
        repository_id, repository_file, "xml_document"
    )
    return [chunk] if chunk is not None else []
