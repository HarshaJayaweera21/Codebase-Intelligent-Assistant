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


HTML_ELEMENT_TYPES = {"element", "script_element", "style_element"}


def chunk_html_file(
    repository_id: str,
    repository_file: RepositoryFile,
) -> list[CodeChunk]:
    if not repository_file.content.strip():
        return []

    source_bytes = repository_file.content.encode("utf-8")
    try:
        tree = get_parser("html").parse(source_bytes)
    except LanguagePackError:
        return []

    if tree.root_node.has_error:
        return whole_html_document(repository_id, repository_file)

    root_elements = [
        child
        for child in tree.root_node.named_children
        if child.type in HTML_ELEMENT_TYPES
    ]
    if not root_elements:
        return whole_html_document(repository_id, repository_file)

    chunks: list[CodeChunk] = []
    for root_element in root_elements:
        if get_tag_name(root_element, source_bytes).lower() != "html":
            chunks.append(
                create_html_element_chunk(
                    root_element,
                    (),
                    (),
                    repository_id,
                    repository_file,
                    source_bytes,
                )
            )
            continue

        html_descriptor = describe_element(root_element, source_bytes)
        html_start_tag = get_start_tag(root_element)
        html_context = (html_start_tag,) if html_start_tag is not None else ()

        for child in direct_element_children(root_element):
            child_tag = get_tag_name(child, source_bytes).lower()
            if child_tag != "body":
                chunks.append(
                    create_html_element_chunk(
                        child,
                        (html_descriptor,),
                        html_context,
                        repository_id,
                        repository_file,
                        source_bytes,
                    )
                )
                continue

            body_children = direct_element_children(child)
            if not body_children:
                chunks.append(
                    create_html_element_chunk(
                        child,
                        (html_descriptor,),
                        html_context,
                        repository_id,
                        repository_file,
                        source_bytes,
                    )
                )
                continue

            body_descriptor = describe_element(child, source_bytes)
            body_start_tag = get_start_tag(child)
            body_context = html_context + (
                (body_start_tag,) if body_start_tag is not None else ()
            )
            for body_child in body_children:
                chunks.append(
                    create_html_element_chunk(
                        body_child,
                        (html_descriptor, body_descriptor),
                        body_context,
                        repository_id,
                        repository_file,
                        source_bytes,
                    )
                )

    return chunks or whole_html_document(repository_id, repository_file)


def create_html_element_chunk(
    node: Node,
    ancestor_names: tuple[str, ...],
    context_nodes: tuple[Node, ...],
    repository_id: str,
    repository_file: RepositoryFile,
    source_bytes: bytes,
) -> CodeChunk:
    descriptor = describe_element(node, source_bytes)
    return create_node_chunk(
        node=node,
        repository_id=repository_id,
        repository_file=repository_file,
        source_bytes=source_bytes,
        chunk_type="html_element",
        symbol_name=" > ".join((*ancestor_names, descriptor)),
        context_nodes=context_nodes,
    )


def direct_element_children(node: Node) -> list[Node]:
    return [
        child for child in node.named_children if child.type in HTML_ELEMENT_TYPES
    ]


def get_start_tag(node: Node) -> Node | None:
    return next(
        (
            child
            for child in node.named_children
            if child.type in {"start_tag", "self_closing_tag"}
        ),
        None,
    )


def get_tag_name(node: Node, source_bytes: bytes) -> str:
    start_tag = get_start_tag(node)
    if start_tag is None:
        return node.type.removesuffix("_element")

    tag_name = next(
        (child for child in start_tag.named_children if child.type == "tag_name"),
        None,
    )
    return node_text(tag_name, source_bytes) if tag_name is not None else "element"


def describe_element(node: Node, source_bytes: bytes) -> str:
    tag_name = get_tag_name(node, source_bytes)
    start_tag = get_start_tag(node)
    if start_tag is None:
        return tag_name

    attributes: dict[str, str] = {}
    for attribute in start_tag.named_children:
        if attribute.type != "attribute":
            continue

        children = attribute.named_children
        if not children:
            continue

        name = node_text(children[0], source_bytes)
        value = (
            node_text(children[1], source_bytes).strip('"\'')
            if len(children) > 1
            else ""
        )
        attributes[name.lower()] = value

    descriptor = tag_name
    if attributes.get("id"):
        descriptor += f"#{attributes['id']}"
    if attributes.get("class"):
        classes = ".".join(attributes["class"].split())
        if classes:
            descriptor += f".{classes}"
    return descriptor


def whole_html_document(
    repository_id: str,
    repository_file: RepositoryFile,
) -> list[CodeChunk]:
    chunk = create_whole_file_chunk(
        repository_id, repository_file, "html_document"
    )
    return [chunk] if chunk is not None else []
