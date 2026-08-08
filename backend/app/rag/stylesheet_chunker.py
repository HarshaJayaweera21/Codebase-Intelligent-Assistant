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


def chunk_stylesheet_file(
    repository_id: str,
    repository_file: RepositoryFile,
) -> list[CodeChunk]:
    if not repository_file.content.strip():
        return []

    parser_name = "scss" if repository_file.language == "SCSS" else "css"
    source_bytes = repository_file.content.encode("utf-8")
    try:
        tree = get_parser(parser_name).parse(source_bytes)
    except LanguagePackError:
        return []

    chunks = [
        create_node_chunk(
            node=node,
            repository_id=repository_id,
            repository_file=repository_file,
            source_bytes=source_bytes,
            chunk_type=classify_stylesheet_node(node, source_bytes),
            symbol_name=get_stylesheet_symbol(node, source_bytes),
        )
        for node in tree.root_node.named_children
        if node.type != "comment"
    ]
    if chunks:
        return chunks

    chunk = create_whole_file_chunk(
        repository_id, repository_file, "stylesheet_document"
    )
    return [chunk] if chunk is not None else []


def classify_stylesheet_node(node: Node, source_bytes: bytes) -> str:
    node_types = {
        "rule_set": "css_rule",
        "media_statement": "css_media_query",
        "keyframes_statement": "css_keyframes",
        "import_statement": "css_import",
        "mixin_statement": "scss_mixin",
        "function_statement": "scss_function",
        "use_statement": "scss_use",
        "forward_statement": "scss_forward",
    }
    if node.type in node_types:
        return node_types[node.type]

    if node.type == "declaration" and node_text(node, source_bytes).lstrip().startswith("$"):
        return "scss_variable"

    if node.type.endswith("statement") or node_text(node, source_bytes).lstrip().startswith("@"):
        return "css_at_rule"

    return "css_declaration"


def get_stylesheet_symbol(node: Node, source_bytes: bytes) -> str | None:
    if node.type == "rule_set":
        selectors = next(
            (child for child in node.named_children if child.type == "selectors"),
            None,
        )
        return node_text(selectors, source_bytes).strip() if selectors else None

    name_node = node.child_by_field_name("name")
    if name_node is not None:
        return node_text(name_node, source_bytes)

    if node.type == "keyframes_statement":
        keyframes_name = next(
            (
                child
                for child in node.named_children
                if child.type == "keyframes_name"
            ),
            None,
        )
        if keyframes_name is not None:
            return node_text(keyframes_name, source_bytes)

    if node.type == "declaration":
        property_node = next(
            (
                child
                for child in node.named_children
                if child.type == "property_name"
            ),
            None,
        )
        return node_text(property_node, source_bytes) if property_node else None

    content = node_text(node, source_bytes).strip()
    header = content.split("{", 1)[0].rstrip(" ;")
    return header or None
