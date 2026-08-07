from tree_sitter import Language, Parser
import tree_sitter_java

from app.models.code_chunk import CodeChunk
from app.models.repository_file import RepositoryFile


JAVA_LANGUAGE = Language(tree_sitter_java.language())

JAVA_PARSER = Parser(JAVA_LANGUAGE)

JAVA_CHUNK_TYPES = {
    "class_declaration": "class_context",
    "interface_declaration": "interface",
    "method_declaration": "method",
    "constructor_declaration": "constructor",
    "enum_declaration": "enum",
    "record_declaration": "record",
}

def parse_java_file(
    repository_file: RepositoryFile,
):
    source_bytes = repository_file.content.encode("utf-8")

    tree = JAVA_PARSER.parse(source_bytes)

    return tree, source_bytes

def print_java_tree(
    repository_file: RepositoryFile,
) -> None:
    tree, _ = parse_java_file(repository_file)

    print(tree.root_node)

def get_node_text(
    node,
    source_bytes: bytes,
) -> str:
    return source_bytes[
        node.start_byte:node.end_byte
    ].decode("utf-8")

def get_symbol_name(
    node,
    source_bytes: bytes,
) -> str | None:
    name_node = node.child_by_field_name("name")

    if name_node is None:
        return None

    return get_node_text(
        name_node,
        source_bytes,
    )

def collect_java_nodes(node) -> list:
    nodes = []

    if node.type in JAVA_CHUNK_TYPES:
        nodes.append(node)

    for child in node.children:
        nodes.extend(
            collect_java_nodes(child)
        )

    return nodes


def chunk_java_file(
    repository_id: str,
    repository_file: RepositoryFile,
) -> list[CodeChunk]:
    tree, source_bytes = parse_java_file(
        repository_file
    )

    nodes = collect_java_nodes(
        tree.root_node
    )

    chunks: list[CodeChunk] = []

    for node in nodes:
        if node.type == "class_declaration":
            content = get_java_class_context(
                node,
                source_bytes,
            )
        else:
            content = get_node_text(
                node,
                source_bytes,
            )

        symbol_name = get_symbol_name(
            node,
            source_bytes,
        )

        chunk = CodeChunk(
            repository_id=repository_id,
            file_path=repository_file.relative_path,
            language="Java",
            chunk_type=JAVA_CHUNK_TYPES[node.type],
            symbol_name=symbol_name,
            start_line=node.start_point.row + 1,
            end_line=node.end_point.row + 1,
            content=content,
        )

        chunks.append(chunk)

    return chunks


def get_java_class_context(
    node,
    source_bytes: bytes,
) -> str:
    body_node = node.child_by_field_name("body")

    if body_node is None:
        return get_node_text(node, source_bytes)

    # Keep everything from the beginning of the class
    # until and including the opening "{".
    header = source_bytes[
        node.start_byte:body_node.start_byte + 1
    ].decode("utf-8")

    class_members: list[str] = []

    for child in body_node.named_children:
        if child.type == "field_declaration":
            class_members.append(
                get_node_text(
                    child,
                    source_bytes,
                )
            )

    parts = [header.rstrip()]

    if class_members:
        parts.append(
            "\n\n".join(class_members)
        )

    parts.append("}")

    return "\n".join(parts)