import re

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


SQL_CHUNK_TYPES = {
    "create_table": "sql_create_table",
    "create_view": "sql_create_view",
    "create_function": "sql_create_function",
    "create_procedure": "sql_create_procedure",
    "alter_table": "sql_alter_table",
    "drop_table": "sql_drop_table",
    "insert": "sql_insert",
    "update": "sql_update",
    "delete": "sql_delete",
    "select": "sql_select",
}

NAMED_SQL_OPERATIONS = {
    "create_table",
    "create_view",
    "create_function",
    "create_procedure",
    "alter_table",
    "drop_table",
    "insert",
    "update",
    "delete",
}


def chunk_sql_file(
    repository_id: str,
    repository_file: RepositoryFile,
) -> list[CodeChunk]:
    if not repository_file.content.strip():
        return []

    source_bytes = repository_file.content.encode("utf-8")
    try:
        tree = get_parser("sql").parse(source_bytes)
    except LanguagePackError:
        return []

    statement_nodes = [
        child
        for child in tree.root_node.named_children
        if child.type == "statement"
    ]
    chunks: list[CodeChunk] = []
    for statement_node in statement_nodes:
        operation = first_named_child(statement_node)
        operation_type = operation.type if operation is not None else "statement"
        chunks.append(
            create_node_chunk(
                node=statement_node,
                repository_id=repository_id,
                repository_file=repository_file,
                source_bytes=source_bytes,
                chunk_type=SQL_CHUNK_TYPES.get(operation_type, "sql_statement"),
                symbol_name=(
                    get_sql_object_name(operation, source_bytes)
                    if operation_type in NAMED_SQL_OPERATIONS
                    else None
                ),
            )
        )

    if chunks:
        return chunks

    chunk = create_whole_file_chunk(
        repository_id, repository_file, "sql_document"
    )
    return [chunk] if chunk is not None else []


def get_sql_object_name(node: Node | None, source_bytes: bytes) -> str | None:
    if node is None:
        return None

    object_reference = find_descendant(node, "object_reference")
    if object_reference is not None:
        return node_text(object_reference, source_bytes).strip()

    # Conservative fallback for dialect-specific nodes not covered by the
    # common grammar's object_reference structure.
    content = node_text(node, source_bytes)
    match = re.match(
        r"(?is)^\s*(?:CREATE\s+(?:OR\s+REPLACE\s+)?(?:TABLE|VIEW|FUNCTION|PROCEDURE)|"
        r"ALTER\s+TABLE|DROP\s+TABLE|INSERT\s+INTO|UPDATE|DELETE\s+FROM)\s+"
        r"(?:IF\s+(?:NOT\s+)?EXISTS\s+)?([^\s(;]+)",
        content,
    )
    return match.group(1) if match else None


def find_descendant(node: Node, node_type: str) -> Node | None:
    if node.type == node_type:
        return node
    for child in node.named_children:
        result = find_descendant(child, node_type)
        if result is not None:
            return result
    return None


def first_named_child(node: Node) -> Node | None:
    return node.named_child(0) if node.named_child_count else None
