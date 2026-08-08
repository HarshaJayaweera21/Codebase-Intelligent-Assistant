from functools import lru_cache

from tree_sitter import Node, Query, QueryCursor
from tree_sitter_language_pack import Error as LanguagePackError
from tree_sitter_language_pack import get_language, get_parser

from app.models.code_chunk import CodeChunk, SourceRange
from app.models.repository_file import RepositoryFile
from app.rag.tree_sitter_profiles import LANGUAGE_PROFILES, LanguageProfile


IDENTIFIER_NODE_TYPES = {
    "constant",
    "field_identifier",
    "identifier",
    "name",
    "namespace_identifier",
    "property_identifier",
    "simple_identifier",
    "type_identifier",
    "word",
}

CONTEXT_BODY_NODE_TYPES = {
    "body_statement",
    "class_body",
    "declaration_list",
    "enum_class_body",
    "enum_variant_list",
    "enumerator_list",
    "field_declaration_list",
    "interface_body",
    "protocol_body",
}

JAVASCRIPT_LANGUAGES = {
    "JavaScript",
    "JavaScript JSX",
    "TypeScript",
    "TypeScript TSX",
}

TOP_LEVEL_VARIABLE_QUERY = r"""
(program
  (lexical_declaration
    (variable_declarator
      name: (identifier) @name
      value: [(arrow_function) (function_expression)
              (call_expression) (new_expression) (object) (array)] @value)
  ) @content)

(program
  (variable_declaration
    (variable_declarator
      name: (identifier) @name
      value: [(arrow_function) (function_expression)
              (call_expression) (new_expression) (object) (array)] @value)
  ) @content)

(program
  (export_statement
    declaration: (lexical_declaration
      (variable_declarator
        name: (identifier) @name
        value: [(arrow_function) (function_expression)
                (call_expression) (new_expression) (object) (array)] @value)
    )
  ) @content)

(program
  (export_statement
    declaration: (variable_declaration
      (variable_declarator
        name: (identifier) @name
        value: [(arrow_function) (function_expression)
                (call_expression) (new_expression) (object) (array)] @value)
    )
  ) @content)
"""


@lru_cache(maxsize=3)
def get_top_level_variable_query(parser_name: str) -> Query:
    """Compile once per JS/TS grammar and reuse across repository files."""
    return Query(get_language(parser_name), TOP_LEVEL_VARIABLE_QUERY)


def parse_source(
    repository_file: RepositoryFile,
    profile: LanguageProfile,
):
    parser = get_parser(profile.parser_name)
    source_bytes = repository_file.content.encode("utf-8")
    return parser.parse(source_bytes), source_bytes


def get_node_text(node: Node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte : node.end_byte].decode("utf-8")


def get_node_source_range(node: Node) -> SourceRange:
    return SourceRange(
        start_line=node.start_point.row + 1,
        end_line=get_node_end_line(node),
    )


def get_node_end_line(node: Node) -> int:
    end_line = node.end_point.row + (1 if node.end_point.column else 0)
    return max(node.start_point.row + 1, end_line)


def find_identifier(node: Node) -> Node | None:
    if node.type in IDENTIFIER_NODE_TYPES:
        return node

    for child in node.named_children:
        identifier = find_identifier(child)
        if identifier is not None:
            return identifier

    return None


def get_symbol_name(node: Node, source_bytes: bytes) -> str | None:
    for field_name in ("name", "declarator", "type"):
        field_node = node.child_by_field_name(field_name)
        if field_node is None:
            continue

        identifier = find_identifier(field_node)
        if identifier is not None:
            return get_node_text(identifier, source_bytes)

    # Some grammars (notably Kotlin) do not expose a `name` field.
    for child in node.named_children:
        if child.type in IDENTIFIER_NODE_TYPES:
            return get_node_text(child, source_bytes)

    # Go type declarations wrap the named type inside a type_spec.
    if node.type == "type_declaration":
        type_spec = next(
            (child for child in node.named_children if child.type == "type_spec"),
            None,
        )
        if type_spec is not None:
            return get_symbol_name(type_spec, source_bytes)

    if node.type in {"init_declaration", "secondary_constructor"}:
        return "init" if node.type == "init_declaration" else "constructor"

    return None


def first_ancestor(node: Node, node_types: set[str]) -> Node | None:
    ancestor = node.parent
    while ancestor is not None:
        if ancestor.type in node_types:
            return ancestor
        ancestor = ancestor.parent
    return None


def has_direct_child_type(node: Node, node_type: str) -> bool:
    return any(child.type == node_type for child in node.children)


def classify_structural_node(
    node: Node,
    profile: LanguageProfile,
    source_bytes: bytes,
) -> str:
    chunk_type = profile.node_types[node.type]

    if node.type == "type_spec":
        declared_type = node.child_by_field_name("type")
        if declared_type is not None:
            return {
                "struct_type": "struct_context",
                "interface_type": "interface_context",
            }.get(declared_type.type, chunk_type)

    if node.type == "class_declaration":
        keyword_types = {
            "struct": "struct_context",
            "enum": "enum_context",
            "interface": "interface_context",
            "extension": "extension_context",
        }
        for keyword, classified_type in keyword_types.items():
            if has_direct_child_type(node, keyword):
                return classified_type

    symbol_name = get_symbol_name(node, source_bytes)

    if node.type == "method_definition" and symbol_name == "constructor":
        return "constructor"

    if node.type == "method_declaration" and symbol_name == "__construct":
        return "constructor"

    if node.type in {"method", "singleton_method"} and symbol_name == "initialize":
        return "constructor"

    if node.type == "function_definition":
        python_scope = first_ancestor(
            node, {"class_definition", "function_definition"}
        )
        if python_scope is not None and python_scope.type == "class_definition":
            return "constructor" if symbol_name == "__init__" else "method"

        cpp_container = first_ancestor(node, {"class_specifier", "struct_specifier"})
        if cpp_container is not None:
            container_name = get_symbol_name(cpp_container, source_bytes)
            return "constructor" if symbol_name == container_name else "method"

    if node.type == "function_item" and first_ancestor(
        node, {"impl_item", "trait_item"}
    ) is not None:
        return "method"

    if node.type == "function_declaration":
        containing_scope = first_ancestor(
            node,
            {
                "function_declaration",
                "method_definition",
                "class_declaration",
                "object_declaration",
                "protocol_declaration",
            },
        )
        if containing_scope is not None and containing_scope.type in {
            "class_declaration",
            "object_declaration",
            "protocol_declaration",
        }:
            return "method"

    return chunk_type


def collect_structural_nodes(node: Node, profile: LanguageProfile) -> list[Node]:
    nodes: list[Node] = []
    if node.type in profile.node_types:
        nodes.append(node)

    for child in node.named_children:
        nodes.extend(collect_structural_nodes(child, profile))

    return nodes


def find_context_body(node: Node) -> Node | None:
    body = node.child_by_field_name("body")
    if body is not None:
        return body

    stack = list(reversed(node.named_children))
    while stack:
        candidate = stack.pop()
        if candidate.type in CONTEXT_BODY_NODE_TYPES:
            return candidate
        stack.extend(reversed(candidate.named_children))

    return None


def contains_separate_symbol(node: Node, profile: LanguageProfile) -> bool:
    if node.type in profile.node_types:
        return True
    return any(
        contains_separate_symbol(child, profile) for child in node.named_children
    )


def create_context_content(
    node: Node,
    source_bytes: bytes,
    profile: LanguageProfile,
) -> tuple[str, tuple[SourceRange, ...]]:
    body_node = find_context_body(node)
    if body_node is None:
        return get_node_text(node, source_bytes), (get_node_source_range(node),)

    content_parts: list[str] = []
    source_ranges: list[SourceRange] = []

    header = source_bytes[node.start_byte : body_node.start_byte].decode("utf-8").rstrip()
    if header:
        content_parts.append(header)
        source_ranges.append(
            SourceRange(
                start_line=node.start_point.row + 1,
                end_line=max(node.start_point.row + 1, body_node.start_point.row),
            )
        )

    for child in body_node.named_children:
        # Wrappers such as decorators and parser recovery ERROR nodes may contain
        # a symbol that is independently chunked.
        if contains_separate_symbol(child, profile):
            continue

        content = get_node_text(child, source_bytes).strip()
        if content:
            content_parts.append(content)
            source_ranges.append(get_node_source_range(child))

    return "\n\n".join(content_parts), tuple(source_ranges)


def chunk_code_file(
    repository_id: str,
    repository_file: RepositoryFile,
) -> list[CodeChunk]:
    profile = LANGUAGE_PROFILES.get(repository_file.language)
    if profile is None:
        return []

    try:
        tree, source_bytes = parse_source(repository_file, profile)
    except LanguagePackError:
        # A missing or unloadable optional grammar should not fail repository
        # ingestion. The router can apply a fallback chunker later.
        return []

    chunks = [
        create_structural_chunk(
            node=node,
            repository_id=repository_id,
            repository_file=repository_file,
            source_bytes=source_bytes,
            profile=profile,
        )
        for node in collect_structural_nodes(tree.root_node, profile)
    ]

    if repository_file.language in JAVASCRIPT_LANGUAGES:
        chunks.extend(
            extract_top_level_variable_chunks(
                root_node=tree.root_node,
                repository_id=repository_id,
                repository_file=repository_file,
                source_bytes=source_bytes,
                parser_name=profile.parser_name,
            )
        )

    return remove_duplicate_chunks([chunk for chunk in chunks if chunk is not None])


def create_structural_chunk(
    node: Node,
    repository_id: str,
    repository_file: RepositoryFile,
    source_bytes: bytes,
    profile: LanguageProfile,
) -> CodeChunk | None:
    chunk_type = classify_structural_node(node, profile, source_bytes)

    if node.type in profile.context_node_types or chunk_type.endswith("_context"):
        content, source_ranges = create_context_content(node, source_bytes, profile)
    else:
        content = get_node_text(node, source_bytes)
        source_ranges = (get_node_source_range(node),)

    if not content.strip():
        return None

    return CodeChunk(
        repository_id=repository_id,
        file_path=repository_file.relative_path,
        language=repository_file.language,
        chunk_type=chunk_type,
        symbol_name=get_symbol_name(node, source_bytes),
        symbol_start_line=node.start_point.row + 1,
        symbol_end_line=get_node_end_line(node),
        source_ranges=source_ranges,
        content=content,
    )


def extract_top_level_variable_chunks(
    root_node: Node,
    repository_id: str,
    repository_file: RepositoryFile,
    source_bytes: bytes,
    parser_name: str,
) -> list[CodeChunk]:
    chunks: list[CodeChunk] = []
    # Query matching finds all candidates in one native traversal. Besides
    # being concise, this avoids instability seen when repeatedly descending
    # very large JavaScript arrow-function subtrees through Node wrappers.
    query_cursor = QueryCursor(get_top_level_variable_query(parser_name))
    for _, captures in query_cursor.matches(root_node):
        content_node = captures["content"][0]
        name_node = captures["name"][0]
        value_node = captures["value"][0]
        chunk = create_variable_declaration_chunk(
            repository_id=repository_id,
            repository_file=repository_file,
            content_node=content_node,
            name_node=name_node,
            value_node=value_node,
            source_bytes=source_bytes,
        )
        if chunk is not None:
            chunks.append(chunk)

    return chunks


def create_variable_declaration_chunk(
    repository_id: str,
    repository_file: RepositoryFile,
    content_node: Node,
    name_node: Node,
    value_node: Node,
    source_bytes: bytes,
) -> CodeChunk | None:
    # Destructuring declarations usually create several local-style bindings
    # and do not have one stable symbol name for retrieval.
    if name_node.type not in {"identifier", "type_identifier"}:
        return None

    chunk_type = classify_variable_declaration(value_node)
    if chunk_type is None:
        return None

    return CodeChunk(
        repository_id=repository_id,
        file_path=repository_file.relative_path,
        language=repository_file.language,
        chunk_type=chunk_type,
        symbol_name=get_node_text(name_node, source_bytes),
        symbol_start_line=content_node.start_point.row + 1,
        symbol_end_line=content_node.end_point.row + 1,
        source_ranges=(get_node_source_range(content_node),),
        content=get_node_text(content_node, source_bytes),
    )


def classify_variable_declaration(value_node: Node) -> str | None:
    if value_node.type in {"arrow_function", "function_expression", "function"}:
        return "function"
    if value_node.type in {"object", "array", "call_expression", "new_expression"}:
        return "declaration"
    return None


def remove_duplicate_chunks(chunks: list[CodeChunk]) -> list[CodeChunk]:
    unique_chunks: list[CodeChunk] = []
    seen: set[tuple[str, str, str | None, int, int]] = set()

    for chunk in chunks:
        key = (
            chunk.file_path,
            chunk.chunk_type,
            chunk.symbol_name,
            chunk.symbol_start_line,
            chunk.symbol_end_line,
        )
        if key not in seen:
            seen.add(key)
            unique_chunks.append(chunk)

    return sorted(
        unique_chunks,
        key=lambda chunk: (
            chunk.symbol_start_line,
            chunk.symbol_end_line,
            chunk.chunk_type,
            chunk.symbol_name or "",
        ),
    )
