from tree_sitter import Node
from tree_sitter_language_pack import get_parser

from app.models.code_chunk import (
    CodeChunk,
    SourceRange,
)
from app.models.repository_file import RepositoryFile
from app.rag.tree_sitter_profiles import (
    LANGUAGE_PROFILES,
    LanguageProfile,
)

def parse_source(
    repository_file: RepositoryFile,
    profile: LanguageProfile,
):
    parser = get_parser(profile.parser_name)

    source_bytes = repository_file.content.encode("utf-8")

    tree = parser.parse(source_bytes)

    return tree, source_bytes


def get_node_text(
    node: Node,
    source_bytes: bytes,
) -> str:
    return source_bytes[
        node.start_byte:node.end_byte
    ].decode("utf-8")

def get_node_source_range(
    node: Node,
) -> SourceRange:
    return SourceRange(
        start_line=node.start_point.row + 1,
        end_line=node.end_point.row + 1,
    )

def get_symbol_name(
    node: Node,
    source_bytes: bytes,
) -> str | None:
    name_node = node.child_by_field_name("name")

    if name_node is not None:
        return get_node_text(
            name_node,
            source_bytes,
        )

    declarator_node = node.child_by_field_name(
        "declarator"
    )

    if declarator_node is not None:
        identifier = find_identifier(
            declarator_node
        )

        if identifier is not None:
            return get_node_text(
                identifier,
                source_bytes,
            )

    return None

IDENTIFIER_NODE_TYPES = {
    "identifier",
    "field_identifier",
    "type_identifier",
}


def find_identifier(
    node: Node,
) -> Node | None:
    if node.type in IDENTIFIER_NODE_TYPES:
        return node

    for child in node.named_children:
        result = find_identifier(child)

        if result is not None:
            return result

    return None


def collect_structural_nodes(
    node: Node,
    profile: LanguageProfile,
) -> list[Node]:
    result: list[Node] = []

    if node.type in profile.node_types:
        result.append(node)

    for child in node.named_children:
        result.extend(
            collect_structural_nodes(
                child,
                profile,
            )
        )

    return result


def create_context_content(
    node: Node,
    source_bytes: bytes,
    profile: LanguageProfile,
) -> tuple[str, tuple[SourceRange, ...]]:
    body_node = node.child_by_field_name("body")

    if body_node is None:
        return (
            get_node_text(node, source_bytes),
            (get_node_source_range(node),),
        )

    content_parts: list[str] = []
    source_ranges: list[SourceRange] = []

    # ------------------------------------------------------------
    # 1. Keep the declaration/header.
    # ------------------------------------------------------------

    header_bytes = source_bytes[
        node.start_byte:body_node.start_byte
    ]

    header = header_bytes.decode("utf-8").rstrip()

    if header:
        content_parts.append(header)

        header_end_line = max(
            node.start_point.row + 1,
            body_node.start_point.row,
        )

        source_ranges.append(
            SourceRange(
                start_line=node.start_point.row + 1,
                end_line=header_end_line,
            )
        )

    # ------------------------------------------------------------
    # 2. Keep direct class/container members that aren't themselves
    #    separately chunked symbols.
    # ------------------------------------------------------------

    for child in body_node.named_children:

        if child.type in profile.node_types:
            continue

        content = get_node_text(
            child,
            source_bytes,
        ).strip()

        if not content:
            continue

        content_parts.append(content)

        source_ranges.append(
            get_node_source_range(child)
        )

    return (
        "\n\n".join(content_parts),
        tuple(source_ranges),
    )


def chunk_code_file(
    repository_id: str,
    repository_file: RepositoryFile,
) -> list[CodeChunk]:

    profile = LANGUAGE_PROFILES.get(
        repository_file.language
    )

    if profile is None:
        return []

    tree, source_bytes = parse_source(
        repository_file,
        profile,
    )

    nodes = collect_structural_nodes(
        tree.root_node,
        profile,
    )

    chunks: list[CodeChunk] = []

    for node in nodes:

        chunk_type = profile.node_types[
            node.type
        ]

        symbol_name = get_symbol_name(
            node,
            source_bytes,
        )

        # --------------------------------------------------------
        # Structural container: class/interface/struct/etc.
        # --------------------------------------------------------

        if node.type in profile.context_node_types:

            content, source_ranges = (
                create_context_content(
                    node,
                    source_bytes,
                    profile,
                )
            )

        # --------------------------------------------------------
        # Normal function/method/etc.
        # --------------------------------------------------------

        else:
            content = get_node_text(
                node,
                source_bytes,
            )

            source_ranges = (
                get_node_source_range(node),
            )

        if not content.strip():
            continue

        chunks.append(
            CodeChunk(
                repository_id=repository_id,
                file_path=repository_file.relative_path,
                language=repository_file.language,
                chunk_type=chunk_type,
                symbol_name=symbol_name,
                symbol_start_line=(
                    node.start_point.row + 1
                ),
                symbol_end_line=(
                    node.end_point.row + 1
                ),
                source_ranges=source_ranges,
                content=content,
            )
        )

        if repository_file.language in {
            "JavaScript",
            "JavaScript JSX",
            "TypeScript",
            "TypeScript TSX",
        }:
            variable_chunks = (
                extract_top_level_variable_chunks(
                    root_node=tree.root_node,
                    repository_id=repository_id,
                    repository_file=repository_file,
                    source_bytes=source_bytes,
                )
            )

            chunks.extend(variable_chunks)

    return remove_duplicate_chunks(
        chunks
    )

    return chunks


def is_top_level_declaration(node: Node) -> bool:
    parent = node.parent

    if parent is None:
        return False

    if parent.type in {
        "program",
        "source_file",
    }:
        return True

    # Handles:
    #
    # export const foo = ...
    # export default ...
    #
    if parent.type in {
        "export_statement",
    }:
        grandparent = parent.parent

        return (
            grandparent is not None
            and grandparent.type
            in {
                "program",
                "source_file",
            }
        )

    return False


def extract_top_level_variable_chunks(
    root_node: Node,
    repository_id: str,
    repository_file: RepositoryFile,
    source_bytes: bytes,
) -> list[CodeChunk]:
    chunks: list[CodeChunk] = []

    for child in root_node.named_children:

        declaration_node = child

        # Handle:
        # export const foo = ...
        if child.type == "export_statement":
            declaration_node = next(
                (
                    nested_child
                    for nested_child
                    in child.named_children
                    if nested_child.type
                    in {
                        "lexical_declaration",
                        "variable_declaration",
                    }
                ),
                None,
            )

            if declaration_node is None:
                continue

        if declaration_node.type not in {
            "lexical_declaration",
            "variable_declaration",
        }:
            continue

        for declarator in declaration_node.named_children:

            if declarator.type != "variable_declarator":
                continue

            chunk = create_variable_declaration_chunk(
                repository_id=repository_id,
                repository_file=repository_file,
                declaration_node=declaration_node,
                declarator=declarator,
                source_bytes=source_bytes,
            )

            if chunk is not None:
                chunks.append(chunk)

    return chunks


def create_variable_declaration_chunk(
    repository_id: str,
    repository_file: RepositoryFile,
    declaration_node: Node,
    declarator: Node,
    source_bytes: bytes,
) -> CodeChunk | None:

    name_node = declarator.child_by_field_name(
        "name"
    )

    value_node = declarator.child_by_field_name(
        "value"
    )

    if name_node is None or value_node is None:
        return None

    symbol_name = get_node_text(
        name_node,
        source_bytes,
    )

    chunk_type = classify_variable_declaration(
        value_node
    )

    # Some declarations are not useful enough
    # to create their own RAG chunk.
    if chunk_type is None:
        return None

    content = get_node_text(
        declaration_node,
        source_bytes,
    )

    return CodeChunk(
        repository_id=repository_id,
        file_path=repository_file.relative_path,
        language=repository_file.language,
        chunk_type=chunk_type,
        symbol_name=symbol_name,

        symbol_start_line=(
            declaration_node.start_point.row + 1
        ),
        symbol_end_line=(
            declaration_node.end_point.row + 1
        ),

        source_ranges=(
            get_node_source_range(
                declaration_node
            ),
        ),

        content=content,
    )


def classify_variable_declaration(
    value_node: Node,
) -> str | None:

    if value_node.type in {
        "arrow_function",
        "function_expression",
        "function",
    }:
        return "function"

    if value_node.type in {
        "object",
        "object_pattern",
        "array",
    }:
        return "declaration"

    if value_node.type in {
        "call_expression",
        "new_expression",
    }:
        return "declaration"

    return None


def remove_duplicate_chunks(
    chunks: list[CodeChunk],
) -> list[CodeChunk]:

    unique_chunks: list[CodeChunk] = []
    seen: set[
        tuple[str, int, int]
    ] = set()

    for chunk in chunks:

        key = (
            chunk.chunk_type,
            chunk.symbol_start_line,
            chunk.symbol_end_line,
        )

        if key in seen:
            continue

        seen.add(key)
        unique_chunks.append(chunk)

    unique_chunks = remove_duplicate_chunks(
        chunks
    )

    return sorted(
        unique_chunks,
        key=lambda chunk: (
            chunk.symbol_start_line,
            chunk.symbol_end_line,
        ),
    )