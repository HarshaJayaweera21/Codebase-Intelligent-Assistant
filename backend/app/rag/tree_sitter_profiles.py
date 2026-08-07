from dataclasses import dataclass


@dataclass(frozen=True)
class LanguageProfile:
    parser_name: str

    node_types: dict[str, str]

    # Node types where we want context only,
    # not the entire body duplicated.
    context_node_types: frozenset[str]


LANGUAGE_PROFILES: dict[str, LanguageProfile] = {

    # ------------------------------------------------------------------
    # Python
    # ------------------------------------------------------------------
    "Python": LanguageProfile(
        parser_name="python",
        node_types={
            "class_definition": "class_context",
            "function_definition": "function",
        },
        context_node_types=frozenset({
            "class_definition",
        }),
    ),

    # ------------------------------------------------------------------
    # Java
    # ------------------------------------------------------------------
    "Java": LanguageProfile(
        parser_name="java",
        node_types={
            "class_declaration": "class_context",
            "interface_declaration": "interface_context",
            "method_declaration": "method",
            "constructor_declaration": "constructor",
            "enum_declaration": "enum_context",
            "record_declaration": "record_context",
        },
        context_node_types=frozenset({
            "class_declaration",
            "interface_declaration",
            "enum_declaration",
            "record_declaration",
        }),
    ),

    # ------------------------------------------------------------------
    # JavaScript
    # ------------------------------------------------------------------
    "JavaScript": LanguageProfile(
        parser_name="javascript",
        node_types={
            "class_declaration": "class_context",
            "function_declaration": "function",
            "method_definition": "method",
        },
        context_node_types=frozenset({
            "class_declaration",
        }),
    ),

    "JavaScript JSX": LanguageProfile(
        parser_name="javascript",
        node_types={
            "class_declaration": "class_context",
            "function_declaration": "function",
            "method_definition": "method",
        },
        context_node_types=frozenset({
            "class_declaration",
        }),
    ),

    # ------------------------------------------------------------------
    # TypeScript
    # ------------------------------------------------------------------
    "TypeScript": LanguageProfile(
        parser_name="typescript",
        node_types={
            "class_declaration": "class_context",
            "interface_declaration": "interface_context",
            "function_declaration": "function",
            "method_definition": "method",
            "enum_declaration": "enum_context",
            "type_alias_declaration": "type_alias",
        },
        context_node_types=frozenset({
            "class_declaration",
            "interface_declaration",
            "enum_declaration",
        }),
    ),

    "TypeScript TSX": LanguageProfile(
        parser_name="tsx",
        node_types={
            "class_declaration": "class_context",
            "interface_declaration": "interface_context",
            "function_declaration": "function",
            "method_definition": "method",
            "enum_declaration": "enum_context",
            "type_alias_declaration": "type_alias",
        },
        context_node_types=frozenset({
            "class_declaration",
            "interface_declaration",
            "enum_declaration",
        }),
    ),

    # ------------------------------------------------------------------
    # C
    # ------------------------------------------------------------------
    "C": LanguageProfile(
        parser_name="c",
        node_types={
            "function_definition": "function",
            "struct_specifier": "struct_context",
            "union_specifier": "union_context",
            "enum_specifier": "enum_context",
        },
        context_node_types=frozenset({
            "struct_specifier",
            "union_specifier",
            "enum_specifier",
        }),
    ),

    "C Header": LanguageProfile(
        parser_name="c",
        node_types={
            "function_definition": "function",
            "struct_specifier": "struct_context",
            "union_specifier": "union_context",
            "enum_specifier": "enum_context",
        },
        context_node_types=frozenset({
            "struct_specifier",
            "union_specifier",
            "enum_specifier",
        }),
    ),

    # ------------------------------------------------------------------
    # C++
    # ------------------------------------------------------------------
    "C++": LanguageProfile(
        parser_name="cpp",
        node_types={
            "function_definition": "function",
            "class_specifier": "class_context",
            "struct_specifier": "struct_context",
            "namespace_definition": "namespace_context",
            "enum_specifier": "enum_context",
        },
        context_node_types=frozenset({
            "class_specifier",
            "struct_specifier",
            "namespace_definition",
            "enum_specifier",
        }),
    ),

    "C++ Header": LanguageProfile(
        parser_name="cpp",
        node_types={
            "function_definition": "function",
            "class_specifier": "class_context",
            "struct_specifier": "struct_context",
            "namespace_definition": "namespace_context",
            "enum_specifier": "enum_context",
        },
        context_node_types=frozenset({
            "class_specifier",
            "struct_specifier",
            "namespace_definition",
            "enum_specifier",
        }),
    ),

    # ------------------------------------------------------------------
    # C#
    # ------------------------------------------------------------------
    "C#": LanguageProfile(
        parser_name="c_sharp",
        node_types={
            "class_declaration": "class_context",
            "interface_declaration": "interface_context",
            "struct_declaration": "struct_context",
            "enum_declaration": "enum_context",
            "method_declaration": "method",
            "constructor_declaration": "constructor",
        },
        context_node_types=frozenset({
            "class_declaration",
            "interface_declaration",
            "struct_declaration",
            "enum_declaration",
        }),
    ),

    # ------------------------------------------------------------------
    # Go
    # ------------------------------------------------------------------
    "Go": LanguageProfile(
        parser_name="go",
        node_types={
            "function_declaration": "function",
            "method_declaration": "method",
            "type_declaration": "type",
        },
        context_node_types=frozenset(),
    ),

    # ------------------------------------------------------------------
    # Rust
    # ------------------------------------------------------------------
    "Rust": LanguageProfile(
        parser_name="rust",
        node_types={
            "function_item": "function",
            "struct_item": "struct_context",
            "enum_item": "enum_context",
            "impl_item": "impl_context",
            "trait_item": "trait_context",
            "mod_item": "module_context",
        },
        context_node_types=frozenset({
            "struct_item",
            "enum_item",
            "impl_item",
            "trait_item",
            "mod_item",
        }),
    ),

    # ------------------------------------------------------------------
    # PHP
    # ------------------------------------------------------------------
    "PHP": LanguageProfile(
        parser_name="php",
        node_types={
            "class_declaration": "class_context",
            "interface_declaration": "interface_context",
            "trait_declaration": "trait_context",
            "function_definition": "function",
            "method_declaration": "method",
        },
        context_node_types=frozenset({
            "class_declaration",
            "interface_declaration",
            "trait_declaration",
        }),
    ),

    # ------------------------------------------------------------------
    # Ruby
    # ------------------------------------------------------------------
    "Ruby": LanguageProfile(
        parser_name="ruby",
        node_types={
            "class": "class_context",
            "module": "module_context",
            "method": "method",
            "singleton_method": "method",
        },
        context_node_types=frozenset({
            "class",
            "module",
        }),
    ),

    # ------------------------------------------------------------------
    # Kotlin
    # ------------------------------------------------------------------
    "Kotlin": LanguageProfile(
        parser_name="kotlin",
        node_types={
            "class_declaration": "class_context",
            "object_declaration": "object_context",
            "function_declaration": "function",
        },
        context_node_types=frozenset({
            "class_declaration",
            "object_declaration",
        }),
    ),

    "Kotlin Script": LanguageProfile(
        parser_name="kotlin",
        node_types={
            "class_declaration": "class_context",
            "object_declaration": "object_context",
            "function_declaration": "function",
        },
        context_node_types=frozenset({
            "class_declaration",
            "object_declaration",
        }),
    ),

    # ------------------------------------------------------------------
    # Swift
    # ------------------------------------------------------------------
    "Swift": LanguageProfile(
        parser_name="swift",
        node_types={
            "class_declaration": "class_context",
            "protocol_declaration": "protocol_context",
            "struct_declaration": "struct_context",
            "enum_declaration": "enum_context",
            "function_declaration": "function",
        },
        context_node_types=frozenset({
            "class_declaration",
            "protocol_declaration",
            "struct_declaration",
            "enum_declaration",
        }),
    ),

    # ------------------------------------------------------------------
    # Shell / Bash
    # ------------------------------------------------------------------
    "Shell": LanguageProfile(
        parser_name="bash",
        node_types={
            "function_definition": "function",
        },
        context_node_types=frozenset(),
    ),
}