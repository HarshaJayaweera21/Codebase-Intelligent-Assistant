import unittest

from app.models.repository_file import RepositoryFile
from app.rag.tree_sitter_chunker import chunk_code_file


def chunk(language: str, source: str):
    return chunk_code_file(
        repository_id="repo_test",
        repository_file=RepositoryFile(
            relative_path="src/example.txt",
            language=language,
            size_bytes=len(source.encode()),
            content=source,
        ),
    )


class TreeSitterChunkerTests(unittest.TestCase):
    def assert_symbols(self, language, source, expected):
        chunks = chunk(language, source)
        actual = {(item.chunk_type, item.symbol_name) for item in chunks}
        self.assertTrue(expected.issubset(actual), (language, expected - actual, actual))
        self.assertEqual(len(chunks), len(set(chunks)), language)
        self.assertEqual(
            [item.symbol_start_line for item in chunks],
            sorted(item.symbol_start_line for item in chunks),
            language,
        )

        for item in chunks:
            self.assertEqual(item.repository_id, "repo_test")
            self.assertEqual(item.file_path, "src/example.txt")
            self.assertEqual(item.language, language)
            self.assertTrue(item.source_ranges)

    def test_python(self):
        self.assert_symbols(
            "Python",
            "class Service:\n    value = 1\n    def __init__(self): pass\n    async def run(self):\n        def helper(): pass\n        return helper()\n",
            {
                ("class_context", "Service"),
                ("constructor", "__init__"),
                ("method", "run"),
                ("function", "helper"),
            },
        )

    def test_java(self):
        self.assert_symbols(
            "Java",
            "record User(int id) {} interface Api { void run(); } class App { App() {} void go() {} }",
            {
                ("record_context", "User"),
                ("interface_context", "Api"),
                ("class_context", "App"),
                ("constructor", "App"),
                ("method", "go"),
            },
        )

    def test_javascript_and_jsx_top_level_declarations(self):
        source = """export const App = () => <View />;
const schema = new Schema({x: 1});
const login = function () { return true; };
class User { constructor() {} save() { const local = {}; } }
"""
        for language in ("JavaScript", "JavaScript JSX"):
            with self.subTest(language=language):
                chunks = chunk(language, source)
                actual = {(item.chunk_type, item.symbol_name) for item in chunks}
                self.assertIn(("function", "App"), actual)
                self.assertIn(("declaration", "schema"), actual)
                self.assertIn(("function", "login"), actual)
                self.assertIn(("constructor", "constructor"), actual)
                self.assertIn(("method", "save"), actual)
                self.assertNotIn(("declaration", "local"), actual)
                self.assertEqual(sum(item.symbol_name == "App" for item in chunks), 1)

    def test_typescript_and_tsx(self):
        sources = {
            "TypeScript": "interface User { id: number } type ID = string; export const run = (): void => {}; class A { constructor() {} m(): void {} }",
            "TypeScript TSX": "interface Props { id: number } export const Card = () => <View />;",
        }
        expected = {
            "TypeScript": {
                ("interface_context", "User"),
                ("type_alias", "ID"),
                ("function", "run"),
                ("constructor", "constructor"),
                ("method", "m"),
            },
            "TypeScript TSX": {
                ("interface_context", "Props"),
                ("function", "Card"),
            },
        }
        for language, source in sources.items():
            with self.subTest(language=language):
                self.assert_symbols(language, source, expected[language])

    def test_c_and_cpp(self):
        self.assert_symbols(
            "C",
            "struct User { int id; }; enum Role { ADMIN }; int run(int x) { return x; }",
            {
                ("struct_context", "User"),
                ("enum_context", "Role"),
                ("function", "run"),
            },
        )
        self.assert_symbols(
            "C++",
            "namespace api { class User { public: User() {} void run() {} }; struct Data { int x; }; }",
            {
                ("namespace_context", "api"),
                ("class_context", "User"),
                ("constructor", "User"),
                ("method", "run"),
                ("struct_context", "Data"),
            },
        )

    def test_csharp(self):
        self.assert_symbols(
            "C#",
            "namespace App; public record User(int Id); interface IRun { void Run(); } class A { A() {} void Go() {} }",
            {
                ("namespace_context", "App"),
                ("record_context", "User"),
                ("interface_context", "IRun"),
                ("class_context", "A"),
                ("constructor", "A"),
                ("method", "Go"),
            },
        )

    def test_go(self):
        self.assert_symbols(
            "Go",
            "type User struct { ID int }; type Runner interface { Run() }; func NewUser() User { return User{} }; func (u User) Run() {}",
            {
                ("struct_context", "User"),
                ("interface_context", "Runner"),
                ("function", "NewUser"),
                ("method", "Run"),
            },
        )

    def test_rust(self):
        self.assert_symbols(
            "Rust",
            "mod api { trait Run { fn run(&self); } struct User { id: i32 } enum Role { Admin } impl Run for User { fn run(&self) {} } }",
            {
                ("module_context", "api"),
                ("trait_context", "Run"),
                ("struct_context", "User"),
                ("enum_context", "Role"),
                ("impl_context", "User"),
                ("method", "run"),
            },
        )

    def test_php_and_ruby(self):
        self.assert_symbols(
            "PHP",
            "<?php namespace App; trait T { public function t() {} } interface I { public function x(); } class A { public function __construct() {} public function x() {} } function top() {}",
            {
                ("namespace_context", "App"),
                ("trait_context", "T"),
                ("interface_context", "I"),
                ("class_context", "A"),
                ("constructor", "__construct"),
                ("function", "top"),
            },
        )
        self.assert_symbols(
            "Ruby",
            "class User\n def initialize; end\n def run; end\n def self.build; end\nend",
            {
                ("class_context", "User"),
                ("constructor", "initialize"),
                ("method", "run"),
                ("method", "build"),
            },
        )

    def test_kotlin(self):
        self.assert_symbols(
            "Kotlin",
            "class User { fun run() {} }\nenum class Role { ADMIN }",
            {
                ("class_context", "User"),
                ("method", "run"),
                ("enum_context", "Role"),
            },
        )

    def test_swift(self):
        self.assert_symbols(
            "Swift",
            "protocol Run { func run() } struct User { init() {} func run() {} } class App { func go() {} } enum Role { case admin } extension User { func extra() {} }",
            {
                ("protocol_context", "Run"),
                ("struct_context", "User"),
                ("constructor", "init"),
                ("class_context", "App"),
                ("enum_context", "Role"),
                ("extension_context", "User"),
                ("method", "extra"),
            },
        )

    def test_shell(self):
        self.assert_symbols(
            "Shell",
            "function build() { echo ok; }\nrun() { echo run; }",
            {("function", "build"), ("function", "run")},
        )

    def test_class_context_does_not_duplicate_method_body(self):
        chunks = chunk(
            "Java",
            "class App { int value = 1; void uniqueMethodBody() { System.out.println(value); } }",
        )
        class_chunk = next(item for item in chunks if item.chunk_type == "class_context")
        self.assertIn("int value = 1", class_chunk.content)
        self.assertNotIn("uniqueMethodBody", class_chunk.content)
        self.assertGreaterEqual(len(class_chunk.source_ranges), 2)


if __name__ == "__main__":
    unittest.main()
