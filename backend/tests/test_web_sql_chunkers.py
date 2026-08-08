import unittest

from app.models.code_chunk import SourceRange
from app.models.repository_file import RepositoryFile
from app.rag.html_chunker import chunk_html_file
from app.rag.sql_chunker import chunk_sql_file
from app.rag.stylesheet_chunker import chunk_stylesheet_file


def repository_file(language: str, source: str, path: str) -> RepositoryFile:
    return RepositoryFile(
        relative_path=path,
        language=language,
        size_bytes=len(source.encode()),
        content=source,
    )


class HtmlChunkerTests(unittest.TestCase):
    def test_head_and_body_children_become_contextual_elements(self):
        source = """<!doctype html>
<html lang="en">
<head><title>Demo</title></head>
<body>
<header id="top">Header</header>
<main class="app shell"><section>Text</section></main>
<script>run();</script>
</body>
</html>
"""
        chunks = chunk_html_file(
            "repo_html", repository_file("HTML", source, "index.html")
        )
        self.assertEqual(
            [chunk.symbol_name for chunk in chunks],
            [
                "html > head",
                "html > body > header#top",
                "html > body > main.app.shell",
                "html > body > script",
            ],
        )
        self.assertEqual(
            chunks[1].source_ranges,
            (SourceRange(2, 2), SourceRange(4, 4), SourceRange(5, 5)),
        )
        self.assertIn('<html lang="en">', chunks[1].content)
        self.assertIn("<body>", chunks[1].content)

    def test_fragment_element_and_invalid_html_fallback(self):
        fragment = chunk_html_file(
            "repo_html",
            repository_file("HTML", '<section id="intro">Text</section>', "part.html"),
        )
        self.assertEqual(fragment[0].symbol_name, "section#intro")

        invalid = chunk_html_file(
            "repo_html", repository_file("HTML", "<main>", "broken.html")
        )
        self.assertEqual(invalid[0].chunk_type, "html_document")


class StylesheetChunkerTests(unittest.TestCase):
    def test_css_top_level_rules_and_at_rules(self):
        source = """@import url("base.css");
:root { --brand: blue; }
.card, .panel { color: red; }
@media (max-width: 600px) { .card { display: none; } }
@keyframes fade { from { opacity: 0; } to { opacity: 1; } }
"""
        chunks = chunk_stylesheet_file(
            "repo_css", repository_file("CSS", source, "styles.css")
        )
        self.assertEqual(
            [chunk.chunk_type for chunk in chunks],
            [
                "css_import",
                "css_rule",
                "css_rule",
                "css_media_query",
                "css_keyframes",
            ],
        )
        self.assertEqual(chunks[1].symbol_name, ":root")
        self.assertEqual(chunks[2].symbol_name, ".card, .panel")
        self.assertEqual(chunks[4].symbol_name, "fade")

    def test_scss_variables_mixins_functions_and_nested_rules(self):
        source = """$brand: blue;
@mixin button($color) { color: $color; }
@function spacing($n) { @return $n * 4px; }
.card { color: $brand; &:hover { opacity: .8; } }
"""
        chunks = chunk_stylesheet_file(
            "repo_scss", repository_file("SCSS", source, "styles.scss")
        )
        self.assertEqual(
            [(chunk.chunk_type, chunk.symbol_name) for chunk in chunks],
            [
                ("scss_variable", "$brand"),
                ("scss_mixin", "button"),
                ("scss_function", "spacing"),
                ("css_rule", ".card"),
            ],
        )


class SqlChunkerTests(unittest.TestCase):
    def test_top_level_sql_statements_are_semantic_chunks(self):
        source = """CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(100));
CREATE VIEW active_users AS SELECT * FROM users WHERE active = true;
INSERT INTO users (id, name) VALUES (1, 'Ada');
SELECT id, name FROM users;
CREATE FUNCTION user_count() RETURNS INT AS $$ SELECT COUNT(*) FROM users; $$ LANGUAGE SQL;
"""
        chunks = chunk_sql_file(
            "repo_sql", repository_file("SQL", source, "schema.sql")
        )
        self.assertEqual(
            [(chunk.chunk_type, chunk.symbol_name) for chunk in chunks],
            [
                ("sql_create_table", "users"),
                ("sql_create_view", "active_users"),
                ("sql_insert", "users"),
                ("sql_select", None),
                ("sql_create_function", "user_count"),
            ],
        )
        self.assertEqual(chunks[0].source_ranges, (SourceRange(1, 1),))

    def test_unrecognized_or_invalid_sql_falls_back_safely(self):
        chunks = chunk_sql_file(
            "repo_sql", repository_file("SQL", "THIS IS NOT SQL", "broken.sql")
        )
        self.assertEqual(len(chunks), 1)
        self.assertIn(chunks[0].chunk_type, {"sql_statement", "sql_document"})


if __name__ == "__main__":
    unittest.main()
