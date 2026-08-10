import unittest

from langchain_core.documents import Document

from app.rag.retrieval_text import (
    build_document_embedding_text,
    build_query_embedding_text,
    lexical_relevance,
)


class RetrievalTextTests(unittest.TestCase):
    def test_document_embedding_text_adds_structure_without_mutation(self):
        document = Document(
            page_content="boolean login(String email, String password) { ... }",
            metadata={
                "file_path": "src/UserService.java",
                "language": "Java",
                "chunk_type": "method",
                "symbol_name": "login",
                "symbol_start_line": 10,
                "symbol_end_line": 12,
            },
        )

        text = build_document_embedding_text(document)

        self.assertIn("File: src/UserService.java", text)
        self.assertIn("Symbol: login", text)
        self.assertTrue(text.endswith(document.page_content))
        self.assertEqual(document.page_content, "boolean login(String email, String password) { ... }")

    def test_authentication_query_expands_to_code_concepts(self):
        text = build_query_embedding_text("Where is authentication implemented?")

        self.assertIn("Question: Where is authentication implemented?", text)
        self.assertIn("login", text)
        self.assertIn("password", text)
        self.assertIn("session", text)

    def test_lexical_relevance_connects_authentication_to_login(self):
        login_document = Document(
            page_content="session.setAttribute(\"user\", user);",
            metadata={
                "file_path": "controller/UserServlet.java",
                "chunk_type": "method",
                "symbol_name": "login",
            },
        )
        unrelated_document = Document(
            page_content="void manageUsers() {}",
            metadata={
                "file_path": "model/Admin.java",
                "chunk_type": "method",
                "symbol_name": "manageUsers",
            },
        )

        self.assertGreater(
            lexical_relevance("Where is authentication implemented?", login_document),
            lexical_relevance("Where is authentication implemented?", unrelated_document),
        )


if __name__ == "__main__":
    unittest.main()
