import unittest

from langchain_core.documents import Document

from app.rag.retrieval_text import (
    build_document_embedding_text,
    build_query_embedding_text,
    lexical_relevance,
    minimum_structural_relevance,
    retrieval_relevance,
    tokenize_text,
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

    def test_tokenization_splits_common_code_identifier_styles(self):
        self.assertEqual(
            tokenize_text("CustomOrderQueue add_to_queue order-queue"),
            {"custom", "order", "queue", "add", "to"},
        )

    def test_tokenization_normalizes_common_word_forms(self):
        self.assertEqual(
            tokenize_text("filtering filtered filters prices categories sorting"),
            {"filter", "price", "category", "sort"},
        )

    def test_filter_query_expands_across_frontend_and_backend_terms(self):
        text = build_query_embedding_text(
            "How does the food price filtering work?"
        )

        self.assertIn("input", text)
        self.assertIn("parameter", text)
        self.assertIn("range", text)
        self.assertIn("request", text)
        self.assertIn("slider", text)
        self.assertEqual(
            minimum_structural_relevance(
                "How does the food price filtering work?"
            ),
            0.45,
        )
        self.assertIsNone(
            minimum_structural_relevance("Where is the FoodItem model?")
        )

    def test_implementation_query_boosts_matching_method_and_path(self):
        method = Document(
            page_content=(
                "public boolean addOrder(Order order) { "
                "orders[rear] = order; }"
            ),
            metadata={
                "file_path": "src/model/CustomOrderQueue.java",
                "chunk_type": "method",
                "symbol_name": "addOrder",
            },
        )

        relevance = retrieval_relevance(
            "How is Order queue implemented?",
            method,
        )

        self.assertEqual(relevance.lexical_score, 1.0)
        self.assertEqual(relevance.exact_match_score, 0.9)
        self.assertEqual(relevance.structural_score, 1.0)

    def test_implementation_query_deprioritizes_trivial_getter(self):
        getter = Document(
            page_content=(
                "public CustomOrderQueue getOrderQueue() {\n"
                "    return orderQueue;\n}"
            ),
            metadata={
                "file_path": "src/service/OrderQueueService.java",
                "chunk_type": "method",
                "symbol_name": "getOrderQueue",
            },
        )

        relevance = retrieval_relevance(
            "How is Order queue implemented?",
            getter,
        )

        self.assertEqual(relevance.structural_score, 0.25)

    def test_filter_query_prefers_filter_and_ui_over_sorting(self):
        query = "How does food price filtering work?"
        filter_method = Document(
            page_content="return items.stream().filter(item -> item.getPrice() <= maxPrice);",
            metadata={
                "file_path": "FoodItemService.java",
                "chunk_type": "method",
                "symbol_name": "filterFoodItems",
            },
        )
        sorting_method = Document(
            page_content="quickSort(items, 0, items.size() - 1);",
            metadata={
                "file_path": "FoodItemService.java",
                "chunk_type": "method",
                "symbol_name": "sortFoodItemsByPrice",
            },
        )
        price_slider = Document(
            page_content='<input type="range" name="priceRange" max="500">',
            metadata={
                "file_path": "foods.jsp",
                "chunk_type": "html_document",
                "symbol_name": None,
            },
        )
        unrelated_admin_form = Document(
            page_content=(
                '<input type="number" name="price">'
                '<button class="apply-filter">Add item</button>'
            ),
            metadata={
                "file_path": "adminDashboard.jsp",
                "chunk_type": "html_document",
                "symbol_name": None,
            },
        )

        self.assertEqual(
            retrieval_relevance(query, filter_method).structural_score,
            1.0,
        )
        self.assertEqual(
            retrieval_relevance(query, price_slider).structural_score,
            1.0,
        )
        self.assertEqual(
            retrieval_relevance(query, sorting_method).structural_score,
            0.1,
        )
        self.assertEqual(
            retrieval_relevance(query, unrelated_admin_form).structural_score,
            0.25,
        )


if __name__ == "__main__":
    unittest.main()
