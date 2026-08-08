import unittest

from langchain_core.embeddings import Embeddings

from app.embeddings.embedding_service import (
    EmbeddingValidationError,
    run_embedding_smoke_test,
    validate_embedding_vectors,
    validate_pinecone_dimension,
)


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(index), 0.5] for index, _ in enumerate(texts)]

    def embed_query(self, text: str) -> list[float]:
        return [0.25, 0.75]


class EmbeddingValidationTests(unittest.TestCase):
    def test_smoke_test_checks_query_and_multiple_documents(self):
        result = run_embedding_smoke_test(
            FakeEmbeddings(),
            expected_dimension=2,
            query="question",
            documents=["first", "second"],
        )

        self.assertTrue(result.model_loaded)
        self.assertEqual(result.query_dimension, 2)
        self.assertEqual(result.document_dimensions, [2, 2])
        self.assertTrue(result.all_values_are_floats)

    def test_dimension_validation_mentions_pinecone(self):
        with self.assertRaisesRegex(
            EmbeddingValidationError,
            "Pinecone index dimension",
        ):
            validate_embedding_vectors(
                [[0.1]],
                expected_count=1,
                expected_dimension=2,
            )

    def test_pinecone_dimension_must_match_model(self):
        with self.assertRaisesRegex(
            EmbeddingValidationError,
            "before index creation or vector upsert",
        ):
            validate_pinecone_dimension(
                embedding_dimension=2560,
                pinecone_dimension=1024,
            )


if __name__ == "__main__":
    unittest.main()
