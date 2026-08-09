import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.core.config import Settings
from app.vectorstores.pinecone_vector_store import (
    PineconeConfigurationError,
    PineconeVectorStore,
    create_pinecone_vector_store,
    create_vector_id,
)


class FakeEmbeddings(Embeddings):
    def __init__(self) -> None:
        self.document_inputs: list[str] = []
        self.query_input: str | None = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_inputs = texts
        return [
            [float(index), float(len(text)), 0.5]
            for index, text in enumerate(texts)
        ]

    def embed_query(self, text: str) -> list[float]:
        self.query_input = text
        return [1.0, float(len(text)), 0.5]


class FakeIndex:
    def __init__(self) -> None:
        self.upserts: list[dict] = []
        self.deletes: list[dict] = []
        self.queries: list[dict] = []
        self.matches: list[SimpleNamespace] = []
        self.closed = False

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)

    def delete(self, **kwargs):
        self.deletes.append(kwargs)

    def query(self, **kwargs):
        self.queries.append(kwargs)
        return SimpleNamespace(matches=self.matches)

    def close(self):
        self.closed = True


class FakeIndexes:
    def __init__(self, *, exists: bool, dimension: int = 3) -> None:
        self._exists = exists
        self.dimension = dimension
        self.created: list[dict] = []

    def exists(self, name: str) -> bool:
        return self._exists

    def create(self, **kwargs):
        self.created.append(kwargs)
        self._exists = True

    def describe(self, name: str):
        return SimpleNamespace(dimension=self.dimension, metric="cosine")


class FakePineconeClient:
    def __init__(self, *, exists: bool, dimension: int = 3) -> None:
        self.indexes = FakeIndexes(exists=exists, dimension=dimension)
        self.data_index = FakeIndex()
        self.requested_index_name: str | None = None
        self.closed = False

    def index(self, name: str):
        self.requested_index_name = name
        return self.data_index

    def close(self):
        self.closed = True


def make_document(content: str, *, repository_id: str = "repo_a1b2c3d4"):
    return Document(
        page_content=content,
        metadata={
            "repository_id": repository_id,
            "file_path": "src/example.py",
            "language": "Python",
            "chunk_type": "function",
            "symbol_name": None,
            "symbol_start_line": 4,
            "symbol_end_line": 8,
            "source_ranges": [{"start_line": 4, "end_line": 8}],
        },
    )


class PineconeVectorStoreTests(unittest.TestCase):
    def setUp(self):
        self.index = FakeIndex()
        self.embeddings = FakeEmbeddings()
        self.store = PineconeVectorStore(
            index=self.index,
            embeddings=self.embeddings,
            dimension=3,
            upsert_batch_size=2,
        )

    def test_indexes_batches_in_repository_namespace(self):
        documents = [make_document(f"content {index}") for index in range(3)]

        count = self.store.index_documents("repo_a1b2c3d4", documents)

        self.assertEqual(count, 3)
        self.assertEqual(
            self.index.deletes,
            [{"delete_all": True, "namespace": "repo_a1b2c3d4"}],
        )
        self.assertEqual([len(call["vectors"]) for call in self.index.upserts], [2, 1])
        self.assertTrue(
            all(
                call["namespace"] == "repo_a1b2c3d4"
                for call in self.index.upserts
            )
        )
        metadata = self.index.upserts[0]["vectors"][0]["metadata"]
        self.assertNotIn("symbol_name", metadata)
        self.assertIn("document_metadata_json", metadata)
        self.assertEqual(metadata["content"], "content 0")
        self.assertIn("File: src/example.py", self.embeddings.document_inputs[0])
        self.assertIn("Chunk type: function", self.embeddings.document_inputs[0])
        self.assertTrue(self.embeddings.document_inputs[0].endswith("content 0"))

    def test_search_uses_same_namespace_and_restores_exact_metadata(self):
        document = make_document("retrieved source")
        self.store.index_documents("repo_a1b2c3d4", [document])
        vector = self.index.upserts[0]["vectors"][0]
        self.index.matches = [
            SimpleNamespace(
                id=vector["id"],
                score=0.91,
                metadata=vector["metadata"],
            )
        ]

        results = self.store.search(
            "repo_a1b2c3d4",
            "where is the example?",
            top_k=3,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].document, document)
        self.assertEqual(results[0].vector_score, 0.91)
        self.assertGreater(results[0].score, results[0].vector_score)
        self.assertEqual(self.index.queries[0]["namespace"], "repo_a1b2c3d4")
        self.assertEqual(self.index.queries[0]["top_k"], 9)
        self.assertTrue(self.index.queries[0]["include_metadata"])
        self.assertEqual(self.embeddings.query_input, "where is the example?")

    def test_rejects_document_from_another_repository(self):
        with self.assertRaisesRegex(ValueError, "must match"):
            self.store.index_documents(
                "repo_a1b2c3d4",
                [make_document("wrong", repository_id="repo_ffffffff")],
            )

        self.assertEqual(self.index.deletes, [])
        self.assertEqual(self.index.upserts, [])

    def test_empty_replacement_deletes_namespace(self):
        count = self.store.index_documents("repo_a1b2c3d4", [])

        self.assertEqual(count, 0)
        self.assertEqual(
            self.index.deletes,
            [{"delete_all": True, "namespace": "repo_a1b2c3d4"}],
        )

    def test_vector_ids_are_stable_and_content_sensitive(self):
        document = make_document("same content")

        self.assertEqual(create_vector_id(document), create_vector_id(document))
        self.assertNotEqual(
            create_vector_id(document),
            create_vector_id(make_document("different content")),
        )

    def test_close_releases_index(self):
        self.store.close()

        self.assertTrue(self.index.closed)


class PineconeVectorStoreFactoryTests(unittest.TestCase):
    def test_creates_missing_dense_index_with_local_embedding_dimension(self):
        settings = Settings(
            embedding_dimension=3,
            pinecone_enabled=True,
            pinecone_api_key="test-key",
            pinecone_index_dimension=3,
            pinecone_index_name="codebase-test",
            pinecone_upsert_batch_size=2,
        )
        client = FakePineconeClient(exists=False, dimension=3)

        with patch("pinecone.Pinecone", return_value=client):
            store = create_pinecone_vector_store(settings, FakeEmbeddings())

        self.assertEqual(client.requested_index_name, "codebase-test")
        self.assertEqual(len(client.indexes.created), 1)
        creation = client.indexes.created[0]
        self.assertEqual(creation["dimension"], 3)
        self.assertEqual(creation["metric"], "cosine")
        self.assertEqual(creation["spec"].cloud, "aws")
        self.assertEqual(creation["spec"].region, "us-east-1")
        self.assertEqual(store.dimension, 3)

    def test_rejects_existing_index_with_wrong_dimension(self):
        settings = Settings(
            embedding_dimension=3,
            pinecone_enabled=True,
            pinecone_api_key="test-key",
            pinecone_index_dimension=3,
        )
        client = FakePineconeClient(exists=True, dimension=1024)

        with (
            patch("pinecone.Pinecone", return_value=client),
            self.assertRaisesRegex(PineconeConfigurationError, "dimension 1024"),
        ):
            create_pinecone_vector_store(settings, FakeEmbeddings())

        self.assertTrue(client.closed)

    def test_requires_api_key_when_enabled(self):
        settings = Settings(
            embedding_dimension=3,
            pinecone_enabled=True,
            pinecone_api_key="",
            pinecone_index_dimension=3,
        )

        with self.assertRaisesRegex(PineconeConfigurationError, "API_KEY"):
            create_pinecone_vector_store(settings, FakeEmbeddings())


if __name__ == "__main__":
    unittest.main()
