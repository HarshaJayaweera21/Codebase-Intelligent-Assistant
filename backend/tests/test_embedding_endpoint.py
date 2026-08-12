import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings

from app.api.embedding_routes import router
from app.core.config import Settings
from app.core.lifespan import create_lifespan


class ManagedFakeEmbeddings(Embeddings):
    dimension = 2

    def __init__(self) -> None:
        self.closed = False

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(index), 0.5] for index, _ in enumerate(texts)]

    def embed_query(self, text: str) -> list[float]:
        return [0.25, 0.75]

    def close(self) -> None:
        self.closed = True


class EmbeddingEndpointTests(unittest.TestCase):
    def test_development_endpoint_runs_query_and_document_embeddings(self):
        settings = Settings(
            embedding_dimension=2,
            pinecone_index_dimension=2,
            pinecone_enabled=False,
            gemini_enabled=False,
        )
        service = ManagedFakeEmbeddings()
        lifespan = create_lifespan(
            settings_provider=lambda: settings,
            embedding_factory=lambda _: service,
        )
        app = FastAPI(lifespan=lifespan)
        app.include_router(router, prefix="/api")

        with TestClient(app) as client:
            response = client.post(
                "/api/development/embeddings/test",
                json={
                    "query": "Where is authentication?",
                    "documents": ["AuthService", "config.yaml"],
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "model_loaded": True,
                "expected_dimension": 2,
                "query_dimension": 2,
                "document_dimensions": [2, 2],
                "all_values_are_floats": True,
            },
        )
        self.assertTrue(service.closed)


if __name__ == "__main__":
    unittest.main()
