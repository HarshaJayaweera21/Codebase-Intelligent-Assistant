import unittest

from fastapi import FastAPI

from app.core.config import Settings
from app.core.lifespan import create_lifespan
from app.embeddings.embedding_service import EmbeddingValidationError
from app.embeddings.qwen_embeddings import QwenEmbeddings


class FakeLlamaEmbeddingModel:
    def __init__(self) -> None:
        self.closed = False

    def n_embd(self) -> int:
        return 3

    def embed(
        self,
        input: list[str],
        *,
        normalize: bool,
        truncate: bool,
    ) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in input]

    def close(self) -> None:
        self.closed = True


class FakeVectorStore:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeChatModel:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class EmbeddingLifespanTests(unittest.IsolatedAsyncioTestCase):
    async def test_loads_once_stores_service_and_closes_at_shutdown(self):
        settings = Settings(
            embedding_dimension=3,
            pinecone_index_dimension=3,
            pinecone_enabled=False,
            gemini_enabled=False,
        )
        model = FakeLlamaEmbeddingModel()
        service = QwenEmbeddings(
            model=model,
            expected_dimension=3,
            batch_size=2,
            query_instruction="instruction",
        )
        load_count = 0

        def factory(_: Settings) -> QwenEmbeddings:
            nonlocal load_count
            load_count += 1
            return service

        lifespan = create_lifespan(
            settings_provider=lambda: settings,
            embedding_factory=factory,
        )
        app = FastAPI(lifespan=lifespan)

        async with lifespan(app):
            self.assertEqual(load_count, 1)
            self.assertIs(app.state.embedding_service, service)
            self.assertIs(app.state.settings, settings)
            self.assertTrue(hasattr(app.state, "repository_processing_store"))
            self.assertIsNone(app.state.rag_service)
            self.assertFalse(model.closed)

        self.assertTrue(model.closed)
        self.assertFalse(hasattr(app.state, "embedding_service"))
        self.assertFalse(hasattr(app.state, "vector_store"))
        self.assertFalse(hasattr(app.state, "repository_processing_store"))
        self.assertFalse(hasattr(app.state, "rag_service"))

    async def test_enabled_pinecone_store_is_managed_by_lifespan(self):
        settings = Settings(
            embedding_dimension=3,
            pinecone_index_dimension=3,
            pinecone_enabled=True,
            pinecone_api_key="test-key",
            gemini_enabled=False,
        )
        model = FakeLlamaEmbeddingModel()
        embedding_service = QwenEmbeddings(
            model=model,
            expected_dimension=3,
            batch_size=2,
            query_instruction="instruction",
        )
        vector_store = FakeVectorStore()
        factory_calls = 0

        def vector_store_factory(settings_arg, embeddings_arg):
            nonlocal factory_calls
            factory_calls += 1
            self.assertIs(settings_arg, settings)
            self.assertIs(embeddings_arg, embedding_service)
            return vector_store

        lifespan = create_lifespan(
            settings_provider=lambda: settings,
            embedding_factory=lambda _: embedding_service,
            vector_store_factory=vector_store_factory,
        )
        app = FastAPI(lifespan=lifespan)

        async with lifespan(app):
            self.assertEqual(factory_calls, 1)
            self.assertIs(app.state.vector_store, vector_store)
            self.assertFalse(vector_store.closed)

        self.assertTrue(vector_store.closed)
        self.assertTrue(model.closed)
        self.assertFalse(hasattr(app.state, "vector_store"))

    async def test_dimension_mismatch_closes_model_during_failed_startup(self):
        settings = Settings(
            embedding_dimension=3,
            pinecone_index_dimension=2,
            pinecone_enabled=False,
            gemini_enabled=False,
        )
        model = FakeLlamaEmbeddingModel()
        service = QwenEmbeddings(
            model=model,
            expected_dimension=3,
            batch_size=2,
            query_instruction="instruction",
        )
        lifespan = create_lifespan(
            settings_provider=lambda: settings,
            embedding_factory=lambda _: service,
        )
        app = FastAPI(lifespan=lifespan)

        with self.assertRaises(EmbeddingValidationError):
            async with lifespan(app):
                pass

        self.assertTrue(model.closed)
        self.assertFalse(hasattr(app.state, "embedding_service"))

    async def test_enabled_gemini_rag_is_managed_by_lifespan(self):
        settings = Settings(
            embedding_dimension=3,
            pinecone_index_dimension=3,
            pinecone_enabled=True,
            pinecone_api_key="test-key",
            gemini_enabled=True,
            gemini_api_key="test-gemini-key",
        )
        model = FakeLlamaEmbeddingModel()
        embedding_service = QwenEmbeddings(
            model=model,
            expected_dimension=3,
            batch_size=2,
            query_instruction="instruction",
        )
        vector_store = FakeVectorStore()
        chat_model = FakeChatModel()
        lifespan = create_lifespan(
            settings_provider=lambda: settings,
            embedding_factory=lambda _: embedding_service,
            vector_store_factory=lambda *_: vector_store,
            chat_model_factory=lambda _: chat_model,
        )
        app = FastAPI(lifespan=lifespan)

        async with lifespan(app):
            self.assertIsNotNone(app.state.rag_service)
            self.assertFalse(chat_model.closed)

        self.assertTrue(chat_model.closed)
        self.assertTrue(vector_store.closed)
        self.assertTrue(model.closed)
        self.assertFalse(hasattr(app.state, "rag_service"))


if __name__ == "__main__":
    unittest.main()
