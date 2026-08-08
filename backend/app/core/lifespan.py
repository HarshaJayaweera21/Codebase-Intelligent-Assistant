import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool

from app.core.config import Settings, get_settings
from app.embeddings.embedding_service import validate_pinecone_dimension
from app.embeddings.qwen_embeddings import QwenEmbeddings
from app.vectorstores.pinecone_vector_store import (
    PineconeVectorStore,
    create_pinecone_vector_store,
)


EmbeddingFactory = Callable[[Settings], QwenEmbeddings]
VectorStoreFactory = Callable[
    [Settings, QwenEmbeddings],
    PineconeVectorStore,
]
SettingsProvider = Callable[[], Settings]
Lifespan = Callable[[FastAPI], AbstractAsyncContextManager[None]]
logger = logging.getLogger(__name__)


def create_lifespan(
    *,
    settings_provider: SettingsProvider = get_settings,
    embedding_factory: EmbeddingFactory = QwenEmbeddings.load,
    vector_store_factory: VectorStoreFactory = create_pinecone_vector_store,
) -> Lifespan:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        settings = settings_provider()
        logger.info(
            "Loading the embedding model from %s",
            settings.resolved_embedding_model_path,
        )
        embedding_service = await run_in_threadpool(
            embedding_factory,
            settings,
        )
        try:
            validate_pinecone_dimension(
                embedding_dimension=embedding_service.dimension,
                pinecone_dimension=settings.pinecone_index_dimension,
            )
        except Exception:
            await run_in_threadpool(embedding_service.close)
            raise

        app.state.settings = settings
        app.state.embedding_service = embedding_service
        vector_store: PineconeVectorStore | None = None
        if settings.pinecone_enabled:
            try:
                vector_store = await run_in_threadpool(
                    vector_store_factory,
                    settings,
                    embedding_service,
                )
            except Exception:
                await run_in_threadpool(embedding_service.close)
                del app.state.embedding_service
                raise
        app.state.vector_store = vector_store
        logger.info(
            "Embedding model loaded with dimension %s",
            embedding_service.dimension,
        )

        try:
            yield
        finally:
            if vector_store is not None:
                await run_in_threadpool(vector_store.close)
            await run_in_threadpool(embedding_service.close)
            del app.state.vector_store
            del app.state.embedding_service
            logger.info("Embedding model resources released")

    return lifespan


application_lifespan = create_lifespan()
