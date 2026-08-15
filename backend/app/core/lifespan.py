import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from langchain_core.language_models.chat_models import BaseChatModel

from app.core.config import Settings, get_settings
from app.embeddings.embedding_service import validate_pinecone_dimension
from app.embeddings.qwen_embeddings import QwenEmbeddings
from app.llms.gemini_chat_model import create_gemini_chat_model
from app.persistence.sqlite_database import SQLiteDatabase
from app.rag.rag_service import RagService
from app.services.chat_service import ChatService
from app.services.repository_lifecycle import RepositoryLifecycleService
from app.services.repository_processing import (
    RepositoryJobCoordinator,
    RepositoryProcessingLimits,
    RepositoryProcessingStore,
)
from app.vectorstores.pinecone_vector_store import (
    PineconeVectorStore,
    create_pinecone_vector_store,
)


EmbeddingFactory = Callable[[Settings], QwenEmbeddings]
VectorStoreFactory = Callable[
    [Settings, QwenEmbeddings],
    PineconeVectorStore,
]
ChatModelFactory = Callable[[Settings], BaseChatModel]
SettingsProvider = Callable[[], Settings]
Lifespan = Callable[[FastAPI], AbstractAsyncContextManager[None]]
logger = logging.getLogger(__name__)


def create_sqlite_database(settings: Settings) -> SQLiteDatabase:
    return SQLiteDatabase(settings.resolved_database_path)


def create_lifespan(
    *,
    settings_provider: SettingsProvider = get_settings,
    embedding_factory: EmbeddingFactory = QwenEmbeddings.load,
    vector_store_factory: VectorStoreFactory = create_pinecone_vector_store,
    chat_model_factory: ChatModelFactory = create_gemini_chat_model,
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

        database = create_sqlite_database(settings)
        try:
            await run_in_threadpool(database.initialize)
        except Exception:
            await run_in_threadpool(embedding_service.close)
            raise

        app.state.settings = settings
        app.state.embedding_service = embedding_service
        app.state.database = database
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
                del app.state.database
                raise
        app.state.vector_store = vector_store
        rag_service: RagService | None = None
        if settings.gemini_enabled:
            if vector_store is None:
                await run_in_threadpool(embedding_service.close)
                del app.state.embedding_service
                del app.state.vector_store
                del app.state.database
                raise RuntimeError(
                    "GEMINI_ENABLED requires PINECONE_ENABLED=true because "
                    "answers must be grounded in retrieved repository evidence."
                )
            try:
                chat_model = chat_model_factory(settings)
                rag_service = RagService(
                    vector_store=vector_store,
                    chat_model=chat_model,
                    default_top_k=settings.rag_top_k,
                    max_context_characters=(
                        settings.rag_max_context_characters
                    ),
                    stream_retry_attempts=(
                        settings.rag_stream_retry_attempts
                    ),
                )
            except Exception:
                await run_in_threadpool(vector_store.close)
                await run_in_threadpool(embedding_service.close)
                del app.state.embedding_service
                del app.state.vector_store
                del app.state.database
                raise
        app.state.rag_service = rag_service
        processing_store = RepositoryProcessingStore(database)
        interrupted_count = await run_in_threadpool(
            processing_store.recover_interrupted
        )
        if interrupted_count:
            logger.warning(
                "Marked %s interrupted repository job(s) as failed",
                interrupted_count,
            )
        app.state.repository_processing_store = processing_store
        app.state.repository_processing_limits = (
            RepositoryProcessingLimits.from_settings(settings)
        )
        app.state.repository_job_coordinator = RepositoryJobCoordinator(
            settings.repository_max_concurrent_jobs
        )
        app.state.chat_service = ChatService(database)
        app.state.repository_lifecycle_service = (
            RepositoryLifecycleService(
                processing_store=processing_store,
                vector_store=vector_store,
            )
            if vector_store is not None
            else None
        )
        logger.info(
            "Embedding model loaded with dimension %s",
            embedding_service.dimension,
        )

        try:
            yield
        finally:
            if rag_service is not None:
                await run_in_threadpool(rag_service.close)
            if vector_store is not None:
                await run_in_threadpool(vector_store.close)
            await run_in_threadpool(embedding_service.close)
            database.close()
            del app.state.vector_store
            del app.state.rag_service
            del app.state.repository_processing_store
            del app.state.repository_processing_limits
            del app.state.repository_job_coordinator
            del app.state.chat_service
            del app.state.repository_lifecycle_service
            del app.state.database
            del app.state.embedding_service
            logger.info("Embedding model resources released")

    return lifespan


application_lifespan = create_lifespan()
