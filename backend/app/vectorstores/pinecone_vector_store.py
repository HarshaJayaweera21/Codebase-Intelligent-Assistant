from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Protocol

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.core.config import Settings
from app.embeddings.embedding_service import validate_embedding_vectors
from app.rag.retrieval_text import (
    build_document_embedding_text,
    build_query_embedding_text,
    lexical_relevance,
)


class PineconeConfigurationError(RuntimeError):
    """Raised when Pinecone is missing or incompatible with local embeddings."""


class PineconeIndex(Protocol):
    def upsert(
        self,
        *,
        vectors: Sequence[Mapping[str, Any]],
        namespace: str,
        **kwargs: Any,
    ) -> Any: ...

    def query(self, **kwargs: Any) -> Any: ...

    def delete(self, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class RetrievedDocument:
    vector_id: str
    score: float
    vector_score: float
    lexical_score: float
    document: Document


class PineconeVectorStore:
    """Store locally generated embeddings in repository-scoped namespaces."""

    def __init__(
        self,
        *,
        index: PineconeIndex,
        embeddings: Embeddings,
        dimension: int,
        upsert_batch_size: int,
        candidate_multiplier: int = 3,
        semantic_weight: float = 0.8,
        client: Any | None = None,
    ) -> None:
        self._index = index
        self._embeddings = embeddings
        self._client = client
        self.dimension = dimension
        self.upsert_batch_size = upsert_batch_size
        self.candidate_multiplier = candidate_multiplier
        self.semantic_weight = semantic_weight

    def index_documents(
        self,
        repository_id: str,
        documents: list[Document],
        *,
        replace_namespace: bool = True,
    ) -> int:
        namespace = _require_repository_id(repository_id)
        _validate_document_repository_ids(namespace, documents)

        if not documents:
            if replace_namespace:
                self.delete_repository(namespace)
            return 0

        vectors = self._embeddings.embed_documents(
            [build_document_embedding_text(document) for document in documents]
        )
        validated_vectors = validate_embedding_vectors(
            vectors,
            expected_count=len(documents),
            expected_dimension=self.dimension,
        )

        pinecone_vectors = [
            {
                "id": create_vector_id(document),
                "values": vector,
                "metadata": _document_to_pinecone_metadata(document),
            }
            for document, vector in zip(
                documents,
                validated_vectors,
                strict=True,
            )
        ]

        # Embed successfully before replacing existing data. This prevents a
        # local inference failure from erasing a working repository namespace.
        if replace_namespace:
            self.delete_repository(namespace)

        for start in range(0, len(pinecone_vectors), self.upsert_batch_size):
            batch = pinecone_vectors[start : start + self.upsert_batch_size]
            self._index.upsert(vectors=batch, namespace=namespace)

        return len(pinecone_vectors)

    def search(
        self,
        repository_id: str,
        query: str,
        *,
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedDocument]:
        namespace = _require_repository_id(repository_id)
        if not query.strip():
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_vector = self._embeddings.embed_query(
            build_query_embedding_text(query)
        )
        validated_query = validate_embedding_vectors(
            [query_vector],
            expected_count=1,
            expected_dimension=self.dimension,
        )[0]
        candidate_count = min(100, max(top_k, top_k * self.candidate_multiplier))
        response = self._index.query(
            vector=validated_query,
            top_k=candidate_count,
            namespace=namespace,
            filter=metadata_filter,
            include_values=False,
            include_metadata=True,
        )

        candidates = [
            _match_to_retrieved_document(match) for match in response.matches
        ]
        reranked = [self._rerank(query, candidate) for candidate in candidates]
        reranked.sort(
            key=lambda result: (
                result.score,
                result.vector_score,
                result.vector_id,
            ),
            reverse=True,
        )
        return reranked[:top_k]

    def _rerank(
        self,
        query: str,
        candidate: RetrievedDocument,
    ) -> RetrievedDocument:
        lexical_score = lexical_relevance(query, candidate.document)
        combined_score = (
            self.semantic_weight * candidate.vector_score
            + (1 - self.semantic_weight) * lexical_score
        )
        return RetrievedDocument(
            vector_id=candidate.vector_id,
            score=combined_score,
            vector_score=candidate.vector_score,
            lexical_score=lexical_score,
            document=candidate.document,
        )

    def delete_repository(self, repository_id: str) -> None:
        namespace = _require_repository_id(repository_id)
        self._index.delete(delete_all=True, namespace=namespace)

    def close(self) -> None:
        for resource in (self._index, self._client):
            close = getattr(resource, "close", None)
            if callable(close):
                close()


def create_pinecone_vector_store(
    settings: Settings,
    embeddings: Embeddings,
) -> PineconeVectorStore:
    if settings.pinecone_api_key is None or not (
        api_key := settings.pinecone_api_key.get_secret_value().strip()
    ):
        raise PineconeConfigurationError(
            "PINECONE_ENABLED is true, but PINECONE_API_KEY is missing. "
            "Create a Pinecone API key and add it to backend/.env."
        )

    try:
        from pinecone import Pinecone, ServerlessSpec
    except ImportError as error:
        raise PineconeConfigurationError(
            "The 'pinecone' package is required. Install backend requirements."
        ) from error

    client = Pinecone(api_key=api_key)
    index_name = settings.pinecone_index_name

    if not client.indexes.exists(index_name):
        client.indexes.create(
            name=index_name,
            dimension=settings.pinecone_index_dimension,
            metric=settings.pinecone_metric,
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
            timeout=120,
        )

    description = client.indexes.describe(index_name)
    actual_dimension = int(description.dimension)
    if actual_dimension != settings.pinecone_index_dimension:
        client.close()
        raise PineconeConfigurationError(
            f"Pinecone index '{index_name}' has dimension {actual_dimension}, "
            f"but local embeddings use {settings.pinecone_index_dimension}. "
            "Use a matching index or change PINECONE_INDEX_NAME."
        )

    actual_metric = str(description.metric).lower()
    if actual_metric != settings.pinecone_metric:
        client.close()
        raise PineconeConfigurationError(
            f"Pinecone index '{index_name}' uses metric '{actual_metric}', "
            f"but PINECONE_METRIC is '{settings.pinecone_metric}'."
        )

    return PineconeVectorStore(
        index=client.index(index_name),
        embeddings=embeddings,
        dimension=actual_dimension,
        upsert_batch_size=settings.pinecone_upsert_batch_size,
        candidate_multiplier=settings.retrieval_candidate_multiplier,
        semantic_weight=settings.retrieval_semantic_weight,
        client=client,
    )


def create_vector_id(document: Document) -> str:
    identity = {
        "metadata": document.metadata,
        "page_content": document.page_content,
    }
    serialized = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"chunk_{sha256(serialized.encode('utf-8')).hexdigest()}"


def _document_to_pinecone_metadata(document: Document) -> dict[str, Any]:
    metadata_json = json.dumps(
        document.metadata,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    metadata: dict[str, Any] = {
        "content": document.page_content,
        "document_metadata_json": metadata_json,
    }

    # Keep useful fields flat for future Pinecone metadata filtering. The JSON
    # copy preserves nested source_ranges and None values exactly.
    for key in (
        "repository_id",
        "file_path",
        "language",
        "chunk_type",
        "symbol_name",
        "symbol_start_line",
        "symbol_end_line",
    ):
        value = document.metadata.get(key)
        if isinstance(value, (str, int, float, bool)):
            metadata[key] = value

    return metadata


def _match_to_retrieved_document(match: Any) -> RetrievedDocument:
    metadata = dict(match.metadata or {})
    content = str(metadata.pop("content", ""))
    serialized_metadata = metadata.pop("document_metadata_json", "{}")
    try:
        document_metadata = json.loads(str(serialized_metadata))
    except (TypeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Pinecone vector '{match.id}' contains invalid document metadata."
        ) from error
    if not isinstance(document_metadata, dict):
        raise RuntimeError(
            f"Pinecone vector '{match.id}' contains invalid document metadata."
        )

    return RetrievedDocument(
        vector_id=str(match.id),
        score=float(match.score),
        vector_score=float(match.score),
        lexical_score=0.0,
        document=Document(
            page_content=content,
            metadata=document_metadata,
        ),
    )


def _require_repository_id(repository_id: str) -> str:
    normalized = repository_id.strip()
    if not normalized:
        raise ValueError("repository_id must not be empty")
    return normalized


def _validate_document_repository_ids(
    repository_id: str,
    documents: list[Document],
) -> None:
    for document in documents:
        if document.metadata.get("repository_id") != repository_id:
            raise ValueError(
                "Every document repository_id must match the Pinecone namespace."
            )
