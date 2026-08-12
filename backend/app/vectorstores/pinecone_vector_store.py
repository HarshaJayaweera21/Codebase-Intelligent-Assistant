from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
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
    minimum_structural_relevance,
    retrieval_relevance,
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
    exact_match_score: float
    structural_score: float
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
        candidate_multiplier: int = 8,
        semantic_weight: float = 0.8,
        exact_match_boost: float = 0.12,
        structural_boost: float = 0.05,
        diversity_weight: float = 0.1,
        max_chunks_per_file: int = 2,
        max_upsert_payload_bytes: int = 750_000,
        client: Any | None = None,
    ) -> None:
        self._index = index
        self._embeddings = embeddings
        self._client = client
        self.dimension = dimension
        self.upsert_batch_size = upsert_batch_size
        self.candidate_multiplier = candidate_multiplier
        self.semantic_weight = semantic_weight
        self.exact_match_boost = exact_match_boost
        self.structural_boost = structural_boost
        self.diversity_weight = diversity_weight
        self.max_chunks_per_file = max_chunks_per_file
        self.max_upsert_payload_bytes = max_upsert_payload_bytes

    def index_documents(
        self,
        repository_id: str,
        documents: list[Document],
        *,
        replace_namespace: bool = True,
        progress_callback: Callable[[str], None] | None = None,
    ) -> int:
        namespace = _require_repository_id(repository_id)
        _validate_document_repository_ids(namespace, documents)

        if not documents:
            if replace_namespace:
                self.delete_repository(namespace)
            return 0

        if progress_callback is not None:
            progress_callback("embedding")
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
            if progress_callback is not None:
                progress_callback("indexing")
            self.delete_repository(namespace)
        elif progress_callback is not None:
            progress_callback("indexing")

        for batch in _build_upsert_batches(
            pinecone_vectors,
            max_count=self.upsert_batch_size,
            max_payload_bytes=self.max_upsert_payload_bytes,
        ):
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
        reranked = _deduplicate_candidates(reranked)
        minimum_structural_score = minimum_structural_relevance(query)
        if minimum_structural_score is not None:
            reranked = [
                candidate
                for candidate in reranked
                if candidate.structural_score >= minimum_structural_score
            ]
        reranked.sort(
            key=lambda result: (
                result.score,
                result.vector_score,
                result.vector_id,
            ),
            reverse=True,
        )
        return _select_diverse_candidates(
            reranked,
            top_k=top_k,
            diversity_weight=self.diversity_weight,
            max_chunks_per_file=self.max_chunks_per_file,
        )

    def _rerank(
        self,
        query: str,
        candidate: RetrievedDocument,
    ) -> RetrievedDocument:
        relevance = retrieval_relevance(query, candidate.document)
        combined_score = (
            self.semantic_weight * candidate.vector_score
            + (1 - self.semantic_weight) * relevance.lexical_score
            + self.exact_match_boost * relevance.exact_match_score
            + self.structural_boost * relevance.structural_score
        )
        return RetrievedDocument(
            vector_id=candidate.vector_id,
            score=combined_score,
            vector_score=candidate.vector_score,
            lexical_score=relevance.lexical_score,
            exact_match_score=relevance.exact_match_score,
            structural_score=relevance.structural_score,
            document=candidate.document,
        )

    def delete_repository(self, repository_id: str) -> None:
        namespace = _require_repository_id(repository_id)
        try:
            self._index.delete(delete_all=True, namespace=namespace)
        except Exception as error:
            # Pinecone returns 404 when delete_all targets a namespace that has
            # never been populated (or was already removed). Deletion is used
            # as an idempotent reset before indexing, so that response already
            # represents the desired state and must not fail the pipeline.
            if not _is_not_found_error(error):
                raise

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
        exact_match_boost=settings.retrieval_exact_match_boost,
        structural_boost=settings.retrieval_structural_boost,
        diversity_weight=settings.retrieval_diversity_weight,
        max_chunks_per_file=settings.retrieval_max_chunks_per_file,
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


def _build_upsert_batches(
    vectors: list[dict[str, Any]],
    *,
    max_count: int,
    max_payload_bytes: int,
) -> list[list[dict[str, Any]]]:
    """Bound batches by count and JSON size to avoid slow oversized writes."""
    batches: list[list[dict[str, Any]]] = []
    batch: list[dict[str, Any]] = []
    batch_bytes = 0

    for vector in vectors:
        vector_bytes = len(
            json.dumps(vector, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        if batch and (
            len(batch) >= max_count
            or batch_bytes + vector_bytes > max_payload_bytes
        ):
            batches.append(batch)
            batch = []
            batch_bytes = 0
        batch.append(vector)
        batch_bytes += vector_bytes

    if batch:
        batches.append(batch)
    return batches


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
        exact_match_score=0.0,
        structural_score=0.0,
        document=Document(
            page_content=content,
            metadata=document_metadata,
        ),
    )


def _deduplicate_candidates(
    candidates: list[RetrievedDocument],
) -> list[RetrievedDocument]:
    deduplicated: list[RetrievedDocument] = []
    seen_vector_ids: set[str] = set()
    seen_chunks: set[tuple[str, str | None, str]] = set()
    for candidate in candidates:
        metadata = candidate.document.metadata
        chunk_identity = (
            str(metadata.get("file_path", "")),
            metadata.get("symbol_name"),
            " ".join(candidate.document.page_content.split()),
        )
        if (
            candidate.vector_id in seen_vector_ids
            or chunk_identity in seen_chunks
        ):
            continue
        seen_vector_ids.add(candidate.vector_id)
        seen_chunks.add(chunk_identity)
        deduplicated.append(candidate)
    return deduplicated


def _select_diverse_candidates(
    candidates: list[RetrievedDocument],
    *,
    top_k: int,
    diversity_weight: float,
    max_chunks_per_file: int,
) -> list[RetrievedDocument]:
    remaining = list(candidates)
    selected: list[RetrievedDocument] = []
    file_counts: dict[str, int] = {}

    while remaining and len(selected) < top_k:
        eligible = [
            candidate
            for candidate in remaining
            if file_counts.get(_candidate_file(candidate), 0)
            < max_chunks_per_file
        ]
        if not eligible:
            eligible = remaining

        candidate = max(
            eligible,
            key=lambda item: (
                _diversity_adjusted_score(
                    item,
                    selected,
                    diversity_weight=diversity_weight,
                ),
                item.score,
                item.vector_score,
                item.vector_id,
            ),
        )
        selected.append(candidate)
        remaining.remove(candidate)
        file_path = _candidate_file(candidate)
        file_counts[file_path] = file_counts.get(file_path, 0) + 1

    return selected


def _diversity_adjusted_score(
    candidate: RetrievedDocument,
    selected: list[RetrievedDocument],
    *,
    diversity_weight: float,
) -> float:
    if not selected:
        return candidate.score
    similarity = max(
        _document_token_similarity(candidate.document, item.document)
        for item in selected
    )
    return (1 - diversity_weight) * candidate.score - diversity_weight * similarity


def _document_token_similarity(left: Document, right: Document) -> float:
    left_terms = set(left.page_content.lower().split())
    right_terms = set(right.page_content.lower().split())
    union = left_terms | right_terms
    if not union:
        return 0.0
    return len(left_terms & right_terms) / len(union)


def _candidate_file(candidate: RetrievedDocument) -> str:
    return str(candidate.document.metadata.get("file_path", ""))


def _is_not_found_error(error: Exception) -> bool:
    return (
        type(error).__name__ in {"NotFoundError", "NotFoundException"}
        and getattr(error, "status_code", None) == 404
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
