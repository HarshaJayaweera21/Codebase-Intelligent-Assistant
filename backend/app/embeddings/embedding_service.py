from dataclasses import dataclass

from langchain_core.embeddings import Embeddings


class EmbeddingValidationError(RuntimeError):
    """Raised when an embedding implementation returns invalid vectors."""


def validate_pinecone_dimension(
    *,
    embedding_dimension: int,
    pinecone_dimension: int,
) -> None:
    if embedding_dimension != pinecone_dimension:
        raise EmbeddingValidationError(
            f"The local embedding model outputs {embedding_dimension} "
            f"dimensions, but PINECONE_INDEX_DIMENSION is "
            f"{pinecone_dimension}. They must match before index creation or "
            "vector upsert."
        )


@dataclass(frozen=True)
class EmbeddingSmokeTestResult:
    model_loaded: bool
    expected_dimension: int
    query_dimension: int
    document_dimensions: list[int]
    all_values_are_floats: bool


def validate_embedding_vectors(
    vectors: list[list[float]],
    *,
    expected_count: int,
    expected_dimension: int,
) -> list[list[float]]:
    if len(vectors) != expected_count:
        raise EmbeddingValidationError(
            f"Expected {expected_count} embedding vectors, received "
            f"{len(vectors)}."
        )

    validated: list[list[float]] = []
    for index, vector in enumerate(vectors):
        if len(vector) != expected_dimension:
            raise EmbeddingValidationError(
                f"Embedding {index} has dimension {len(vector)}; expected "
                f"{expected_dimension}. The Pinecone index dimension must "
                "match the local model output dimension."
            )
        try:
            validated.append([float(value) for value in vector])
        except (TypeError, ValueError) as error:
            raise EmbeddingValidationError(
                f"Embedding {index} contains a non-numeric value."
            ) from error

    return validated


def run_embedding_smoke_test(
    embeddings: Embeddings,
    *,
    expected_dimension: int,
    query: str,
    documents: list[str],
) -> EmbeddingSmokeTestResult:
    query_vector = embeddings.embed_query(query)
    document_vectors = embeddings.embed_documents(documents)

    validate_embedding_vectors(
        [query_vector],
        expected_count=1,
        expected_dimension=expected_dimension,
    )
    validate_embedding_vectors(
        document_vectors,
        expected_count=len(documents),
        expected_dimension=expected_dimension,
    )

    all_values = query_vector + [
        value for vector in document_vectors for value in vector
    ]
    return EmbeddingSmokeTestResult(
        model_loaded=True,
        expected_dimension=expected_dimension,
        query_dimension=len(query_vector),
        document_dimensions=[len(vector) for vector in document_vectors],
        all_values_are_floats=all(
            isinstance(value, float) for value in all_values
        ),
    )
