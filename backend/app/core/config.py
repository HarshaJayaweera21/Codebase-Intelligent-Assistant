from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = Path(
    "models/embeddings/Qwen3-Embedding-4B-Q4_K_M.gguf"
)
DEFAULT_QUERY_INSTRUCTION = (
    "Retrieve relevant source code, configuration, and technical "
    "documentation from a software repository that can answer the given "
    "question."
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    embedding_model_path: Path = DEFAULT_MODEL_PATH
    embedding_cuda_dll_directory: Path | None = None
    embedding_n_gpu_layers: int = -1
    embedding_n_ctx: int = 4096
    embedding_n_batch: int = 512
    embedding_batch_size: int = 4
    embedding_dimension: int = 2560
    embedding_query_instruction: str = DEFAULT_QUERY_INSTRUCTION
    embedding_verbose: bool = False
    pinecone_enabled: bool = False
    pinecone_api_key: SecretStr | None = None
    pinecone_index_name: str = "codebase-assistant"
    pinecone_index_dimension: int = 2560
    pinecone_metric: str = "cosine"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_upsert_batch_size: int = 50
    retrieval_candidate_multiplier: int = 3
    retrieval_semantic_weight: float = 0.8

    @field_validator(
        "embedding_n_ctx",
        "embedding_n_batch",
        "embedding_batch_size",
        "embedding_dimension",
        "pinecone_index_dimension",
        "pinecone_upsert_batch_size",
        "retrieval_candidate_multiplier",
    )
    @classmethod
    def require_positive_integer(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be greater than zero")
        return value

    @field_validator("embedding_n_gpu_layers")
    @classmethod
    def validate_gpu_layers(cls, value: int) -> int:
        if value < -1:
            raise ValueError("must be -1 for all layers or zero/greater")
        return value

    @field_validator("pinecone_index_name")
    @classmethod
    def validate_pinecone_index_name(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("must not be empty")
        return normalized

    @field_validator("pinecone_metric")
    @classmethod
    def validate_pinecone_metric(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"cosine", "euclidean", "dotproduct"}:
            raise ValueError("must be cosine, euclidean, or dotproduct")
        return normalized

    @field_validator("retrieval_semantic_weight")
    @classmethod
    def validate_semantic_weight(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("must be between zero and one")
        return value

    @property
    def resolved_embedding_model_path(self) -> Path:
        path = self.embedding_model_path.expanduser()
        if not path.is_absolute():
            path = BACKEND_ROOT / path
        return path.resolve()

    @property
    def resolved_embedding_cuda_dll_directory(self) -> Path | None:
        if self.embedding_cuda_dll_directory is None:
            return None
        path = self.embedding_cuda_dll_directory.expanduser()
        if not path.is_absolute():
            path = BACKEND_ROOT / path
        return path.resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()
