import os
from collections.abc import Sequence
from pathlib import Path
from threading import Lock
from typing import Any, Protocol

from langchain_core.embeddings import Embeddings

from app.core.config import Settings
from app.embeddings.embedding_service import validate_embedding_vectors


class LlamaEmbeddingModel(Protocol):
    def n_embd(self) -> int: ...

    def embed(
        self,
        input: list[str],
        *,
        normalize: bool,
        truncate: bool,
    ) -> list[list[float]]: ...

    def close(self) -> None: ...


class EmbeddingModelLoadError(RuntimeError):
    """Raised when the configured local GGUF model cannot be loaded."""


class EmbeddingInputTooLongError(RuntimeError):
    """Raised when one input exceeds the configured llama.cpp capacity."""


class QwenEmbeddings(Embeddings):
    """In-process LangChain embeddings backed by one llama.cpp model."""

    def __init__(
        self,
        *,
        model: LlamaEmbeddingModel,
        expected_dimension: int,
        batch_size: int,
        query_instruction: str,
        model_batch_capacity: int | None = None,
        model_context_size: int | None = None,
        dll_directory_handle: Any | None = None,
    ) -> None:
        if expected_dimension <= 0:
            raise ValueError("expected_dimension must be greater than zero")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        native_dimension = model.n_embd()
        if native_dimension != expected_dimension:
            model.close()
            if dll_directory_handle is not None:
                dll_directory_handle.close()
            raise EmbeddingModelLoadError(
                f"The GGUF model reports embedding dimension "
                f"{native_dimension}, but EMBEDDING_DIMENSION is "
                f"{expected_dimension}. Pinecone must use the actual model "
                "dimension."
            )

        self.dimension = native_dimension
        self.batch_size = batch_size
        self.model_batch_capacity = model_batch_capacity
        self.model_context_size = model_context_size
        self.query_instruction = query_instruction.strip()
        self._model: LlamaEmbeddingModel | None = model
        self._dll_directory_handle = dll_directory_handle
        self._inference_lock = Lock()

    @classmethod
    def load(cls, settings: Settings) -> "QwenEmbeddings":
        model_path = settings.resolved_embedding_model_path
        _require_local_model(model_path)
        dll_directory_handle = _add_cuda_dll_directory(settings)

        try:
            from llama_cpp import LLAMA_POOLING_TYPE_LAST, Llama
        except ImportError as error:
            if dll_directory_handle is not None:
                dll_directory_handle.close()
            raise EmbeddingModelLoadError(
                "llama-cpp-python is not installed. Install its CUDA-enabled "
                "build before starting FastAPI."
            ) from error

        try:
            model = Llama(
                model_path=str(model_path),
                embedding=True,
                n_gpu_layers=settings.embedding_n_gpu_layers,
                n_ctx=settings.embedding_n_ctx,
                n_batch=settings.embedding_n_batch,
                pooling_type=LLAMA_POOLING_TYPE_LAST,
                verbose=settings.embedding_verbose,
            )
        except Exception as error:
            if dll_directory_handle is not None:
                dll_directory_handle.close()
            raise EmbeddingModelLoadError(
                f"Failed to load the embedding model from '{model_path}'. "
                "Check the GGUF file and GPU settings."
            ) from error

        return cls(
            model=model,
            expected_dimension=settings.embedding_dimension,
            batch_size=settings.embedding_batch_size,
            query_instruction=settings.embedding_query_instruction,
            model_batch_capacity=settings.embedding_n_batch,
            model_context_size=settings.embedding_n_ctx,
            dll_directory_handle=dll_directory_handle,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed_texts(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed_texts([self.format_query(text)])[0]

    def format_query(self, query: str) -> str:
        if not self.query_instruction:
            return query
        return f"Instruct: {self.query_instruction}\nQuery: {query}"

    def close(self) -> None:
        with self._inference_lock:
            if self._model is not None:
                self._model.close()
                self._model = None
            if self._dll_directory_handle is not None:
                self._dll_directory_handle.close()
                self._dll_directory_handle = None

    def _embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        embeddings: list[list[float]] = []
        with self._inference_lock:
            model = self._require_open_model()
            for start in range(0, len(texts), self.batch_size):
                batch = list(texts[start : start + self.batch_size])
                self._validate_token_capacity(model, batch, start)
                try:
                    raw_vectors = model.embed(
                        batch,
                        normalize=True,
                        truncate=False,
                    )
                except ValueError as error:
                    if "exceed batch size" not in str(error):
                        raise
                    raise EmbeddingInputTooLongError(
                        f"An embedding input exceeded EMBEDDING_N_BATCH="
                        f"{self.model_batch_capacity}. Increase "
                        "EMBEDDING_N_BATCH and restart FastAPI. The input was "
                        "not truncated."
                    ) from error
                vectors = _coerce_sequence_embeddings(raw_vectors)
                embeddings.extend(
                    validate_embedding_vectors(
                        vectors,
                        expected_count=len(batch),
                        expected_dimension=self.dimension,
                    )
                )

        return embeddings

    def _validate_token_capacity(
        self,
        model: LlamaEmbeddingModel,
        texts: list[str],
        start_index: int,
    ) -> None:
        tokenize = getattr(model, "tokenize", None)
        if not callable(tokenize):
            return
        for offset, text in enumerate(texts):
            token_count = len(tokenize(text.encode("utf-8")))
            input_number = start_index + offset + 1
            if (
                self.model_context_size is not None
                and token_count > self.model_context_size
            ):
                raise EmbeddingInputTooLongError(
                    f"Embedding input {input_number} requires {token_count} "
                    f"tokens, exceeding EMBEDDING_N_CTX="
                    f"{self.model_context_size}. Reduce the chunk size or "
                    "increase EMBEDDING_N_CTX, then restart FastAPI. The "
                    "input was not truncated."
                )
            if (
                self.model_batch_capacity is not None
                and token_count > self.model_batch_capacity
            ):
                raise EmbeddingInputTooLongError(
                    f"Embedding input {input_number} requires {token_count} "
                    f"tokens, exceeding EMBEDDING_N_BATCH="
                    f"{self.model_batch_capacity}. Increase "
                    "EMBEDDING_N_BATCH and restart FastAPI. The input was "
                    "not truncated."
                )

    def _require_open_model(self) -> LlamaEmbeddingModel:
        if self._model is None:
            raise RuntimeError("The embedding model has already been closed.")
        return self._model


def _require_local_model(model_path: Path) -> None:
    if not model_path.is_file():
        raise EmbeddingModelLoadError(
            f"Embedding model file not found at '{model_path}'. Download "
            "Qwen3-Embedding-4B-Q4_K_M.gguf manually and set "
            "EMBEDDING_MODEL_PATH to that local file before starting FastAPI."
        )


def _add_cuda_dll_directory(settings: Settings) -> Any | None:
    dll_directory = settings.resolved_embedding_cuda_dll_directory
    if dll_directory is None or os.name != "nt":
        return None
    if not dll_directory.is_dir():
        raise EmbeddingModelLoadError(
            f"CUDA DLL directory not found at '{dll_directory}'. Set "
            "EMBEDDING_CUDA_DLL_DIRECTORY to the installed CUDA runtime bin "
            "directory."
        )

    # llama.dll has transitive CUDA dependencies. Registering the directory is
    # not always enough for those nested lookups, so update this process's PATH
    # before importing llama_cpp as well.
    dll_directory_text = str(dll_directory)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if dll_directory_text.casefold() not in {
        entry.casefold() for entry in path_entries if entry
    }:
        os.environ["PATH"] = os.pathsep.join([dll_directory_text, *path_entries])

    return os.add_dll_directory(dll_directory_text)


def _coerce_sequence_embeddings(raw_vectors: Any) -> list[list[float]]:
    if not isinstance(raw_vectors, list) or not all(
        isinstance(vector, list) for vector in raw_vectors
    ):
        raise RuntimeError("llama.cpp returned an invalid embedding result.")
    return raw_vectors
