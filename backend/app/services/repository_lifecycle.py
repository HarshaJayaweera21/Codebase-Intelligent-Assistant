from dataclasses import dataclass
from pathlib import Path

from app.models.repository import RepositoryProcessingStatus
from app.services.repository_processing import (
    RepositoryProcessingRecord,
    RepositoryProcessingStore,
    safe_remove_repository_directory,
)
from app.vectorstores.pinecone_vector_store import PineconeVectorStore


class RepositoryLifecycleConflict(RuntimeError):
    """Raised when an active repository cannot be mutated safely."""


@dataclass(frozen=True)
class RepositoryDeletionResult:
    repository_id: str
    chat_id: str


class RepositoryLifecycleService:
    """Coordinates external cleanup before deleting persistent metadata."""

    def __init__(
        self,
        *,
        processing_store: RepositoryProcessingStore,
        vector_store: PineconeVectorStore,
    ) -> None:
        self._processing_store = processing_store
        self._vector_store = vector_store

    def delete_repository(
        self,
        repository_id: str,
    ) -> RepositoryDeletionResult | None:
        record = self._processing_store.get(repository_id)
        if record is None:
            return None
        self._require_not_processing(record)

        # Keep SQLite metadata until both external resources are removed. If
        # either step fails, the same DELETE request can be retried safely.
        self._vector_store.delete_repository(repository_id)
        safe_remove_repository_directory(Path(record.local_path))
        if not self._processing_store.delete(repository_id):
            raise RuntimeError("Repository metadata disappeared during deletion.")
        return RepositoryDeletionResult(
            repository_id=record.repository_id,
            chat_id=record.chat_id,
        )

    def delete_chat(self, chat_id: str) -> RepositoryDeletionResult | None:
        record = next(
            (
                item
                for item in self._processing_store.list()
                if item.chat_id == chat_id
            ),
            None,
        )
        if record is None:
            return None
        return self.delete_repository(record.repository_id)

    @staticmethod
    def _require_not_processing(record: RepositoryProcessingRecord) -> None:
        if record.status not in {
            RepositoryProcessingStatus.READY,
            RepositoryProcessingStatus.FAILED,
        }:
            raise RepositoryLifecycleConflict(
                "Repository cannot be deleted while processing is active."
            )
