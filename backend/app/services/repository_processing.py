from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace
from datetime import UTC, datetime
import json
import logging
import os
from pathlib import Path
import shutil
import stat
from threading import BoundedSemaphore, Lock, Thread
from typing import TYPE_CHECKING, Callable

from app.core.config import Settings
from app.core.exceptions import (
    RepositoryCloneError,
    RepositoryLimitError,
    RepositoryScanError,
)
from app.models.repository import RepositoryProcessingStatus
from app.models.repository_scan import RepositoryScanSummary
from app.rag.chunking_router import chunk_repository_files
from app.rag.document_processor import create_langchain_documents
from app.services.repository_scanner import scan_repository
from app.services.repository_service import RepositoryDetails, clone_repository
from app.vectorstores.pinecone_vector_store import PineconeVectorStore

if TYPE_CHECKING:
    from app.persistence.sqlite_database import SQLiteDatabase


logger = logging.getLogger(__name__)

STATUS_PROGRESS = {
    RepositoryProcessingStatus.QUEUED: 0,
    RepositoryProcessingStatus.CLONING: 10,
    RepositoryProcessingStatus.SCANNING: 25,
    RepositoryProcessingStatus.CHUNKING: 45,
    RepositoryProcessingStatus.EMBEDDING: 60,
    RepositoryProcessingStatus.INDEXING: 85,
    RepositoryProcessingStatus.READY: 100,
}
STATUS_MESSAGES = {
    RepositoryProcessingStatus.QUEUED: "Repository processing is queued.",
    RepositoryProcessingStatus.CLONING: "Cloning the GitHub repository.",
    RepositoryProcessingStatus.SCANNING: "Scanning supported repository files.",
    RepositoryProcessingStatus.CHUNKING: "Creating structural source chunks.",
    RepositoryProcessingStatus.EMBEDDING: "Generating local document embeddings.",
    RepositoryProcessingStatus.INDEXING: "Uploading vectors to Pinecone.",
    RepositoryProcessingStatus.READY: "Repository is ready for questions.",
    RepositoryProcessingStatus.FAILED: "Repository processing failed.",
}


@dataclass(frozen=True)
class RepositoryProcessingRecord:
    repository_id: str
    chat_id: str
    repository_name: str
    repository_owner: str
    repository_url: str
    local_path: str
    status: RepositoryProcessingStatus
    progress_percent: int
    status_message: str
    created_at: datetime
    updated_at: datetime
    scan_summary: RepositoryScanSummary | None = None
    chunk_count: int | None = None
    indexed_document_count: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class RepositoryProcessingLimits:
    clone_timeout_seconds: int = 120
    max_repository_size_bytes: int = 262_144_000
    max_files: int = 20_000
    max_chunks: int = 10_000

    @classmethod
    def from_settings(cls, settings: Settings) -> RepositoryProcessingLimits:
        return cls(
            clone_timeout_seconds=settings.repository_clone_timeout_seconds,
            max_repository_size_bytes=settings.repository_max_size_bytes,
            max_files=settings.repository_max_files,
            max_chunks=settings.repository_max_chunks,
        )


class RepositoryJobCoordinator:
    """Limits heavy repository jobs sharing the in-process embedding model."""

    def __init__(self, max_concurrent_jobs: int) -> None:
        self._semaphore = BoundedSemaphore(max_concurrent_jobs)
        self._running_repository_ids: set[str] = set()
        self._running_lock = Lock()

    def slot(self):
        return self._semaphore

    def submit(
        self,
        repository_id: str,
        job: Callable[..., None],
        *args: object,
    ) -> bool:
        """Start one durable in-process worker per repository.

        FastAPI response background tasks can be lost when a development client
        disconnects or the response lifecycle is interrupted. Starting the
        worker here makes submission explicit while the persisted status still
        allows startup recovery after a process restart.
        """
        with self._running_lock:
            if repository_id in self._running_repository_ids:
                return False
            self._running_repository_ids.add(repository_id)

        worker = Thread(
            target=self._run_submitted_job,
            args=(repository_id, job, args),
            daemon=True,
            name=f"repository-job-{repository_id}",
        )
        try:
            worker.start()
        except Exception:
            with self._running_lock:
                self._running_repository_ids.discard(repository_id)
            raise
        return True

    def is_running(self, repository_id: str) -> bool:
        with self._running_lock:
            return repository_id in self._running_repository_ids

    def _run_submitted_job(
        self,
        repository_id: str,
        job: Callable[..., None],
        args: tuple[object, ...],
    ) -> None:
        try:
            job(*args)
        finally:
            with self._running_lock:
                self._running_repository_ids.discard(repository_id)


class RepositoryProcessingStore:
    """Repository processing state backed by SQLite in the running app."""

    def __init__(self, database: SQLiteDatabase | None = None) -> None:
        self._database = database
        self._records: dict[str, RepositoryProcessingRecord] = {}
        self._lock = Lock()

    def create(self, repository: RepositoryDetails) -> RepositoryProcessingRecord:
        now = datetime.now(UTC)
        record = RepositoryProcessingRecord(
            repository_id=repository.repository_id,
            chat_id=repository.chat_id,
            repository_name=repository.repository_name,
            repository_owner=repository.repository_owner,
            repository_url=repository.repository_url,
            local_path=repository.local_path,
            status=RepositoryProcessingStatus.QUEUED,
            progress_percent=STATUS_PROGRESS[RepositoryProcessingStatus.QUEUED],
            status_message=STATUS_MESSAGES[RepositoryProcessingStatus.QUEUED],
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            if self._database is not None:
                with self._database.connect() as connection:
                    existing = connection.execute(
                        "SELECT 1 FROM repositories WHERE repository_id = ?",
                        (record.repository_id,),
                    ).fetchone()
                    if existing is not None:
                        raise ValueError(
                            "Repository processing record already exists."
                        )
                    connection.execute(
                        """
                        INSERT INTO repositories (
                            repository_id, repository_name, repository_owner,
                            repository_url, local_path, status,
                            progress_percent, status_message, created_at,
                            updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            record.repository_id,
                            record.repository_name,
                            record.repository_owner,
                            record.repository_url,
                            record.local_path,
                            record.status.value,
                            record.progress_percent,
                            record.status_message,
                            record.created_at.isoformat(),
                            record.updated_at.isoformat(),
                        ),
                    )
                    connection.execute(
                        """
                        INSERT INTO chats (
                            chat_id, repository_id, title, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            record.chat_id,
                            record.repository_id,
                            record.repository_name,
                            record.created_at.isoformat(),
                            record.updated_at.isoformat(),
                        ),
                    )
                return record
            if record.repository_id in self._records:
                raise ValueError("Repository processing record already exists.")
            self._records[record.repository_id] = record
        return record

    def get(self, repository_id: str) -> RepositoryProcessingRecord | None:
        with self._lock:
            if self._database is not None:
                with self._database.connect() as connection:
                    row = connection.execute(
                        """
                        SELECT repositories.*, chats.chat_id
                        FROM repositories
                        JOIN chats USING (repository_id)
                        WHERE repositories.repository_id = ?
                        """,
                        (repository_id,),
                    ).fetchone()
                return _record_from_row(row) if row is not None else None
            return self._records.get(repository_id)

    def list(self) -> list[RepositoryProcessingRecord]:
        with self._lock:
            if self._database is not None:
                with self._database.connect() as connection:
                    rows = connection.execute(
                        """
                        SELECT repositories.*, chats.chat_id
                        FROM repositories
                        JOIN chats USING (repository_id)
                        ORDER BY repositories.updated_at DESC
                        """
                    ).fetchall()
                return [_record_from_row(row) for row in rows]
            return sorted(
                self._records.values(),
                key=lambda record: record.updated_at,
                reverse=True,
            )

    def find_by_url(self, repository_url: str) -> RepositoryProcessingRecord | None:
        normalized_url = repository_url.rstrip("/")
        with self._lock:
            if self._database is not None:
                with self._database.connect() as connection:
                    row = connection.execute(
                        """
                        SELECT repositories.*, chats.chat_id
                        FROM repositories
                        JOIN chats USING (repository_id)
                        WHERE repositories.repository_url = ?
                        ORDER BY repositories.created_at DESC
                        LIMIT 1
                        """,
                        (normalized_url,),
                    ).fetchone()
                return _record_from_row(row) if row is not None else None
            return next(
                (
                    record
                    for record in self._records.values()
                    if record.repository_url == normalized_url
                ),
                None,
            )

    def reset(self, repository_id: str) -> RepositoryProcessingRecord:
        with self._lock:
            current = self._require(repository_id)
            record = replace(
                current,
                status=RepositoryProcessingStatus.QUEUED,
                progress_percent=STATUS_PROGRESS[RepositoryProcessingStatus.QUEUED],
                status_message=STATUS_MESSAGES[RepositoryProcessingStatus.QUEUED],
                updated_at=datetime.now(UTC),
                scan_summary=None,
                chunk_count=None,
                indexed_document_count=None,
                error=None,
            )
            if self._database is not None:
                self._update_database_record(record)
            else:
                self._records[repository_id] = record
            return record

    def recover_interrupted(self) -> int:
        active_statuses = {
            RepositoryProcessingStatus.QUEUED,
            RepositoryProcessingStatus.CLONING,
            RepositoryProcessingStatus.SCANNING,
            RepositoryProcessingStatus.CHUNKING,
            RepositoryProcessingStatus.EMBEDDING,
            RepositoryProcessingStatus.INDEXING,
        }
        recovered = 0
        with self._lock:
            records = self.list_unlocked()
            for current in records:
                if current.status not in active_statuses:
                    continue
                record = replace(
                    current,
                    status=RepositoryProcessingStatus.FAILED,
                    status_message=STATUS_MESSAGES[RepositoryProcessingStatus.FAILED],
                    updated_at=datetime.now(UTC),
                    error=(
                        "Repository processing was interrupted by an application "
                        "restart. Retry the repository to continue."
                    ),
                )
                if self._database is not None:
                    self._update_database_record(record)
                else:
                    self._records[record.repository_id] = record
                recovered += 1
        return recovered

    def delete(self, repository_id: str) -> bool:
        with self._lock:
            if self._database is not None:
                with self._database.connect() as connection:
                    cursor = connection.execute(
                        "DELETE FROM repositories WHERE repository_id = ?",
                        (repository_id,),
                    )
                return cursor.rowcount > 0
            return self._records.pop(repository_id, None) is not None

    def list_unlocked(self) -> list[RepositoryProcessingRecord]:
        if self._database is not None:
            with self._database.connect() as connection:
                rows = connection.execute(
                    """
                    SELECT repositories.*, chats.chat_id
                    FROM repositories JOIN chats USING (repository_id)
                    """
                ).fetchall()
            return [_record_from_row(row) for row in rows]
        return list(self._records.values())

    def transition(
        self,
        repository_id: str,
        status: RepositoryProcessingStatus,
        *,
        scan_summary: RepositoryScanSummary | None = None,
        chunk_count: int | None = None,
        indexed_document_count: int | None = None,
    ) -> RepositoryProcessingRecord:
        with self._lock:
            current = self._require(repository_id)
            record = replace(
                current,
                status=status,
                progress_percent=STATUS_PROGRESS[status],
                status_message=STATUS_MESSAGES[status],
                updated_at=datetime.now(UTC),
                scan_summary=(scan_summary or current.scan_summary),
                chunk_count=(
                    chunk_count if chunk_count is not None else current.chunk_count
                ),
                indexed_document_count=(
                    indexed_document_count
                    if indexed_document_count is not None
                    else current.indexed_document_count
                ),
                error=None,
            )
            if self._database is not None:
                self._update_database_record(record)
                return record
            self._records[repository_id] = record
            return record

    def fail(self, repository_id: str, error: str) -> RepositoryProcessingRecord:
        with self._lock:
            current = self._require(repository_id)
            record = replace(
                current,
                status=RepositoryProcessingStatus.FAILED,
                status_message=STATUS_MESSAGES[RepositoryProcessingStatus.FAILED],
                updated_at=datetime.now(UTC),
                error=error,
            )
            if self._database is not None:
                self._update_database_record(record)
                return record
            self._records[repository_id] = record
            return record

    def _require(self, repository_id: str) -> RepositoryProcessingRecord:
        if self._database is not None:
            with self._database.connect() as connection:
                row = connection.execute(
                    """
                    SELECT repositories.*, chats.chat_id
                    FROM repositories
                    JOIN chats USING (repository_id)
                    WHERE repositories.repository_id = ?
                    """,
                    (repository_id,),
                ).fetchone()
            if row is not None:
                return _record_from_row(row)
            raise KeyError("Repository processing record was not found.")
        try:
            return self._records[repository_id]
        except KeyError as error:
            raise KeyError("Repository processing record was not found.") from error

    def _update_database_record(
        self,
        record: RepositoryProcessingRecord,
    ) -> None:
        scan_summary_json = (
            record.scan_summary.model_dump_json()
            if record.scan_summary is not None
            else None
        )
        with self._database.connect() as connection:
            connection.execute(
                """
                UPDATE repositories
                SET status = ?, progress_percent = ?, status_message = ?,
                    updated_at = ?, scan_summary_json = ?, chunk_count = ?,
                    indexed_document_count = ?, error = ?
                WHERE repository_id = ?
                """,
                (
                    record.status.value,
                    record.progress_percent,
                    record.status_message,
                    record.updated_at.isoformat(),
                    scan_summary_json,
                    record.chunk_count,
                    record.indexed_document_count,
                    record.error,
                    record.repository_id,
                ),
            )


def _record_from_row(row: object) -> RepositoryProcessingRecord:
    scan_summary_json = row["scan_summary_json"]
    scan_summary = (
        RepositoryScanSummary.model_validate(json.loads(scan_summary_json))
        if scan_summary_json
        else None
    )
    return RepositoryProcessingRecord(
        repository_id=row["repository_id"],
        chat_id=row["chat_id"],
        repository_name=row["repository_name"],
        repository_owner=row["repository_owner"],
        repository_url=row["repository_url"],
        local_path=row["local_path"],
        status=RepositoryProcessingStatus(row["status"]),
        progress_percent=row["progress_percent"],
        status_message=row["status_message"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        scan_summary=scan_summary,
        chunk_count=row["chunk_count"],
        indexed_document_count=row["indexed_document_count"],
        error=row["error"],
    )


def process_repository_pipeline(
    repository: RepositoryDetails,
    processing_store: RepositoryProcessingStore,
    vector_store: PineconeVectorStore,
    limits: RepositoryProcessingLimits | None = None,
    coordinator: RepositoryJobCoordinator | None = None,
) -> None:
    _run_repository_pipeline(
        repository,
        processing_store,
        vector_store,
        limits=limits or RepositoryProcessingLimits(),
        coordinator=coordinator,
        clone_first=True,
        clean_on_failure=True,
    )


def reindex_repository_pipeline(
    repository: RepositoryDetails,
    processing_store: RepositoryProcessingStore,
    vector_store: PineconeVectorStore,
    limits: RepositoryProcessingLimits | None = None,
    coordinator: RepositoryJobCoordinator | None = None,
) -> None:
    _run_repository_pipeline(
        repository,
        processing_store,
        vector_store,
        limits=limits or RepositoryProcessingLimits(),
        coordinator=coordinator,
        clone_first=False,
        clean_on_failure=False,
    )


def _run_repository_pipeline(
    repository: RepositoryDetails,
    processing_store: RepositoryProcessingStore,
    vector_store: PineconeVectorStore,
    *,
    limits: RepositoryProcessingLimits,
    coordinator: RepositoryJobCoordinator | None,
    clone_first: bool,
    clean_on_failure: bool,
) -> None:
    repository_path = Path(repository.local_path)
    slot = coordinator.slot() if coordinator is not None else nullcontext()
    with slot:
        try:
            if clone_first:
                safe_remove_repository_directory(repository_path)
                processing_store.transition(
                    repository.repository_id,
                    RepositoryProcessingStatus.CLONING,
                )
                clone_repository(
                    repository.repository_url,
                    repository_path,
                    timeout_seconds=limits.clone_timeout_seconds,
                )
            elif not repository_path.is_dir():
                raise RepositoryScanError(
                    "The local repository is missing. Retry cloning before reindexing."
                )

            _validate_repository_size(repository_path, limits)
            processing_store.transition(
                repository.repository_id,
                RepositoryProcessingStatus.SCANNING,
            )
            try:
                scan_result = scan_repository(repository_path)
            except (OSError, FileNotFoundError, NotADirectoryError) as error:
                raise RepositoryScanError(
                    "The repository was cloned, but its files could not be scanned."
                ) from error
            if scan_result.summary.total_files > limits.max_files:
                raise RepositoryLimitError(
                    f"Repository contains {scan_result.summary.total_files} files, "
                    f"exceeding the configured limit of {limits.max_files}."
                )

            processing_store.transition(
                repository.repository_id,
                RepositoryProcessingStatus.CHUNKING,
                scan_summary=scan_result.summary,
            )
            chunks = chunk_repository_files(
                repository.repository_id,
                scan_result.files,
            )
            if len(chunks) > limits.max_chunks:
                raise RepositoryLimitError(
                    f"Repository produced {len(chunks)} chunks, exceeding the "
                    f"configured limit of {limits.max_chunks}."
                )
            documents = create_langchain_documents(chunks)

            processing_store.transition(
                repository.repository_id,
                RepositoryProcessingStatus.EMBEDDING,
                chunk_count=len(chunks),
            )

            def report_vector_stage(stage: str) -> None:
                if stage == "indexing":
                    processing_store.transition(
                        repository.repository_id,
                        RepositoryProcessingStatus.INDEXING,
                    )

            indexed_document_count = vector_store.index_documents(
                repository.repository_id,
                documents,
                replace_namespace=True,
                progress_callback=report_vector_stage,
            )
            processing_store.transition(
                repository.repository_id,
                RepositoryProcessingStatus.READY,
                indexed_document_count=indexed_document_count,
            )
        except Exception as error:
            logger.exception(
                "Repository processing failed for %s",
                repository.repository_id,
            )
            if clean_on_failure:
                _cleanup_failed_repository(repository, vector_store)
            processing_store.fail(
                repository.repository_id,
                _public_processing_error(error),
            )


def _cleanup_failed_repository(
    repository: RepositoryDetails,
    vector_store: PineconeVectorStore,
) -> None:
    try:
        safe_remove_repository_directory(Path(repository.local_path))
    except Exception:
        # Cleanup is best-effort here. Never mask the processing error or leave
        # the persisted job in an active state merely because Windows/OneDrive
        # temporarily holds a cloned Git object.
        logger.exception(
            "Failed to clean local repository directory %s",
            repository.repository_id,
        )

    try:
        vector_store.delete_repository(repository.repository_id)
    except Exception:
        logger.exception(
            "Failed to clean Pinecone namespace %s",
            repository.repository_id,
        )


def _public_processing_error(error: Exception) -> str:
    if isinstance(
        error,
        (RepositoryCloneError, RepositoryScanError, RepositoryLimitError),
    ):
        return error.message
    return "Repository processing failed unexpectedly. Check backend logs."


def safe_remove_repository_directory(repository_path: Path) -> bool:
    resolved_path = repository_path.resolve()
    storage_root = Path("storage/repositories").resolve()
    if resolved_path.parent != storage_root:
        raise ValueError("Refusing to remove a path outside repository storage.")
    if not resolved_path.exists():
        return False
    shutil.rmtree(resolved_path, onexc=_remove_readonly_path)
    return True


def _remove_readonly_path(
    remove: Callable[[str], object],
    path: str,
    error: BaseException,
) -> None:
    """Clear the Windows read-only bit on Git objects and retry deletion."""
    try:
        os.chmod(path, stat.S_IWRITE)
        remove(path)
    except OSError:
        raise error


def repository_details_from_record(
    record: RepositoryProcessingRecord,
) -> RepositoryDetails:
    return RepositoryDetails(
        repository_id=record.repository_id,
        chat_id=record.chat_id,
        repository_name=record.repository_name,
        repository_owner=record.repository_owner,
        repository_url=record.repository_url,
        local_path=record.local_path,
        status=record.status.value,
        scan_summary=record.scan_summary,
    )


def _validate_repository_size(
    repository_path: Path,
    limits: RepositoryProcessingLimits,
) -> None:
    total_bytes = 0
    for path in repository_path.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            total_bytes += path.stat().st_size
        except OSError:
            continue
        if total_bytes > limits.max_repository_size_bytes:
            raise RepositoryLimitError(
                f"Repository size exceeds the configured limit of "
                f"{limits.max_repository_size_bytes} bytes."
            )
