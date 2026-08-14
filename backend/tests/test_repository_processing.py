import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.repository_routes import router
from app.core.exceptions import RepositoryCloneError
from app.models.repository import RepositoryProcessingStatus
from app.models.repository_file import RepositoryFile
from app.models.repository_scan import RepositoryScanResult, RepositoryScanSummary
from app.services.repository_processing import (
    RepositoryJobCoordinator,
    RepositoryProcessingLimits,
    RepositoryProcessingStore,
    process_repository_pipeline,
)
from app.services.repository_service import RepositoryDetails


class FakePipelineVectorStore:
    def __init__(self) -> None:
        self.indexed_repository_id: str | None = None
        self.deleted_repository_ids: list[str] = []

    def index_documents(
        self,
        repository_id,
        documents,
        *,
        replace_namespace,
        progress_callback,
    ):
        self.indexed_repository_id = repository_id
        progress_callback("embedding")
        progress_callback("indexing")
        return len(documents)

    def delete_repository(self, repository_id: str) -> None:
        self.deleted_repository_ids.append(repository_id)


def repository_details(local_path: Path) -> RepositoryDetails:
    return RepositoryDetails(
        repository_id="repo_1234abcd",
        chat_id="chat_1234abcd",
        repository_name="example",
        repository_owner="owner",
        repository_url="https://github.com/owner/example",
        local_path=str(local_path),
        status="queued",
    )


class RepositoryProcessingPipelineTests(unittest.TestCase):
    def test_pipeline_reaches_ready_with_stage_counts(self):
        storage_root = Path("storage/repositories")
        storage_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=storage_root) as temporary_directory:
            repository = repository_details(Path(temporary_directory))
            shutil.rmtree(temporary_directory)
            store = RepositoryProcessingStore()
            store.create(repository)
            vector_store = FakePipelineVectorStore()
            scan_result = RepositoryScanResult(
                summary=RepositoryScanSummary(
                    total_files=1,
                    supported_files=1,
                    ignored_files=0,
                    languages={"Python": 1},
                ),
                files=[
                    RepositoryFile(
                        relative_path="app.py",
                        language="Python",
                        size_bytes=20,
                        content="def run():\n    pass\n",
                    )
                ],
            )

            def fake_clone(
                _url: str,
                destination: Path,
                *,
                timeout_seconds: int,
            ) -> None:
                destination.mkdir(parents=True)

            with (
                patch(
                    "app.services.repository_processing.clone_repository",
                    side_effect=fake_clone,
                ),
                patch(
                    "app.services.repository_processing.scan_repository",
                    return_value=scan_result,
                ),
            ):
                process_repository_pipeline(repository, store, vector_store)

            record = store.get(repository.repository_id)
            self.assertEqual(record.status, RepositoryProcessingStatus.READY)
            self.assertEqual(record.progress_percent, 100)
            self.assertEqual(record.chunk_count, 1)
            self.assertEqual(record.indexed_document_count, 1)
            self.assertEqual(record.scan_summary.supported_files, 1)
            self.assertEqual(
                vector_store.indexed_repository_id,
                repository.repository_id,
            )

    def test_failure_marks_job_and_cleans_local_and_pinecone_data(self):
        storage_root = Path("storage/repositories")
        storage_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=storage_root) as temporary_directory:
            repository = repository_details(Path(temporary_directory))
            store = RepositoryProcessingStore()
            store.create(repository)
            vector_store = FakePipelineVectorStore()

            with patch(
                "app.services.repository_processing.clone_repository",
                side_effect=RepositoryCloneError("Clone failed safely."),
            ):
                process_repository_pipeline(repository, store, vector_store)

            record = store.get(repository.repository_id)
            self.assertEqual(record.status, RepositoryProcessingStatus.FAILED)
            self.assertEqual(record.error, "Clone failed safely.")
            self.assertFalse(Path(repository.local_path).exists())
            self.assertEqual(
                vector_store.deleted_repository_ids,
                [repository.repository_id],
            )

    def test_cleanup_failure_does_not_leave_job_in_active_state(self):
        storage_root = Path("storage/repositories")
        storage_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=storage_root) as temporary_directory:
            repository = repository_details(Path(temporary_directory))
            store = RepositoryProcessingStore()
            store.create(repository)
            vector_store = FakePipelineVectorStore()

            with (
                patch(
                    "app.services.repository_processing.clone_repository",
                    side_effect=RepositoryCloneError("Clone failed safely."),
                ),
                patch(
                    "app.services.repository_processing.safe_remove_repository_directory",
                    side_effect=[None, PermissionError("Git object is locked")],
                ),
            ):
                process_repository_pipeline(repository, store, vector_store)

            record = store.get(repository.repository_id)
            self.assertEqual(record.status, RepositoryProcessingStatus.FAILED)
            self.assertEqual(record.error, "Clone failed safely.")
            self.assertEqual(
                vector_store.deleted_repository_ids,
                [repository.repository_id],
            )

    def test_file_limit_fails_before_chunking(self):
        storage_root = Path("storage/repositories")
        storage_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=storage_root) as temporary_directory:
            repository = repository_details(Path(temporary_directory))
            shutil.rmtree(temporary_directory)
            store = RepositoryProcessingStore()
            store.create(repository)
            vector_store = FakePipelineVectorStore()
            scan_result = RepositoryScanResult(
                summary=RepositoryScanSummary(
                    total_files=2,
                    supported_files=0,
                    ignored_files=2,
                    languages={},
                ),
                files=[],
            )

            def fake_clone(_url, destination, *, timeout_seconds):
                destination.mkdir(parents=True)

            with (
                patch(
                    "app.services.repository_processing.clone_repository",
                    side_effect=fake_clone,
                ),
                patch(
                    "app.services.repository_processing.scan_repository",
                    return_value=scan_result,
                ),
            ):
                process_repository_pipeline(
                    repository,
                    store,
                    vector_store,
                    RepositoryProcessingLimits(max_files=1),
                )

            record = store.get(repository.repository_id)
            self.assertEqual(record.status, RepositoryProcessingStatus.FAILED)
            self.assertIn("configured limit of 1", record.error)


class RepositoryProcessingRouteTests(unittest.TestCase):
    def test_creation_returns_accepted_job_and_polling_url(self):
        app = FastAPI()
        app.include_router(router, prefix="/api")
        app.state.repository_processing_store = RepositoryProcessingStore()
        app.state.repository_processing_limits = RepositoryProcessingLimits()
        app.state.repository_job_coordinator = RepositoryJobCoordinator(1)
        app.state.vector_store = FakePipelineVectorStore()

        with (
            patch("app.api.repository_routes.process_repository_pipeline"),
            TestClient(app) as client,
        ):
            response = client.post(
                "/api/repositories",
                json={"repository_url": "https://github.com/owner/example"},
            )

            self.assertEqual(response.status_code, 202)
            payload = response.json()
            self.assertEqual(payload["status"], "queued")
            self.assertEqual(
                payload["status_url"],
                f"/api/repositories/{payload['repository_id']}/status",
            )

            status_response = client.get(payload["status_url"])

        self.assertEqual(status_response.status_code, 200)
        self.assertEqual(status_response.json()["progress_percent"], 0)
        self.assertEqual(status_response.json()["chat_id"], payload["chat_id"])

    def test_unknown_processing_job_returns_not_found(self):
        app = FastAPI()
        app.include_router(router, prefix="/api")
        app.state.repository_processing_store = RepositoryProcessingStore()
        app.state.repository_processing_limits = RepositoryProcessingLimits()
        app.state.repository_job_coordinator = RepositoryJobCoordinator(1)
        app.state.vector_store = FakePipelineVectorStore()

        with TestClient(app) as client:
            response = client.get("/api/repositories/repo_ffffffff/status")

        self.assertEqual(response.status_code, 404)

    def test_duplicate_repository_url_returns_existing_ids(self):
        app = FastAPI()
        app.include_router(router, prefix="/api")
        store = RepositoryProcessingStore()
        existing = repository_details(Path("storage/repositories/repo_1234abcd"))
        store.create(existing)
        app.state.repository_processing_store = store
        app.state.vector_store = FakePipelineVectorStore()
        app.state.repository_processing_limits = RepositoryProcessingLimits()
        app.state.repository_job_coordinator = RepositoryJobCoordinator(1)

        with TestClient(app) as client:
            response = client.post(
                "/api/repositories",
                json={"repository_url": existing.repository_url},
            )

        self.assertEqual(response.status_code, 409)
        detail = response.json()["detail"]
        self.assertEqual(detail["repository_id"], existing.repository_id)
        self.assertEqual(detail["chat_id"], existing.chat_id)

    def test_failed_repository_can_be_queued_for_retry(self):
        app = FastAPI()
        app.include_router(router, prefix="/api")
        store = RepositoryProcessingStore()
        repository = repository_details(
            Path("storage/repositories/repo_1234abcd")
        )
        store.create(repository)
        store.fail(repository.repository_id, "Previous failure")
        app.state.repository_processing_store = store
        app.state.vector_store = FakePipelineVectorStore()
        app.state.repository_processing_limits = RepositoryProcessingLimits()
        app.state.repository_job_coordinator = RepositoryJobCoordinator(1)

        with (
            patch("app.api.repository_routes.process_repository_pipeline") as job,
            TestClient(app) as client,
        ):
            response = client.post(
                f"/api/repositories/{repository.repository_id}/retry"
            )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["status"], "queued")
        self.assertIsNone(response.json()["error"])
        job.assert_called_once()

    def test_lists_and_returns_repository_details(self):
        app = FastAPI()
        app.include_router(router, prefix="/api")
        store = RepositoryProcessingStore()
        repository = repository_details(
            Path("storage/repositories/repo_1234abcd")
        )
        store.create(repository)
        app.state.repository_processing_store = store

        with TestClient(app) as client:
            list_response = client.get("/api/repositories")
            detail_response = client.get(
                f"/api/repositories/{repository.repository_id}"
            )

        self.assertEqual(list_response.status_code, 200)
        self.assertEqual(list_response.json()[0]["repository_id"], repository.repository_id)
        self.assertEqual(detail_response.status_code, 200)
        self.assertEqual(detail_response.json()["chat_id"], repository.chat_id)

    def test_ready_repository_can_be_queued_for_reindexing(self):
        app = FastAPI()
        app.include_router(router, prefix="/api")
        store = RepositoryProcessingStore()
        repository = repository_details(
            Path("storage/repositories/repo_1234abcd")
        )
        store.create(repository)
        store.transition(
            repository.repository_id,
            RepositoryProcessingStatus.READY,
        )
        app.state.repository_processing_store = store
        app.state.vector_store = FakePipelineVectorStore()
        app.state.repository_processing_limits = RepositoryProcessingLimits()
        app.state.repository_job_coordinator = RepositoryJobCoordinator(1)

        with (
            patch("app.api.repository_routes.reindex_repository_pipeline") as job,
            TestClient(app) as client,
        ):
            response = client.post(
                f"/api/repositories/{repository.repository_id}/reindex"
            )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["status"], "queued")
        job.assert_called_once()


if __name__ == "__main__":
    unittest.main()
