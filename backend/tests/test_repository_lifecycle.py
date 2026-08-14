from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.chat_routes import router as chat_router
from app.models.repository import RepositoryProcessingStatus
from app.persistence.sqlite_database import SQLiteDatabase
from app.services.chat_service import ChatService
from app.services.repository_lifecycle import (
    RepositoryLifecycleConflict,
    RepositoryLifecycleService,
)
from app.services.repository_processing import RepositoryProcessingStore
from app.services.repository_service import RepositoryDetails


class FakeVectorStore:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    def delete_repository(self, repository_id: str) -> None:
        self.deleted.append(repository_id)


class RepositoryLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.database_directory = TemporaryDirectory()
        self.database = SQLiteDatabase(
            Path(self.database_directory.name) / "assistant.db"
        )
        self.database.initialize()
        self.store = RepositoryProcessingStore(self.database)
        self.storage_root = Path("storage/repositories")
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.repository_temp = TemporaryDirectory(dir=self.storage_root)
        self.repository_directory = Path(self.repository_temp.name)
        self.repository = RepositoryDetails(
            repository_id="repo_1234abcd",
            chat_id="chat_1234abcd",
            repository_name="example",
            repository_owner="owner",
            repository_url="https://github.com/owner/example",
            local_path=str(self.repository_directory),
            status="queued",
        )
        self.store.create(self.repository)
        self.vector_store = FakeVectorStore()
        self.service = RepositoryLifecycleService(
            processing_store=self.store,
            vector_store=self.vector_store,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.repository_directory, ignore_errors=True)
        self.repository_temp.cleanup()
        self.database_directory.cleanup()

    def test_delete_removes_pinecone_folder_and_sqlite_cascade(self):
        self.store.transition(
            self.repository.repository_id,
            RepositoryProcessingStatus.READY,
        )
        result = self.service.delete_chat(self.repository.chat_id)

        self.assertEqual(result.repository_id, self.repository.repository_id)
        self.assertEqual(self.vector_store.deleted, [self.repository.repository_id])
        self.assertFalse(self.repository_directory.exists())
        self.assertIsNone(self.store.get(self.repository.repository_id))
        self.assertIsNone(ChatService(self.database).get_chat(self.repository.chat_id))

    def test_active_repository_cannot_be_deleted(self):
        self.store.transition(
            self.repository.repository_id,
            RepositoryProcessingStatus.EMBEDDING,
        )

        with self.assertRaises(RepositoryLifecycleConflict):
            self.service.delete_repository(self.repository.repository_id)

        self.assertTrue(self.repository_directory.exists())
        self.assertEqual(self.vector_store.deleted, [])

    def test_interrupted_processing_is_marked_failed_and_retryable(self):
        self.store.transition(
            self.repository.repository_id,
            RepositoryProcessingStatus.INDEXING,
        )

        recovered = RepositoryProcessingStore(
            self.database
        ).recover_interrupted()
        record = self.store.get(self.repository.repository_id)

        self.assertEqual(recovered, 1)
        self.assertEqual(record.status, RepositoryProcessingStatus.FAILED)
        self.assertIn("interrupted", record.error)

    def test_delete_chat_endpoint_returns_deleted_repository_ids(self):
        self.store.transition(
            self.repository.repository_id,
            RepositoryProcessingStatus.READY,
        )
        app = FastAPI()
        app.include_router(chat_router, prefix="/api")
        app.state.chat_service = ChatService(self.database)
        app.state.repository_lifecycle_service = self.service

        with TestClient(app) as client:
            response = client.delete(
                f"/api/chats/{self.repository.chat_id}"
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["repository_id"], self.repository.repository_id)
        self.assertTrue(response.json()["deleted"])


if __name__ == "__main__":
    unittest.main()
