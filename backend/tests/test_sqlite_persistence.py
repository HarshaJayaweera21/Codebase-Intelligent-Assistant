from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.chat_routes import router as chat_router
from app.models.repository import RepositoryProcessingStatus
from app.models.repository_scan import RepositoryScanSummary
from app.persistence.sqlite_database import SQLiteDatabase
from app.services.chat_service import ChatService
from app.services.repository_processing import RepositoryProcessingStore
from app.services.repository_service import RepositoryDetails


def repository_details() -> RepositoryDetails:
    return RepositoryDetails(
        repository_id="repo_1234abcd",
        chat_id="chat_1234abcd",
        repository_name="example",
        repository_owner="owner",
        repository_url="https://github.com/owner/example",
        local_path="storage/repositories/repo_1234abcd",
        status="queued",
    )


class SQLitePersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.database = SQLiteDatabase(
            Path(self.temporary_directory.name) / "assistant.db"
        )
        self.database.initialize()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_processing_state_survives_store_recreation(self):
        store = RepositoryProcessingStore(self.database)
        repository = repository_details()
        store.create(repository)
        store.transition(
            repository.repository_id,
            RepositoryProcessingStatus.CHUNKING,
            scan_summary=RepositoryScanSummary(
                total_files=4,
                supported_files=3,
                ignored_files=1,
                languages={"Python": 3},
            ),
            chunk_count=12,
        )

        restored = RepositoryProcessingStore(self.database).get(
            repository.repository_id
        )

        self.assertIsNotNone(restored)
        self.assertEqual(restored.status, RepositoryProcessingStatus.CHUNKING)
        self.assertEqual(restored.chunk_count, 12)
        self.assertEqual(restored.scan_summary.languages, {"Python": 3})

    def test_chat_service_saves_question_answer_and_sources(self):
        store = RepositoryProcessingStore(self.database)
        store.create(repository_details())
        service = ChatService(self.database)

        service.save_exchange(
            chat_id="chat_1234abcd",
            question="Where is authentication?",
            answer="Authentication is handled by login [S1].",
            sources=[{"citation_id": "S1", "file_path": "src/auth.py"}],
        )
        chat = service.get_chat("chat_1234abcd")

        self.assertIsNotNone(chat)
        self.assertEqual(chat.summary.message_count, 2)
        self.assertEqual([message.role for message in chat.messages], ["user", "assistant"])
        self.assertEqual(
            chat.messages[1].sources[0]["file_path"],
            "src/auth.py",
        )

    def test_chat_routes_list_and_return_history(self):
        store = RepositoryProcessingStore(self.database)
        store.create(repository_details())
        service = ChatService(self.database)
        service.save_exchange(
            chat_id="chat_1234abcd",
            question="How does it work?",
            answer="It works through the service [S1].",
            sources=[{"citation_id": "S1"}],
        )
        app = FastAPI()
        app.include_router(chat_router, prefix="/api")
        app.state.chat_service = service

        with TestClient(app) as client:
            list_response = client.get("/api/chats")
            detail_response = client.get("/api/chats/chat_1234abcd")

        self.assertEqual(list_response.status_code, 200)
        self.assertEqual(list_response.json()[0]["message_count"], 2)
        self.assertEqual(detail_response.status_code, 200)
        self.assertEqual(len(detail_response.json()["messages"]), 2)


if __name__ == "__main__":
    unittest.main()
