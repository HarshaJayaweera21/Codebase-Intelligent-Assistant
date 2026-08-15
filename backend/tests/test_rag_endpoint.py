import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.main import create_app
from app.models.repository import RepositoryProcessingStatus
from app.persistence.sqlite_database import SQLiteDatabase
from app.rag.rag_service import RagAnswer, RagAnswerStream, RagSource
from app.services.chat_service import ChatService
from app.services.repository_processing import RepositoryProcessingStore
from app.services.repository_service import RepositoryDetails


class FakeRagService:
    def __init__(self):
        self.calls = []
        self.stream_calls = []

    def answer(self, repository_id, question, *, top_k=None):
        self.calls.append((repository_id, question, top_k))
        return RagAnswer(
            answer="The login function handles authentication [S1].",
            sources=[
                RagSource(
                    citation_id="S1",
                    vector_id="chunk_123",
                    score=0.9,
                    vector_score=0.85,
                    lexical_score=1.0,
                    exact_match_score=1.0,
                    structural_score=1.0,
                    file_path="src/auth.py",
                    language="python",
                    chunk_type="function",
                    symbol_name="login",
                    symbol_start_line=10,
                    symbol_end_line=11,
                    source_ranges=[{"start_line": 10, "end_line": 11}],
                    content="def login(): pass",
                )
            ],
        )

    def stream_answer(self, repository_id, question, *, top_k=None):
        self.stream_calls.append((repository_id, question, top_k))
        answer = self.answer(repository_id, question, top_k=top_k)
        return RagAnswerStream(
            chunks=iter(["The login function ", "handles authentication [S1]."]),
            sources=answer.sources,
        )


def test_lifespan(rag_service):
    @asynccontextmanager
    async def lifespan(app):
        app.state.rag_service = rag_service
        app.state.repository_processing_store = RepositoryProcessingStore()
        app.state.settings = SimpleNamespace()
        yield

    return lifespan


class RagEndpointTests(unittest.TestCase):
    def test_returns_grounded_answer_and_sources(self):
        service = FakeRagService()
        app = create_app(lifespan=test_lifespan(service))

        with TestClient(app) as client:
            response = client.post(
                "/api/repositories/repo_babb91aa/ask",
                json={
                    "question": "Where is authentication implemented?",
                    "top_k": 5,
                },
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["sources"][0]["citation_id"], "S1")
        self.assertEqual(body["sources"][0]["file_path"], "src/auth.py")
        self.assertEqual(
            service.calls,
            [
                (
                    "repo_babb91aa",
                    "Where is authentication implemented?",
                    5,
                )
            ],
        )

    def test_returns_503_when_gemini_rag_is_disabled(self):
        app = create_app(lifespan=test_lifespan(None))

        with TestClient(app) as client:
            response = client.post(
                "/api/repositories/repo_babb91aa/ask",
                json={"question": "Where is authentication?"},
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn("GEMINI_ENABLED", response.json()["detail"])

    def test_configured_question_length_limit_is_enforced(self):
        service = FakeRagService()

        @asynccontextmanager
        async def lifespan(app):
            app.state.rag_service = service
            app.state.repository_processing_store = RepositoryProcessingStore()
            app.state.settings = SimpleNamespace(
                rag_max_question_characters=10
            )
            yield

        app = create_app(lifespan=lifespan)
        with TestClient(app) as client:
            response = client.post(
                "/api/repositories/repo_babb91aa/ask",
                json={"question": "This question is too long"},
            )

        self.assertEqual(response.status_code, 422)
        self.assertIn("10 characters", response.json()["detail"])

    def test_successful_answer_is_saved_to_the_repository_chat(self):
        service = FakeRagService()
        with TemporaryDirectory() as temporary_directory:
            database = SQLiteDatabase(
                Path(temporary_directory) / "assistant.db"
            )
            database.initialize()
            processing_store = RepositoryProcessingStore(database)
            processing_store.create(
                RepositoryDetails(
                    repository_id="repo_babb91aa",
                    chat_id="chat_babb91aa",
                    repository_name="example",
                    repository_owner="owner",
                    repository_url="https://github.com/owner/example",
                    local_path="storage/repositories/repo_babb91aa",
                    status="queued",
                )
            )
            processing_store.transition(
                "repo_babb91aa",
                RepositoryProcessingStatus.READY,
            )
            chat_service = ChatService(database)

            @asynccontextmanager
            async def lifespan(app):
                app.state.rag_service = service
                app.state.repository_processing_store = processing_store
                app.state.chat_service = chat_service
                app.state.settings = SimpleNamespace()
                yield

            app = create_app(lifespan=lifespan)
            with TestClient(app) as client:
                response = client.post(
                    "/api/repositories/repo_babb91aa/ask",
                    json={"question": "Where is authentication?"},
                )

            chat = chat_service.get_chat("chat_babb91aa")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["chat_id"], "chat_babb91aa")
        self.assertEqual(chat.summary.message_count, 2)
        self.assertEqual(chat.messages[0].role, "user")
        self.assertEqual(chat.messages[1].sources[0]["citation_id"], "S1")

    def test_stream_returns_sse_events_and_saves_completed_answer(self):
        service = FakeRagService()
        with TemporaryDirectory() as temporary_directory:
            database = SQLiteDatabase(Path(temporary_directory) / "assistant.db")
            database.initialize()
            processing_store = RepositoryProcessingStore(database)
            processing_store.create(
                RepositoryDetails(
                    repository_id="repo_babb91aa",
                    chat_id="chat_babb91aa",
                    repository_name="example",
                    repository_owner="owner",
                    repository_url="https://github.com/owner/example",
                    local_path="storage/repositories/repo_babb91aa",
                    status="queued",
                )
            )
            processing_store.transition(
                "repo_babb91aa",
                RepositoryProcessingStatus.READY,
            )
            chat_service = ChatService(database)

            @asynccontextmanager
            async def lifespan(app):
                app.state.rag_service = service
                app.state.repository_processing_store = processing_store
                app.state.chat_service = chat_service
                app.state.settings = SimpleNamespace(
                    rag_max_question_characters=4_000
                )
                yield

            app = create_app(lifespan=lifespan)
            with TestClient(app) as client:
                response = client.post(
                    "/api/repositories/repo_babb91aa/ask/stream",
                    json={"question": "Where is authentication?", "top_k": 5},
                )
            chat = chat_service.get_chat("chat_babb91aa")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
        self.assertIn("event: sources", response.text)
        self.assertIn("event: token", response.text)
        self.assertIn("event: done", response.text)
        self.assertEqual(chat.summary.message_count, 2)
        self.assertEqual(
            chat.messages[1].content,
            "The login function handles authentication [S1].",
        )


if __name__ == "__main__":
    unittest.main()
