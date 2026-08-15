import unittest

import httpx
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from app.rag.rag_service import (
    AnswerStreamConnectionError,
    NO_EVIDENCE_ANSWER,
    RagService,
)
from app.vectorstores.pinecone_vector_store import RetrievedDocument


class FakeVectorStore:
    def __init__(self, results):
        self.results = results
        self.search_calls = []

    def search(self, repository_id, query, *, top_k):
        self.search_calls.append((repository_id, query, top_k))
        return self.results


class FakeChatModel:
    def __init__(self, answer="Authentication is handled in login [S1]."):
        self.answer = answer
        self.invocations = []

    def invoke(self, messages):
        self.invocations.append(messages)
        return AIMessage(content=self.answer)

    def stream(self, messages):
        self.invocations.append(messages)
        yield AIMessage(content="Authentication is ")
        yield AIMessage(content="handled in login [S1].")


class DisconnectOnceChatModel(FakeChatModel):
    def __init__(self):
        super().__init__()
        self.stream_attempts = 0

    def stream(self, messages):
        self.invocations.append(messages)
        self.stream_attempts += 1
        if self.stream_attempts == 1:
            raise httpx.RemoteProtocolError(
                "Server disconnected without sending a response."
            )
        yield AIMessage(content="Recovered answer [S1].")


class PartialDisconnectChatModel(FakeChatModel):
    def __init__(self):
        super().__init__()
        self.stream_attempts = 0

    def stream(self, messages):
        self.invocations.append(messages)
        self.stream_attempts += 1
        yield AIMessage(content="Partial answer")
        raise httpx.RemoteProtocolError("Stream closed early.")


def retrieved_document():
    return RetrievedDocument(
        vector_id="chunk_123",
        score=0.91,
        vector_score=0.88,
        lexical_score=1.0,
        exact_match_score=1.0,
        structural_score=1.0,
        document=Document(
            page_content="def login(user):\n    authenticate(user)",
            metadata={
                "repository_id": "repo_babb91aa",
                "file_path": "src/auth.py",
                "language": "python",
                "chunk_type": "function",
                "symbol_name": "login",
                "symbol_start_line": 10,
                "symbol_end_line": 11,
                "source_ranges": [{"start_line": 10, "end_line": 11}],
            },
        ),
    )


class RagServiceTests(unittest.TestCase):
    def test_retrieves_evidence_and_prompts_model_with_citation_id(self):
        vector_store = FakeVectorStore([retrieved_document()])
        chat_model = FakeChatModel()
        service = RagService(
            vector_store=vector_store,
            chat_model=chat_model,
            default_top_k=5,
            max_context_characters=20_000,
        )

        result = service.answer(
            "repo_babb91aa",
            " Where is authentication implemented? ",
        )

        self.assertEqual(
            vector_store.search_calls,
            [("repo_babb91aa", "Where is authentication implemented?", 5)],
        )
        self.assertEqual(result.answer, chat_model.answer)
        self.assertEqual(result.sources[0].citation_id, "S1")
        self.assertEqual(result.sources[0].file_path, "src/auth.py")
        self.assertEqual(
            result.sources[0].source_ranges,
            [{"start_line": 10, "end_line": 11}],
        )
        prompt = chat_model.invocations[0][1].content
        self.assertIn("[S1]", prompt)
        self.assertIn("file: src/auth.py", prompt)
        self.assertIn("symbol: login", prompt)
        self.assertIn("lines: 10-11", prompt)
        self.assertIn("authenticate(user)", prompt)

    def test_empty_retrieval_returns_safe_answer_without_calling_model(self):
        vector_store = FakeVectorStore([])
        chat_model = FakeChatModel()
        service = RagService(
            vector_store=vector_store,
            chat_model=chat_model,
            default_top_k=5,
            max_context_characters=20_000,
        )

        result = service.answer("repo_babb91aa", "Unknown behavior?")

        self.assertEqual(result.answer, NO_EVIDENCE_ANSWER)
        self.assertEqual(result.sources, [])
        self.assertEqual(chat_model.invocations, [])

    def test_rejects_blank_question(self):
        service = RagService(
            vector_store=FakeVectorStore([]),
            chat_model=FakeChatModel(),
            default_top_k=5,
            max_context_characters=20_000,
        )

        with self.assertRaisesRegex(ValueError, "question must not be empty"):
            service.answer("repo_babb91aa", "   ")

    def test_stream_reuses_retrieval_and_yields_model_chunks(self):
        vector_store = FakeVectorStore([retrieved_document()])
        chat_model = FakeChatModel()
        service = RagService(
            vector_store=vector_store,
            chat_model=chat_model,
            default_top_k=5,
            max_context_characters=20_000,
        )

        result = service.stream_answer(
            "repo_babb91aa",
            "Where is authentication?",
            top_k=3,
        )

        self.assertEqual(
            "".join(result.chunks),
            "Authentication is handled in login [S1].",
        )
        self.assertEqual(result.sources[0].citation_id, "S1")
        self.assertEqual(
            vector_store.search_calls,
            [("repo_babb91aa", "Where is authentication?", 3)],
        )

    def test_stream_retries_disconnect_before_first_token(self):
        chat_model = DisconnectOnceChatModel()
        service = RagService(
            vector_store=FakeVectorStore([retrieved_document()]),
            chat_model=chat_model,
            default_top_k=5,
            max_context_characters=20_000,
            stream_retry_attempts=2,
        )

        result = service.stream_answer(
            "repo_babb91aa",
            "Where is authentication?",
        )

        self.assertEqual("".join(result.chunks), "Recovered answer [S1].")
        self.assertEqual(chat_model.stream_attempts, 2)

    def test_stream_does_not_retry_after_emitting_partial_answer(self):
        chat_model = PartialDisconnectChatModel()
        service = RagService(
            vector_store=FakeVectorStore([retrieved_document()]),
            chat_model=chat_model,
            default_top_k=5,
            max_context_characters=20_000,
            stream_retry_attempts=2,
        )
        result = service.stream_answer(
            "repo_babb91aa",
            "Where is authentication?",
        )

        with self.assertRaises(AnswerStreamConnectionError):
            list(result.chunks)

        self.assertEqual(chat_model.stream_attempts, 1)


if __name__ == "__main__":
    unittest.main()
