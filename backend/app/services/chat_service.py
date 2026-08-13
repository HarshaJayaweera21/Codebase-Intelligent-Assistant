from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import json
from typing import Any
from uuid import uuid4

from app.models.repository import RepositoryProcessingStatus
from app.persistence.sqlite_database import SQLiteDatabase


class ChatNotFoundError(KeyError):
    """Raised when a requested persisted chat does not exist."""


@dataclass(frozen=True)
class ChatSummary:
    chat_id: str
    repository_id: str
    title: str
    repository_name: str
    repository_owner: str
    repository_url: str
    repository_status: RepositoryProcessingStatus
    message_count: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class ChatMessage:
    message_id: str
    role: str
    content: str
    sources: list[dict[str, object]]
    created_at: datetime


@dataclass(frozen=True)
class ChatDetail:
    summary: ChatSummary
    messages: list[ChatMessage]


class ChatService:
    def __init__(self, database: SQLiteDatabase) -> None:
        self._database = database

    def list_chats(self) -> list[ChatSummary]:
        with self._database.connect() as connection:
            rows = connection.execute(
                """
                SELECT chats.*, repositories.repository_name,
                       repositories.repository_owner,
                       repositories.repository_url,
                       repositories.status AS repository_status,
                       COUNT(messages.message_id) AS message_count
                FROM chats
                JOIN repositories USING (repository_id)
                LEFT JOIN messages USING (chat_id)
                GROUP BY chats.chat_id
                ORDER BY chats.updated_at DESC
                """
            ).fetchall()
        return [_summary_from_row(row) for row in rows]

    def get_chat(self, chat_id: str) -> ChatDetail | None:
        with self._database.connect() as connection:
            chat_row = connection.execute(
                """
                SELECT chats.*, repositories.repository_name,
                       repositories.repository_owner,
                       repositories.repository_url,
                       repositories.status AS repository_status,
                       (SELECT COUNT(*) FROM messages
                        WHERE messages.chat_id = chats.chat_id) AS message_count
                FROM chats
                JOIN repositories USING (repository_id)
                WHERE chats.chat_id = ?
                """,
                (chat_id,),
            ).fetchone()
            if chat_row is None:
                return None
            message_rows = connection.execute(
                """
                SELECT * FROM messages
                WHERE chat_id = ?
                ORDER BY created_at, message_id
                """,
                (chat_id,),
            ).fetchall()
        return ChatDetail(
            summary=_summary_from_row(chat_row),
            messages=[_message_from_row(row) for row in message_rows],
        )

    def save_exchange(
        self,
        *,
        chat_id: str,
        question: str,
        answer: str,
        sources: list[dict[str, Any]],
    ) -> tuple[ChatMessage, ChatMessage]:
        question_time = datetime.now(UTC)
        answer_time = question_time + timedelta(microseconds=1)
        user_message = ChatMessage(
            message_id=f"msg_{uuid4().hex}",
            role="user",
            content=question,
            sources=[],
            created_at=question_time,
        )
        assistant_message = ChatMessage(
            message_id=f"msg_{uuid4().hex}",
            role="assistant",
            content=answer,
            sources=sources,
            created_at=answer_time,
        )
        with self._database.connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM chats WHERE chat_id = ?",
                (chat_id,),
            ).fetchone()
            if exists is None:
                raise ChatNotFoundError(chat_id)
            connection.executemany(
                """
                INSERT INTO messages (
                    message_id, chat_id, role, content, sources_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        user_message.message_id,
                        chat_id,
                        user_message.role,
                        user_message.content,
                        None,
                        user_message.created_at.isoformat(),
                    ),
                    (
                        assistant_message.message_id,
                        chat_id,
                        assistant_message.role,
                        assistant_message.content,
                        json.dumps(sources, ensure_ascii=False),
                        assistant_message.created_at.isoformat(),
                    ),
                ],
            )
            connection.execute(
                "UPDATE chats SET updated_at = ? WHERE chat_id = ?",
                (assistant_message.created_at.isoformat(), chat_id),
            )
        return user_message, assistant_message


def _summary_from_row(row: object) -> ChatSummary:
    return ChatSummary(
        chat_id=row["chat_id"],
        repository_id=row["repository_id"],
        title=row["title"],
        repository_name=row["repository_name"],
        repository_owner=row["repository_owner"],
        repository_url=row["repository_url"],
        repository_status=RepositoryProcessingStatus(row["repository_status"]),
        message_count=row["message_count"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def _message_from_row(row: object) -> ChatMessage:
    sources_json = row["sources_json"]
    sources = json.loads(sources_json) if sources_json else []
    return ChatMessage(
        message_id=row["message_id"],
        role=row["role"],
        content=row["content"],
        sources=sources,
        created_at=datetime.fromisoformat(row["created_at"]),
    )
