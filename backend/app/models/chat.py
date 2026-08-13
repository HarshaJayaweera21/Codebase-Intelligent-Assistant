from datetime import datetime

from pydantic import BaseModel

from app.models.repository import RepositoryProcessingStatus


class ChatSummaryResponse(BaseModel):
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


class ChatMessageResponse(BaseModel):
    message_id: str
    role: str
    content: str
    sources: list[dict[str, object]]
    created_at: datetime


class ChatDetailResponse(ChatSummaryResponse):
    messages: list[ChatMessageResponse]
