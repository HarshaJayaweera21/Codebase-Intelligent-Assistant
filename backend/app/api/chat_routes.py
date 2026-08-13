import re

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool

from app.models.chat import (
    ChatDetailResponse,
    ChatMessageResponse,
    ChatSummaryResponse,
)
from app.models.repository import RepositoryDeleteResponse
from app.services.chat_service import ChatDetail, ChatService, ChatSummary
from app.services.repository_lifecycle import (
    RepositoryLifecycleConflict,
    RepositoryLifecycleService,
)


router = APIRouter(prefix="/chats", tags=["Chats"])
CHAT_ID_PATTERN = re.compile(r"^chat_[a-f0-9]{8}$")


@router.get("", response_model=list[ChatSummaryResponse])
async def list_chats(request: Request) -> list[ChatSummaryResponse]:
    chats = await run_in_threadpool(_get_chat_service(request).list_chats)
    return [
        _summary_response(chat)
        for chat in chats
    ]


@router.get("/{chat_id}", response_model=ChatDetailResponse)
async def get_chat(chat_id: str, request: Request) -> ChatDetailResponse:
    if CHAT_ID_PATTERN.fullmatch(chat_id) is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Invalid chat_id.",
        )
    chat = await run_in_threadpool(
        _get_chat_service(request).get_chat,
        chat_id,
    )
    if chat is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat was not found.",
        )
    return _detail_response(chat)


@router.delete("/{chat_id}", response_model=RepositoryDeleteResponse)
async def delete_chat(
    chat_id: str,
    request: Request,
) -> RepositoryDeleteResponse:
    if CHAT_ID_PATTERN.fullmatch(chat_id) is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Invalid chat_id.",
        )
    lifecycle_service: RepositoryLifecycleService | None = getattr(
        request.app.state,
        "repository_lifecycle_service",
        None,
    )
    if lifecycle_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat deletion requires repository lifecycle management.",
        )
    try:
        result = await run_in_threadpool(
            lifecycle_service.delete_chat,
            chat_id,
        )
    except RepositoryLifecycleConflict as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat was not found.",
        )
    return RepositoryDeleteResponse(**result.__dict__)


def _get_chat_service(request: Request) -> ChatService:
    return request.app.state.chat_service


def _summary_response(chat: ChatSummary) -> ChatSummaryResponse:
    return ChatSummaryResponse(**chat.__dict__)


def _detail_response(chat: ChatDetail) -> ChatDetailResponse:
    return ChatDetailResponse(
        **chat.summary.__dict__,
        messages=[
            ChatMessageResponse(**message.__dict__)
            for message in chat.messages
        ],
    )
