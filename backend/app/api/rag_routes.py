import json
import logging
import re
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from app.models.repository import RepositoryProcessingStatus
from app.rag.rag_service import (
    AnswerStreamConnectionError,
    RagAnswerStream,
    RagService,
    RagSource,
)
from app.services.chat_service import ChatService
from app.services.repository_processing import (
    RepositoryProcessingRecord,
    RepositoryProcessingStore,
)


router = APIRouter(prefix="/repositories", tags=["Repository questions"])
REPOSITORY_ID_PATTERN = re.compile(r"^repo_[a-f0-9]{8}$")
logger = logging.getLogger(__name__)


class RepositoryQuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=100_000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class RagSourceResponse(BaseModel):
    citation_id: str
    vector_id: str
    score: float
    vector_score: float
    lexical_score: float
    exact_match_score: float
    structural_score: float
    file_path: str
    language: str
    chunk_type: str
    symbol_name: str | None
    symbol_start_line: int | None
    symbol_end_line: int | None
    source_ranges: list[dict[str, int]]
    content: str


class RepositoryAnswerResponse(BaseModel):
    repository_id: str
    chat_id: str | None
    question: str
    answer: str
    sources: list[RagSourceResponse]


def get_rag_service(request: Request) -> RagService:
    rag_service = getattr(request.app.state, "rag_service", None)
    if rag_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Gemini RAG is disabled. Set GEMINI_ENABLED=true and add "
                "GEMINI_API_KEY in backend/.env, then restart FastAPI."
            ),
        )
    return rag_service


RagServiceDependency = Annotated[RagService, Depends(get_rag_service)]


@router.post(
    "/{repository_id}/ask",
    response_model=RepositoryAnswerResponse,
)
async def ask_repository(
    repository_id: str,
    payload: RepositoryQuestionRequest,
    request: Request,
    rag_service: RagServiceDependency,
) -> RepositoryAnswerResponse:
    _validate_repository_id(repository_id)
    _validate_question_length(request, payload.question)
    record = await run_in_threadpool(
        _require_repository_ready,
        request,
        repository_id,
    )
    result = await run_in_threadpool(
        rag_service.answer,
        repository_id,
        payload.question,
        top_k=payload.top_k,
    )
    if record is not None:
        chat_service: ChatService = request.app.state.chat_service
        await run_in_threadpool(
            chat_service.save_exchange,
            chat_id=record.chat_id,
            question=payload.question.strip(),
            answer=result.answer,
            sources=[source.__dict__ for source in result.sources],
        )
    return RepositoryAnswerResponse(
        repository_id=repository_id,
        chat_id=record.chat_id if record is not None else None,
        question=payload.question.strip(),
        answer=result.answer,
        sources=[_source_response(source) for source in result.sources],
    )


@router.post("/{repository_id}/ask/stream")
async def stream_repository_answer(
    repository_id: str,
    payload: RepositoryQuestionRequest,
    request: Request,
    rag_service: RagServiceDependency,
) -> StreamingResponse:
    _validate_repository_id(repository_id)
    _validate_question_length(request, payload.question)
    record = await run_in_threadpool(
        _require_repository_ready,
        request,
        repository_id,
    )
    stream = await run_in_threadpool(
        rag_service.stream_answer,
        repository_id,
        payload.question,
        top_k=payload.top_k,
    )
    chat_service: ChatService | None = (
        request.app.state.chat_service if record is not None else None
    )
    return StreamingResponse(
        _stream_sse_events(
            stream=stream,
            repository_id=repository_id,
            chat_id=record.chat_id if record is not None else None,
            question=payload.question.strip(),
            chat_service=chat_service,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _require_repository_ready(
    request: Request,
    repository_id: str,
) -> RepositoryProcessingRecord | None:
    processing_store: RepositoryProcessingStore = (
        request.app.state.repository_processing_store
    )
    record = processing_store.get(repository_id)
    # Namespaces indexed before SQLite persistence was introduced have no local
    # record. Pinecone remains the source of truth for those legacy namespaces.
    if record is None:
        return None
    if record.status is not RepositoryProcessingStatus.READY:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Repository is not ready for questions. Current status: "
                f"{record.status.value}."
            ),
        )
    return record


def _validate_repository_id(repository_id: str) -> None:
    if REPOSITORY_ID_PATTERN.fullmatch(repository_id) is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Invalid repository_id.",
        )


def _validate_question_length(request: Request, question: str) -> None:
    if not question.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Question must not be blank.",
        )
    maximum = getattr(
        request.app.state.settings,
        "rag_max_question_characters",
        4_000,
    )
    if len(question.strip()) > maximum:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Question must not exceed {maximum} characters.",
        )


def _source_response(source: RagSource) -> RagSourceResponse:
    return RagSourceResponse(**source.__dict__)


def _stream_sse_events(
    *,
    stream: RagAnswerStream,
    repository_id: str,
    chat_id: str | None,
    question: str,
    chat_service: ChatService | None,
):
    sources = [source.__dict__ for source in stream.sources]
    yield _sse_event(
        "sources",
        {
            "repository_id": repository_id,
            "chat_id": chat_id,
            "question": question,
            "sources": sources,
        },
    )
    answer_parts: list[str] = []
    try:
        for text in stream.chunks:
            answer_parts.append(text)
            yield _sse_event("token", {"text": text})
        answer = "".join(answer_parts).strip()
        if not answer:
            raise RuntimeError("The language model returned an empty answer.")
        if chat_service is not None and chat_id is not None:
            chat_service.save_exchange(
                chat_id=chat_id,
                question=question,
                answer=answer,
                sources=sources,
            )
        yield _sse_event("done", {"answer": answer})
    except GeneratorExit:
        raise
    except AnswerStreamConnectionError:
        logger.exception("Model connection failed while streaming an answer")
        yield _sse_event(
            "error",
            {
                "message": (
                    "The model connection was interrupted. Please try the "
                    "question again."
                )
            },
        )
    except Exception:
        logger.exception("Streaming repository answer failed")
        yield _sse_event(
            "error",
            {"message": "The answer stream failed unexpectedly."},
        )


def _sse_event(event: str, data: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
