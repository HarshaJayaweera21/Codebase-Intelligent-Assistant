from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from app.core.config import Settings


class GeminiConfigurationError(RuntimeError):
    """Raised when Gemini generation is enabled but cannot be configured."""


def create_gemini_chat_model(settings: Settings) -> BaseChatModel:
    if settings.gemini_api_key is None or not (
        api_key := settings.gemini_api_key.get_secret_value().strip()
    ):
        raise GeminiConfigurationError(
            "GEMINI_ENABLED is true, but GEMINI_API_KEY is missing. "
            "Create a Gemini API key in Google AI Studio, add it to "
            "backend/.env, and restart FastAPI."
        )

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as error:
        raise GeminiConfigurationError(
            "The 'langchain-google-genai' package is required. "
            "Install backend requirements."
        ) from error

    class Gemini36ChatModel(ChatGoogleGenerativeAI):
        """Omit a legacy field that Gemini 3.6 no longer accepts."""

        def _build_base_generation_config(
            self,
            stop: list[str] | None,
            **kwargs: Any,
        ) -> dict[str, Any]:
            config = super()._build_base_generation_config(stop, **kwargs)
            config.pop("candidate_count", None)
            return config

    return Gemini36ChatModel(
        model=settings.gemini_model,
        api_key=api_key,
        thinking_level=settings.gemini_thinking_level,
        # Gemini 3.6 does not use the older sampling controls. Passing None
        # prevents the LangChain class default from sending a temperature.
        temperature=None,
    )
