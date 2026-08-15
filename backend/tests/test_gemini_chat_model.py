import unittest

from app.core.config import Settings
from app.llms.gemini_chat_model import (
    GeminiConfigurationError,
    create_gemini_chat_model,
)


class GeminiChatModelTests(unittest.TestCase):
    def test_requires_api_key(self):
        settings = Settings(
            gemini_enabled=True,
            gemini_api_key=None,
        )

        with self.assertRaisesRegex(
            GeminiConfigurationError,
            "GEMINI_API_KEY is missing",
        ):
            create_gemini_chat_model(settings)

    def test_creates_configured_langchain_model_without_network_call(self):
        settings = Settings(
            gemini_enabled=True,
            gemini_api_key="test-api-key",
            gemini_model="gemini-3.6-flash",
            gemini_thinking_level="low",
        )

        model = create_gemini_chat_model(settings)

        self.assertEqual(model.model, "gemini-3.6-flash")
        self.assertEqual(model.thinking_level, "low")
        self.assertIsNone(model.temperature)
        request_config = model._prepare_params(None).model_dump(
            exclude_unset=True,
            exclude_none=True,
        )
        self.assertNotIn("candidate_count", request_config)
        self.assertNotIn("temperature", request_config)
        self.assertNotIn("top_p", request_config)
        self.assertNotIn("top_k", request_config)


if __name__ == "__main__":
    unittest.main()
