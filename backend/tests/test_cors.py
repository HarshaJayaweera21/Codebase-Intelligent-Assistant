from contextlib import asynccontextmanager
import unittest

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


class CorsTests(unittest.TestCase):
    def test_configured_frontend_origin_is_allowed(self):
        @asynccontextmanager
        async def lifespan(_app):
            yield

        settings = Settings(
            cors_allowed_origins=(
                "http://localhost:5173,http://127.0.0.1:5173"
            )
        )
        app = create_app(lifespan=lifespan, settings=settings)

        with TestClient(app) as client:
            response = client.options(
                "/api/chats",
                headers={
                    "Origin": "http://localhost:5173",
                    "Access-Control-Request-Method": "GET",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["access-control-allow-origin"],
            "http://localhost:5173",
        )


if __name__ == "__main__":
    unittest.main()
