import asyncio
import pytest
from unittest.mock import MagicMock, patch


class TestGetHeadersAndCookies:
    """Test header construction for outbound LLM API calls."""

    @pytest.fixture(autouse=True)
    def _patch_env(self):
        """Mock environment imports so we don't need the full app context."""
        with patch(
            "open_webui.routers.openai.ENABLE_FORWARD_USER_INFO_HEADERS", False
        ), patch(
            "open_webui.routers.openai.VERSION", "0.0.0-test"
        ):
            from open_webui.routers.openai import get_headers_and_cookies

            self.get_headers_and_cookies = get_headers_and_cookies
            yield

    @pytest.mark.asyncio
    async def test_includes_user_agent(self):
        """User-Agent header is set with open-webui version."""
        request = MagicMock()
        headers, _ = await self.get_headers_and_cookies(
            request=request,
            url="https://api.anthropic.com/v1/messages",
            key="test-key",
            config={"auth_type": "bearer"},
        )
        assert headers["User-Agent"] == "open-webui/0.0.0-test"

    @pytest.mark.asyncio
    async def test_includes_content_type(self):
        """Content-Type header is always set to application/json."""
        request = MagicMock()
        headers, _ = await self.get_headers_and_cookies(
            request=request,
            url="https://api.openai.com/v1/chat/completions",
            key="test-key",
            config={"auth_type": "bearer"},
        )
        assert headers["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_bearer_auth(self):
        """Bearer token is set in Authorization header."""
        request = MagicMock()
        headers, _ = await self.get_headers_and_cookies(
            request=request,
            url="https://api.openai.com/v1/chat/completions",
            key="sk-test-123",
            config={"auth_type": "bearer"},
        )
        assert headers["Authorization"] == "Bearer sk-test-123"

    @pytest.mark.asyncio
    async def test_openrouter_extra_headers(self):
        """OpenRouter URLs get extra HTTP-Referer and X-Title headers."""
        request = MagicMock()
        headers, _ = await self.get_headers_and_cookies(
            request=request,
            url="https://openrouter.ai/api/v1/chat/completions",
            key="test-key",
            config={"auth_type": "bearer"},
        )
        assert headers["HTTP-Referer"] == "https://openwebui.com/"
        assert headers["X-Title"] == "Open WebUI"

    @pytest.mark.asyncio
    async def test_non_openrouter_no_extra_headers(self):
        """Non-OpenRouter URLs do not get HTTP-Referer or X-Title."""
        request = MagicMock()
        headers, _ = await self.get_headers_and_cookies(
            request=request,
            url="https://api.anthropic.com/v1/messages",
            key="test-key",
            config={"auth_type": "bearer"},
        )
        assert "HTTP-Referer" not in headers
        assert "X-Title" not in headers
