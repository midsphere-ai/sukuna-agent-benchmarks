#!/usr/bin/env python
"""Vertex AI LLM provider — extends OpenRouterLLM with Google OAuth2 credentials.

Vertex AI exposes an OpenAI-compatible API but requires OAuth2 tokens
that expire and need auto-refreshing. This subclass handles credential
resolution and transparent token refresh before each LLM call.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

from google.auth.transport.requests import Request  # pyright: ignore[reportMissingImports]
from google.oauth2 import service_account  # pyright: ignore[reportMissingImports]

from orbiter.models.types import (  # pyright: ignore[reportMissingImports]
    ModelResponse,
    StreamChunk,
)
from orbiter.types import Message  # pyright: ignore[reportMissingImports]

from .openrouter_llm import OpenRouterLLM

logger = logging.getLogger("gaia-llm")

_VERTEX_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


class VertexLLM(OpenRouterLLM):
    """Vertex AI LLM provider with auto-refreshing OAuth2 credentials.

    Resolves Google credentials in priority order:
      1. ``GOOGLE_SERVICE_ACCOUNT_B64`` env var (base64-encoded SA JSON)
      2. ``GOOGLE_APPLICATION_CREDENTIALS`` file path
      3. ``google.auth.default()`` (ADC / gcloud auth)

    The resolved token is passed as ``api_key`` to the parent
    ``OpenRouterLLM``, and refreshed transparently before each call.

    Args:
        api_base: Vertex AI OpenAI-compatible endpoint URL.
        model_name: Model identifier on the Vertex endpoint.
        timeout: Request timeout in seconds.
        **kwargs: Additional fields forwarded to ``OpenRouterConfig``.
    """

    def __init__(
        self,
        api_base: str,
        model_name: str,
        timeout: int = 600,
        **kwargs: Any,
    ):
        self._credentials = self._resolve_credentials()
        self._auth_request = Request()
        self._credentials.refresh(self._auth_request)
        token = self._credentials.token or ""

        super().__init__(
            api_key=token,
            api_base=api_base,
            model_name=model_name,
            timeout=timeout,
            disable_cache_control=True,
            **kwargs,
        )

    @staticmethod
    def _resolve_credentials():
        """Resolve Google credentials in priority order."""
        # 1. Base64-encoded service account JSON
        sa_b64 = os.environ.get("GOOGLE_SERVICE_ACCOUNT_B64")
        if sa_b64:
            sa_info = json.loads(base64.b64decode(sa_b64))
            return service_account.Credentials.from_service_account_info(
                sa_info, scopes=_VERTEX_SCOPES
            )

        # 2. Service account key file path
        sa_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if sa_path:
            return service_account.Credentials.from_service_account_file(
                sa_path, scopes=_VERTEX_SCOPES
            )

        # 3. Application Default Credentials (ADC)
        import google.auth  # pyright: ignore[reportMissingImports]

        credentials, _ = google.auth.default(scopes=_VERTEX_SCOPES)
        return credentials

    def _refresh_token(self) -> None:
        """Refresh the OAuth2 token if expired and update the client."""
        if not self._credentials.valid:
            self._credentials.refresh(self._auth_request)
            self._client.api_key = self._credentials.token or ""
            logger.debug("Vertex AI OAuth2 token refreshed")

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ModelResponse:
        """Refresh token then delegate to parent."""
        self._refresh_token()
        return await super().complete(
            messages, tools=tools, temperature=temperature, max_tokens=max_tokens
        )

    async def stream(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Refresh token then delegate to parent."""
        self._refresh_token()
        async for chunk in super().stream(
            messages, tools=tools, temperature=temperature, max_tokens=max_tokens
        ):
            yield chunk
