#!/usr/bin/env python
"""LLM provider factory.

Selects the appropriate LLM class based on a ``provider`` string.
"""

from __future__ import annotations

from typing import Any

from .openrouter_llm import OpenRouterLLM


def create_llm(
    provider: str = "openrouter",
    api_key: str = "",
    api_base: str = "https://inference.do-ai.run/v1",
    model_name: str = "anthropic-claude-opus-4.7",
    timeout: int = 600,
    **kwargs: Any,
) -> OpenRouterLLM:
    """Create an LLM provider instance based on the provider string.

    Args:
        provider: Provider identifier (``"openrouter"``, ``"vertex"``, etc.).
        api_key: API key (unused for vertex, which uses OAuth2).
        api_base: Provider base URL.
        model_name: Model identifier.
        timeout: Request timeout in seconds.
        **kwargs: Additional provider-specific options.

    Returns:
        An ``OpenRouterLLM`` (or subclass) instance.
    """
    if provider == "vertex":
        # MiroThinker / Vertex AI recommended defaults.
        kwargs.setdefault("temperature", 1.0)
        kwargs.setdefault("top_p", 0.95)
        kwargs.setdefault("repetition_penalty", 1.05)
        kwargs.setdefault("max_tokens", 16384)
        kwargs.setdefault("max_context_length", 262144)

        # MiroThinker uses XML-based tool calling instead of OpenAI's
        # function-calling API — use the specialised provider.
        if "mirothinker" in model_name.lower() or "miromind" in model_name.lower():
            from .mirothinker_llm import MiroThinkerLLM

            return MiroThinkerLLM(
                api_base=api_base,
                model_name=model_name,
                timeout=timeout,
                **kwargs,
            )

        from .vertex_llm import VertexLLM

        return VertexLLM(
            api_base=api_base,
            model_name=model_name,
            timeout=timeout,
            **kwargs,
        )

    if provider == "azure":
        # Azure OpenAI via standard OpenAI client.
        # api_base should be the full deployment URL, e.g.:
        #   https://<resource>.openai.azure.com/openai/deployments/<deployment>
        # Pass azure_api_version in kwargs (default: 2024-12-01-preview).
        azure_api_version = kwargs.pop("azure_api_version", "2024-12-01-preview")
        llm = OpenRouterLLM(
            api_key=api_key,
            api_base=api_base,
            model_name=model_name,
            timeout=timeout,
            disable_cache_control=True,
            **kwargs,
        )
        # Reconfigure the client with Azure-specific auth and query params
        from openai import AsyncOpenAI

        llm._client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=timeout,
            default_query={"api-version": azure_api_version},
            default_headers={"api-key": api_key},
        )
        return llm

    return OpenRouterLLM(
        api_key=api_key,
        api_base=api_base,
        model_name=model_name,
        timeout=timeout,
        **kwargs,
    )
