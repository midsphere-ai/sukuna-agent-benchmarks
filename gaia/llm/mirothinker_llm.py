#!/usr/bin/env python
"""MiroThinker LLM provider — extends VertexLLM with XML-based tool calling.

MiroThinker uses an XML MCP-style format for tool calling instead of
the standard OpenAI function calling API.  This class translates between
the standard OpenAI tool interface used by the agent framework and
MiroThinker's ``<use_mcp_tool>`` XML format so the rest of the agent
code (ReActAgent, ToolCallHandler) can work unchanged.

Translation overview:
  * Tool schemas  → injected into the system prompt as markdown / JSON
  * ``tools=`` API parameter  → omitted (model doesn't use it)
  * ``<use_mcp_tool>`` in response text  → parsed into ``ToolCall`` objects
  * ``role: tool`` messages on next turn  → converted to ``role: user``
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

import httpx  # transitive dep of openai
from openai.types.chat import ChatCompletion

from orbiter.models.types import (  # pyright: ignore[reportMissingImports]
    ModelError,
    ModelResponse,
    StreamChunk,
)
from orbiter.types import (  # pyright: ignore[reportMissingImports]
    Message,
    ToolCall,
    Usage,
)

from .openrouter_llm import ContextLimitError, _to_openai_messages
from .vertex_llm import VertexLLM

logger = logging.getLogger("gaia-llm")

# ---------------------------------------------------------------------------
# Tool-description system-prompt (matches MiroThinker's training format)
# ---------------------------------------------------------------------------

_TOOL_PROMPT_HEADER = """\
In this environment you have access to a set of tools you can use to answer the user's question. Today is: {today_date}

## CRITICAL RULES
- You can ONLY use the tools listed below. No other tools exist.
- You MUST use the EXACT tool_name as listed. Do NOT invent, guess, or use tool names from your training data (e.g. do NOT use "google_search", "scrape_and_extract_info", etc. unless they are explicitly listed below).
- If a tool is not listed below, it does not exist and you cannot call it.
- You can only use one tool per message.
- You will receive the result of that tool in the user's next response.

## Tool-Use Format

<use_mcp_tool>
<server_name>tools</server_name>
<tool_name>EXACT_TOOL_NAME_FROM_LIST_BELOW</tool_name>
<arguments>
{{"param": "value"}}
</arguments>
</use_mcp_tool>

- Tool-use must be placed at the end of your response, top-level.
- arguments must be valid JSON matching the tool's input schema.

## Available Tools

The ONLY tools you can use are listed below. Use the tool_name EXACTLY as shown:
"""

_TOOL_PROMPT_FOOTER = """
## REMINDER: You MUST only use tool names from the list above. Any other tool name will fail.

# General Objective

You accomplish a given task iteratively, breaking it down into clear steps and working through them methodically.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tools_to_xml_descriptions(tools: list[dict[str, Any]]) -> str:
    """Convert OpenAI-format tool schemas to MiroThinker description blocks."""
    names: list[str] = []
    sections: list[str] = []

    for tool in tools:
        if tool.get("type") != "function":
            continue
        func = tool["function"]
        name: str = func.get("name", "unknown")
        description: str = func.get("description", "")
        parameters: dict = func.get("parameters", {})
        names.append(name)

        sections.append(
            f"### tool_name: `{name}`\n"
            f"Description: {description}\n"
            f"Input JSON schema: {json.dumps(parameters, ensure_ascii=False)}"
        )

    # Put an explicit list of valid names at the top
    name_list = ", ".join(f"`{n}`" for n in names)
    header = f"Valid tool_name values: {name_list}\n"

    return header + "\n".join(sections)


def _build_tool_system_prompt(tools: list[dict[str, Any]]) -> str:
    """Build the full tool-description system prompt."""
    today = datetime.now().strftime("%Y-%m-%d")
    header = _TOOL_PROMPT_HEADER.replace("{today_date}", today)
    body = _tools_to_xml_descriptions(tools)
    return header + body + _TOOL_PROMPT_FOOTER


# ---------------------------------------------------------------------------
# XML tool-call parser
# ---------------------------------------------------------------------------

_MCP_PATTERN = re.compile(
    r"<use_mcp_tool>(.*?)</use_mcp_tool>", re.DOTALL
)

# Format 1 (MCP style): <server_name>...</server_name> <tool_name>...</tool_name> <arguments>...</arguments>
_TOOL_PATTERN = re.compile(
    r"<tool_name>(.*?)</tool_name>", re.DOTALL
)
_ARGS_PATTERN = re.compile(
    r"<arguments>(.*?)</arguments>", re.DOTALL
)

# Format 2 (functions style): <recipient_name>functions.NAME</recipient_name> <parameters>...</parameters>
_RECIPIENT_PATTERN = re.compile(
    r"<recipient_name>\s*(?:functions\.)?(.+?)\s*</recipient_name>", re.DOTALL
)
_PARAMS_PATTERN = re.compile(
    r"<parameters>(.*?)</parameters>", re.DOTALL
)


def _parse_json_best_effort(raw: str, tool_name: str) -> dict[str, Any]:
    """Try to parse JSON, with a fallback for trailing-comma cleanup."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            cleaned = re.sub(r",\s*([}\]])", r"\1", raw)
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning(
                "Failed to parse MiroThinker tool arguments for %s: %s",
                tool_name,
                raw[:200],
            )
            return {}


def _parse_mcp_tool_calls(text: str) -> list[ToolCall]:
    """Parse ``<use_mcp_tool>`` blocks from *text* into ``ToolCall`` objects.

    Handles two XML variants the model may produce:

    **Format 1 (MCP style)**::

        <tool_name>google_search</tool_name>
        <arguments>{"q": "..."}</arguments>

    **Format 2 (functions style)**::

        <recipient_name>functions.agent-browsing</recipient_name>
        <parameters>{"subtask": "..."}</parameters>
    """
    results: list[ToolCall] = []

    for match in _MCP_PATTERN.finditer(text):
        content = match.group(1)

        # --- Resolve tool name (try both formats) ---
        tool_name = ""

        tool_m = _TOOL_PATTERN.search(content)
        if tool_m:
            tool_name = tool_m.group(1).strip()

        if not tool_name:
            recip_m = _RECIPIENT_PATTERN.search(content)
            if recip_m:
                tool_name = recip_m.group(1).strip()

        if not tool_name:
            continue

        # --- Resolve arguments (try both formats) ---
        arguments: dict[str, Any] = {}

        args_m = _ARGS_PATTERN.search(content)
        if args_m:
            arguments = _parse_json_best_effort(args_m.group(1).strip(), tool_name)
        else:
            params_m = _PARAMS_PATTERN.search(content)
            if params_m:
                arguments = _parse_json_best_effort(params_m.group(1).strip(), tool_name)

        call_id = f"miro_{uuid.uuid4().hex[:12]}"
        results.append(
            ToolCall(
                id=call_id,
                name=tool_name,
                arguments=json.dumps(arguments, ensure_ascii=False),
            )
        )

    return results


# ---------------------------------------------------------------------------
# Provider class
# ---------------------------------------------------------------------------


class MiroThinkerLLM(VertexLLM):
    """MiroThinker LLM provider with XML-based MCP-style tool calling.

    Subclasses ``VertexLLM`` (which handles Vertex AI OAuth2 auth) and
    overrides ``complete()`` / ``stream()`` to:

    1. Inject tool descriptions into the system prompt.
    2. Strip the ``tools`` API parameter (model ignores it).
    3. Convert ``role: tool`` messages to ``role: user`` messages.
    4. Parse ``<use_mcp_tool>`` XML from response text into ``ToolCall``
       objects so the agent framework can execute them normally.

    Supports both Vertex AI ``rawPredict`` endpoints (direct HTTP via
    httpx) and OpenAI-compatible ``/chat/completions`` endpoints
    (via the OpenAI SDK).
    """

    def __init__(
        self,
        api_base: str,
        model_name: str,
        timeout: int = 600,
        **kwargs: Any,
    ) -> None:
        # Detect rawPredict endpoint — these don't support the
        # /chat/completions sub-path that the OpenAI SDK appends.
        self._is_raw_predict = ":rawPredict" in api_base
        self._raw_predict_url = api_base if self._is_raw_predict else ""
        super().__init__(
            api_base=api_base,
            model_name=model_name,
            timeout=timeout,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ModelResponse:
        """Send a completion using MiroThinker's XML tool-calling format."""
        try:
            self._refresh_token()

            openai_messages = _to_openai_messages(messages)
            processed = self._prepare_messages(openai_messages)

            # --- MiroThinker message transformation ---
            transformed = self._transform_messages(processed, tools)

            body: dict[str, Any] = {
                "model": self.or_config.model_name,
                "messages": transformed,
                "temperature": (
                    temperature
                    if temperature is not None
                    else self.or_config.temperature
                ),
                "top_p": self.or_config.top_p,
                "max_tokens": max_tokens or self.or_config.max_tokens,
                "stream": False,
            }
            if self.or_config.repetition_penalty is not None:
                body["repetition_penalty"] = self.or_config.repetition_penalty

            # NOTE: no ``tools=`` key — MiroThinker reads tools from the
            # system prompt, not from the OpenAI tools API.

            if self._is_raw_predict:
                response = await self._raw_predict(body)
            else:
                # Standard OpenAI-compatible endpoint
                extra_body: dict[str, Any] = {}
                if self.or_config.repetition_penalty is not None:
                    extra_body["repetition_penalty"] = (
                        self.or_config.repetition_penalty
                    )
                    body.pop("repetition_penalty", None)
                params = {k: v for k, v in body.items()}
                if extra_body:
                    params["extra_body"] = extra_body
                response = await self._client.chat.completions.create(
                    **params
                )

            if response.usage:
                self._update_token_usage(response.usage)

            return self._parse_mirothinker_response(response)

        except ContextLimitError:
            raise
        except Exception as e:
            error_str = str(e)
            if any(
                phrase in error_str
                for phrase in [
                    "Input is too long",
                    "context limit",
                    "maximum context length",
                ]
            ):
                logger.error("Context limit exceeded: %s", error_str)
                raise ContextLimitError(
                    f"Context limit exceeded: {error_str}"
                ) from e

            logger.error("MiroThinker LLM call failed: %s", error_str)
            raise ModelError(
                error_str,
                model=f"vertex:{self.or_config.model_name}",
            ) from e

    async def _raw_predict(self, body: dict[str, Any]) -> ChatCompletion:
        """POST directly to the Vertex AI rawPredict endpoint.

        The OpenAI SDK can't be used here because it appends
        ``/chat/completions`` to the base URL, which doesn't exist on
        rawPredict endpoints.  We use httpx and parse the response with
        the OpenAI SDK's ``ChatCompletion`` model for type consistency.
        """
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(float(self.or_config.timeout)),
        ) as client:
            resp = await client.post(
                self._raw_predict_url,
                headers={
                    "Authorization": f"Bearer {self._credentials.token}",
                    "Content-Type": "application/json",
                },
                json=body,
            )
            resp.raise_for_status()
            return ChatCompletion.model_validate(resp.json())

    async def stream(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Delegate to ``complete()`` and yield a single chunk."""
        result = await self.complete(
            messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        yield StreamChunk(
            delta=result.content,
            finish_reason=result.finish_reason,
            usage=result.usage,
        )

    # ------------------------------------------------------------------
    # Message transformation
    # ------------------------------------------------------------------

    @staticmethod
    def _transform_messages(
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        """Rewrite a message list for MiroThinker's expectations.

        1. Prepend tool descriptions to the first system message (or
           insert a new system message if none exists).
        2. Strip ``tool_calls`` from assistant messages (the XML is
           already in the text ``content``).
        3. Convert ``role: tool`` messages to ``role: user`` so the
           model sees tool results as ordinary user turns.
        """
        tool_prompt = _build_tool_system_prompt(tools) if tools else ""

        result: list[dict[str, Any]] = []
        injected_tools = False

        for msg in messages:
            role = msg.get("role", "")

            if role == "system":
                content = msg.get("content", "")
                if tool_prompt and not injected_tools:
                    content = tool_prompt + "\n\n" + content
                    injected_tools = True
                result.append({"role": "system", "content": content})

            elif role == "assistant":
                # Keep content (includes XML tool calls), drop tool_calls key
                result.append({
                    "role": "assistant",
                    "content": msg.get("content", "") or "",
                })

            elif role == "tool":
                # Model expects tool results as user messages
                content = msg.get("content", "")
                result.append({
                    "role": "user",
                    "content": f"[Tool Result]\n{content}",
                })

            else:
                # user / other roles — pass through
                result.append(msg)

        # If there was no system message, prepend one
        if tool_prompt and not injected_tools:
            result.insert(0, {"role": "system", "content": tool_prompt})

        return result

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_mirothinker_response(self, response: Any) -> ModelResponse:
        """Parse a MiroThinker API response, extracting XML tool calls.

        The full response text (including any ``<use_mcp_tool>`` blocks)
        is kept in ``content`` so it round-trips correctly when stored in
        the conversation history and later stripped by
        ``_transform_messages``.
        """
        if not response or not getattr(response, "choices", None):
            raise ValueError("LLM did not return a valid response")

        choice = response.choices[0]
        message = choice.message
        content: str = getattr(message, "content", "") or ""

        # Extract XML tool calls from the text
        tool_calls = _parse_mcp_tool_calls(content)

        finish_reason = "tool_calls" if tool_calls else "stop"

        usage = Usage()
        if getattr(response, "usage", None):
            usage = Usage(
                input_tokens=response.usage.prompt_tokens or 0,
                output_tokens=response.usage.completion_tokens or 0,
                total_tokens=response.usage.total_tokens or 0,
            )

        return ModelResponse(
            id=getattr(response, "id", "") or "",
            model=(
                getattr(response, "model", "")
                or self.or_config.model_name
            ),
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=finish_reason,
        )
