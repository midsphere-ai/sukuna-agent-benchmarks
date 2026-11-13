import os
import traceback
import concurrent.futures
from functools import partial
from fastmcp import FastMCP
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters
import asyncio
import httpx
from tool.logger import bootstrap_logger
from typing import List, Optional
try:
    from orbiter.perplexica import search_with_details
    from orbiter.perplexica.config import PerplexicaConfig
    from orbiter.perplexica.tools.searxng import clear_collected_results
    _PERPLEXICA_AVAILABLE = True
except ImportError:
    _PERPLEXICA_AVAILABLE = False
    search_with_details = None  # type: ignore
    PerplexicaConfig = None  # type: ignore
    def clear_collected_results():  # type: ignore
        pass
from .utils.query_enhancer import enhance_and_classify
from .utils.search_content_judge import search_content_judge
from dataclasses import dataclass
import time

SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")
JINA_API_KEY = os.environ.get("JINA_API_KEY", "")
JINA_DEEPSEARCH_URL = os.environ.get("JINA_DEEPSEARCH_URL", "https://deepsearch.jina.ai/v1/chat/completions")
# Initialize FastMCP server
mcp = FastMCP("searching-mcp-server")

logger = bootstrap_logger()
DEFAULT_ENGINE_TIMEOUT = 300.0

# ---------------------------------------------------------------------------
# Perplexica stability: serialise calls and expand the thread pool so the
# pipeline's many asyncio.to_thread() calls (Serper, Jina, embeddings, web
# fetch) never exhaust workers and deadlock.
# ---------------------------------------------------------------------------
_perplexica_sem = asyncio.Semaphore(1)
_search_thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=64)
_pool_installed = False


def _ensure_thread_pool() -> None:
    """Install a larger thread pool on the running event loop (once)."""
    global _pool_installed
    if not _pool_installed:
        try:
            asyncio.get_running_loop().set_default_executor(_search_thread_pool)
        except RuntimeError:
            pass
        _pool_installed = True


def _run_pipeline_in_isolation(
    search_query: str,
    resolved_mode: str,
    config,
    timeout: float,
):
    """Run the Perplexica pipeline in a throwaway event loop.

    ``search_with_details`` spawns fire-and-forget ``asyncio.create_task``
    work (suggestion generator, parallel sub-researchers) that outlives the
    top-level coroutine when ``asyncio.wait_for`` cancels it on timeout.
    Those orphaned tasks hold aiohttp sessions and anyio cancel scopes that
    crash the MCP server during shutdown.  Running in a dedicated loop
    ensures every task and async generator is torn down before returning.
    """
    loop = asyncio.new_event_loop()
    # Dedicated executor so loop.close() won't shut down the shared pool.
    loop.set_default_executor(
        concurrent.futures.ThreadPoolExecutor(max_workers=64)
    )
    try:
        return loop.run_until_complete(
            asyncio.wait_for(
                search_with_details(search_query, mode=resolved_mode, config=config),
                timeout=timeout,
            )
        )
    finally:
        _cleanup_event_loop(loop)


def _cleanup_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Cancel remaining tasks and close *loop* cleanly."""
    try:
        pending = asyncio.all_tasks(loop)
        for t in pending:
            t.cancel()
        if pending:
            loop.run_until_complete(
                asyncio.gather(*pending, return_exceptions=True)
            )
    except Exception:
        pass
    try:
        loop.run_until_complete(loop.shutdown_asyncgens())
    except Exception:
        pass
    try:
        loop.close()
    except Exception:
        pass


_exception_handler_installed = False


def _install_exception_handler() -> None:
    """Suppress anyio cancel-scope errors during event-loop shutdown.

    These are harmless cleanup errors from ``stdio_client`` async generators
    being finalized after the owning task has already ended.
    """
    global _exception_handler_installed
    if _exception_handler_installed:
        return
    try:
        loop = asyncio.get_running_loop()

        def _handler(loop, context):
            exc = context.get("exception")
            if isinstance(exc, RuntimeError) and "cancel scope" in str(exc):
                logger.debug("Suppressed cancel-scope cleanup error: %s", exc)
                return
            loop.default_exception_handler(context)

        loop.set_exception_handler(_handler)
        _exception_handler_installed = True
    except RuntimeError:
        pass


@dataclass
class SearchResult:
    """search results from a specific engine"""

    engine: str
    query: str
    content: str
    success: bool
    error: Optional[str] = None


async def jina_deep_search(query: str, timeout: float = DEFAULT_ENGINE_TIMEOUT):
    url = JINA_DEEPSEARCH_URL
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {JINA_API_KEY}",
    }

    data = {
        "model": "jina-deepsearch-v1",
        "messages": [{"role": "user", "content": query}],
        "stream": False,
        "reasoning_effort": "low",
        "team_size": 4,
        "no_direct_answer": True,
    }
    logger.info("Jina DeepSearch request -- query: %s", query)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, headers=headers, json=data)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        logger.info("Jina DeepSearch success -- %d chars returned", len(content))
        return SearchResult("JINA", query, content, True)

    # httpx raises TimeoutException; catching httpx.Timeout (a config class) triggers
    # "catching classes that do not inherit from BaseException" in some environments.
    except httpx.TimeoutException:
        logger.error("Jina DeepSearch timed out after %.0fs", timeout)
        return SearchResult("JINA", query, "Request timed out", False)
    except httpx.RequestError as e:
        logger.error("Jina DeepSearch request failed: %s", e)
        return SearchResult("JINA", query, f"Request failed: {e}", False)
    except asyncio.CancelledError:
        logger.warning("Jina DeepSearch cancelled")
        raise
    except Exception as e:
        logger.error("Jina DeepSearch unexpected error: %s", e)
        return SearchResult("JINA", query, f"Unexpected error: {e}", False)


async def perplexica_search(
    query: str,
    mode: str = "auto",
    call_timeout: float = DEFAULT_ENGINE_TIMEOUT,
) -> SearchResult:
    """Perform a web search using Perplexica (open-source, backed by Jina search and Serper API).

    Args:
        query: a natural language question about the web
        mode: search mode — 'speed' (2 iterations), 'balanced' (6 iterations),
              'quality' (25 iterations), or 'auto' (pick from query classification)
        call_timeout: maximum seconds to wait for a response

    Returns:
        SearchResult with answer and citations
    """
    if not _PERPLEXICA_AVAILABLE:
        return SearchResult("Perplexica", query, "[ERROR]: Perplexica is disabled (orbiter.perplexica not installed)", False, "disabled")

    _ensure_thread_pool()
    logger.info("Perplexica search called -- query: %s", query)

    # Serialise Perplexica invocations: the pipeline fans out to many
    # asyncio.to_thread() workers (Serper, Jina, embeddings, web fetch).
    # Running two pipelines concurrently can exhaust the thread pool and
    # deadlock.  The semaphore ensures only one pipeline runs at a time.
    async with _perplexica_sem:
        # Clear stale results left by previous invocations.
        clear_collected_results()

        try:
            enhanced = await asyncio.to_thread(enhance_and_classify, query)
            search_query = enhanced["enhanced_query"]

            resolved_mode = mode
            if mode == "auto":
                resolved_mode = enhanced.get("perplexica_mode", "balanced")

            logger.info(
                "Query enhanced: %s (type=%s, topic=%s, mode=%s)",
                search_query,
                enhanced["query_type"],
                enhanced["query_topic"],
                resolved_mode,
            )

            config = PerplexicaConfig(
                model=os.getenv("PERPLEXICA_MODEL"),
                fast_model=os.getenv("PERPLEXICA_FAST_MODEL"),
                jina_api_key=JINA_API_KEY,
                serper_api_key=SERPER_API_KEY
            )
            result = await asyncio.to_thread(
                _run_pipeline_in_isolation,
                search_query,
                resolved_mode,
                config,
                call_timeout,
            )
            content = result.answer
            if result.sources:
                citations = "\n".join(
                    f"[{i + 1}] {s.title} ({s.url})" for i, s in enumerate(result.sources)
                )
                content += f"\n\n## Citations\n{citations}"
            return SearchResult("Perplexica", query, content, True)
        except asyncio.TimeoutError:
            logger.warning("Perplexica timed out after %.0fs -- query: %s", call_timeout, query)
            return SearchResult("Perplexica", query, "Request timed out", False, "timeout")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("Perplexica error: %s\n%s", e, traceback.format_exc())
            return SearchResult("Perplexica", query, f"[ERROR]: {e}", False, str(e))
        finally:
            # Always clean up after ourselves so the next call starts fresh.
            clear_collected_results()


async def google_search(
    q: str,
    gl: str = "us",
    hl: str = "en",
    location: str = None,
    num: int = 10,
    tbs: str = None,
    page: int = 1,
) -> str:
    """Perform google searches via Serper API and retrieve rich results.
    It is able to retrieve organic search results, people also ask, related searches, and knowledge graph.

    Args:
        q: Search query string.
        location: Location for search results (e.g., 'SoHo, New York, United States', 'California, United States').
        num: The number of results to return (default: 10).
        tbs: Time-based search filter ('qdr:h' for past hour, 'qdr:d' for past day, 'qdr:w' for past week, 'qdr:m' for past month, 'qdr:y' for past year).
        page: The page number of results to return (default: 1).

    Returns:
        The search results.
    """
    if not SERPER_API_KEY:
        return "SERPER_API_KEY is not set, google_search tool is not available."

    if not q or not q.strip():
        return "[ERROR]: Search query 'q' cannot be empty."

    serper_tool_name = "google_search"
    serper_search_arguments = {
        "q": q,
        "gl": gl,
        "hl": hl,
        "num": num,
        "page": page,
        "autocorrect": False,
    }
    if location:
        serper_search_arguments["location"] = location
    if tbs:
        serper_search_arguments["tbs"] = tbs
    serper_server_params = StdioServerParameters(
        command="npx",
        args=["-y", "serper-search-scrape-mcp-server"],
        env={"SERPER_API_KEY": SERPER_API_KEY},
    )
    max_retries = 5
    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            async with stdio_client(serper_server_params) as (read, write):
                async with ClientSession(
                    read, write, sampling_callback=None
                ) as session:
                    await session.initialize()
                    serper_tool_result = await session.call_tool(
                        serper_tool_name, arguments=serper_search_arguments
                    )
                    search_result_content = (
                        serper_tool_result.content[-1].text if serper_tool_result.content else ""
                    )
                    if not search_result_content or not search_result_content.strip():
                        raise ValueError(
                            "Empty result from google_search tool, please try again."
                        )
                    return search_result_content
        except Exception as error:
            last_error = error
            if attempt >= max_retries:
                break
            await asyncio.sleep(min(2**attempt, 60))

    error_detail = f": {last_error}" if last_error else ""
    return f"[ERROR]: Tool execution failed after {max_retries} attempts{error_detail}"


async def search_all(
    query: str, engines: List[str] = None, **kwargs
) -> List[SearchResult]:
    """search all engines in parallel and return results"""

    # 并行执行搜索
    tasks = [engine(query, **kwargs) for engine in engines]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 处理异常
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed_results.append(
                SearchResult(
                    engine=engines[i],
                    query=query,
                    content="",
                    success=False,
                    error=str(result),
                )
            )
        else:
            processed_results.append(result)

    return processed_results


def format_search_results(
    results: List[SearchResult], query: str, include_metadata: bool = True
) -> str:
    """format search results"""
    if not results:
        return f"No search results found for query: '{query}'"

    formatted_output = []
    successful_results = [r for r in results if r.success]
    # failed_results = [r for r in results if not r.success]

    if successful_results:
        for result in successful_results:
            formatted_output.append(f"### {result.engine.title()} Search Results:")
            formatted_output.append(result.content)
            formatted_output.append("\n" + "=" * 50 + "\n")

    # if failed_results:
    #     formatted_output.append("## Failed Searches\n")
    #     for result in failed_results:
    #         formatted_output.append(f"- **{result.engine.title()}**: {result.error}")
    #     formatted_output.append("")
    return "\n".join(formatted_output)


@mcp.tool()
async def general_search(
    query: str,
    mode: str = "auto",
    parallel_search: bool = False,
    with_trust: bool = True,
    engine_timeout: float = DEFAULT_ENGINE_TIMEOUT,
) -> str:
    """Perform a deep search across multiple search engines (Perplexity AI, Google, Bing, DuckDuckGo) and return unified results.

    This tool searches across multiple engines and provides a unified, comprehensive response with results from all available sources.

    Args:
        query: a natural language search query
        mode: search depth — "speed" (fast factual, 2 iterations), "balanced"
              (moderate, 6 iterations), "quality" (deep research, 25 iterations),
              or "auto" (classify query and pick automatically). Default: "auto".

    Returns:
        Unified search results from all specified engines with performance metrics and citations
    """
    query = (query or "").strip()
    if not query:
        return "[ERROR]: Search query cannot be empty."

    _install_exception_handler()

    engines = [jina_deep_search]
    # functools.partial objects lack __name__; add it so engine logging works
    for engine in engines:
        if not hasattr(engine, "__name__"):
            engine.__name__ = getattr(getattr(engine, "func", None), "__name__", "unknown")
    logger.info(
        "general_search called -- query: %s, engines: %d", query, len(engines)
    )
    start_time = time.time()

    try:
        if parallel_search:
            logger.info("Starting parallel search across %d engines", len(engines))
            wrapped = [
                asyncio.wait_for(engine(query), timeout=engine_timeout)
                for engine in engines
            ]
            raw_results = await asyncio.gather(*wrapped, return_exceptions=True)
            results = []
            for i, res in enumerate(raw_results):
                if isinstance(res, Exception):
                    results.append(
                        SearchResult(
                            engine=getattr(engines[i], "__name__", "unknown"),
                            query=query,
                            content="",
                            success=False,
                            error=str(res),
                        )
                    )
                else:
                    results.append(res)

        else:
            logger.info(
                "Starting sequential search across %d engines", len(engines)
            )
            results = []
            for engine in engines:
                logger.info("%s started", engine.__name__)
                try:
                    result = await asyncio.wait_for(
                        engine(query), timeout=engine_timeout
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "%s timed out after %.0fs", engine.__name__, engine_timeout
                    )
                    continue
                except asyncio.CancelledError:
                    logger.warning("%s cancelled", engine.__name__)
                    raise
                except Exception as e:
                    # Log full traceback for easier debugging of engine-specific failures
                    logger.error(
                        "%s failed with error: %s\n%s",
                        engine.__name__,
                        e,
                        traceback.format_exc(),
                    )
                    continue

                logger.info("%s completed", result.engine.title())
                logger.info("Search results: %s", result.content)
                gpt_judge = await search_content_judge(query, result.content)
                if gpt_judge:
                    logger.info("Content judge passed")
                    results.append(result)
                    break
                else:
                    logger.info("Content judge failed, skipping result")
            if len(results) == 0:
                logger.info(
                    "No valid search results from any engine, falling back to Google"
                )
                google_result_str = await google_search(query)
                # google_search returns a string, not a SearchResult, so wrap it
                google_result = SearchResult(
                    engine="Google",
                    query=query,
                    content=google_result_str,
                    success=not google_result_str.startswith("[ERROR]"),
                    error=google_result_str
                    if google_result_str.startswith("[ERROR]")
                    else None,
                )
                if google_result.success and len(google_result.content) > 0:
                    logger.info("Google search completed")
                else:
                    logger.info("Google search failed")
                results.append(google_result)

        # 格式化并返回结果
        formatted_results = format_search_results(results, query)

        elapsed_time = time.time() - start_time
        logger.info("general_search completed in %.2fs", elapsed_time)
        # logger.info("general_search inner formatted_results:")
        # logger.info(formatted_results)
        return formatted_results

    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error("general_search error in %.2fs: %s", elapsed_time, e)
        return f"[ERROR]: General search failed: {str(e)}"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Searching MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="sse",
        help="Transport method: 'stdio' or 'sse' (default: sse)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to use when running with SSE transport (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8936,
        help="Port to use when running with SSE transport (default: 8936)",
    )
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="sse", host=args.host, port=args.port)
