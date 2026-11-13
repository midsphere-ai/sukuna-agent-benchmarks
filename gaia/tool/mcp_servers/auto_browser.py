import os
import base64
import subprocess
import atexit
import httpx
import signal
import inspect
from urllib.parse import urlsplit
from dotenv import load_dotenv
load_dotenv()
import sys, socket, time

# Ensure browser-use never reuses an existing Playwright instance; set before importing library.
os.environ["BROWSER_USE_REUSE"] = "0"

from browser_use import Agent, Browser, BrowserConfig
from tool.browser import Controller
from tool.browser.action_memory import BrowserSessionContext
import requests
from tool.logger import bootstrap_logger
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from tool.mcp_servers import model_manager
from pathlib import Path
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from browser_use.browser.context import BrowserContext
import asyncio

class CacheLoggingCallbackHandler(BaseCallbackHandler):
    """Logs Anthropic prompt caching metrics from each LLM call."""

    def __init__(self):
        self.total_prompt = 0
        self.total_cached = 0
        self.total_cache_write = 0
        self.call_count = 0

    def on_llm_end(self, response: LLMResult, **kwargs):
        self.call_count += 1
        token_usage = (response.llm_output or {}).get("token_usage") or {}
        prompt = token_usage.get("prompt_tokens", 0)
        details = token_usage.get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens", 0)
        cache_write = details.get("cache_write_tokens", 0)

        self.total_prompt += prompt
        self.total_cached += cached
        self.total_cache_write += cache_write

        if prompt:
            logger.info(
                "[Cache] Call %d: prompt=%d cached=%d cache_write=%d hit_rate=%.1f%%",
                self.call_count, prompt, cached, cache_write, cached / prompt * 100,
            )
        else:
            logger.info("[Cache] Call %d: prompt=0 (n/a)", self.call_count)

    def summary(self):
        if self.total_prompt:
            logger.info(
                "[Cache Summary] calls=%d total_prompt=%d total_cached=%d "
                "total_cache_write=%d overall_hit_rate=%.1f%%",
                self.call_count, self.total_prompt, self.total_cached,
                self.total_cache_write, self.total_cached / self.total_prompt * 100,
            )
        else:
            logger.info("[Cache Summary] calls=%d total_prompt=0 (no data)", self.call_count)


PROXIES = None
VERIFY_SSL = False if os.environ.get("PPLX_VERIFY_SSL", "0").lower() in {"", "0", "false", "no"} else True
DEFAULT_TIMEOUT = float(os.environ.get("PPLX_HTTP_TIMEOUT", 30))
RETRY_MAX_TRIES = int(os.environ.get("PPLX_HTTP_RETRIES", 3))
CHROME_PATH = os.environ.get("CHROME_PATH", None)
CHROME_USER_PROFILE_DIR = os.environ.get("CHROME_USER_PROFILE_DIR", None)
API_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "openai-gpt-5.5"
MAX_VISION_IMAGE_BYTES = 4_800_000  # stay under common 5MB vision payload limits
logger = bootstrap_logger()
def _get_env_timeout(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default

AUTO_BROWSER_TIMEOUT_SECONDS = _get_env_timeout("AUTO_BROWSER_TIMEOUT_SECONDS", 0)
AUTO_BROWSER_CANCEL_GRACE_SECONDS = _get_env_timeout("AUTO_BROWSER_CANCEL_GRACE_SECONDS", 10)

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def get_project_root():
    current = Path(__file__).resolve()
    for candidate in current.parents:
        if (candidate / ".git").exists():
            return candidate
    return current.parents[-1]

def assemble_project_path(path):
    path_obj = Path(path)
    if not path_obj.is_absolute():
        path_obj = get_project_root() / path_obj
    return path_obj


def find_chrome_executable():
    """Return the Chrome executable path from the CHROME_PATH environment variable."""
    return CHROME_PATH


def get_browseruse_user_data_dir():
    """Get the browser-use user data directory path."""
    return CHROME_USER_PROFILE_DIR or str(Path.home() / ".config" / "browseruse" / "profiles" / "default")


def _is_google_maps_url(url: str | None) -> bool:
    """Return True when URL belongs to Google Maps (including /maps paths)."""
    if not url:
        return False
    try:
        parsed = urlsplit(url)
    except ValueError:
        return False
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    return (
        host.endswith("google.com")
        and ("maps." in host or path.startswith("/maps"))
    )


def _patch_browser_context_screenshot():
    """Extend screenshot handling to wait longer before timing out."""
    if getattr(BrowserContext, "_patched_screenshot", False):
        return

    async def _capture(page, *, full_page: bool, timeout_ms: int, quality: int):
        try:
            return await page.screenshot(
                full_page=full_page,
                animations="disabled",
                caret="initial",
                timeout=timeout_ms,
                type="jpeg",
                quality=quality,
            )
        except PlaywrightTimeoutError:
            logger.warning("Screenshot timed out; retrying with relaxed options.")
            try:
                return await page.screenshot(
                    full_page=full_page,
                    animations=None,
                    caret="initial",
                    timeout=timeout_ms,
                    type="jpeg",
                    quality=quality,
                )
            except PlaywrightTimeoutError:
                logger.error("Screenshot timed out on retry; falling back to minimal capture.")
                return None

    async def take_screenshot(self, full_page: bool = False) -> str:  # type: ignore[override]
        page = await self.get_agent_current_page()

        timeout_ms = 60_000
        current_url = getattr(page, "url", None)
        skip_network_idle = _is_google_maps_url(current_url)

        if skip_network_idle:
            # Google Maps keeps the network busy forever; waiting for networkidle blocks steps.
            try:
                await page.wait_for_load_state(timeout=min(15_000, timeout_ms))
            except PlaywrightTimeoutError:
                logger.debug("Load wait timed out on Google Maps; proceeding with screenshot.")
        else:
            try:
                await page.wait_for_load_state(state="networkidle", timeout=timeout_ms)
            except PlaywrightTimeoutError:
                try:
                    await page.wait_for_load_state(timeout=timeout_ms)
                except PlaywrightTimeoutError:
                    logger.warning("Load state wait timed out before screenshot; continuing anyway.")

        # Try a reasonably high JPEG quality first; degrade if payload exceeds limits.
        screenshot = await _capture(page, full_page=full_page, timeout_ms=timeout_ms, quality=80)

        if screenshot and len(screenshot) > MAX_VISION_IMAGE_BYTES:
            for quality in (65, 50, 40, 30, 20):
                candidate = await _capture(
                    page, full_page=full_page, timeout_ms=timeout_ms, quality=quality
                )
                if candidate and len(candidate) <= MAX_VISION_IMAGE_BYTES:
                    screenshot = candidate
                    break

        if screenshot and len(screenshot) > MAX_VISION_IMAGE_BYTES and full_page:
            # Drop full_page to shrink the shot if it is still too large.
            for quality in (65, 50, 40, 30, 20):
                candidate = await _capture(
                    page, full_page=False, timeout_ms=timeout_ms, quality=quality
                )
                if candidate and len(candidate) <= MAX_VISION_IMAGE_BYTES:
                    screenshot = candidate
                    break

        if screenshot is None:
            screenshot = await _capture(page, full_page=False, timeout_ms=0, quality=40) or b""

        return base64.b64encode(screenshot).decode("utf-8")

    BrowserContext.take_screenshot = take_screenshot  # type: ignore[assignment]
    BrowserContext._patched_screenshot = True  # type: ignore[attr-defined]


def _clear_browser_use_reuse_marker():
    """Remove reuse marker so browser_use won't attach to an existing Playwright instance."""
    try:
        reuse_file = Path.home() / ".browser_use" / "reuse.json"
        if reuse_file.exists():
            reuse_file.unlink()
    except Exception as e:
        logger.debug("Failed to clear browser_use reuse marker: %s", e)


class AutoBrowserUseTool():
    def __init__(self,
                api_key,
                model_id: str = "gemini",
                timeout_seconds: float | None = None,
                cancel_grace_seconds: float | None = None,
                 ):

        super(AutoBrowserUseTool, self).__init__()

        self.api_key = api_key
        self.model_id = model_id if model_id else DEFAULT_MODEL
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else AUTO_BROWSER_TIMEOUT_SECONDS
        self.cancel_grace_seconds = cancel_grace_seconds if cancel_grace_seconds is not None else AUTO_BROWSER_CANCEL_GRACE_SECONDS
        self.url = API_URL
        self.http_server_path = assemble_project_path("./browser/http_server")
        self.http_save_path = assemble_project_path("./browser/http_server/local")
        os.makedirs(self.http_save_path, exist_ok=True)

        self.http_server_port = self._find_free_port()  # 新增：可配置端口
        self.server_proc = None       # 新增：保存进程句柄

        self.browser = None
        self._started_at: float | None = None

        self._init_pdf_server()

    def _find_free_port(self, start=8080):
        s = socket.socket()
        for p in range(start, start+200):
            try:
                s.bind(("127.0.0.1", p))
                s.close()
                return p
            except OSError:
                continue
        return 0

    def _is_port_open(self, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.3)
            return s.connect_ex(("127.0.0.1", port)) == 0

    def _init_pdf_server(self):
        # 若端口被占用则直接认为可用；否则尝试启动
        if not self._is_port_open(self.http_server_port):
            cmd = [sys.executable, "-m", "http.server", str(self.http_server_port)]
        python_exec = sys.executable
        server_proc = subprocess.Popen(
            [python_exec, "-m", "http.server", str(self.http_server_port)],
            cwd=self.http_server_path,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        self.server_proc = server_proc
        
        # 健康检查：最多等 5 秒
        for _ in range(25):
            time.sleep(0.2)
            if self._is_port_open(self.http_server_port):
                break

        @atexit.register
        def shutdown_server():
            logger.info("Shutting down http.server")
            if self.server_proc and self.server_proc.poll() is None:
                try:
                    if os.name == "nt":
                        server_proc.terminate()
                    else:
                        server_proc.send_signal(signal.SIGTERM)
                    server_proc.wait(timeout=5)
                except Exception as e:
                    server_proc.kill()

    def _format_session_visit_summary(self, session_context: BrowserSessionContext | None) -> str:
        """Render a human-readable summary of sites and actions visited this run."""
        if not session_context:
            return "Visited sites summary: No website activity recorded during this session."
        memory = getattr(session_context, "memory", None)
        if not memory or not hasattr(memory, "get_session_observations"):
            return "Visited sites summary: Tracking is not available."
        site_actions = memory.get_session_observations()
        if not site_actions:
            return "Visited sites summary: No website interactions recorded during this run."
        lines: list[str] = ["Visited sites summary:"]
        for idx, (site, observations) in enumerate(site_actions.items(), start=1):
            lines.append(f"{idx}. {site}")
            for observation in observations:
                flattened = " ".join(
                    part.strip() for part in observation.splitlines() if part.strip()
                )
                lines.append(f"   - {flattened}")
        return "\n".join(lines)

    async def _close_context_and_tabs(self, ctx) -> None:
        """Close all pages within a Playwright context, then the context itself."""
        if not ctx:
            return
        pages = getattr(ctx, "pages", None)
        if pages:
            for page in list(pages):
                try:
                    await page.close()
                except Exception as e:
                    logger.debug("Error closing page: %s", e)
        close_ctx = getattr(ctx, "close", None)
        if callable(close_ctx):
            try:
                maybe_ctx = close_ctx()
                if inspect.isawaitable(maybe_ctx):
                    await maybe_ctx
            except Exception as e:
                logger.debug("Error closing context: %s", e)

    async def _safe_close_browser(self) -> None:
        """Close all tabs and the browser instance (best-effort)."""
        browser = getattr(self, "browser", None)
        if browser is None:
            return

        # Collect likely contexts (browser, browser.session, etc.)
        contexts = []
        for holder in (browser, getattr(browser, "session", None)):
            if not holder:
                continue
            for attr in ("context", "playwright_context", "playwright_browser"):
                ctx = getattr(holder, attr, None)
                if ctx and ctx not in contexts:
                    contexts.append(ctx)
            ctxs = getattr(holder, "contexts", None) or getattr(holder, "browser_contexts", None)
            if ctxs:
                for ctx in ctxs:
                    if ctx and ctx not in contexts:
                        contexts.append(ctx)

        for ctx in contexts:
            try:
                await self._close_context_and_tabs(ctx)
            except Exception as e:
                logger.debug("Error while closing context: %s", e)

        close_fn = getattr(browser, "close", None)
        if callable(close_fn):
            try:
                maybe_result = close_fn()
                if inspect.isawaitable(maybe_result):
                    await maybe_result
            except Exception as e:
                logger.warning("Error closing browser instance: %s", e)

        self.browser = None

    async def _browser_task(self, task):
        try:
            _clear_browser_use_reuse_marker()
            controller = Controller(http_save_path=self.http_save_path, http_server_port=self.http_server_port)
            storage_root = assemble_project_path("./website_database")
            task_id = ""
            if task_id:
                storage_path = storage_root / f"{task_id}_site_action_memory.jsonl"
            else:
                storage_path = storage_root / "site_action_memory.jsonl"
            session_context = BrowserSessionContext(
                storage_path=storage_path,
                session_id=task_id,
                # Persist browsing history across sub-agent invocations within the same task.
                load_existing=False,
            )
        except asyncio.CancelledError:
            # Propagate cancellation so upstream timeout logic can stop the run.
            raise
        except Exception as e:
            return f"Cannot instantiate controller: {e}"
            
        try:
            # Not working with the browser-use because of ainvoke introduction
            """client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.url,
                http_client=httpx.AsyncClient(verify=False),
            )
            model = LiteLLMModel(
                    model_id=self.model_id,
                    http_client=client,
                    custom_role_conversions={"tool-call": "assistant", "tool-response": "user"},
            )"""
            

            _gradient_base = os.getenv("GRADIENT_BASE_URL", "https://inference.do-ai.run/v1")
            _gradient_key = os.getenv("GRADIENT_API_KEY", self.api_key)
            cache_logger = CacheLoggingCallbackHandler()
            _disabled_params = {"parallel_tool_calls": None}
            slow_model = ChatOpenAI(
                model="anthropic-claude-4.6-sonnet",
                api_key=_gradient_key,
                base_url=_gradient_base,
                temperature=0.8,
                max_tokens=8192,
                timeout=240,
                callbacks=[cache_logger],
                disabled_params=_disabled_params,
            )
            fast_model = ChatOpenAI(
                model="anthropic-claude-4.6-sonnet",
                api_key=_gradient_key,
                base_url=_gradient_base,
                temperature=0.8,
                max_tokens=8192,
                timeout=120,
                callbacks=[cache_logger],
                disabled_params=_disabled_params,
            )
            planner_model = slow_model

            controller.bind_memory_summarizer(fast_model)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.info("Cannot instantiate model")
            return f"Error while instantiating model for browser use: {e}"

        # Code to check model has the .ainvoke attribute
        """try:
            _ = await model.ainvoke("ping")
        except Exception as e:
            return f"LLM probe failed: {e}"""


        # Find Chrome executable dynamically
        chrome_path = find_chrome_executable()
        if not chrome_path:
            logger.warning("Chrome executable not found at common locations. Trying without chrome_instance_path.")
        
        # Get user data directory dynamically
        user_data_dir = get_browseruse_user_data_dir()
        os.makedirs(user_data_dir, exist_ok=True)

        # Build BrowserConfig with conditional chrome_instance_path
        browser_config_kwargs = {
                # Let Chrome control window size via launch args (start maximized)
                "viewport": None,
                "headless": False,
                "channel": 'chrome',
                "keep_alive": False,
                "disable_security": True,
                "navigation_timeout_ms": 8000,       # 导航超时 ↑
                "screenshot_timeout_ms": 15000,       # 截图超时 ↑
                "wait_until": "domcontentloaded",
                "user_data_dir": user_data_dir,  # Dynamically determined path
                "launch_args": [
                    f"--user-data-dir={user_data_dir}",
                    "--profile-directory=Default",
                    "--start-maximized",
                    "--enable-gpu",
                    "--ignore-gpu-blocklist",
                    "--enable-accelerated-2d-canvas",
                    "--enable-gpu-rasterization",
                    "--enable-features=CanvasOopRasterization,UseSkiaRenderer",
                    # On Windows, ANGLE D3D11 is best for WebGL:
                    "--use-angle=d3d11",
                    # If d3d11 is problematic on your machine, try one of:
                    # "--use-angle=gl", "--use-gl=desktop"
                    # "--high-dpi-support=1",
                    # "--disable-renderer-backgrounding",
                    # "--disable-background-timer-throttling",
                    # "--disable-backgrounding-occluded-windows",
                    "--no-first-run",
                    "--no-default-browser-check",
                    # "--disable-features=IsolateOrigins,site-per-process",
                ],
                "new_context_kwargs": {
                    "ignore_https_errors": True,
                    # Use the full window size Playwright creates (maximized)
                    "viewport": None,
                    "locale": "en-US",
                    "timezone_id": "Asia/Singapore",
                    "permissions": ["clipboard-read","clipboard-write", "geolocation"],
                },
                "user_agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.7444.163 Safari/537.36',
        }
        
        # Only include chrome_instance_path if Chrome was found
        if chrome_path:
            browser_config_kwargs["chrome_instance_path"] = chrome_path
        
        browser = Browser(
            config=BrowserConfig(**browser_config_kwargs)
        )
        self.browser = browser
        # This is to ensure go_back function can run - might slow down google maps/street view but it shouldn't be significant problem
        context_config = browser.config.new_context_config
        context_config.minimum_wait_page_load_time = 3
        context_config.wait_for_network_idle_page_load_time = 3.0
        context_config.maximum_wait_page_load_time = 5.0
        context_config.wait_between_actions = 2
        logger.info("Browser agent starting")
        try:
            browser_agent = Agent(
                task=task,
                llm=fast_model,
                enable_memory=True,
                controller=controller,
                page_extraction_llm=slow_model,
                planner_llm = planner_model,
                browser=browser,      
                max_failures = 3,
                context=session_context,
                max_actions_per_step = 4,
                use_vision = True,
                use_vision_for_planner=True,
                planner_interval = 4,
                generate_gif=False,
                save_conversation_path=None,
                validate_output=False,

            )
            session_context.bind_message_manager(browser_agent.message_manager)
            logger.info("Browser agent running")
            history = await browser_agent.run(max_steps=30)
            """contents = []
            contents.extend(r.extracted_content or "" for r in history.action_results())
            contents.append(f"steps={history.number_of_steps()}")
            contents.append(f"is_done={history.is_done()}")
            return "\n".join(contents)"""
            contents = [
                entry.strip()
                for entry in history.extracted_content()
                if entry and entry.strip()
            ]
            if not contents:
                reuse_count = session_context.memory.existing_site_hits
                if reuse_count:
                    logger.info(
                        "Browser-use reused site memory %d time(s) during this run.",
                        reuse_count,
                    )
                visits_summary = self._format_session_visit_summary(session_context)
                return (
                    "AutoBrowser agent returned no extracted content.\n\n"
                    f"Site memory reuse count: {reuse_count}\n\n{visits_summary}"
                )

            reuse_count = session_context.memory.existing_site_hits
            if reuse_count:
                logger.info(
                    "Browser-use reused site memory %d time(s) during this run.",
                    reuse_count,
                )
            visits_summary = self._format_session_visit_summary(session_context)
            final_text = contents[-1]
            final_text = (
                f"{final_text}\n\nSite memory reuse count: {reuse_count}\n\n{visits_summary}"
            )
            return final_text
        except asyncio.CancelledError:
            logger.warning("Browser task cancelled; propagating.")
            raise
        except Exception as e:
            return f"Agent couldn't run due to : {e}"
        finally:
            # Always tear down tabs and browser when the run ends.
            try:
                await self._safe_close_browser()
            except Exception as e:
                logger.debug("Error during browser shutdown: %s", e)

    async def forward(self, task: str):
        """
        Automatically browse the web and extract information based on a given task.

        Args:
            task: The task to perform
        Returns:
            ToolResult with the task result
        """
        self._started_at = time.monotonic()
        return await self._browser_task(task)

async def browse(model_id: str, api_key: str, task: str, timeout: float = AUTO_BROWSER_TIMEOUT_SECONDS) -> str:
    """
    Convenience wrapper to run AutoBrowserUseTool in one call with timeout.
    
    Args:
        model_id: Model identifier (gemini, gpt, claude)
        api_key: API key for the model
        task: The browsing task to perform
        timeout: Maximum time in seconds (default uses AUTO_BROWSER_TIMEOUT_SECONDS env, 15 minutes if unset)
    
    Returns:
        Task result or timeout error message
    """
    tool = None
    try:
        tool = AutoBrowserUseTool(
            api_key=api_key,
            model_id=model_id,
            timeout_seconds=timeout,
            cancel_grace_seconds=AUTO_BROWSER_CANCEL_GRACE_SECONDS,
        )
        result = await tool.forward(task)
        return result

    except Exception as e:
        logger.error("Error in browse function: %s", e, exc_info=True)
        return f"Browser task failed: {str(e)}"
        
    finally:
        # Cleanup: destroy the tool instance
        if tool is not None:
            await _cleanup_browser_tool(tool)
            del tool


async def _cleanup_browser_tool(tool: AutoBrowserUseTool):
    """
    Clean up browser tool resources.
    
    Args:
        tool: The AutoBrowserUseTool instance to clean up
    """
    try:
        # Terminate the HTTP server if it's running
        if hasattr(tool, 'server_proc') and tool.server_proc is not None:
            if tool.server_proc.poll() is None:  # Process is still running
                logger.info("Terminating HTTP server on port %s", tool.http_server_port)
                try:
                    if os.name == "nt":
                        tool.server_proc.terminate()
                    else:
                        tool.server_proc.send_signal(signal.SIGTERM)
                    
                    # Wait for graceful shutdown
                    try:
                        tool.server_proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        logger.warning("HTTP server didn't terminate gracefully, killing it")
                        tool.server_proc.kill()
                        tool.server_proc.wait()
                except Exception as e:
                    logger.error("Error terminating HTTP server: %s", e)
                    try:
                        tool.server_proc.kill()
                    except:
                        pass
        try:
            # Close any remaining pages and the browser itself.
            await tool._safe_close_browser()
        except Exception as e:
            logger.error("Error closing browser instance: %s", e)

        # Additional cleanup if needed (browser instances, temp files, etc.)
        # You can add more cleanup logic here if necessary
        
    except Exception as e:
        logger.error("Error during browser tool cleanup: %s", e, exc_info=True)

# Just for sanity check to see if tools are registered
if __name__=='__main__':
    import asyncio
    def _find_free_port(start=8080):
        s = socket.socket()
        for p in range(start, start+200):
            try:
                s.bind(("127.0.0.1", p))
                s.close()
                return p
            except OSError:
                continue
        return 0

    async def main():
        tmp = AutoBrowserUseTool(
            os.getenv("GRADIENT_KEY"),
            'claude'
            )
        result = await tmp.forward("""You are a research assistant. Complete ALL of the following steps carefully and report your findings:

1. Go to Wikipedia and find the article on "Logarska Dolina" (a valley in Slovenia). Note the exact elevation listed, the river that flows through it, and the year it was designated a landscape park.

2. From that Wikipedia article, find the nearest major town mentioned. Click through to that town's Wikipedia page. Record its population (most recent census), elevation, and the municipality it belongs to.

3. Go back to the Logarska Dolina article. Find any mentioned peaks or mountains surrounding the valley. Click through to the Wikipedia article for the highest peak mentioned. Record its exact elevation in meters, the mountain range it belongs to, and the first ascent date if listed.

4. Now go to Google Maps and search for "Logarska Dolina, Slovenia". Switch to satellite view. Zoom in to identify the main parking area at the entrance of the valley. Note the approximate coordinates of that parking area.

5. From the parking area, switch to Street View. Navigate along the main road INTO the valley (heading roughly northwest). Take note of at least 3 distinct landmarks or signs you pass along the way (e.g., information boards, hotels, trail markers, bridges).

6. Continue in Street View until you reach what appears to be the end of the road or a turnaround point. Describe what you see there — any buildings, waterfalls, trail signs, or natural features.

7. Now go to Google Maps and search for "Rinka Waterfall" (the famous waterfall at the head of Logarska Dolina). Switch to the Photos tab or check nearby photos. Describe the waterfall based on any available imagery.

8. Go to timeanddate.com and look up the sunrise and sunset times for Logarska Dolina (or the nearest town Solčava) for today's date. Record the exact sunrise, sunset, and day length.

9. Go to booking.com or a similar travel site. Search for accommodations in "Logarska Dolina" for a 2-night stay next weekend. Find the cheapest and most expensive options available. Record their names, prices, and star ratings.

10. Finally, compile all your findings into a structured summary with sections for: Geography, History, Nearby Peaks, Street View Observations, Waterfall Description, Daylight Hours, and Accommodation Options. Make sure every data point from steps 1-9 is included.""")
        print(result)
    
    asyncio.run(main())
