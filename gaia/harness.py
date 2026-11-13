"""GAIA Benchmark Harness — core logic for task loading, execution, and checkpointing.

Loads GAIA dataset tasks, sets up multi-agent system, runs tasks through
the agent pipeline, and maintains crash-safe JSONL checkpoints.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from urllib.parse import urlsplit
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("gaia-agent")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
# The GAIA dataset is not shipped with this repository. Download it from
# https://huggingface.co/datasets/gaia-benchmark/GAIA and place the 2023/
# folder at <repo>/data/gaia/2023, or point GAIA_DATASET_DIR at it.
BENCHMARK_ROOT = Path(os.getenv("GAIA_DATASET_DIR") or REPO_ROOT.parent / "data" / "gaia" / "2023")
RESULTS_DIR = REPO_ROOT / "runs"


# ---------------------------------------------------------------------------
# GAIATask
# ---------------------------------------------------------------------------
@dataclass
class GAIATask:
    """Single GAIA benchmark task."""

    task_id: str
    question: str
    level: int
    final_answer: str  # "?" for test split
    file_name: str = ""  # original file_name field
    file_path: str = ""  # resolved absolute path


# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------

def load_tasks(
    split: str = "validation",
    level: int | None = None,
    limit: int | None = None,
) -> list[GAIATask]:
    """Load GAIA tasks from the dataset JSONL file.

    Args:
        split: 'validation' or 'test'.
        level: Filter by difficulty level (1, 2, or 3). None = all.
        limit: Maximum number of tasks to return.

    Returns:
        List of GAIATask instances.
    """
    metadata_path = BENCHMARK_ROOT / split / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Dataset not found: {metadata_path}")

    tasks: list[GAIATask] = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)

            # GAIA uses capital-letter keys
            task_id = entry.get("task_id", "")
            question = entry.get("Question", entry.get("question", ""))
            task_level = int(entry.get("Level", entry.get("level", 1)))
            final_answer = entry.get("Final answer", entry.get("ground_truth", "?"))
            file_name = entry.get("file_name", "")

            # Resolve file path
            file_path = ""
            if file_name:
                resolved = BENCHMARK_ROOT / split / file_name
                if resolved.exists():
                    file_path = str(resolved)

            tasks.append(
                GAIATask(
                    task_id=task_id,
                    question=question,
                    level=task_level,
                    final_answer=final_answer,
                    file_name=file_name,
                    file_path=file_path,
                )
            )

    if level is not None:
        tasks = [t for t in tasks if t.level == level]

    if limit is not None:
        tasks = tasks[:limit]

    return tasks


# ---------------------------------------------------------------------------
# Question enrichment
# ---------------------------------------------------------------------------

def enrich_question(task: GAIATask) -> str:
    """Append file path info to question if the task has an attached file."""
    if task.file_path:
        return task.question + f"\n\nHere are the necessary files: {task.file_path}"
    return task.question


# ---------------------------------------------------------------------------
# Checkpoint management (append-only JSONL, crash-safe)
# ---------------------------------------------------------------------------

def load_checkpoint(path: Path) -> dict[str, dict]:
    """Load checkpoint entries keyed by task_id.

    Later entries for the same task_id override earlier ones.
    Returns empty dict if file doesn't exist.
    """
    results: dict[str, dict] = {}
    if not path.exists():
        return results
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                results[entry["task_id"]] = entry
            except (json.JSONDecodeError, KeyError):
                continue
    return results


def save_checkpoint_entry(path: Path, entry: dict) -> None:
    """Append one JSON line to the checkpoint file (crash-safe)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# Submission formatting
# ---------------------------------------------------------------------------

def format_submission_entry(task: GAIATask, agent_result: dict[str, Any]) -> dict:
    """Format a single task result into GAIA submission format.

    Args:
        task: The GAIA task.
        agent_result: Result dict from ReActAgent.invoke().

    Returns:
        Dict in GAIA submission format.
    """
    extracted = agent_result.get("extracted_metadata", {})
    if isinstance(extracted, dict):
        boxed_answer = extracted.get("boxed_answer", "")
    else:
        boxed_answer = ""

    # Score via exact string match (only meaningful for validation split)
    ground_truth = task.final_answer
    if ground_truth and ground_truth != "?":
        score = str(boxed_answer).strip().lower() == str(ground_truth).strip().lower()
    else:
        score = False  # Can't score test split

    return {
        "task_id": task.task_id,
        "question": task.question,
        "level": task.level,
        "model_answer": str(boxed_answer),
        "ground_truth": ground_truth,
        "score": score,
        "steps": None,
        "capabilities": None,
    }


def write_submission(
    path: Path,
    tasks: list[GAIATask],
    checkpoint: dict[str, dict],
) -> None:
    """Write final GAIA-format submission JSONL from checkpoint data.

    Entries are ordered by the original task list.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for task in tasks:
            entry = checkpoint.get(task.task_id)
            if entry:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Agent setup
# ---------------------------------------------------------------------------

# Environment configuration
API_BASE = os.getenv("API_BASE", "https://inference.do-ai.run/v1")
API_KEY = os.getenv("API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "anthropic-claude-opus-4.7")
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openrouter")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
REASONING_MODEL_NAME = os.getenv("REASONING_MODEL_NAME", "openai-gpt-5.5")
REASONING_BASE_URL = os.getenv("REASONING_BASE_URL", "https://inference.do-ai.run/v1")
os.environ.setdefault("LLM_SSL_VERIFY", "false")

# Tool environment
TOOLING_VENV_DIR = os.getenv(
    "MCP_TOOL_VENV_DIR",
    str(REPO_ROOT / "tool" / ".venv-tool"),
)

if os.name == "nt":
    _tooling_python = os.path.join(TOOLING_VENV_DIR, "Scripts", "python.exe")
    _tooling_bin = os.path.join(TOOLING_VENV_DIR, "Scripts")
else:
    _tooling_python = os.path.join(TOOLING_VENV_DIR, "bin", "python")
    _tooling_bin = os.path.join(TOOLING_VENV_DIR, "bin")

MCP_TOOL_PYTHON = _tooling_python if os.path.exists(_tooling_python) else sys.executable
MCP_TOOL_BIN = _tooling_bin if os.path.isdir(_tooling_bin) else None
MCP_TOOL_CWD = str(REPO_ROOT)

_pythonpath_entries = [str(REPO_ROOT)]
_existing_pythonpath = os.getenv("PYTHONPATH")
if _existing_pythonpath:
    _pythonpath_entries.append(_existing_pythonpath)
MCP_TOOL_ENV_BASE = {"PYTHONPATH": os.pathsep.join(_pythonpath_entries)}


def build_tool_env(extra: dict | None = None) -> dict[str, str]:
    """Build environment dict for MCP tool subprocess."""
    env = dict(MCP_TOOL_ENV_BASE)
    if extra:
        for key, value in extra.items():
            if value is None:
                continue
            env[key] = str(value)
    return env


def _get_mcp_tool_groups() -> dict[str, dict]:
    """Return MCP tool group configurations (lazy import to avoid import-time side effects)."""
    from mcp import StdioServerParameters

    # Auto-browser transport config
    auto_browser_transport = (os.getenv("AUTO_BROWSER_TRANSPORT") or "sse").strip().lower()
    auto_browser_sse_url = os.getenv("AUTO_BROWSER_SSE_URL", "http://127.0.0.1:8930/sse")

    if auto_browser_transport == "sse":
        autobrowser_client_type = "sse"
        autobrowser_params = auto_browser_sse_url
    else:
        autobrowser_client_type = "stdio"
        autobrowser_params = StdioServerParameters(
            command=MCP_TOOL_PYTHON,
            args=["-u", "-m", "tool.mcp_servers.browser_use_mcp_server", "--transport", "stdio"],
            env=build_tool_env({
                "GRADIENT_API_KEY": os.getenv("GRADIENT_API_KEY"),
                "GRADIENT_BASE_URL": os.getenv("GRADIENT_BASE_URL"),
                "MCP_STDIO_SAFE": os.getenv("MCP_STDIO_SAFE"),
                "BROWSER_USE_STDIO_SAFE": os.getenv("BROWSER_USE_STDIO_SAFE"),
            }),
            cwd=MCP_TOOL_CWD,
        )

    return {
        "tool-autobrowser": {
            "server_name": "browser-use-server",
            "client_type": autobrowser_client_type,
            "params": autobrowser_params,
        },
        "tool-transcribe": {
            "server_name": "audio-mcp-server",
            "client_type": "stdio",
            "params": StdioServerParameters(
                command=MCP_TOOL_PYTHON,
                args=["-u", "-m", "tool.mcp_servers.audio_mcp_server", "--transport", "stdio"],
                env=build_tool_env({
                    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
                    "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY"),
                    "OPENROUTER_BASE_URL": os.getenv("OPENROUTER_BASE_URL"),
                }),
                cwd=MCP_TOOL_CWD,
            ),
        },
        "tool-reasoning": {
            "server_name": "reasoning-mcp-server",
            "client_type": "stdio",
            "params": StdioServerParameters(
                command=MCP_TOOL_PYTHON,
                args=["-u", "-m", "tool.mcp_servers.reasoning_mcp_server", "--transport", "stdio"],
                env=build_tool_env({
                    "GRADIENT_API_KEY": os.getenv("GRADIENT_API_KEY"),
                    "GRADIENT_BASE_URL": os.getenv("GRADIENT_BASE_URL"),
                    "REASONING_MODEL_NAME": os.getenv("REASONING_MODEL_NAME"),
                }),
                cwd=MCP_TOOL_CWD,
            ),
        },
        "tool-reading": {
            "server_name": "reading-mcp-server",
            "client_type": "stdio",
            "params": StdioServerParameters(
                command=MCP_TOOL_PYTHON,
                args=["-u", "-m", "tool.mcp_servers.reading_mcp_server", "--transport", "stdio"],
                env=build_tool_env({
                    "PATH": os.pathsep.join(
                        p for p in [MCP_TOOL_BIN, os.getenv("PATH", "")] if p
                    ),
                }),
                cwd=MCP_TOOL_CWD,
            ),
        },
        "tool-searching": {
            "server_name": "searching-mcp-server",
            "client_type": "stdio",
            "params": StdioServerParameters(
                command=MCP_TOOL_PYTHON,
                args=["-u", "-m", "tool.mcp_servers.searching_mcp_server", "--transport", "stdio"],
                env=build_tool_env({
                    "SERPER_API_KEY": os.getenv("SERPER_API_KEY"),
                    "JINA_API_KEY": os.getenv("JINA_API_KEY"),
                    "JINA_DEEPSEARCH_URL": os.getenv("JINA_DEEPSEARCH_URL"),
                    "GRADIENT_API_KEY": os.getenv("GRADIENT_API_KEY"),
                    "GRADIENT_BASE_URL": os.getenv("GRADIENT_BASE_URL"),
                    "SEARXNG_URL": os.getenv("SEARXNG_URL"),
                }),
                cwd=MCP_TOOL_CWD,
            ),
        },
        "tool-vqa": {
            "server_name": "vision-mcp-server",
            "client_type": "stdio",
            "params": StdioServerParameters(
                command=MCP_TOOL_PYTHON,
                args=["-u", "-m", "tool.mcp_servers.vision_mcp_server", "--transport", "stdio"],
                env=build_tool_env({
                    "GRADIENT_API_KEY": os.getenv("GRADIENT_API_KEY"),
                    "GRADIENT_BASE_URL": os.getenv("GRADIENT_BASE_URL"),
                    "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
                }),
                cwd=MCP_TOOL_CWD,
            ),
        },
        "tool-code": {
            "server_name": "e2b-python-interpreter",
            "client_type": "stdio",
            "params": StdioServerParameters(
                command=MCP_TOOL_PYTHON,
                args=["-u", "-m", "tool.mcp_servers.python_server", "--transport", "stdio"],
                env=build_tool_env({
                    "E2B_API_KEY": os.getenv("E2B_API_KEY"),
                    "E2B_TEMPLATE_ID": os.getenv("E2B_TEMPLATE_ID"),
                }),
                cwd=MCP_TOOL_CWD,
            ),
        },
    }


# ---------------------------------------------------------------------------
# Auto-browser SSE server management
# ---------------------------------------------------------------------------

AUTO_BROWSER_TRANSPORT = (os.getenv("AUTO_BROWSER_TRANSPORT") or "sse").strip().lower()
AUTO_BROWSER_SSE_URL = os.getenv("AUTO_BROWSER_SSE_URL", "http://127.0.0.1:8930/sse")
AUTO_BROWSER_START_SSE = (os.getenv("AUTO_BROWSER_START_SSE") or "true").strip().lower()

_AUTO_BROWSER_SSE_PROCESS: subprocess.Popen | None = None


def _parse_sse_host_port(url: str) -> tuple[str, int]:
    parsed = urlsplit(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8930
    return host, port


def _is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) == 0


def _stop_autobrowser_sse_server() -> None:
    global _AUTO_BROWSER_SSE_PROCESS
    if _AUTO_BROWSER_SSE_PROCESS and _AUTO_BROWSER_SSE_PROCESS.poll() is None:
        try:
            _AUTO_BROWSER_SSE_PROCESS.terminate()
            _AUTO_BROWSER_SSE_PROCESS.wait(timeout=5)
        except Exception:
            _AUTO_BROWSER_SSE_PROCESS.kill()
    _AUTO_BROWSER_SSE_PROCESS = None


def ensure_autobrowser_sse_server() -> None:
    """Start the browser-use SSE server if needed."""
    global _AUTO_BROWSER_SSE_PROCESS
    if AUTO_BROWSER_TRANSPORT != "sse":
        return
    if AUTO_BROWSER_START_SSE in {"0", "false", "no", "off"}:
        return

    host, port = _parse_sse_host_port(AUTO_BROWSER_SSE_URL)
    if _is_port_open(host, port):
        logger.info("auto-browser SSE server already running on %s:%d", host, port)
        return

    logger.info("starting auto-browser SSE server on %s:%d...", host, port)
    env = dict(os.environ)
    env.update(build_tool_env({
        "GRADIENT_API_KEY": os.getenv("GRADIENT_API_KEY"),
        "GRADIENT_BASE_URL": os.getenv("GRADIENT_BASE_URL"),
        "CHROME_PATH": os.getenv("CHROME_PATH"),
        "CHROME_USER_PROFILE_DIR": os.getenv("CHROME_USER_PROFILE_DIR"),
        "MCP_STDIO_SAFE": os.getenv("MCP_STDIO_SAFE"),
        "BROWSER_USE_STDIO_SAFE": os.getenv("BROWSER_USE_STDIO_SAFE"),
    }))

    args = [
        "-u", "-m", "tool.mcp_servers.browser_use_mcp_server",
        "--transport", "sse",
        "--host", host,
        "--port", str(port),
    ]
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    _AUTO_BROWSER_SSE_PROCESS = subprocess.Popen(
        [MCP_TOOL_PYTHON, *args],
        cwd=MCP_TOOL_CWD,
        env=env,
        creationflags=creationflags,
    )
    atexit.register(_stop_autobrowser_sse_server)

    for _ in range(50):
        if _is_port_open(host, port):
            logger.info("auto-browser SSE server ready")
            return
        time.sleep(0.2)

    logger.warning(
        "failed to detect browser-use SSE server on %s:%d — "
        "check logs and ensure tool venv dependencies are installed",
        host, port,
    )


MAIN_AGENT_TOOL_GROUPS = ["tool-reasoning"]
BROWSING_AGENT_TOOL_GROUPS = [
    "tool-searching",
    "tool-vqa",
    "tool-reading",
    "tool-autobrowser",
    "tool-transcribe",
]
CODING_AGENT_TOOL_GROUPS = ["tool-code", "tool-vqa", "tool-reading"]


async def setup_agents() -> tuple:
    """Create and configure the 3-agent system (main, browsing, coding).

    Returns:
        Tuple of (main_agent, browsing_agent, coding_agent).
    """
    from agent.react_agent import ReActAgent
    from agent.config import AgentConfigFactory, ModelInfo, LLMConfig
    from agent.prompt_templates import (
        get_main_agent_system_prompt,
        get_browsing_agent_system_prompt,
        get_coding_agent_system_prompt,
    )

    model_info = ModelInfo(
        api_key=API_KEY,
        api_base=API_BASE,
        model_name=MODEL_NAME,
        timeout=60,
        provider=MODEL_PROVIDER,
    )
    model_config = LLMConfig(model_info=model_info)

    main_agent_config = AgentConfigFactory.create_main_agent_config(
        agent_id="gaia_main_agent",
        agent_type="main",
        agent_version="1.0",
        description="Main MCP agent with multiple tool groups",
        model=model_config,
        prompt_template=[
            {"role": "system", "content": get_main_agent_system_prompt(datetime.now())}
        ],
        max_iteration=20,
        max_tool_calls_per_turn=1,
        enable_question_hints=True,
        open_api_key=OPENAI_API_KEY,
        enable_extract_final_answer=True,
        reasoning_model=REASONING_MODEL_NAME,
        reasoning_base_url=REASONING_BASE_URL,
        enable_todo_plan=False,
    )

    browsing_agent_config = AgentConfigFactory.create_main_agent_config(
        agent_id="agent-browsing",
        agent_type="sub",
        agent_version="1.0",
        description="Browsing specialist sub-agent",
        model=model_config,
        prompt_template=[
            {"role": "system", "content": get_browsing_agent_system_prompt(datetime.now())}
        ],
        max_iteration=20,
        max_tool_calls_per_turn=1,
        enable_question_hints=True,
        open_api_key=OPENAI_API_KEY,
        enable_extract_final_answer=True,
        enable_todo_plan=False,
    )

    coding_agent_config = AgentConfigFactory.create_main_agent_config(
        agent_id="agent-coding",
        agent_type="sub",
        agent_version="1.0",
        description="Coding specialist sub-agent",
        model=model_config,
        prompt_template=[
            {"role": "system", "content": get_coding_agent_system_prompt(datetime.now())}
        ],
        max_iteration=20,
        max_tool_calls_per_turn=1,
        enable_question_hints=True,
        open_api_key=OPENAI_API_KEY,
        enable_extract_final_answer=True,
        enable_todo_plan=False,
    )

    main_agent = ReActAgent(agent_config=main_agent_config, tools=None)
    browsing_agent = ReActAgent(agent_config=browsing_agent_config, tools=None)
    coding_agent = ReActAgent(agent_config=coding_agent_config, tools=None)

    # Start auto-browser SSE server if configured
    if AUTO_BROWSER_TRANSPORT == "sse":
        ensure_autobrowser_sse_server()

    # Build MCP tool groups via main_agent's connection manager
    mcp_groups = _get_mcp_tool_groups()
    tool_groups: dict[str, list] = {}
    for group_name, cfg in mcp_groups.items():
        t0 = time.monotonic()
        logger.info("connecting %s (%s)...", group_name, cfg["server_name"])
        tools = await main_agent.create_mcp_tools(
            server_name=cfg["server_name"],
            client_type=cfg["client_type"],
            params=cfg["params"],
        )
        elapsed = time.monotonic() - t0
        logger.info("  %s ready  %d tools  %.1fs", group_name, len(tools), elapsed)
        tool_groups[group_name] = tools

    # Assign tool groups to agents
    for group in MAIN_AGENT_TOOL_GROUPS:
        if group in tool_groups:
            main_agent.add_tools(tool_groups[group])

    for group in BROWSING_AGENT_TOOL_GROUPS:
        if group in tool_groups:
            browsing_agent.add_tools(tool_groups[group])

    for group in CODING_AGENT_TOOL_GROUPS:
        if group in tool_groups:
            coding_agent.add_tools(tool_groups[group])

    # Register sub-agents on main
    main_agent.register_sub_agent("agent-browsing", browsing_agent)
    main_agent.register_sub_agent("agent-coding", coding_agent)

    return main_agent, browsing_agent, coding_agent


# ---------------------------------------------------------------------------
# Task execution
# ---------------------------------------------------------------------------

async def run_task(
    main_agent: Any,
    browsing_agent: Any,
    coding_agent: Any,
    task: GAIATask,
    on_event: Callable[..., None] | None = None,
) -> dict:
    """Run a single GAIA task through the agent system.

    Args:
        main_agent: Main ReActAgent.
        browsing_agent: Browsing sub-agent.
        coding_agent: Coding sub-agent.
        task: The GAIA task to execute.
        on_event: Optional callback for TUI updates.

    Returns:
        Formatted submission entry dict.
    """
    # Clear all agent contexts
    main_agent._context_manager.clear()
    browsing_agent._context_manager.clear()
    coding_agent._context_manager.clear()

    enriched_q = enrich_question(task)

    if on_event:
        on_event("task_start", task_id=task.task_id, question=enriched_q)

    try:
        result = await main_agent.invoke({"query": enriched_q, "file_path": task.file_path})
    except Exception as e:
        logger.error("Task %s failed with error: %s", task.task_id, e)
        result = {
            "output": f"Error: {e}",
            "result_type": "error",
            "extracted_metadata": {"boxed_answer": ""},
        }

    entry = format_submission_entry(task, result)

    if on_event:
        on_event("task_complete", task_id=task.task_id, entry=entry)

    return entry


# ---------------------------------------------------------------------------
# CLI entry point (headless mode)
# ---------------------------------------------------------------------------

async def run_benchmark_headless(
    split: str = "validation",
    level: int | None = None,
    limit: int | None = None,
    resume: bool = True,
    checkpoint_file: str | None = None,
) -> None:
    """Run the full benchmark without TUI (headless/CI mode)."""
    from tool.logger import bootstrap_logger
    from agent.utils import setup_file_logging

    bootstrap_logger("INFO")
    setup_file_logging()

    tasks = load_tasks(split=split, level=level, limit=limit)
    logger.info("Loaded %d tasks (split=%s, level=%s)", len(tasks), split, level)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(checkpoint_file) if checkpoint_file else RESULTS_DIR / f"{split}_checkpoint.jsonl"
    checkpoint = load_checkpoint(checkpoint_path) if resume else {}
    logger.info("Checkpoint has %d completed tasks", len(checkpoint))

    main_agent, browsing_agent, coding_agent = await setup_agents()

    correct = sum(1 for e in checkpoint.values() if e.get("score"))
    completed = len(checkpoint)

    try:
        for i, task in enumerate(tasks):
            if task.task_id in checkpoint:
                logger.info("Skipping completed task %s (%d/%d)", task.task_id[:8], i + 1, len(tasks))
                continue

            logger.info("Running task %s (%d/%d) [Level %d]", task.task_id[:8], i + 1, len(tasks), task.level)

            entry = await run_task(main_agent, browsing_agent, coding_agent, task)
            save_checkpoint_entry(checkpoint_path, entry)
            checkpoint[task.task_id] = entry

            completed += 1
            if entry.get("score"):
                correct += 1

            accuracy = (correct / completed * 100) if completed else 0
            logger.info(
                "Task %s: answer=%s | score=%s | Running accuracy: %d/%d (%.1f%%)",
                task.task_id[:8],
                entry.get("model_answer", ""),
                entry.get("score"),
                correct,
                completed,
                accuracy,
            )
    finally:
        # Always write submission file from checkpoint
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        submission_path = RESULTS_DIR / f"{split}_{timestamp}_submission.jsonl"
        write_submission(submission_path, tasks, checkpoint)
        logger.info("Submission written to %s", submission_path)

        # Cleanup MCP connections
        for agent in (main_agent, browsing_agent, coding_agent):
            for conn in getattr(agent, "_mcp_connections", []):
                try:
                    await conn.disconnect()
                except Exception:
                    pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GAIA Benchmark Harness (headless)")
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--level", type=int, default=None, choices=[1, 2, 3])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint JSONL file to resume from")
    args = parser.parse_args()

    asyncio.run(run_benchmark_headless(
        split=args.split,
        level=args.level,
        limit=args.limit,
        resume=not args.no_resume,
        checkpoint_file=args.checkpoint,
    ))
