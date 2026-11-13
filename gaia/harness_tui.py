"""GAIA Benchmark Harness — Textual TUI Dashboard.

Usage:
    cd gaia
    uv run python harness_tui.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import Screen
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    Footer,
    Input,
    Label,
    RichLog,
    Select,
    Static,
)

from harness import (
    RESULTS_DIR,
    GAIATask,
    _stop_autobrowser_sse_server,
    load_checkpoint,
    load_tasks,
    run_task,
    save_checkpoint_entry,
    setup_agents,
    write_submission,
)

logger = logging.getLogger("gaia-agent")


def _find_checkpoints() -> list[tuple[str, str]]:
    """Scan runs/ for checkpoint JSONL files."""
    if not RESULTS_DIR.exists():
        return []
    files = sorted(RESULTS_DIR.glob("*_checkpoint.jsonl"), reverse=True)
    items: list[tuple[str, str]] = []
    for f in files:
        try:
            count = sum(1 for line in open(f) if line.strip())
        except Exception:
            count = 0
        items.append((f"{f.name}  ({count} tasks)", str(f)))
    return items


# ---------------------------------------------------------------------------
# Log handler — posts to main thread via call_from_thread
# ---------------------------------------------------------------------------

class TUILogHandler(logging.Handler):
    def __init__(self, app: GAIAHarnessApp) -> None:
        super().__init__()
        self._app = app

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
            self._app.log_from_thread(f"[dim]{ts}[/] {msg}")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Messages (processed on main thread)
# ---------------------------------------------------------------------------

class LogMessage(Message):
    """Worker -> main thread log message."""
    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text


class StatusUpdate(Message):
    """Worker -> main thread status update."""
    def __init__(
        self,
        *,
        tasks_completed: int | None = None,
        tasks_total: int | None = None,
        correct_count: int | None = None,
        current_task_id: str | None = None,
        current_question: str | None = None,
        current_level: int | None = None,
        is_paused: bool | None = None,
        is_running: bool | None = None,
    ) -> None:
        super().__init__()
        self.tasks_completed = tasks_completed
        self.tasks_total = tasks_total
        self.correct_count = correct_count
        self.current_task_id = current_task_id
        self.current_question = current_question
        self.current_level = current_level
        self.is_paused = is_paused
        self.is_running = is_running


class TaskCompleted(Message):
    def __init__(self, entry: dict, task: GAIATask) -> None:
        super().__init__()
        self.entry = entry
        self.task = task


class BenchmarkFinished(Message):
    def __init__(self, submission_path: str) -> None:
        super().__init__()
        self.submission_path = submission_path


# ---------------------------------------------------------------------------
# Config Screen
# ---------------------------------------------------------------------------

class ConfigScreen(Screen):

    CSS = """
    ConfigScreen {
        align: center middle;
    }
    #config-box {
        width: 60;
        height: auto;
        border: round $accent;
        border-title-color: $accent;
        border-title-style: bold;
        padding: 1 3 2 3;
        background: $surface;
    }
    .cfg-label {
        color: $text-muted;
        margin-top: 1;
    }
    #resume-check {
        margin-top: 1;
    }
    #checkpoint-select {
        display: none;
    }
    #start-btn {
        margin-top: 2;
        width: 100%;
    }
    """

    BINDINGS = [Binding("enter", "start", show=False)]

    def compose(self) -> ComposeResult:
        box = Vertical(id="config-box")
        box.border_title = "Configuration"
        with box:
            yield Label("Split", classes="cfg-label")
            yield Select(
                [("validation  (165 tasks)", "validation"), ("test  (300 tasks)", "test")],
                value="validation",
                id="split-select",
            )
            yield Label("Level", classes="cfg-label")
            yield Select(
                [("All levels", 0), ("1 - easy", 1), ("2 - medium", 2), ("3 - hard", 3)],
                value=0,
                id="level-select",
            )
            yield Label("Task limit (blank = run all)", classes="cfg-label")
            yield Input(placeholder="e.g. 10", id="limit-input")
            yield Checkbox("Resume from checkpoint", value=True, id="resume-check")
            yield Label("Checkpoint file", classes="cfg-label", id="checkpoint-label")
            ckpt_items = _find_checkpoints()
            if ckpt_items:
                yield Select(
                    [("Auto (latest for split)", "")] + ckpt_items,
                    value="",
                    id="checkpoint-select",
                )
            else:
                yield Select(
                    [("No checkpoints found", "")],
                    value="",
                    id="checkpoint-select",
                )
            yield Button("Start", variant="primary", id="start-btn")

    def on_mount(self) -> None:
        self._toggle_checkpoint_visibility()

    @on(Checkbox.Changed, "#resume-check")
    def on_resume_toggled(self, event: Checkbox.Changed) -> None:
        self._toggle_checkpoint_visibility()

    def _toggle_checkpoint_visibility(self) -> None:
        resume = self.query_one("#resume-check", Checkbox).value
        self.query_one("#checkpoint-select", Select).display = resume
        self.query_one("#checkpoint-label", Label).display = resume

    def action_start(self) -> None:
        self._go()

    @on(Button.Pressed, "#start-btn")
    def on_start(self, event: Button.Pressed) -> None:
        self._go()

    def _go(self) -> None:
        split = self.query_one("#split-select", Select).value or "validation"
        level_val = self.query_one("#level-select", Select).value
        level = int(level_val) if level_val and level_val != 0 else None
        limit_text = self.query_one("#limit-input", Input).value.strip()
        limit = int(limit_text) if limit_text.isdigit() else None
        resume = self.query_one("#resume-check", Checkbox).value
        ckpt_path = self.query_one("#checkpoint-select", Select).value if resume else ""

        app = self.app
        assert isinstance(app, GAIAHarnessApp)
        app.bench_split = split
        app.bench_level = level
        app.bench_limit = limit
        app.bench_resume = resume
        app.bench_checkpoint_path = ckpt_path or ""
        app.push_screen("dashboard")


# ---------------------------------------------------------------------------
# Dashboard Screen
# ---------------------------------------------------------------------------

class DashboardScreen(Screen):

    CSS = """
    DashboardScreen {
        layout: vertical;
    }
    #top-row {
        height: auto;
        max-height: 11;
    }
    #status-panel {
        width: 1fr;
        height: auto;
        min-height: 7;
        border: round $primary;
        border-title-color: $primary;
        border-title-style: bold;
        padding: 0 2;
        background: $surface;
    }
    #task-panel {
        width: 2fr;
        height: auto;
        min-height: 7;
        border: round $accent;
        border-title-color: $accent;
        border-title-style: bold;
        padding: 0 2;
        background: $surface;
    }
    #log-panel {
        height: 1fr;
        border: round $primary;
        border-title-color: $primary;
        border-title-style: bold;
        padding: 0;
        background: $surface;
    }
    #agent-log {
        height: 1fr;
        scrollbar-size: 1 1;
        padding: 0 1;
        background: $surface;
    }
    #results-panel {
        height: 12;
        border: round $primary;
        border-title-color: $primary;
        border-title-style: bold;
        padding: 0;
        background: $surface;
    }
    #results-table {
        height: 1fr;
        background: $surface;
    }
    """

    BINDINGS = [
        Binding("p", "toggle_pause", "Pause"),
        Binding("k", "skip_task", "Skip"),
        Binding("q", "quit_harness", "Quit"),
        Binding("c", "clear_log", "Clear"),
    ]

    def compose(self) -> ComposeResult:
        with Horizontal(id="top-row"):
            status = Vertical(id="status-panel")
            status.border_title = "Status"
            with status:
                yield Static("Initializing...", id="status-content")

            task = Vertical(id="task-panel")
            task.border_title = "Current Task"
            with task:
                yield Static("Waiting for agents...", id="task-content")

        log = Vertical(id="log-panel")
        log.border_title = "Log"
        with log:
            yield RichLog(id="agent-log", wrap=True, highlight=False, markup=True)

        results = Vertical(id="results-panel")
        results.border_title = "Results"
        with results:
            yield DataTable(id="results-table")

        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#results-table", DataTable)
        table.add_columns("#", "Task ID", "Lvl", "Model Answer", "Ground Truth", " ")
        table.cursor_type = "none"

        app = self.app
        assert isinstance(app, GAIAHarnessApp)
        app.start_benchmark()

    def action_toggle_pause(self) -> None:
        app = self.app
        assert isinstance(app, GAIAHarnessApp)
        app._is_paused = not app._is_paused
        self.query_one("#agent-log", RichLog).write(
            "[bold]PAUSED[/]" if app._is_paused else "[bold]RESUMED[/]"
        )
        app.refresh_status()

    def action_skip_task(self) -> None:
        app = self.app
        assert isinstance(app, GAIAHarnessApp)
        app.skip_event.set()

    def action_quit_harness(self) -> None:
        app = self.app
        assert isinstance(app, GAIAHarnessApp)
        app.quit_requested = True
        app.skip_event.set()
        # Give worker a moment to finish, then exit
        self.set_timer(2, app.exit)

    def action_clear_log(self) -> None:
        self.query_one("#agent-log", RichLog).clear()


# ---------------------------------------------------------------------------
# App — all widget mutations happen on main thread via message handlers
# ---------------------------------------------------------------------------

class GAIAHarnessApp(App):

    TITLE = "GAIA Benchmark Harness"
    SCREENS = {"config": ConfigScreen, "dashboard": DashboardScreen}

    # Config (set by ConfigScreen before pushing dashboard)
    bench_split: str = "validation"
    bench_level: int | None = None
    bench_limit: int | None = None
    bench_resume: bool = True
    bench_checkpoint_path: str = ""

    # Shared state (thread-safe via atomics / simple reads)
    quit_requested: bool = False
    _is_paused: bool = False
    skip_event: asyncio.Event

    # State for rendering (only touched on main thread)
    _tasks_completed: int = 0
    _tasks_total: int = 0
    _correct_count: int = 0
    _current_task_id: str = ""
    _current_question: str = ""
    _current_level: int = 0
    _is_running: bool = False

    def __init__(self) -> None:
        super().__init__()
        self.skip_event = asyncio.Event()
        self._log_handler: TUILogHandler | None = None
        self._start_time: float = 0

    def on_mount(self) -> None:
        self.push_screen("config")

    def on_unmount(self) -> None:
        """Kill subprocesses when app exits."""
        self._remove_log_handler()
        _stop_autobrowser_sse_server()

    # --- Main-thread message handlers (all widget access here) ---

    @on(LogMessage)
    def on_log_message(self, message: LogMessage) -> None:
        try:
            self.query_one("#agent-log", RichLog).write(message.text)
        except Exception:
            print(message.text, file=sys.stderr)

    @on(StatusUpdate)
    def on_status_update(self, message: StatusUpdate) -> None:
        if message.tasks_completed is not None:
            self._tasks_completed = message.tasks_completed
        if message.tasks_total is not None:
            self._tasks_total = message.tasks_total
        if message.correct_count is not None:
            self._correct_count = message.correct_count
        if message.current_task_id is not None:
            self._current_task_id = message.current_task_id
        if message.current_question is not None:
            self._current_question = message.current_question
        if message.current_level is not None:
            self._current_level = message.current_level
        if message.is_running is not None:
            self._is_running = message.is_running
        self.refresh_status()
        self.refresh_task()

    @on(TaskCompleted)
    def on_task_completed(self, message: TaskCompleted) -> None:
        entry = message.entry
        task = message.task

        score = entry.get("score", False)
        if score:
            ok = "+"
        elif task.final_answer == "?":
            ok = "-"
        else:
            ok = "x"

        try:
            table = self.query_one("#results-table", DataTable)
            table.add_row(
                str(self._tasks_completed),
                entry["task_id"][:12],
                str(entry.get("level", "")),
                str(entry.get("model_answer", ""))[:30],
                str(entry.get("ground_truth", ""))[:30],
                ok,
            )
            table.scroll_end()
        except Exception:
            pass

    @on(BenchmarkFinished)
    def on_benchmark_finished(self, message: BenchmarkFinished) -> None:
        try:
            self.query_one("#agent-log", RichLog).write(f"done. {message.submission_path}")
        except Exception:
            pass
        self._is_running = False
        self.refresh_status()

    # --- Rendering helpers (main thread only) ---

    def _elapsed(self) -> str:
        s = int(time.time() - self._start_time) if self._start_time else 0
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def refresh_status(self) -> None:
        try:
            w = self.query_one("#status-content", Static)
            done = self._tasks_completed
            total = self._tasks_total
            wrong = done - self._correct_count
            acc = (self._correct_count / done * 100) if done else 0
            remaining = total - done

            bar_w = 20
            filled = int(done / total * bar_w) if total else 0
            bar = "█" * filled + "░" * (bar_w - filled)

            lvl = f"Level {self.bench_level}" if self.bench_level else "All levels"
            pause_tag = "  [bold]PAUSED[/]" if self._is_paused else ""

            w.update(
                f"[dim]{self.bench_split}  {lvl}[/]{pause_tag}\n"
                f"[dim]Time[/]  {self._elapsed()}\n\n"
                f"{bar}  {done}/{total}\n\n"
                f"[bold]pass[/] {self._correct_count}  "
                f"[bold]fail[/] {wrong}  "
                f"[dim]left[/] {remaining}  "
                f"[bold]{acc:.1f}%[/]"
            )
        except Exception:
            pass

    def refresh_task(self) -> None:
        try:
            w = self.query_one("#task-content", Static)
            if not self._current_task_id:
                w.update("[dim]Waiting for agents...[/]")
            else:
                q = self._current_question
                lines = []
                while q and len(lines) < 3:
                    if len(q) <= 64:
                        lines.append(q)
                        q = ""
                    else:
                        brk = q.rfind(" ", 0, 64)
                        if brk == -1:
                            brk = 64
                        lines.append(q[:brk])
                        q = q[brk:].lstrip()
                if q:
                    lines[-1] = lines[-1][:61] + "..."
                q_display = "\n".join(lines)
                w.update(
                    f"[bold]{self._current_task_id[:16]}[/]  "
                    f"[dim]Level {self._current_level}[/]\n\n"
                    f"{q_display}"
                )
        except Exception:
            pass

    # --- Timer for elapsed clock (runs on main thread) ---

    def _on_tick(self) -> None:
        if self._is_running:
            self.refresh_status()

    # --- Worker thread helpers ---

    def log_from_thread(self, text: str) -> None:
        """Safe to call from any thread. Posts a LogMessage to main thread."""
        try:
            self.post_message(LogMessage(text))
        except Exception:
            print(text, file=sys.stderr)

    def update_status_from_thread(self, **kwargs: Any) -> None:
        """Safe to call from any thread. Posts a StatusUpdate to main thread."""
        try:
            self.post_message(StatusUpdate(**kwargs))
        except Exception:
            pass

    # --- Log handler management ---

    def _install_log_handler(self) -> None:
        self._log_handler = TUILogHandler(self)
        self._log_handler.setFormatter(logging.Formatter("%(message)s"))
        for name in ("gaia-agent", "agent.react_agent", "agent.tool_call_handler"):
            lg = logging.getLogger(name)
            lg.addHandler(self._log_handler)
            lg.setLevel(logging.DEBUG)

    def _remove_log_handler(self) -> None:
        if self._log_handler:
            for name in ("gaia-agent", "agent.react_agent", "agent.tool_call_handler"):
                logging.getLogger(name).removeHandler(self._log_handler)

    # --- Benchmark worker ---

    @work(thread=True, exclusive=True)
    def start_benchmark(self) -> None:
        # Start the elapsed-time ticker on the main thread
        self.call_from_thread(self.set_interval, 1, self._on_tick)
        self._start_time = time.time()

        try:
            asyncio.run(self._run_benchmark_async())
        except Exception as e:
            tb = traceback.format_exc()
            self.log_from_thread(f"FATAL: {e}\n{tb}")

    async def _run_benchmark_async(self) -> None:
        self._install_log_handler()
        self.update_status_from_thread(is_running=True)

        main_agent = browsing_agent = coding_agent = None

        try:
            tasks = load_tasks(
                split=self.bench_split,
                level=self.bench_level,
                limit=self.bench_limit,
            )
            self.log_from_thread(f"loaded {len(tasks)} tasks")

            RESULTS_DIR.mkdir(parents=True, exist_ok=True)

            # Resolve checkpoint
            if self.bench_resume:
                if self.bench_checkpoint_path:
                    ckpt_path = Path(self.bench_checkpoint_path)
                else:
                    ckpt_path = RESULTS_DIR / f"{self.bench_split}_checkpoint.jsonl"
                ckpt = load_checkpoint(ckpt_path)
                self.log_from_thread(f"checkpoint: {ckpt_path.name}")
            else:
                ckpt_path = RESULTS_DIR / f"{self.bench_split}_checkpoint.jsonl"
                ckpt = {}

            # Count completed
            completed = 0
            correct = 0
            for task in tasks:
                if task.task_id in ckpt:
                    completed += 1
                    if ckpt[task.task_id].get("score"):
                        correct += 1

            self.update_status_from_thread(
                tasks_total=len(tasks),
                tasks_completed=completed,
                correct_count=correct,
            )

            if completed:
                self.log_from_thread(f"resuming: {completed} tasks already done")

            self.log_from_thread("connecting mcp servers...")
            main_agent, browsing_agent, coding_agent = await setup_agents()
            self.log_from_thread("agents ready")

            for i, task in enumerate(tasks):
                if self.quit_requested:
                    break
                if task.task_id in ckpt:
                    continue

                while self._is_paused and not self.quit_requested:
                    await asyncio.sleep(0.5)
                if self.quit_requested:
                    break

                self.skip_event.clear()

                self.update_status_from_thread(
                    current_task_id=task.task_id,
                    current_question=task.question,
                    current_level=task.level,
                )

                self.log_from_thread(
                    f"\n{'─' * 40}\n"
                    f"task {i+1}/{len(tasks)}  {task.task_id[:12]}  L{task.level}"
                )

                entry = await run_task(main_agent, browsing_agent, coding_agent, task)

                save_checkpoint_entry(ckpt_path, entry)
                ckpt[task.task_id] = entry
                completed += 1
                if entry.get("score"):
                    correct += 1

                self.update_status_from_thread(
                    tasks_completed=completed,
                    correct_count=correct,
                )

                marker = "+" if entry.get("score") else "x"
                self.log_from_thread(f"{marker} {entry.get('model_answer', '')[:40]}")
                self.post_message(TaskCompleted(entry=entry, task=task))

        except Exception as e:
            self.log_from_thread(f"error: {e}")
            logger.exception("Benchmark error")
        finally:
            self.update_status_from_thread(is_running=False)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            sub_path = RESULTS_DIR / f"{self.bench_split}_{ts}_submission.jsonl"
            try:
                ckpt = load_checkpoint(ckpt_path)
                all_tasks = load_tasks(split=self.bench_split, level=self.bench_level, limit=self.bench_limit)
                write_submission(sub_path, all_tasks, ckpt)
            except Exception:
                pass

            self._remove_log_handler()
            self.post_message(BenchmarkFinished(submission_path=str(sub_path)))

            if main_agent:
                for agent in (main_agent, browsing_agent, coding_agent):
                    for conn in getattr(agent, "_mcp_connections", []):
                        try:
                            await conn.disconnect()
                        except Exception:
                            pass


def main() -> None:
    GAIAHarnessApp().run()


if __name__ == "__main__":
    main()
