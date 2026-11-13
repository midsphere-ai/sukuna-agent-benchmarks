import logging
import os
from logging.handlers import RotatingFileHandler

logger = logging.getLogger("gaia-llm")

_file_logging_configured = False


def setup_file_logging(log_file_path: str = "gaia_run.log", simple_format: bool = True) -> None:
    """Configure logging for the entire agent process.

    Sets up the **root** logger with a console handler and a rotating file
    handler so that every logger in the process — ``gaia-llm``,
    ``agent.*`` (via ``__name__``), third-party libs (browser-use,
    perplexica, langchain, …) — gets uniform formatting and writes to
    the log file.

    Safe to call multiple times; only the first call has any effect.

    Args:
        log_file_path: Path to the log file (default: "gaia_run.log").
        simple_format: If True, use a compact format; if False, include
            the logger name in the console output too.
    """
    global _file_logging_configured
    if _file_logging_configured:
        return
    _file_logging_configured = True

    _fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    _datefmt = "%Y-%m-%d %H:%M:%S"
    _formatter = logging.Formatter(_fmt, datefmt=_datefmt)

    # ------------------------------------------------------------------
    # Root logger — catches ALL loggers that propagate (the default).
    # Replace any pre-existing handlers (library defaults that may
    # truncate, use Rich boxes, etc.).
    # ------------------------------------------------------------------
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)

    root_console = logging.StreamHandler()
    root_console.setLevel(logging.INFO)
    root_console.setFormatter(_formatter)
    root.addHandler(root_console)
    root.setLevel(logging.DEBUG)

    # Rotating file handler on root — every logger writes here.
    file_handler = RotatingFileHandler(
        filename=log_file_path,
        maxBytes=20 * 1024 * 1024,
        backupCount=20,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(_formatter)
    root.addHandler(file_handler)

    # ------------------------------------------------------------------
    # Named loggers — ensure they propagate to root so the above
    # handlers apply.  ``gaia-llm`` is used by openrouter_llm.py /
    # vertex_llm.py; ``gaia-agent`` by bootstrap_logger() / harness.
    # ------------------------------------------------------------------
    for name in ("gaia-llm", "gaia-agent"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.DEBUG)
        lg.propagate = True
        # Remove private handlers so messages aren't doubled — root
        # already has the console + file handlers.
        for h in lg.handlers[:]:
            lg.removeHandler(h)

    logger.info("File logging enabled: %s", os.path.abspath(log_file_path))


def process_input(task_description, task_file_name):
    """
    Enriches task description with file context information.

    Args:
        task_description: The original task description
        task_file_name: Optional file path associated with the task

    Returns:
        Enhanced task description with file handling instructions
    """
    # TODO: Support URL differentiation (YouTube, Wikipedia, general URLs)

    if not task_file_name:
        return task_description

    # Validate file existence
    if not os.path.isfile(task_file_name):
        raise FileNotFoundError(f"Error: File not found {task_file_name}")

    # Map file extensions to descriptive types
    extension_mappings = {
        'image': ['jpg', 'jpeg', 'png', 'gif', 'webp'],
        'text': ['txt'],
        'json': ['json', 'jsonld'],
        'excel': ['xlsx', 'xls'],
        'pdf': ['pdf'],
        'document': ['docx', 'doc'],
        'html': ['html', 'htm'],
        'ppt': ['pptx', 'ppt'],
        'wav': ['wav'],
        'mp3': ['mp3', 'm4a'],
        'zip': ['zip']
    }

    # Extract and normalize file extension
    ext = task_file_name.rsplit('.', 1)[-1].lower()

    # Determine file type category
    file_category = ext  # Default to extension itself
    for category, extensions in extension_mappings.items():
        if ext in extensions:
            file_category = category.capitalize()
            break

    # Append file context and usage instructions
    file_instruction = (
        f"\nNote: A {file_category} file '{task_file_name}' is associated with this task. "
        f"You should use available tools to read its content if necessary through {task_file_name}. "
        f"Additionally, if you need to analyze this file by Linux commands or python codes, "
        f"you should upload it to the sandbox first. Files in the sandbox cannot be accessed by other tools.\n\n"
    )

    return task_description + file_instruction

