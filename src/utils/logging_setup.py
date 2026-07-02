"""Project-wide logging setup.

We mirror the Python ``logging`` module's API and route logs both to the
console (with rich formatting where available) and to ``logs/<name>.log``.
Importing modules should only call :func:`get_logger`; :func:`setup_logging`
is invoked once at the start of any entry point.
"""
from __future__ import annotations

import logging
import logging.config
import os
import sys
from datetime import datetime
from pathlib import Path

try:
    from rich.logging import RichHandler

    _RICH = True
except Exception:  # pragma: no cover - rich is optional at import time
    _RICH = False


_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    log_dir: str | os.PathLike = "logs",
    run_name: str | None = None,
    level: int = logging.INFO,
) -> Path:
    """Configure root logger.

    Returns the path to the log file used for the current run, so the caller
    can echo it for the user.
    """
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = run_name or "run"
    log_file = log_dir_path / f"{name}_{stamp}.log"

    handlers: list[logging.Handler] = []

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, _DATE_FORMAT))
    handlers.append(file_handler)

    if _RICH and sys.stderr.isatty():
        console_handler = RichHandler(
            rich_tracebacks=True,
            show_path=False,
            log_time_format=_DATE_FORMAT,
        )
    else:
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, _DATE_FORMAT))
    handlers.append(console_handler)

    logging.basicConfig(level=level, handlers=handlers, force=True)
    # Calm noisy third-party loggers.
    for noisy in ("matplotlib", "PIL", "urllib3", "httpx", "transformers", "torch"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger(__name__).info("Logging initialised — file: %s", log_file)
    return log_file


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
