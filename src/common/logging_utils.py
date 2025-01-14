"""Logging setup. Library code logs; only command-line entry points configure handlers."""

from __future__ import annotations

import logging
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """Return the logger for a module; call as ``get_logger(__name__)``."""
    return logging.getLogger(name)


def setup_logging(level: str | int = "INFO", log_file: str | Path | None = None) -> None:
    """Configure root logging once for a command-line run.

    Parameters
    ----------
    level:
        Logging level name or number for the console handler.
    log_file:
        If given, the same records are also appended to this file.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.setLevel(level)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(_FORMAT, _DATE_FORMAT))
    root.addHandler(console)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_FORMAT, "%Y-%m-%d %H:%M:%S"))
        root.addHandler(file_handler)
    # Third-party libraries that are chatty at INFO level.
    for noisy in ("flwr", "matplotlib", "PIL", "opacus"):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, root.level))
