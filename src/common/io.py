"""Result files: run directories, CSV, JSON and run metadata.

Every run writes, next to its raw CSV files, the exact config it used
(``config_used.yaml``) and a ``metadata.json`` with seeds, the command line and
package versions. A number is only allowed to be called ``[Reproduced]`` in the
documentation if such a folder exists for it.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from .config import config_hash, save_yaml

TRACKED_PACKAGES = (
    "numpy",
    "pandas",
    "torch",
    "flwr",
    "opacus",
    "scikit-learn",
    "matplotlib",
    "pyyaml",
)


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) if needed and return it as a ``Path``."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> Path:
    """Write dict rows to CSV with a stable column order.

    If ``fieldnames`` is omitted, columns are the union of keys in first-seen order.
    Missing values are written as empty cells.
    """
    rows = list(rows)
    if fieldnames is None:
        seen: dict[str, None] = {}
        for row in rows:
            for key in row:
                seen.setdefault(key, None)
        fieldnames = list(seen)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(row.get(key, "")) for key in fieldnames})
    return path


def _cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    return value


def read_csv(path: str | Path) -> list[dict[str, str]]:
    """Read a CSV written by :func:`write_csv` back into a list of string dicts."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: str | Path, obj: Any) -> Path:
    """Write ``obj`` as indented JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    return path


def read_json(path: str | Path) -> Any:
    """Read a JSON file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def environment_info() -> dict[str, Any]:
    """Describe the software environment of the current process."""
    packages: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            packages[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            packages[name] = "not installed"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "git_commit": _git_commit(),
    }


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def sanitise_command(text: str) -> str:
    """Strip machine-specific absolute paths from a command line before it is stored.

    Result folders are meant to be committed and shared, so they must not carry the local
    user name or directory layout: the interpreter path becomes ``python``, paths under the
    current working directory become relative, and anything else under the home directory is
    abbreviated with ``~``.
    """
    root = str(Path.cwd())
    for old, new in (
        (sys.executable, "python"),
        (root + os.sep, ""),
        (root, "."),
        (str(Path.home()), "~"),
    ):
        if old:
            text = text.replace(old, new)
    return text


def save_run_metadata(
    run_dir: str | Path,
    config: Any,
    seeds: Sequence[int],
    command: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write ``config_used.yaml`` and ``metadata.json`` into ``run_dir``.

    Returns the path of ``metadata.json``.
    """
    run_dir = ensure_dir(run_dir)
    save_yaml(config, run_dir / "config_used.yaml")
    payload: dict[str, Any] = {
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command": sanitise_command(command if command is not None else " ".join(sys.argv)),
        "seeds": [int(seed) for seed in seeds],
        "config_hash": config_hash(config),
        "environment": environment_info(),
    }
    if extra:
        payload["extra"] = dict(extra)
    return write_json(run_dir / "metadata.json", payload)
