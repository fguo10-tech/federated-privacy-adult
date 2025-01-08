"""Execute the notebooks top to bottom and save executed copies.

The committed notebooks carry no outputs. This script runs each of them in a fresh kernel
that uses the *current* Python interpreter (so the project's virtualenv does not have to be
activated or registered as a Jupyter kernel), with the repository root as working directory,
and writes the executed copies to ``notebooks/_executed/`` (git-ignored).

Usage::

    python -m experiments.run_notebooks                 # all notebooks
    python -m experiments.run_notebooks 01 03           # only those whose name starts with 01 / 03
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from src.common.logging_utils import get_logger, setup_logging

LOGGER = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO_ROOT / "notebooks"
OUTPUT_DIR = NOTEBOOK_DIR / "_executed"


def execute_notebook(path: Path, output_dir: Path = OUTPUT_DIR, timeout_s: int = 1800) -> float:
    """Run one notebook and return the wall-clock seconds it took.

    Raises whatever ``nbclient`` raises if a cell fails, after saving the partial output so
    the failing cell can be inspected.
    """
    import nbformat
    from jupyter_client.manager import KernelManager
    from nbclient import NotebookClient

    notebook = nbformat.read(path, as_version=4)
    manager = KernelManager(kernel_name="python3")
    # Launch the kernel with this interpreter, whatever "python" means on PATH.
    manager.kernel_spec.argv[0] = sys.executable
    environment = dict(os.environ)
    environment.setdefault("MPLBACKEND", "module://matplotlib_inline.backend_inline")
    client = NotebookClient(
        notebook,
        km=manager,
        timeout=timeout_s,
        resources={"metadata": {"path": str(REPO_ROOT)}},
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        manager.start_kernel(cwd=str(REPO_ROOT), env=environment)
        client.kc = manager.client()
        client.kc.start_channels()
        client.kc.wait_for_ready(timeout=120)
        client.execute(reset_kc=False)
    finally:
        nbformat.write(notebook, output_dir / path.name)
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    return time.perf_counter() - started


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "prefixes", nargs="*", help="run only notebooks whose name starts with these"
    )
    parser.add_argument("--timeout", type=int, default=1800, help="per-cell timeout in seconds")
    args = parser.parse_args(argv)
    setup_logging("INFO")
    notebooks = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    if args.prefixes:
        notebooks = [nb for nb in notebooks if nb.name.startswith(tuple(args.prefixes))]
    if not notebooks:
        LOGGER.error("no notebooks found in %s", NOTEBOOK_DIR)
        return 1
    failures = 0
    for notebook in notebooks:
        LOGGER.info("executing %s ...", notebook.name)
        try:
            seconds = execute_notebook(notebook, timeout_s=args.timeout)
        except Exception as error:  # noqa: BLE001 - report every notebook, then fail
            failures += 1
            LOGGER.error("FAILED %s: %s", notebook.name, str(error).splitlines()[-1][:300])
        else:
            LOGGER.info("ok %s in %.1f s -> %s", notebook.name, seconds, OUTPUT_DIR / notebook.name)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
