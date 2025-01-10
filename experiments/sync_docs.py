"""Inject generated Markdown blocks into the documentation, or check that they are current.

Every table that contains a ``[Reproduced]`` number is produced by
``experiments/build_results.py`` from the result CSV files and stored as a small Markdown
file under ``experiments/results/reference/generated/``. Documents include such a block
between two marker lines::

    <!-- BEGIN GENERATED: before_after_summary -->
    ... never edit by hand ...
    <!-- END GENERATED: before_after_summary -->

Because README, RESULTS.md, the project report, the poster and the slides all include the
*same* block files, a value cannot differ between placements.

Usage::

    python -m experiments.sync_docs            # rewrite stale blocks in place
    python -m experiments.sync_docs --check    # exit 1 if any block is stale (used by CI)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from src.common.logging_utils import get_logger, setup_logging

LOGGER = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIR = REPO_ROOT / "experiments" / "results" / "reference" / "generated"
SKIP_DIRS = {".venv", ".git", "node_modules", "runs", "_executed", "generated", "__pycache__"}

MARKER_LINE = re.compile(r"^\s*<!-- (?P<kind>BEGIN|END) GENERATED: (?P<id>[A-Za-z0-9_\-]+) -->\s*$")


class DocSyncError(RuntimeError):
    """Raised for a malformed marker or a block id that has no generated file."""


def markdown_files(root: Path = REPO_ROOT) -> list[Path]:
    """All Markdown files of the repository that may contain generated blocks."""
    files: list[Path] = []
    for path in sorted(root.rglob("*.md")):
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        files.append(path)
    return files


def load_block(block_id: str, generated_dir: Path = GENERATED_DIR) -> str:
    """Return the generated Markdown for ``block_id`` (always ending with one newline)."""
    path = generated_dir / f"{block_id}.md"
    if not path.is_file():
        raise DocSyncError(
            f"No generated block '{block_id}' ({path}). Run `python -m experiments.build_results`."
        )
    return path.read_text(encoding="utf-8").rstrip("\n") + "\n"


def render(
    text: str, generated_dir: Path = GENERATED_DIR, source: str = ""
) -> tuple[str, list[str]]:
    """Replace every generated block in ``text``; return the new text and the ids found.

    A marker counts only when it is alone on its line and outside a fenced code block, so a
    document may show the marker syntax as an example. Everything between a BEGIN line and
    the END line with the same id is replaced, including any code fences the block contains
    (generated diagrams and config listings are fenced), which keeps the operation idempotent.
    """
    where = source or "document"
    output: list[str] = []
    found: list[str] = []
    in_fence = False
    open_block: str | None = None
    for line in text.split("\n"):
        marker = MARKER_LINE.match(line)
        if open_block is not None:
            if marker and marker.group("kind") == "END" and marker.group("id") == open_block:
                output.append(line)
                open_block = None
            elif marker and marker.group("kind") == "BEGIN":
                raise DocSyncError(
                    f"Unbalanced GENERATED markers in {where}: block '{open_block}' is not closed "
                    f"before '{marker.group('id')}' begins."
                )
            continue
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            output.append(line)
            continue
        if marker and not in_fence:
            if marker.group("kind") == "END":
                raise DocSyncError(
                    f"Unbalanced GENERATED markers in {where}: END '{marker.group('id')}' has no "
                    "BEGIN line before it."
                )
            open_block = marker.group("id")
            found.append(open_block)
            output.append(line)
            output.extend(load_block(open_block, generated_dir).rstrip("\n").split("\n"))
            continue
        output.append(line)
    if open_block is not None:
        raise DocSyncError(
            f"Unbalanced GENERATED markers in {where}: block '{open_block}' has no END line."
        )
    return "\n".join(output), found


def sync(check: bool, root: Path = REPO_ROOT, generated_dir: Path = GENERATED_DIR) -> int:
    """Synchronise (or check) all documents. Returns the number of stale files."""
    stale: list[Path] = []
    used: set[str] = set()
    for path in markdown_files(root):
        original = path.read_text(encoding="utf-8")
        if "GENERATED:" not in original:
            continue
        updated, found = render(original, generated_dir, str(path.relative_to(root)))
        used.update(found)
        if updated != original:
            stale.append(path)
            if not check:
                path.write_text(updated, encoding="utf-8")
                LOGGER.info("updated %s (%d blocks)", path.relative_to(root), len(found))
    available = {p.stem for p in generated_dir.glob("*.md")} if generated_dir.is_dir() else set()
    unused = sorted(available - used)
    if unused:
        LOGGER.info("generated blocks not used by any document: %s", ", ".join(unused))
    if check and stale:
        for path in stale:
            LOGGER.error("stale generated block(s) in %s", path.relative_to(root))
        LOGGER.error("Run `make results` (or `python -m experiments.sync_docs`) and commit.")
    elif check:
        LOGGER.info("all generated blocks are up to date (%d block ids in use)", len(used))
    return len(stale)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="do not write; exit 1 if stale")
    args = parser.parse_args(argv)
    setup_logging("INFO")
    try:
        stale = sync(check=args.check)
    except DocSyncError as error:
        LOGGER.error("%s", error)
        return 2
    return 1 if (args.check and stale) else 0


if __name__ == "__main__":
    sys.exit(main())
