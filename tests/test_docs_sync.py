"""The marker mechanism that keeps generated tables identical in every document."""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.sync_docs import DocSyncError, render, sync


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_blocks_are_injected_and_idempotent(tmp_path: Path) -> None:
    generated = tmp_path / "generated"
    _write(generated / "table_a.md", "| a | b |\n|---|---|\n| 1 | 2 |\n")
    document = "# Title\n\n<!-- BEGIN GENERATED: table_a -->\nold text\n<!-- END GENERATED: table_a -->\n\ntail\n"
    updated, found = render(document, generated)
    assert found == ["table_a"]
    assert "old text" not in updated
    assert "| 1 | 2 |\n<!-- END GENERATED: table_a -->" in updated
    again, _ = render(updated, generated)
    assert again == updated


def test_same_block_in_two_documents_is_identical_and_check_detects_drift(tmp_path: Path) -> None:
    generated = tmp_path / "experiments" / "results" / "reference" / "generated"
    _write(generated / "numbers.md", "value: 0.84\n")
    block = "<!-- BEGIN GENERATED: numbers -->\n<!-- END GENERATED: numbers -->\n"
    readme = _write(tmp_path / "README.md", f"intro\n\n{block}")
    report = _write(tmp_path / "docs" / "report.md", f"report\n\n{block}")
    assert sync(check=True, root=tmp_path, generated_dir=generated) == 2
    assert sync(check=False, root=tmp_path, generated_dir=generated) == 2
    assert sync(check=True, root=tmp_path, generated_dir=generated) == 0
    assert "value: 0.84" in readme.read_text() and "value: 0.84" in report.read_text()

    readme.write_text(readme.read_text().replace("0.84", "0.94"), encoding="utf-8")
    assert sync(check=True, root=tmp_path, generated_dir=generated) == 1, "hand edit must be caught"


def test_unknown_block_and_unbalanced_markers_are_errors(tmp_path: Path) -> None:
    generated = tmp_path / "generated"
    generated.mkdir()
    with pytest.raises(DocSyncError, match="No generated block"):
        render("<!-- BEGIN GENERATED: missing -->\n<!-- END GENERATED: missing -->\n", generated)
    _write(generated / "x.md", "x\n")
    with pytest.raises(DocSyncError, match="Unbalanced"):
        render("<!-- BEGIN GENERATED: x -->\nno end marker\n", generated)
    with pytest.raises(DocSyncError, match="Unbalanced"):
        render("<!-- BEGIN GENERATED: x -->\n<!-- END GENERATED: y -->\n", generated)


def test_marker_examples_in_code_fences_are_ignored_and_fenced_content_is_idempotent(
    tmp_path: Path,
) -> None:
    generated = tmp_path / "generated"
    _write(generated / "diagram.md", "```mermaid\nflowchart LR\n    A --> B\n```\n")
    document = (
        "How to include a block:\n\n"
        "```\n<!-- BEGIN GENERATED: block_id -->\n<!-- END GENERATED: block_id -->\n```\n\n"
        "Inline mention of `<!-- BEGIN GENERATED: other -->` is also not a block.\n\n"
        "<!-- BEGIN GENERATED: diagram -->\n<!-- END GENERATED: diagram -->\n\ntail\n"
    )
    updated, found = render(document, generated)
    assert found == ["diagram"], "the fenced example and the inline mention must be ignored"
    assert "flowchart LR" in updated and "BEGIN GENERATED: block_id" in updated
    again, found_again = render(updated, generated)
    assert again == updated and found_again == ["diagram"]
