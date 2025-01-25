"""Every key in the experiment configs must be documented, and every config must load."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from src.common.config import load_config

EXPERIMENT_CONFIGS = (
    "federated_adult.yaml",
    "federated_fedavg.yaml",
    "federated_dp.yaml",
)

KEY_LINE = re.compile(r"^\s*(?!-)(?P<key>[^\s#:][^#:]*?):(?:\s|$)")
INLINE_COMMENT = re.compile(r"\s#")


def undocumented_keys(path: Path) -> list[str]:
    """Keys that have neither an inline comment nor a comment line directly above them.

    List items (``- ...``) are covered by the comment of the key that owns the list.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    missing: list[str] = []
    for index, line in enumerate(lines):
        match = KEY_LINE.match(line)
        if match is None:
            continue
        if INLINE_COMMENT.search(line):
            continue
        previous = next((p for p in reversed(lines[:index]) if p.strip()), "")
        if previous.strip().startswith("#"):
            continue
        missing.append(f"{path.name}:{index + 1}: {match.group('key').strip()}")
    return missing


@pytest.mark.parametrize("name", EXPERIMENT_CONFIGS)
def test_every_key_has_a_comment(configs_dir: Path, name: str) -> None:
    path = configs_dir / name
    assert path.is_file(), f"missing config {name}"
    missing = undocumented_keys(path)
    assert not missing, "undocumented config keys:\n" + "\n".join(missing)


@pytest.mark.parametrize("name", EXPERIMENT_CONFIGS)
def test_configs_carry_provenance_tags_and_load(configs_dir: Path, name: str) -> None:
    path = configs_dir / name
    text = path.read_text(encoding="utf-8")
    assert "[Assumption — not in paper]" in text, f"{name} must label its assumptions"
    assert load_config(path), f"{name} loads to an empty mapping"
    smoke = load_config(path, smoke=True)
    assert "smoke_overrides" not in smoke and "base_config" not in smoke


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that rejects a mapping with a repeated key (PyYAML keeps the last one)."""

    def construct_mapping(self, node, deep=False):  # type: ignore[override]
        keys = [self.construct_object(key_node, deep=deep) for key_node, _ in node.value]
        duplicates = sorted({str(key) for key in keys if keys.count(key) > 1})
        if duplicates:
            raise yaml.constructor.ConstructorError(
                None, None, f"duplicate key(s) {duplicates}", node.start_mark
            )
        return super().construct_mapping(node, deep=deep)


@pytest.mark.parametrize("name", (*EXPERIMENT_CONFIGS, "before_after.yaml"))
def test_no_key_is_duplicated(configs_dir: Path, name: str) -> None:
    """A repeated key would silently override an earlier value; make it an error."""
    text = (configs_dir / name).read_text(encoding="utf-8")
    assert yaml.load(text, Loader=_UniqueKeyLoader)  # noqa: S506 - subclass of SafeLoader
