"""Configuration loading must be strict: no silent defaults, no silent typos."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest

from src.common.config import (
    ConfigError,
    config_hash,
    deep_update,
    from_dict,
    get_path,
    load_config,
    parse_overrides,
    save_yaml,
    to_dict,
)


@dataclass(frozen=True)
class Inner:
    rate: float
    steps: int


@dataclass(frozen=True)
class Outer:
    name: str
    enabled: bool
    inner: Inner
    sizes: list[int]
    mode: Literal["a", "b"]
    limit: float | None
    pair: tuple[float, float]


GOOD = {
    "name": "x",
    "enabled": True,
    "inner": {"rate": 1, "steps": 3},
    "sizes": [1, 2],
    "mode": "a",
    "limit": None,
    "pair": [0.5, 2],
}


def test_from_dict_builds_nested_dataclass_and_converts_numbers() -> None:
    built = from_dict(Outer, GOOD)
    assert built.inner == Inner(rate=1.0, steps=3)
    assert isinstance(built.inner.rate, float)
    assert built.pair == (0.5, 2.0)
    assert built.limit is None
    assert to_dict(built)["inner"] == {"rate": 1.0, "steps": 3}


@pytest.mark.parametrize("missing", ["name", "inner", "limit"])
def test_missing_key_is_an_error_even_for_optional_values(missing: str) -> None:
    data = {key: value for key, value in GOOD.items() if key != missing}
    with pytest.raises(ConfigError, match="missing required key"):
        from_dict(Outer, data)


def test_unknown_key_is_an_error() -> None:
    with pytest.raises(ConfigError, match="unknown key"):
        from_dict(Outer, {**GOOD, "learning_rte": 0.1})
    with pytest.raises(ConfigError, match="unknown key"):
        from_dict(Outer, {**GOOD, "inner": {"rate": 1.0, "steps": 3, "extra": 1}})


@pytest.mark.parametrize(
    "patch",
    [
        {"enabled": 1},
        {"inner": {"rate": "fast", "steps": 3}},
        {"inner": {"rate": 1.0, "steps": 2.5}},
        {"inner": {"rate": 1.0, "steps": True}},
        {"sizes": "1,2"},
        {"mode": "c"},
        {"pair": [1.0]},
        {"name": 3},
    ],
)
def test_wrong_types_are_rejected(patch: dict) -> None:
    with pytest.raises(ConfigError):
        from_dict(Outer, {**GOOD, **patch})


def test_deep_update_merges_mappings_and_replaces_lists() -> None:
    base = {"a": {"b": 1, "c": [1, 2]}, "d": 4}
    merged = deep_update(base, {"a": {"c": [9]}, "e": 5})
    assert merged == {"a": {"b": 1, "c": [9]}, "d": 4, "e": 5}
    assert base == {"a": {"b": 1, "c": [1, 2]}, "d": 4}, "input must not be mutated"


def test_parse_overrides_uses_yaml_scalars() -> None:
    parsed = parse_overrides(["training.lr=1e-3", "seeds=[0, 1]", "flag=true", "name=fedavg"])
    assert parsed == {"training": {"lr": 0.001}, "seeds": [0, 1], "flag": True, "name": "fedavg"}
    with pytest.raises(ConfigError):
        parse_overrides(["no_equals_sign"])


def test_load_config_inheritance_smoke_and_cli_precedence(tmp_path: Path) -> None:
    save_yaml(
        {"rounds": 20, "model": {"hidden": [64, 32]}, "smoke_overrides": {"rounds": 2}},
        tmp_path / "base.yaml",
    )
    save_yaml(
        {
            "base_config": "base.yaml",
            "model": {"dropout": 0.0},
            "smoke_overrides": {"model": {"hidden": [8]}},
        },
        tmp_path / "child.yaml",
    )
    full = load_config(tmp_path / "child.yaml")
    assert full == {"rounds": 20, "model": {"hidden": [64, 32], "dropout": 0.0}}
    smoke = load_config(tmp_path / "child.yaml", smoke=True)
    assert smoke == {"rounds": 2, "model": {"hidden": [8], "dropout": 0.0}}
    overridden = load_config(tmp_path / "child.yaml", overrides=["rounds=5"], smoke=True)
    assert overridden["rounds"] == 5


def test_get_path_and_hash(tmp_path: Path) -> None:
    config = {"a": {"b": 3}}
    assert get_path(config, "a.b") == 3
    with pytest.raises(ConfigError, match="a.c"):
        get_path(config, "a.c")
    assert config_hash({"x": 1, "y": 2}) == config_hash({"y": 2, "x": 1})
    assert config_hash({"x": 1}) != config_hash({"x": 2})
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "absent.yaml")
