"""YAML configuration loading with strict validation.

Every hyperparameter in this repository lives in a YAML file under ``configs/``.
Code never carries a default value for one, so :func:`from_dict` rejects both
*missing* and *unknown* keys: a config file and the dataclass that consumes it
cannot silently drift apart.

Three small conventions are implemented here and used by every entry point:

``base_config``
    A config may name another YAML file (path relative to itself) to inherit
    from. The child is deep-merged over the parent.
``smoke_overrides``
    A mapping that is deep-merged over the config when an entry point is run
    with ``--smoke``. It lives in the YAML, not in code, so the tiny CI-scale
    settings are as visible and reviewable as the full ones.
``--set key.sub=value``
    Command-line overrides, parsed with YAML scalar rules.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import re
import types
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal, TypeVar, Union, get_args, get_origin, get_type_hints

import yaml

T = TypeVar("T")

BASE_KEY = "base_config"
SMOKE_KEY = "smoke_overrides"


class ConfigError(ValueError):
    """Raised when a configuration file is missing a key, has an unknown key or a bad type."""


class _ConfigLoader(yaml.SafeLoader):
    """SafeLoader that also reads ``1e-3`` as a float.

    PyYAML follows YAML 1.1, where a float in scientific notation needs a dot
    (``1.0e-3``); ``1e-3`` would otherwise load as a string. Hyperparameters such
    as learning rates and ``delta`` are routinely written without the dot.
    """


_ConfigLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\.[0-9_]+(?:[eE][-+]?[0-9]+)?
            |[-+]?\.(?:inf|Inf|INF)
            |\.(?:nan|NaN|NAN))$""",
        re.VERBOSE,
    ),
    list("-+0123456789."),
)


def parse_yaml_text(text: str) -> Any:
    """Parse a YAML snippet with the repository's loader (see :class:`_ConfigLoader`)."""
    return yaml.load(text, Loader=_ConfigLoader)  # noqa: S506 - subclass of SafeLoader


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read one YAML file and return its top-level mapping."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = parse_yaml_text(handle.read())
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Top level of {path} must be a mapping, got {type(data).__name__}")
    return data


def deep_update(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Return a new dict with ``overrides`` merged recursively over ``base``.

    Mappings are merged key by key; every other value (including lists) is replaced.
    """
    merged: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def parse_overrides(pairs: Iterable[str] | None) -> dict[str, Any]:
    """Turn ``["agent.learning_rate=0.001", "seeds=[0,1]"]`` into a nested dict.

    Values are parsed with YAML scalar rules, so ``true``, ``3``, ``1e-3`` and
    ``[1, 2]`` become bool, int, float and list respectively.
    """
    result: dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ConfigError(f"Override must look like key.sub=value, got: {pair!r}")
        dotted, raw = pair.split("=", 1)
        keys = [part for part in dotted.strip().split(".") if part]
        if not keys:
            raise ConfigError(f"Override has an empty key: {pair!r}")
        value = parse_yaml_text(raw)
        cursor = result
        for key in keys[:-1]:
            cursor = cursor.setdefault(key, {})
            if not isinstance(cursor, dict):
                raise ConfigError(f"Override {pair!r} conflicts with an earlier override")
        cursor[keys[-1]] = value
    return result


def load_config(
    path: str | Path,
    overrides: Iterable[str] | None = None,
    smoke: bool = False,
) -> dict[str, Any]:
    """Load a config file, resolving inheritance, smoke overrides and CLI overrides.

    Order of precedence (lowest to highest): ``base_config`` parent, the file
    itself, ``smoke_overrides`` (only when ``smoke`` is true), then ``overrides``.
    The ``base_config`` and ``smoke_overrides`` keys are removed from the result.
    """
    config = load_yaml_with_bases(path)
    smoke_block = config.pop(SMOKE_KEY, None) or {}
    if not isinstance(smoke_block, Mapping):
        raise ConfigError(f"{SMOKE_KEY} in {path} must be a mapping")
    if smoke:
        config = deep_update(config, smoke_block)
    if overrides:
        config = deep_update(config, parse_overrides(overrides))
    return config


def load_yaml_with_bases(path: str | Path) -> dict[str, Any]:
    """Load a YAML file merged over its ``base_config`` chain.

    ``smoke_overrides`` blocks are merged like any other mapping, so a child
    config extends (rather than replaces) the smoke settings of its parent.
    """
    path = Path(path)
    config = load_yaml(path)
    base_ref = config.pop(BASE_KEY, None)
    if base_ref is None:
        return config
    parent = load_yaml_with_bases((path.parent / str(base_ref)).resolve())
    return deep_update(parent, config)


def get_path(config: Mapping[str, Any], dotted: str) -> Any:
    """Fetch ``config["a"]["b"]`` for ``dotted == "a.b"``; raise ConfigError if absent."""
    cursor: Any = config
    walked: list[str] = []
    for key in dotted.split("."):
        walked.append(key)
        if not isinstance(cursor, Mapping) or key not in cursor:
            raise ConfigError(f"Missing required config key: {'.'.join(walked)}")
        cursor = cursor[key]
    return cursor


def from_dict(cls: type[T], data: Any, path: str = "") -> T:
    """Build dataclass ``cls`` from a mapping, strictly.

    * every dataclass field must be present in ``data`` (dataclass defaults are ignored,
      because hyperparameters belong in YAML);
    * every key in ``data`` must be a dataclass field;
    * values are checked and converted against the field's type annotation
      (nested dataclasses, ``list``, ``tuple``, ``dict``, ``Optional``, ``Literal``,
      ``int``, ``float``, ``bool``, ``str``).
    """
    where = path or cls.__name__
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    if not isinstance(data, Mapping):
        raise ConfigError(f"{where}: expected a mapping, got {type(data).__name__}")
    hints = get_type_hints(cls)
    fields = {field.name for field in dataclasses.fields(cls) if field.init}
    missing = sorted(fields - set(data))
    unknown = sorted(set(data) - fields)
    if missing:
        raise ConfigError(f"{where}: missing required key(s): {', '.join(missing)}")
    if unknown:
        raise ConfigError(f"{where}: unknown key(s): {', '.join(unknown)}")
    kwargs = {name: _convert(data[name], hints[name], f"{where}.{name}") for name in sorted(fields)}
    return cls(**kwargs)


def _convert(value: Any, annotation: Any, where: str) -> Any:
    """Check ``value`` against ``annotation`` and convert containers and nested dataclasses."""
    if annotation is Any:
        return value
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin in (Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        errors: list[str] = []
        for option in args:
            if option is type(None):
                continue
            try:
                return _convert(value, option, where)
            except ConfigError as exc:
                errors.append(str(exc))
        raise ConfigError(f"{where}: value {value!r} matches none of {annotation}: {errors}")

    if origin is Literal:
        if value not in args:
            raise ConfigError(f"{where}: {value!r} is not one of {list(args)}")
        return value

    if origin in (list, tuple):
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{where}: expected a list, got {type(value).__name__}")
        if origin is tuple and args and args[-1] is not Ellipsis:
            if len(args) != len(value):
                raise ConfigError(f"{where}: expected {len(args)} items, got {len(value)}")
            return tuple(
                _convert(item, arg, f"{where}[{i}]")
                for i, (item, arg) in enumerate(zip(value, args, strict=True))
            )
        item_type = args[0] if args else Any
        items = [_convert(item, item_type, f"{where}[{i}]") for i, item in enumerate(value)]
        return tuple(items) if origin is tuple else items

    if origin is dict:
        if not isinstance(value, Mapping):
            raise ConfigError(f"{where}: expected a mapping, got {type(value).__name__}")
        value_type = args[1] if len(args) == 2 else Any
        return {str(k): _convert(v, value_type, f"{where}.{k}") for k, v in value.items()}

    if dataclasses.is_dataclass(annotation) and isinstance(annotation, type):
        return from_dict(annotation, value, where)

    if annotation is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{where}: expected true/false, got {value!r}")
        return value
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where}: expected an integer, got {value!r}")
        return value
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where}: expected a number, got {value!r}")
        return float(value)
    if annotation is str:
        if not isinstance(value, str):
            raise ConfigError(f"{where}: expected a string, got {value!r}")
        return value
    if isinstance(annotation, type) and isinstance(value, annotation):
        return value
    raise ConfigError(f"{where}: cannot interpret {value!r} as {annotation}")


def to_dict(obj: Any) -> Any:
    """Convert a (nested) dataclass back into plain containers for saving."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Mapping):
        return {str(k): to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    return obj


def save_yaml(config: Any, path: str | Path) -> Path:
    """Write a config (dict or dataclass) to ``path`` as YAML, keeping key order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(to_dict(config), handle, sort_keys=False, allow_unicode=True)
    return path


def config_hash(config: Any) -> str:
    """Short, stable fingerprint of a config (first 12 hex digits of SHA-256)."""
    canonical = json.dumps(to_dict(config), sort_keys=True, default=str, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
