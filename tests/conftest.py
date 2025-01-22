"""Shared pytest fixtures.

Tests never need the network. Tests that need the real UCI Adult files are
marked ``adult`` and skip themselves when ``data/adult`` is absent.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path of the repository checkout."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def configs_dir(repo_root: Path) -> Path:
    """Absolute path of the ``configs/`` folder."""
    return repo_root / "configs"
