"""LEAF-format FEMNIST reader on a tiny synthetic fixture (FEMNIST itself is never downloaded)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.common.config import from_dict
from src.federated.config import EvaluationConfig, FemnistConfig
from src.federated.datasets.femnist import (
    IMAGE_PIXELS,
    NUM_CLASSES,
    load_femnist_dataset,
    read_leaf_directory,
    write_leaf_fixture,
)


@pytest.fixture(scope="module")
def leaf_dir(repo_root: Path) -> Path:
    return repo_root / "tests" / "fixtures" / "federated" / "leaf"


def test_reader_on_committed_fixture(leaf_dir: Path) -> None:
    writers = read_leaf_directory(leaf_dir)
    assert list(writers) == ["f0000_00", "f0001_00", "f0002_00"]
    x, y = writers["f0001_00"]
    assert x.shape == (8, IMAGE_PIXELS) and x.dtype == np.float32
    assert y.min() >= 0 and y.max() < NUM_CLASSES


def test_dataset_writers_are_clients(leaf_dir: Path) -> None:
    cfg = from_dict(
        FemnistConfig,
        {"data_dir": str(leaf_dir), "max_writers": 2, "min_samples_per_writer": 7, "test_fraction": 0.25},  # fmt: skip
    )
    evaluation = from_dict(
        EvaluationConfig, {"eval_on": "test", "validation_fraction": 0.2, "group_attributes": []}
    )
    dataset = load_femnist_dataset(cfg, evaluation, seed=100)
    assert dataset.domain_names == ("f0001_00", "f0002_00")  # f0000 has only 6 samples
    assert dataset.num_classes == 62 and dataset.positive_class is None
    assert len(dataset.train) + len(dataset.test) == 18
    assert set(np.unique(dataset.train.domain)) == {0, 1}


def test_generated_fixture_and_errors(tmp_path: Path) -> None:
    write_leaf_fixture(tmp_path / "a.json", writers=2, samples=3, seed=0)
    assert len(read_leaf_directory(tmp_path)) == 2
    with pytest.raises(FileNotFoundError, match="does not\\s+download FEMNIST"):
        read_leaf_directory(tmp_path / "empty")
