"""Datasets of the federated track behind one small interface.

* :mod:`.adult` — UCI Adult / Census Income (1994 U.S. Census extraction), the primary
  benchmark (``[Assumption — not in paper]``).
* :mod:`.synthetic_domains` — three toy domains for unit tests, smoke runs and ablations.
* :mod:`.femnist` — optional LEAF-format FEMNIST reader (never the headline).

Every loader returns a :class:`FederatedDataset`: preprocessed training and evaluation
matrices, the natural client domain of every row, and evaluation-only group attributes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from src.common.config import to_dict
from src.common.seeding import rng_from_seed

from ..config import DatasetConfig, EvaluationConfig

REPO_ROOT = Path(__file__).resolve().parents[3]
EXCLUDED_DOMAIN = -1


def resolve_path(path: str | Path) -> Path:
    """Resolve a config path: absolute paths stay, relative ones are relative to the repo root."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


@dataclass(frozen=True)
class TabularSplit:
    """Rows of one split: features, labels, natural domain and evaluation-only attributes.

    ``domain[i]`` is the index of row ``i``'s natural client domain, or ``-1`` when the row
    belongs to no client (e.g. Adult rows with a missing workclass). ``groups`` holds the
    evaluation-only attributes (for Adult: sex, race, age band); they are never model inputs.
    """

    x: np.ndarray
    y: np.ndarray
    domain: np.ndarray
    groups: dict[str, np.ndarray] = field(default_factory=dict)
    subdomain: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def subset(self, index: np.ndarray) -> TabularSplit:
        """Rows selected by an integer index array."""
        index = np.asarray(index, dtype=np.int64)
        return TabularSplit(
            x=self.x[index],
            y=self.y[index],
            domain=self.domain[index],
            groups={key: value[index] for key, value in self.groups.items()},
            subdomain=None if self.subdomain is None else self.subdomain[index],
        )


@dataclass(frozen=True)
class FederatedDataset:
    """A preprocessed dataset with natural client domains.

    ``test`` is the shared evaluation set: the real test rows when ``evaluation.eval_on`` is
    ``test``, or a validation split carved from the training rows when it is ``validation``
    (used for tuning, so that the test set never drives a choice).
    """

    name: str
    train: TabularSplit
    test: TabularSplit
    domain_names: tuple[str, ...]
    natural_split_name: str
    feature_names: tuple[str, ...]
    num_classes: int
    positive_class: int | None
    eval_on: str
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def num_features(self) -> int:
        """Width of the feature matrix."""
        return int(self.train.x.shape[1])


def validation_indices(
    domain: np.ndarray, fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Indices of (remaining training rows, validation rows), stratified by natural domain.

    Each domain (including the excluded group ``-1``) loses the same fraction of its rows, so
    the validation split is "carved out of each client's training rows". Seeded with
    ``derive_seed(seed, "validation")``.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError("evaluation.validation_fraction must be in (0, 1)")
    rng = rng_from_seed(seed, "validation")
    keep: list[np.ndarray] = []
    held: list[np.ndarray] = []
    for code in np.unique(domain):
        rows = np.flatnonzero(domain == code)
        rows = rows[rng.permutation(len(rows))]
        count = int(round(fraction * len(rows)))
        held.append(rows[:count])
        keep.append(rows[count:])
    return np.sort(np.concatenate(keep)), np.sort(np.concatenate(held))


def carve_validation(
    train: TabularSplit, fraction: float, seed: int
) -> tuple[TabularSplit, TabularSplit]:
    """Split training rows into (remaining training rows, validation rows); see
    :func:`validation_indices`."""
    keep, held = validation_indices(train.domain, fraction, seed)
    return train.subset(keep), train.subset(held)


def load_dataset(
    dataset: DatasetConfig, evaluation: EvaluationConfig, seed: int
) -> FederatedDataset:
    """Load the configured dataset (cached per process; callers must not mutate arrays)."""
    key = json.dumps(
        {
            "dataset": to_dict(dataset),
            "evaluation": to_dict(evaluation),
            "seed": seed if (dataset.kind == "synthetic" or evaluation.eval_on != "test") else 0,
        },
        sort_keys=True,
    )
    return _load_dataset_cached(key)


@lru_cache(maxsize=8)
def _load_dataset_cached(key: str) -> FederatedDataset:
    from src.common.config import from_dict

    payload = json.loads(key)
    dataset = from_dict(DatasetConfig, payload["dataset"])
    evaluation = from_dict(EvaluationConfig, payload["evaluation"])
    seed = int(payload["seed"])
    if dataset.kind == "adult":
        from .adult import load_adult_dataset

        assert dataset.adult is not None
        return load_adult_dataset(dataset.adult, evaluation, seed)
    if dataset.kind == "synthetic":
        from .synthetic_domains import load_synthetic_dataset

        assert dataset.synthetic is not None
        return load_synthetic_dataset(dataset.synthetic, evaluation, seed)
    if dataset.kind == "femnist":
        from .femnist import load_femnist_dataset

        assert dataset.femnist is not None
        return load_femnist_dataset(dataset.femnist, evaluation, seed)
    raise ValueError(f"Unknown dataset kind {dataset.kind!r}")


__all__ = [
    "EXCLUDED_DOMAIN",
    "REPO_ROOT",
    "FederatedDataset",
    "TabularSplit",
    "carve_validation",
    "load_dataset",
    "resolve_path",
    "validation_indices",
]
