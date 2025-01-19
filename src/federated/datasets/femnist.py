"""OPTIONAL secondary benchmark: reader for LEAF-format FEMNIST JSON shards.

FEMNIST (LEAF; Caldas et al., 2018) is handwritten characters grouped by writer; one writer
is one natural client, there are 62 classes (digits, upper- and lower-case letters), and each
image is 28×28 grey levels, flattened here to 784 features for the same MLP code.

This repository does **not** download FEMNIST. Prepare it yourself with the LEAF scripts
(https://github.com/TalwalkarLab/leaf, ``data/femnist``) and point
``configs/optional/federated_femnist.yaml`` at the folder that holds the JSON shards. Each
shard is ``{"users": [...], "num_samples": [...], "user_data": {user: {"x": [...], "y":
[...]}}}``. FEMNIST is never the headline benchmark and all of its results stay
``[Target / TBD]`` until somebody runs them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.common.seeding import rng_from_seed

if TYPE_CHECKING:
    from src.federated.config import EvaluationConfig, FemnistConfig

    from . import FederatedDataset

NUM_CLASSES = 62
IMAGE_PIXELS = 28 * 28


def read_leaf_directory(directory: str | Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Read every ``*.json`` shard of a LEAF directory into ``{writer: (x, y)}``.

    Writers are returned in sorted order; ``x`` is ``float32 [n, 784]`` and ``y`` ``int64 [n]``.
    """
    directory = Path(directory)
    shards = sorted(directory.glob("*.json"))
    if not shards:
        raise FileNotFoundError(
            f"No LEAF-format FEMNIST JSON shards in {directory}. This repository does not "
            "download FEMNIST; prepare it with the LEAF scripts (see src/federated/datasets/"
            "femnist.py)."
        )
    writers: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for shard in shards:
        payload = json.loads(shard.read_text(encoding="utf-8"))
        for key in ("users", "num_samples", "user_data"):
            if key not in payload:
                raise ValueError(f"{shard}: not a LEAF shard (missing {key!r})")
        for user, count in zip(payload["users"], payload["num_samples"], strict=True):
            data = payload["user_data"][user]
            x = np.asarray(data["x"], dtype=np.float32).reshape(len(data["y"]), -1)
            y = np.asarray(data["y"], dtype=np.int64)
            if len(y) != int(count):
                raise ValueError(f"{shard}: writer {user} has {len(y)} samples, header {count}")
            if x.shape[1] != IMAGE_PIXELS:
                raise ValueError(f"{shard}: writer {user} images have {x.shape[1]} values")
            if y.size and (y.min() < 0 or y.max() >= NUM_CLASSES):
                raise ValueError(f"{shard}: writer {user} has labels outside 0..61")
            if user in writers:
                raise ValueError(f"writer {user} appears in more than one shard")
            writers[user] = (x, y)
    return dict(sorted(writers.items()))


def load_femnist_dataset(
    cfg: FemnistConfig, evaluation: EvaluationConfig, seed: int
) -> FederatedDataset:
    """Writers as natural clients; a per-writer fraction of samples forms the shared test set."""
    from . import FederatedDataset, TabularSplit, carve_validation, resolve_path

    writers = read_leaf_directory(resolve_path(cfg.data_dir))
    eligible = [w for w, (_, y) in writers.items() if len(y) >= cfg.min_samples_per_writer]
    if cfg.max_writers > 0:
        eligible = eligible[: cfg.max_writers]
    if not eligible:
        raise ValueError("No FEMNIST writer has enough samples")
    rng = rng_from_seed(seed, "femnist_test")
    parts: dict[str, list[np.ndarray]] = {key: [] for key in ("tx", "ty", "td", "ex", "ey", "ed")}
    for code, writer in enumerate(eligible):
        x, y = writers[writer]
        order = rng.permutation(len(y))
        n_test = int(round(cfg.test_fraction * len(y)))
        test_idx, train_idx = order[:n_test], order[n_test:]
        parts["tx"].append(x[train_idx])
        parts["ty"].append(y[train_idx])
        parts["td"].append(np.full(len(train_idx), code, dtype=np.int64))
        parts["ex"].append(x[test_idx])
        parts["ey"].append(y[test_idx])
        parts["ed"].append(np.full(len(test_idx), code, dtype=np.int64))
    train = TabularSplit(
        x=np.concatenate(parts["tx"]),
        y=np.concatenate(parts["ty"]),
        domain=np.concatenate(parts["td"]),
    )
    test = TabularSplit(
        x=np.concatenate(parts["ex"]),
        y=np.concatenate(parts["ey"]),
        domain=np.concatenate(parts["ed"]),
    )
    if evaluation.eval_on == "validation":
        train, test = carve_validation(train, evaluation.validation_fraction, seed)
    return FederatedDataset(
        name="femnist",
        train=train,
        test=test,
        domain_names=tuple(eligible),
        natural_split_name="writer",
        feature_names=tuple(f"pixel{i}" for i in range(IMAGE_PIXELS)),
        num_classes=NUM_CLASSES,
        positive_class=None,
        eval_on=evaluation.eval_on,
        info={"writers": len(eligible)},
    )


def write_leaf_fixture(path: str | Path, writers: int, samples: int, seed: int) -> Path:
    """Write a tiny SYNTHETIC LEAF-format shard (random pixels) for tests."""
    rng = np.random.default_rng(seed)
    users = [f"f{index:04d}_00" for index in range(writers)]
    payload = {"users": users, "num_samples": [], "user_data": {}}
    for user in users:
        x = rng.random((samples, IMAGE_PIXELS)).round(3).tolist()
        y = rng.integers(0, NUM_CLASSES, size=samples).tolist()
        payload["num_samples"].append(samples)
        payload["user_data"][user] = {"x": x, "y": y}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
