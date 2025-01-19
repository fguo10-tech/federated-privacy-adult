"""Seeded generator of three toy domains sharing one binary classification task.

The three synthetic toy domains carry healthcare-like, finance-like and user-behaviour-like
names in the configs (``healthcare-like``, ``finance-like``, ``user-behaviour-like``). These
names are **labels on synthetic numbers**, never real data about patients, accounts or users.
The domains exist only for unit tests, smoke runs and ``configs/federated_fedavg.yaml``; they
are never the headline benchmark (that is UCI Adult).

Generative model (all parameters in YAML, ``[Assumption — not in paper]``):

* one shared weight vector ``w ~ N(0, I / d)``;
* domain ``k`` draws features ``x ~ N(feature_shift · u_k, I)`` with a random unit vector
  ``u_k`` (covariate / feature shift);
* score ``s = class_separation · x·w``; the domain intercept ``b_k`` is set so that a fraction
  ``positive_rates[k]`` of the domain's rows has ``s + b_k > 0`` (label-prior shift);
* a fraction ``label_noise`` of labels is flipped.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.common.seeding import derive_seed, rng_from_seed

if TYPE_CHECKING:
    from src.federated.config import EvaluationConfig, SyntheticDomainsConfig

    from . import FederatedDataset

TOY_SEED_LABEL = "synthetic_domains"


def generate_domains(cfg: SyntheticDomainsConfig, seed: int) -> dict[str, np.ndarray]:
    """Generate train and test rows for every domain.

    Returns arrays ``train_x, train_y, train_domain, test_x, test_y, test_domain``.
    """
    names = list(cfg.domain_names)
    if not (len(names) == len(cfg.train_rows) == len(cfg.test_rows) == len(cfg.positive_rates)):
        raise ValueError("synthetic: domain_names, train_rows, test_rows, positive_rates differ")
    rng = rng_from_seed(seed, TOY_SEED_LABEL)
    d = int(cfg.num_features)
    weights = rng.normal(0.0, 1.0 / np.sqrt(d), size=d)
    parts: dict[str, list[np.ndarray]] = {key: [] for key in ("tx", "ty", "td", "ex", "ey", "ed")}
    for k, _name in enumerate(names):
        direction = rng.normal(size=d)
        direction /= np.linalg.norm(direction)
        n_train, n_test = int(cfg.train_rows[k]), int(cfg.test_rows[k])
        x = rng.normal(size=(n_train + n_test, d)) + cfg.feature_shift * direction
        score = cfg.class_separation * (x @ weights)
        intercept = -np.quantile(score, 1.0 - cfg.positive_rates[k])
        y = (score + intercept > 0).astype(np.int64)
        flip = rng.random(len(y)) < cfg.label_noise
        y[flip] = 1 - y[flip]
        parts["tx"].append(x[:n_train])
        parts["ty"].append(y[:n_train])
        parts["td"].append(np.full(n_train, k, dtype=np.int64))
        parts["ex"].append(x[n_train:])
        parts["ey"].append(y[n_train:])
        parts["ed"].append(np.full(n_test, k, dtype=np.int64))
    return {
        "train_x": np.concatenate(parts["tx"]).astype(np.float32),
        "train_y": np.concatenate(parts["ty"]),
        "train_domain": np.concatenate(parts["td"]),
        "test_x": np.concatenate(parts["ex"]).astype(np.float32),
        "test_y": np.concatenate(parts["ey"]),
        "test_domain": np.concatenate(parts["ed"]),
    }


def load_synthetic_dataset(
    cfg: SyntheticDomainsConfig, evaluation: EvaluationConfig, seed: int
) -> FederatedDataset:
    """Toy :class:`~src.federated.datasets.FederatedDataset` (data drawn from the run seed)."""
    from . import FederatedDataset, TabularSplit, carve_validation

    arrays = generate_domains(cfg, derive_seed(seed, "toy_data"))
    train = TabularSplit(x=arrays["train_x"], y=arrays["train_y"], domain=arrays["train_domain"])
    test = TabularSplit(x=arrays["test_x"], y=arrays["test_y"], domain=arrays["test_domain"])
    if evaluation.eval_on == "validation":
        train, test = carve_validation(train, evaluation.validation_fraction, seed)
    return FederatedDataset(
        name="synthetic_domains",
        train=train,
        test=test,
        domain_names=tuple(cfg.domain_names),
        natural_split_name="domain",
        feature_names=tuple(f"x{i}" for i in range(int(cfg.num_features))),
        num_classes=2,
        positive_class=1,
        eval_on=evaluation.eval_on,
        info={"generator_seed": derive_seed(seed, "toy_data")},
    )


def save_domains(
    arrays: dict[str, np.ndarray], domain_names: Sequence[str], out_dir: str | Path
) -> list[Path]:
    """Write one CSV per split (features, label, domain name) into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for split in ("train", "test"):
        x = arrays[f"{split}_x"]
        header = ",".join([f"x{i}" for i in range(x.shape[1])] + ["label", "domain"])
        lines = [header]
        for row, label, code in zip(
            x, arrays[f"{split}_y"], arrays[f"{split}_domain"], strict=True
        ):
            values = ",".join(f"{value:.6f}" for value in row)
            lines.append(f"{values},{int(label)},{domain_names[int(code)]}")
        path = out_dir / f"federated_toy_{split}.csv"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written.append(path)
    return written
