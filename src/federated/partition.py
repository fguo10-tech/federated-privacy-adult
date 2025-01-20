"""Client partitions of the training rows and their statistics.

All clients here are **simulated domains derived from one dataset** (for Adult: one public
census dataset), not real employers, hospitals or banks.

Partitions (``[Assumption — not in paper]``):

``natural`` (named after the dataset: ``workclass`` for Adult, ``domain`` for the toy data,
``writer`` for FEMNIST)
    One client per natural domain. For Adult, from the **training rows only**: ``Private``
    (workclass Private), ``Government`` (Federal-gov, State-gov, Local-gov) and
    ``Self-employed`` (Self-emp-inc, Self-emp-not-inc). Rows with workclass ``?``,
    ``Without-pay`` or ``Never-worked`` are excluded from the client split.
``iid`` (the IID control)
    The same client rows (the union of the natural clients), shuffled with the run seed and
    dealt into clients of equal size (sizes differ by at most one row).
``iid_size_matched``
    Shuffled like ``iid`` but with the natural split's client sizes.
``label_skew``
    For each label, client shares are drawn from ``Dirichlet(α)`` (``partitions.label_skew_alpha``).
``quantity_skew``
    Shuffled rows dealt into clients whose sizes follow ``Dirichlet(α)`` or a power law.

Every partition is a pure function of (dataset, partition settings, seed).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from src.common.seeding import rng_from_seed

from .config import PartitionConfig
from .datasets import EXCLUDED_DOMAIN, FederatedDataset

PARTITION_STATS_FIELDS = (
    "split",
    "seed",
    "client",
    "rows",
    "positives",
    "positive_rate",
    "share_of_client_rows",
    "composition",
)
"""Columns of ``partition_stats.csv``. ``composition`` lists the sub-domain counts (for the
Adult workclass split: the workclass values inside each client); the row with
``client == "(excluded)"`` counts training rows that belong to no client."""

ABLATION_PARTITIONS = ("iid_size_matched", "label_skew", "quantity_skew")


@dataclass(frozen=True)
class Partition:
    """Training-row indices of every client of one split."""

    name: str
    client_names: tuple[str, ...]
    indices: tuple[np.ndarray, ...]

    @property
    def num_clients(self) -> int:
        """Number of clients."""
        return len(self.indices)

    def sizes(self) -> list[int]:
        """Rows per client."""
        return [int(len(index)) for index in self.indices]


def client_row_pool(domain: np.ndarray) -> np.ndarray:
    """Indices of rows that belong to some natural client (domain >= 0), sorted."""
    return np.flatnonzero(np.asarray(domain) != EXCLUDED_DOMAIN)


def natural_partition(domain: np.ndarray, num_domains: int) -> list[np.ndarray]:
    """One client per natural domain (rows with domain ``-1`` are excluded)."""
    domain = np.asarray(domain)
    return [np.flatnonzero(domain == code) for code in range(num_domains)]


def _deal(rows: np.ndarray, sizes: list[int]) -> list[np.ndarray]:
    bounds = np.cumsum([0, *sizes])
    return [np.sort(rows[bounds[i] : bounds[i + 1]]) for i in range(len(sizes))]


def iid_partition(pool: np.ndarray, num_clients: int, seed: int) -> list[np.ndarray]:
    """Shuffle ``pool`` with the run seed and deal it into equal clients (±1 row)."""
    rng = rng_from_seed(seed, "partition", "iid")
    shuffled = pool[rng.permutation(len(pool))]
    return [np.sort(part) for part in np.array_split(shuffled, num_clients)]


def iid_size_matched_partition(pool: np.ndarray, sizes: list[int], seed: int) -> list[np.ndarray]:
    """Shuffle ``pool`` and deal it into clients with the given ``sizes``."""
    if sum(sizes) != len(pool):
        raise ValueError("client sizes must add up to the number of client rows")
    rng = rng_from_seed(seed, "partition", "iid_size_matched")
    return _deal(pool[rng.permutation(len(pool))], list(sizes))


def label_skew_partition(
    pool: np.ndarray,
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
    min_rows: int,
    max_attempts: int,
) -> list[np.ndarray]:
    """Per label, split its rows over clients with ``Dirichlet(alpha)`` proportions.

    Draws are repeated (same generator) until every client holds at least ``min_rows`` rows,
    at most ``max_attempts`` times.
    """
    rng = rng_from_seed(seed, "partition", "label_skew")
    pool_labels = np.asarray(labels)[pool]
    for _attempt in range(max_attempts):
        parts: list[list[np.ndarray]] = [[] for _ in range(num_clients)]
        for label in np.unique(pool_labels):
            rows = pool[pool_labels == label]
            rows = rows[rng.permutation(len(rows))]
            shares = rng.dirichlet(np.full(num_clients, alpha))
            cuts = (np.cumsum(shares)[:-1] * len(rows)).astype(np.int64)
            for client, chunk in enumerate(np.split(rows, cuts)):
                parts[client].append(chunk)
        clients = [np.sort(np.concatenate(chunks)) for chunks in parts]
        if min(len(client) for client in clients) >= min_rows:
            return clients
    raise RuntimeError(
        f"label_skew: no draw gave every client >= {min_rows} rows in {max_attempts} attempts"
    )


def quantity_skew_partition(
    pool: np.ndarray,
    num_clients: int,
    kind: str,
    alpha: float,
    exponent: float,
    seed: int,
    min_rows: int,
    max_attempts: int,
) -> list[np.ndarray]:
    """Shuffled rows dealt into clients with skewed sizes.

    ``kind == "dirichlet"``: sizes ∝ a ``Dirichlet(alpha)`` draw (redrawn until every client
    has ``min_rows``). ``kind == "power_law"``: sizes ∝ ``(k + 1) ** -exponent``.
    """
    rng = rng_from_seed(seed, "partition", "quantity_skew")
    shuffled = pool[rng.permutation(len(pool))]
    total = len(pool)
    for _attempt in range(max_attempts):
        if kind == "dirichlet":
            shares = rng.dirichlet(np.full(num_clients, alpha))
        elif kind == "power_law":
            raw = (np.arange(num_clients) + 1.0) ** (-exponent)
            shares = raw / raw.sum()
        else:
            raise ValueError(f"Unknown quantity_skew kind {kind!r}")
        sizes = np.floor(shares * total).astype(np.int64)
        sizes[np.argmax(sizes)] += total - int(sizes.sum())
        if sizes.min() >= min_rows:
            return _deal(shuffled, sizes.tolist())
    raise RuntimeError(
        f"quantity_skew: no draw gave every client >= {min_rows} rows in {max_attempts} attempts"
    )


def make_partition(
    name: str,
    dataset: FederatedDataset,
    cfg: PartitionConfig,
    seed: int,
    num_clients: int | None = None,
) -> Partition:
    """Build the partition called ``name`` over ``dataset.train``.

    ``num_clients`` defaults to the number of natural domains (it is only overridden by the
    robustness suite, which uses ``robustness.num_clients`` IID clients).
    """
    domain = dataset.train.domain
    domains = len(dataset.domain_names)
    clients = domains if num_clients is None else int(num_clients)
    pool = client_row_pool(domain)
    if name in (dataset.natural_split_name, "natural"):
        return Partition(
            dataset.natural_split_name,
            tuple(dataset.domain_names),
            tuple(natural_partition(domain, domains)),
        )
    if name == "iid":
        indices = iid_partition(pool, clients, seed)
    elif name == "iid_size_matched":
        sizes = [len(rows) for rows in natural_partition(domain, domains)]
        indices = iid_size_matched_partition(pool, sizes, seed)
        clients = len(sizes)
    elif name == "label_skew":
        indices = label_skew_partition(
            pool,
            dataset.train.y,
            clients,
            cfg.label_skew_alpha,
            seed,
            cfg.min_client_rows,
            cfg.max_resample_attempts,
        )
    elif name == "quantity_skew":
        indices = quantity_skew_partition(
            pool,
            clients,
            cfg.quantity_skew.kind,
            cfg.quantity_skew.alpha,
            cfg.quantity_skew.exponent,
            seed,
            cfg.min_client_rows,
            cfg.max_resample_attempts,
        )
    else:
        raise ValueError(f"Unknown partition {name!r}")
    names = tuple(f"{name}_{k}" for k in range(clients))
    return Partition(name, names, tuple(indices))


def partition_statistics(
    partition: Partition, dataset: FederatedDataset, seed: int
) -> list[dict[str, object]]:
    """Rows, positives, positive rate and sub-domain composition per client.

    A final row ``client == "(excluded)"`` reports the training rows that belong to no client,
    with the reason (sub-domain value, e.g. the Adult workclass) in ``composition``.
    """
    train = dataset.train
    positive = dataset.positive_class
    total_client_rows = sum(partition.sizes())
    rows: list[dict[str, object]] = []

    def describe(index: np.ndarray, client: str) -> dict[str, object]:
        labels = train.y[index]
        positives = int((labels == positive).sum()) if positive is not None else ""
        composition = ""
        if train.subdomain is not None and len(index):
            counts = Counter(str(value) for value in train.subdomain[index])
            composition = ";".join(f"{key}:{counts[key]}" for key in sorted(counts))
        return {
            "split": partition.name,
            "seed": seed,
            "client": client,
            "rows": int(len(index)),
            "positives": positives,
            "positive_rate": (
                float(positives / len(index)) if positive is not None and len(index) else ""
            ),
            "share_of_client_rows": (
                float(len(index) / total_client_rows) if client != "(excluded)" else ""
            ),
            "composition": composition,
        }

    for name, index in zip(partition.client_names, partition.indices, strict=True):
        rows.append(describe(index, name))
    rows.append(describe(np.flatnonzero(train.domain == EXCLUDED_DOMAIN), "(excluded)"))
    return rows
