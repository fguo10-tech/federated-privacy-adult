"""Write the UCI Adult partition statistics used by the dataset card.

Reads the downloaded files (``python data/download_adult.py`` first), builds the workclass
split and the IID control for seed 0 (the IID partition depends on the seed; the workclass
split does not), and writes ``adult_partition_stats.csv`` and ``adult_partition_stats.md``
with rows, positives and positive rate per client, the workclass composition, the rows
excluded from the client split (by workclass value) and the attribute roles. Only the
training rows are partitioned; nothing here is tuned or evaluated.

Usage::

    python -m experiments.describe_adult [--config configs/federated_adult.yaml] [--out DIR]

The default ``--out`` is ``experiments/results/runs/adult_description`` (git-ignored).
"""

from __future__ import annotations

import argparse
import sys

from src.common.io import write_csv
from src.common.logging_utils import get_logger, setup_logging
from src.federated.config import load_federated_config
from src.federated.datasets import load_dataset, resolve_path
from src.federated.datasets.adult import ATTRIBUTE_ROLES, CITATION, EXTRACTION_YEAR
from src.federated.partition import PARTITION_STATS_FIELDS, make_partition, partition_statistics

LOGGER = get_logger("experiments.describe_adult")
DESCRIBE_SEED = 0


def markdown_table(rows: list[dict], fields: tuple[str, ...]) -> str:
    """Render dict rows as a GitHub Markdown table."""
    lines = ["| " + " | ".join(fields) + " |", "|" + "---|" * len(fields)]
    for row in rows:
        cells = []
        for field in fields:
            value = row.get(field, "")
            cells.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/federated_adult.yaml")
    parser.add_argument("--out", default="experiments/results/runs/adult_description")
    args = parser.parse_args(argv)
    setup_logging()
    config, _ = load_federated_config(args.config, overrides=["evaluation.eval_on=test"])
    if config.dataset.kind != "adult":
        LOGGER.error("describe_adult needs an Adult config")
        return 2
    try:
        dataset = load_dataset(config.dataset, config.evaluation, DESCRIBE_SEED)
    except FileNotFoundError as exc:
        LOGGER.error("%s", exc)
        return 2
    rows = []
    for split in ("workclass", "iid"):
        partition = make_partition(split, dataset, config.partitions, DESCRIBE_SEED)
        rows.extend(partition_statistics(partition, dataset, DESCRIBE_SEED))
    out = resolve_path(args.out)
    write_csv(out / "adult_partition_stats.csv", rows, PARTITION_STATS_FIELDS)
    raw = dataset.info["raw"]
    roles = [{"attribute": k, "role": v[0], "reason": v[1]} for k, v in ATTRIBUTE_ROLES.items()]
    text = [
        "# UCI Adult — client partition statistics",
        "",
        f"Source: {CITATION} (extracted from the {EXTRACTION_YEAR} U.S. Census database).",
        f"Raw files: {raw['train_rows']} training rows (positive rate "
        f"{raw['train_positive_rate']:.4f}), {raw['test_rows']} test rows (positive rate "
        f"{raw['test_positive_rate']:.4f}). Only training rows are partitioned. Clients are "
        "simulated domains derived from one public census dataset, not real organisations.",
        f"IID control shown for seed {DESCRIBE_SEED}.",
        "",
        markdown_table(rows, PARTITION_STATS_FIELDS),
        "",
        "## Attribute roles",
        "",
        markdown_table(roles, ("attribute", "role", "reason")),
        "",
    ]
    (out / "adult_partition_stats.md").write_text("\n".join(text), encoding="utf-8")
    LOGGER.info("Wrote %s", out)
    for row in rows:
        LOGGER.info("%s", row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
