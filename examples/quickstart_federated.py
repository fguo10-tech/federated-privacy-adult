"""Quickstart: federated averaging on the toy domains through the real Flower apps.

Runs (in under a minute on a laptop CPU) weighted FedAvg on the three synthetic domains of
``configs/federated_fedavg.yaml`` — labels on synthetic numbers, not real data — printing
accuracy and F1 of the global model after every round, then one run with per-client DP-SGD
and its ε. All settings come from the YAML file; they are ``[Assumption — not in paper]``.

Usage::

    python examples/quickstart_federated.py [--seed 100]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.logging_utils import get_logger, setup_logging  # noqa: E402
from src.federated.config import load_federated_config  # noqa: E402
from src.federated.privacy import GUARANTEE  # noqa: E402
from src.federated.runner import build_tasks, run_task  # noqa: E402

LOGGER = get_logger("examples.quickstart_federated")
CONFIG = REPO_ROOT / "configs" / "federated_fedavg.yaml"


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=100, help="development seed (>= 100)")
    args = parser.parse_args(argv)
    setup_logging()
    config, raw = load_federated_config(CONFIG)
    natural = config.partitions.splits[0]
    tasks = [
        t
        for t in build_tasks(config, "main", [args.seed])
        if t.split == natural and t.condition in ("fedavg", config.dp.sweep[0].name)
    ]
    LOGGER.info("config %s, seed %d, split %s", CONFIG.relative_to(REPO_ROOT), args.seed, natural)
    for task in tasks:
        output = run_task(raw, task)
        for row in output.per_round:
            LOGGER.info(
                "%-16s round %2d  accuracy %.4f  F1 %.4f  epsilon %s",
                task.condition,
                row["round"],
                row["accuracy"],
                row["f1"],
                row["epsilon_max"],
            )
        if task.dp is not None:
            LOGGER.info(
                "%s: max per-client epsilon %.3f at delta %g — %s",
                task.condition,
                output.final[0]["epsilon_max"],
                config.dp.delta,
                GUARANTEE,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
