"""Write the toy three-domain federated dataset to CSV.

The domain names are labels on synthetic numbers, not data about patients, accounts or
users. Settings come from ``configs/federated_fedavg.yaml`` (``dataset.synthetic``).

Usage::

    python data/synthetic/generate_federated_toy.py --seed 0 --out data/synthetic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.logging_utils import get_logger, setup_logging  # noqa: E402
from src.common.seeding import derive_seed  # noqa: E402
from src.federated.config import load_federated_config  # noqa: E402
from src.federated.datasets.synthetic_domains import generate_domains, save_domains  # noqa: E402

LOGGER = get_logger("data.generate_federated_toy")


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, required=True, help="run seed")
    parser.add_argument("--out", required=True, help="output folder")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "federated_fedavg.yaml"))
    parser.add_argument("--smoke", action="store_true", help="use the smoke-size domains")
    args = parser.parse_args(argv)
    setup_logging()
    config, _ = load_federated_config(args.config, smoke=args.smoke)
    if config.dataset.synthetic is None:
        LOGGER.error("%s has no dataset.synthetic block", args.config)
        return 2
    # Same derivation as the benchmark, so the CSV equals the data a run with this seed uses.
    arrays = generate_domains(config.dataset.synthetic, derive_seed(args.seed, "toy_data"))
    for path in save_domains(arrays, config.dataset.synthetic.domain_names, args.out):
        LOGGER.info("Wrote %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
