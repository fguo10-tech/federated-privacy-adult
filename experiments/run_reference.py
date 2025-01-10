"""Re-create the reference run that the documentation cites.

Runs the benchmark exactly as fixed in the configs (reporting seeds 0-4), the ablation suites
and the extended privacy study, writes everything under ``experiments/results/reference/`` and
records the measured wall-clock time of each part in that part's ``metadata.json``. Afterwards
run ``make results`` to rebuild the tables.

This overwrites the committed reference results. Do it on purpose, not by accident::

    python -m experiments.run_reference --workers 8                 # everything
    python -m experiments.run_reference --skip-ablations            # headline run only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from src.common.io import read_json, write_json
from src.common.logging_utils import get_logger, setup_logging

LOGGER = get_logger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE = REPO_ROOT / "experiments" / "results" / "reference"
FEDERATED_CONFIG = "configs/federated_adult.yaml"
FEDERATED_DP_CONFIG = "configs/federated_dp.yaml"


def _run(module: str, *arguments: str) -> float:
    """Run ``python -m module arguments`` from the repository root; return the seconds it took."""
    command = [sys.executable, "-m", module, *arguments]
    LOGGER.info("running: %s", " ".join(command[1:]))
    started = time.perf_counter()
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    return time.perf_counter() - started


def _record_timing(run_dir: Path, seconds: float, workers: int) -> None:
    """Add the measured wall-clock time to a run's ``metadata.json``."""
    path = run_dir / "metadata.json"
    if not path.is_file():
        LOGGER.warning("no metadata.json in %s; timing not recorded", run_dir)
        return
    metadata = read_json(path)
    extra = dict(metadata.get("extra") or {})
    extra.update({"wall_clock_seconds": round(seconds, 1), "workers": workers})
    metadata["extra"] = extra
    write_json(path, metadata)


def run_federated(workers: int, ablations: bool) -> None:
    """The locked Adult protocol (and, optionally, the ablation suites) and figures."""
    out = REFERENCE / "federated_adult"
    seconds = _run(
        "experiments.run_federated_benchmark",
        "--config", FEDERATED_CONFIG, "--suite", "all" if ablations else "main",
        "--out", str(out), "--workers", str(workers),
    )  # fmt: skip
    _record_timing(out, seconds, workers)
    _run("experiments.plot_federated", "--run-dir", str(out))
    if not ablations:
        return
    # Extended privacy study: a finer epsilon grid, clipping norms and fixed-noise runs.
    extended = REFERENCE / "federated_adult_dp"
    seconds = _run(
        "experiments.run_federated_benchmark",
        "--config", FEDERATED_DP_CONFIG, "--suite", "main",
        "--out", str(extended), "--workers", str(workers),
    )  # fmt: skip
    _record_timing(extended, seconds, workers)
    _run("experiments.plot_federated", "--run-dir", str(extended))


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--workers", type=int, default=4, help="worker processes")
    parser.add_argument("--skip-ablations", action="store_true", help="headline run only")
    args = parser.parse_args(argv)
    setup_logging("INFO")
    run_federated(args.workers, not args.skip_ablations)
    LOGGER.info(
        "reference run written to %s; now run `make results`", REFERENCE.relative_to(REPO_ROOT)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
