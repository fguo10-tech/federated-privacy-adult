"""Download UCI Adult / Census Income into ``data/adult/`` (git-ignored) and pin checksums.

Adult (Census Income) was extracted by Barry Becker from the 1994 U.S. Census database and
donated to the UCI Machine Learning Repository on 1996-04-30 (Becker & Kohavi, 1996,
https://doi.org/10.24432/C5XW20, CC BY 4.0; http://archive.ics.uci.edu/ml/datasets/Adult).

Only the canonical UCI host is contacted, with the standard library. UCI publishes no
checksums, so ``data/checksums/adult.sha256`` is a trust-on-first-use pin recorded when this
draft was written; later downloads are verified against it.

Usage (from the repository root)::

    python data/download_adult.py            # same as: python -m src.federated.datasets.adult --download
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.federated.datasets.adult import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(
        main(
            [
                "--download",
                "--data-dir",
                str(REPO_ROOT / "data" / "adult"),
                "--checksums",
                str(REPO_ROOT / "data" / "checksums" / "adult.sha256"),
                *sys.argv[1:],
            ]
        )
    )
