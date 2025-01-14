"""Paper-reported values, transcribed once.

This module is the single place where numbers printed in the source paper
enter the code base. Table builders read the ``Paper-reported`` column from
here and the ``Reproduced`` column from result CSV files, so the two can never
be mixed up by a copy-paste slip.

All values below are ``[Paper-reported]``. None of them was produced by this
repository. Values read off a figure (rather than a table) are approximate and
are marked as such.

Source
------
Y. Zhang, J. Liu, J. Wang, L. Dai, F. Guo, G. Cai, "Federated Learning
for Cross-Domain Data Privacy: A Distributed Approach to Secure Collaboration",
arXiv:2504.00282 (v1, 31 March 2025). Table 1.
"""

from __future__ import annotations

from typing import Final

PAPER_ARXIV: Final = "https://arxiv.org/abs/2504.00282"
PAPER_IEEE: Final = "https://ieeexplore.ieee.org/document/11116917/"

# Table 1 "Experimental results" of the source paper.
PAPER_TABLE1: Final[dict[str, dict[str, float]]] = {
    "SVM": {"accuracy": 0.3124, "precision": 0.1032, "recall": 0.2987, "f1": 0.1543},
    "RF": {"accuracy": 0.3278, "precision": 0.1147, "recall": 0.3156, "f1": 0.1679},
    "DT": {"accuracy": 0.2945, "precision": 0.0978, "recall": 0.2764, "f1": 0.1432},
    "Ours (federated)": {"accuracy": 0.35, "precision": 0.1255, "recall": 0.35, "f1": 0.1815},
}

# Figures 2 and 3 of the source paper: training length shown on the x axis ("Epoch", 0 to 100).
PAPER_TRAINING_EPOCHS: Final = 100
