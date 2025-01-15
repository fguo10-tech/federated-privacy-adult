"""Deterministic seeding helpers.

Every experiment in this repository is driven by an explicit integer seed.
Child seeds (per client, per round, per evaluation episode) are derived with
:func:`derive_seed`, a hash of the parent seed and a label, so that adding a
new consumer of randomness never shifts the random stream of an existing one.
"""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np


def derive_seed(base_seed: int, *labels: int | str) -> int:
    """Return a stable 32-bit child seed for ``base_seed`` and a sequence of labels.

    ``derive_seed(3, "client", 1)`` is the same number on every machine and in every
    process, and is unrelated to ``derive_seed(3, "client", 2)``.
    """
    payload = "|".join([str(int(base_seed)), *[str(label) for label in labels]])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def rng_from_seed(base_seed: int, *labels: int | str) -> np.random.Generator:
    """NumPy generator seeded with ``derive_seed(base_seed, *labels)``."""
    return np.random.default_rng(derive_seed(base_seed, *labels))


def seed_everything(seed: int, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy and (if installed) PyTorch global generators.

    Library code in this repository prefers explicit generators; this function
    exists for entry points and for third-party code that reads global state.
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import torch
    except ImportError:  # code paths that do not use torch still work
        return
    torch.manual_seed(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (RuntimeError, TypeError):  # very old torch builds
            pass
