"""Secure-aggregation interface and a pairwise-masking **SIMULATION**.

THIS IS NOT SECURE AGGREGATION. It is an in-process simulation that shows the arithmetic of
pairwise additive masking (Bonawitz et al., 2017): every pair of clients ``(i, j)`` shares a
mask ``m_ij``; client ``i`` adds ``+m_ij`` and client ``j`` adds ``−m_ij``, so each masked
update looks random on its own but the masks cancel in the sum. Here every mask is derived
from **one shared seed known to the whole process** — there is no key agreement, no dropout
recovery, no authenticated channel, no threshold secret sharing and no homomorphic
encryption. Anyone holding the seed (including the "server" in this process) can unmask any
individual update. It is **not cryptographically secure** and must not be described as such.

Sample-count-weighted case: client ``k`` sends masked ``n_k · θ_k`` and a masked ``n_k``; the
server sums both and divides, recovering exactly the weighted FedAvg mean of
:func:`src.federated.aggregation.weighted_average` without seeing any single ``θ_k`` or
``n_k`` in the clear (in a real protocol; here only arithmetically).

Masks are real-valued (float64) rather than elements of a finite ring, so cancellation is
exact only up to floating-point rounding; tests compare with a tolerance.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from src.common.seeding import rng_from_seed

IMPLEMENTED_VS_STUBBED: dict[str, str] = {
    "pairwise additive masks that cancel in the sum": "implemented (simulation)",
    "sample-count weighting via masked n_k·θ_k and masked n_k": "implemented (simulation)",
    "pairwise key agreement (e.g. Diffie-Hellman)": "NOT implemented — one shared seed",
    "dropout recovery / threshold secret sharing of mask seeds": "NOT implemented",
    "authenticated, encrypted channels": "NOT implemented",
    "finite-field arithmetic (exact cancellation, no float leakage)": "NOT implemented — float64",
    "homomorphic encryption": "NOT implemented",
    "protection against a malicious server": "NOT implemented",
}
"""What this module does and does not do. Anything marked NOT implemented is absent."""


class SecureAggregator(Protocol):
    """Interface of a secure-aggregation scheme: mask one update, unmask the sum."""

    def mask(self, client_id: int, update: Sequence[np.ndarray], weight: float) -> list[np.ndarray]:
        """Return the masked (weighted) update that the client would send."""

    def unmask_sum(self, masked: Sequence[Sequence[np.ndarray]]) -> list[np.ndarray]:
        """Return the sum of the unmasked (weighted) updates from all masked updates."""


class PairwiseMaskingSimulator:
    """Pairwise additive masking among ``client_ids`` with masks drawn from ``shared_seed``.

    ``mask`` returns ``[weight · θ_k + Σ_j ±m_kj ..., weight + Σ_j ±s_kj]``: the parameter arrays
    followed by one extra scalar array carrying the masked weight. ``unmask_sum`` adds all
    masked messages (the masks cancel) and returns the weighted sum of the parameters followed
    by the total weight. :meth:`weighted_mean` divides the two.
    """

    def __init__(self, client_ids: Sequence[int], shared_seed: int, mask_scale: float) -> None:
        self.client_ids = tuple(sorted(int(c) for c in client_ids))
        if len(set(self.client_ids)) != len(self.client_ids):
            raise ValueError("client ids must be unique")
        self.shared_seed = int(shared_seed)
        self.mask_scale = float(mask_scale)

    def _pair_mask(
        self, low: int, high: int, shapes: Sequence[tuple[int, ...]]
    ) -> list[np.ndarray]:
        rng = rng_from_seed(self.shared_seed, "pairmask", low, high)
        return [rng.normal(0.0, self.mask_scale, size=shape) for shape in shapes]

    def mask(self, client_id: int, update: Sequence[np.ndarray], weight: float) -> list[np.ndarray]:
        """Masked ``[weight · θ, weight]`` of one client (float64)."""
        if client_id not in self.client_ids:
            raise ValueError(f"unknown client {client_id}")
        payload = [np.asarray(a, dtype=np.float64) * float(weight) for a in update]
        payload.append(np.asarray([float(weight)], dtype=np.float64))
        shapes = [a.shape for a in payload]
        for other in self.client_ids:
            if other == client_id:
                continue
            low, high = min(client_id, other), max(client_id, other)
            sign = 1.0 if client_id == low else -1.0
            for array, mask in zip(payload, self._pair_mask(low, high, shapes), strict=True):
                array += sign * mask
        return payload

    def unmask_sum(self, masked: Sequence[Sequence[np.ndarray]]) -> list[np.ndarray]:
        """Sum of all masked messages; pairwise masks cancel (every client must report)."""
        if len(masked) != len(self.client_ids):
            raise ValueError("this simulation has no dropout recovery: every client must report")
        total = [np.zeros_like(np.asarray(a, dtype=np.float64)) for a in masked[0]]
        for message in masked:
            for acc, array in zip(total, message, strict=True):
                acc += array
        return total

    def weighted_mean(self, masked: Sequence[Sequence[np.ndarray]]) -> list[np.ndarray]:
        """``Σ n_k θ_k / Σ n_k`` recovered from the masked messages."""
        total = self.unmask_sum(masked)
        weight = float(total[-1][0])
        if weight <= 0:
            raise ValueError("total weight must be positive")
        return [array / weight for array in total[:-1]]
