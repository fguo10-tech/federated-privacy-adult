"""Secure-aggregation masking simulation, Byzantine attacks and communication accounting."""

from __future__ import annotations

import numpy as np
import pytest

from src.federated.aggregation import weighted_average
from src.federated.attacks import (
    apply_update_attack,
    byzantine_clients,
    gaussian_noise,
    label_flip,
    scale,
    sign_flip,
)
from src.federated.communication import CommunicationLedger, arrayrecord_bytes, raw_parameter_bytes
from src.federated.model import weights_to_arrayrecord
from src.federated.secure_agg import IMPLEMENTED_VS_STUBBED, PairwiseMaskingSimulator


def _updates(seed: int = 0):
    rng = np.random.default_rng(seed)
    return [[rng.normal(size=(4, 3)), rng.normal(size=5)] for _ in range(4)]


def test_masks_cancel_in_weighted_sum() -> None:
    updates, sizes = _updates(), [10, 200, 35, 7]
    sim = PairwiseMaskingSimulator([0, 1, 2, 3], shared_seed=42, mask_scale=100.0)
    masked = [sim.mask(k, update, sizes[k]) for k, update in enumerate(updates)]
    recovered = sim.weighted_mean(masked)
    expected = weighted_average(updates, sizes)
    for a, b in zip(recovered, expected, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9)
    total = sim.unmask_sum(masked)
    assert total[-1][0] == pytest.approx(sum(sizes))


def test_individual_masked_update_is_far_from_raw() -> None:
    updates = _updates(1)
    sim = PairwiseMaskingSimulator([0, 1, 2, 3], shared_seed=7, mask_scale=100.0)
    masked = sim.mask(2, updates[2], 1.0)
    distance = np.linalg.norm(masked[0] - updates[2][0])
    assert distance > 10 * np.linalg.norm(updates[2][0])
    with pytest.raises(ValueError):
        sim.unmask_sum([masked])  # no dropout recovery


def test_simulation_is_labelled_not_secure() -> None:
    import src.federated.secure_agg as module

    assert "NOT cryptographically secure" in module.__doc__ or "not cryptographically secure" in module.__doc__  # fmt: skip
    assert any("NOT implemented" in value for value in IMPLEMENTED_VS_STUBBED.values())


def test_attacks() -> None:
    global_w = [np.array([1.0, 1.0])]
    honest = [np.array([2.0, 0.0])]  # update (+1, -1)
    np.testing.assert_allclose(sign_flip(honest, global_w)[0], [0.0, 2.0])
    np.testing.assert_allclose(scale(honest, global_w, 10.0)[0], [11.0, -9.0])
    noisy = gaussian_noise(honest, global_w, 1.0, seed=3, client=1, server_round=2)
    again = gaussian_noise(honest, global_w, 1.0, seed=3, client=1, server_round=2)
    assert np.array_equal(noisy[0], again[0]) and not np.allclose(noisy[0], honest[0])
    np.testing.assert_array_equal(label_flip(np.array([0, 1, 1]), 2), [1, 0, 0])
    assert apply_update_attack("label_flip", honest, global_w, 1.0, 1.0, 0, 0, 1) is honest
    assert byzantine_clients(10, 2) == (8, 9)
    with pytest.raises(ValueError):
        apply_update_attack("bogus", honest, global_w, 1.0, 1.0, 0, 0, 1)


def test_communication_byte_counts() -> None:
    from flwr.app import RecordDict

    weights = [np.zeros((10, 4), dtype=np.float32), np.zeros(4, dtype=np.float32)]
    record = weights_to_arrayrecord(weights, ["w", "b"])
    assert raw_parameter_bytes(weights) == 44 * 4
    assert arrayrecord_bytes(record) >= raw_parameter_bytes(weights)
    assert arrayrecord_bytes(record) < raw_parameter_bytes(weights) + 400  # .npy headers + keys
    ledger = CommunicationLedger()
    content = RecordDict({"arrays": record})
    ledger.record_down(1, content, receivers=3)
    ledger.record_up(1, content)
    ledger.record_up(2, content)
    up, down = ledger.cumulative(1)
    assert down == 3 * arrayrecord_bytes(record) and up == arrayrecord_bytes(record)
    assert ledger.cumulative(2)[0] == 2 * arrayrecord_bytes(record)
