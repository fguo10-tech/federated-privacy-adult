"""DP accounting (Opacus RDP) and DP-SGD local training."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.federated.client import LocalTrainingSpec, train_local
from src.federated.model import initial_weights
from src.federated.privacy import (
    GUARANTEE,
    calibrate_noise_multiplier,
    dp_schedule,
    epsilon_spent,
)


def test_schedule_matches_opacus_dataloader() -> None:
    schedule = dp_schedule(num_rows=1000, batch_size=64, local_epochs=2, rounds=20)
    assert schedule.sample_rate == pytest.approx(1 / 16)  # ceil(1000/64) = 16 batches
    assert schedule.steps_per_round == 32
    assert schedule.total_steps == 640


def test_accountant_monotonicity() -> None:
    q, delta = 0.01, 1e-5
    eps_100 = epsilon_spent(1.0, q, 100, delta, "rdp")
    eps_1000 = epsilon_spent(1.0, q, 1000, delta, "rdp")
    eps_more_noise = epsilon_spent(2.0, q, 1000, delta, "rdp")
    assert 0 < eps_100 < eps_1000 < math.inf
    assert eps_more_noise < eps_1000
    assert epsilon_spent(0.0, q, 10, delta, "rdp") == math.inf
    assert epsilon_spent(1.0, q, 0, delta, "rdp") == 0.0


@pytest.mark.parametrize("target", [1.0, 3.0, 8.0])
def test_calibration_meets_target_after_last_round(target: float) -> None:
    schedule = dp_schedule(num_rows=3000, batch_size=64, local_epochs=1, rounds=20)
    sigma = calibrate_noise_multiplier(target, 1e-5, schedule.sample_rate, schedule.total_steps, "rdp", 0.01)  # fmt: skip
    final = epsilon_spent(sigma, schedule.sample_rate, schedule.total_steps, 1e-5, "rdp")
    assert final <= target + 1e-9
    assert final >= target - 0.05  # calibrated, not wildly over-noised
    halfway = epsilon_spent(sigma, schedule.sample_rate, schedule.total_steps // 2, 1e-5, "rdp")
    assert halfway < final


def _spec(mode: str, server_round: int = 1) -> LocalTrainingSpec:
    return LocalTrainingSpec(
        seed=100,
        server_round=server_round,
        num_rounds=3,
        local_epochs=1,
        batch_size=16,
        learning_rate=0.1,
        momentum=0.0,
        weight_decay=0.0,
        hidden_sizes=(8,),
        num_classes=2,
        dp_mode=mode,
        target_epsilon=3.0,
        noise_multiplier=0.0,
        max_grad_norm=1.0,
        delta=1e-4,
        accountant="rdp",
        epsilon_tolerance=0.01,
        run_label="main|domain|test_condition",
    )


def test_dp_training_changes_weights_reports_epsilon_and_is_reproducible() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 5)).astype(np.float32)
    y = (x[:, 0] > 0).astype(np.int64)
    w0 = initial_weights(5, [8], 2, seed=100)
    first = train_local(w0, x, y, _spec("target_epsilon"), client_id=0, dp_secret=7)
    again = train_local(w0, x, y, _spec("target_epsilon"), client_id=0, dp_secret=7)
    plain = train_local(w0, x, y, _spec("none"), client_id=0, dp_secret=7)
    assert any(not np.allclose(a, b) for a, b in zip(first.weights, w0, strict=True))
    assert all(np.array_equal(a, b) for a, b in zip(first.weights, again.weights, strict=True))
    assert any(not np.allclose(a, b) for a, b in zip(first.weights, plain.weights, strict=True))
    assert 0 < first.epsilon < 3.0  # one of three rounds spent
    assert first.noise_multiplier > 0 and plain.epsilon == math.inf
    last = train_local(w0, x, y, _spec("target_epsilon", server_round=3), client_id=0, dp_secret=7)
    assert first.epsilon < last.epsilon <= 3.0 + 1e-9
    other_round = train_local(
        w0, x, y, _spec("target_epsilon", server_round=2), client_id=0, dp_secret=7
    )
    assert any(not np.array_equal(a, b) for a, b in zip(first.weights, other_round.weights, strict=True))  # fmt: skip


def test_guarantee_text_states_scope() -> None:
    assert "record-level" in GUARANTEE and "NOT client-level" in GUARANTEE
    assert "num-examples" in GUARANTEE


def test_empty_poisson_batches_take_a_noise_only_step() -> None:
    import torch

    from src.federated.model import MLP, get_weights
    from src.federated.privacy import train_dp_sgd

    n, batch, epochs = 20, 1, 2  # q = 1/20, P(empty batch) = 0.95**20 ~ 0.36 per step
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.normal(size=(n, 4)).astype(np.float32))
    y = torch.from_numpy((rng.random(n) > 0.5).astype(np.int64))
    model = MLP(4, [3], 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss, steps, empty = train_dp_sgd(
        model, optimizer, x, y, batch, epochs, 1.0, 1.0, sample_seed=5, noise_seed=6
    )
    planned = dp_schedule(n, batch, epochs, rounds=1)
    assert empty >= 1  # the fixed sample seed produces empty batches
    assert steps == epochs * math.ceil(n / batch) == planned.steps_per_round
    assert all(np.isfinite(w).all() for w in get_weights(model))
    assert math.isfinite(loss)  # NaN losses of empty batches are left out of the mean


def test_executed_steps_equal_accounted_steps_with_empty_batches() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(20, 5)).astype(np.float32)
    y = (x[:, 0] > 0).astype(np.int64)
    spec = _spec("fixed_noise")
    spec = LocalTrainingSpec(**{**spec.__dict__, "batch_size": 1, "noise_multiplier": 1.0})
    w0 = initial_weights(5, [8], 2, seed=100)
    result = train_local(w0, x, y, spec, client_id=0, dp_secret=7)
    schedule = dp_schedule(20, 1, 1, rounds=3)
    assert result.empty_batches >= 1
    assert result.steps == schedule.steps_per_round
    charged = epsilon_spent(1.0, schedule.sample_rate, result.steps * spec.server_round, 1e-4, "rdp")  # fmt: skip
    assert result.epsilon == pytest.approx(charged)
    assert all(np.isfinite(w).all() for w in result.weights)


def test_huge_noise_destroys_utility_and_dp_runs_are_reproducible() -> None:
    from src.federated.client import evaluate_local

    rng = np.random.default_rng(3)
    x = rng.normal(size=(400, 5)).astype(np.float32)
    y = (x[:, 0] + x[:, 1] > 0).astype(np.int64)  # balanced, linearly separable
    base = _spec("none")
    base = LocalTrainingSpec(**{**base.__dict__, "local_epochs": 3, "batch_size": 32})
    noisy = LocalTrainingSpec(
        **{**base.__dict__, "dp_mode": "fixed_noise", "noise_multiplier": 1000.0}
    )
    w0 = initial_weights(5, [8], 2, seed=100)
    plain = train_local(w0, x, y, base, client_id=0, dp_secret=7)
    private = train_local(w0, x, y, noisy, client_id=0, dp_secret=7)
    again = train_local(w0, x, y, noisy, client_id=0, dp_secret=7)
    plain_acc = evaluate_local(plain.weights, x, y, (8,), 2)["accuracy"]
    private_acc = evaluate_local(private.weights, x, y, (8,), 2)["accuracy"]
    assert plain_acc > 0.85
    assert abs(private_acc - 0.5) < 0.2  # near chance on a balanced task
    distance = np.sqrt(sum(float(np.sum((a - b) ** 2)) for a, b in zip(private.weights, plain.weights, strict=True)))  # fmt: skip
    scale = np.sqrt(sum(float(np.sum(b**2)) for b in plain.weights))
    assert distance > 10 * scale
    assert all(np.array_equal(a, b) for a, b in zip(private.weights, again.weights, strict=True))


def _one_step_update(label: str, sigma: float) -> list[np.ndarray]:
    """One DP-SGD step with q = 1 (every row in the batch), so only the noise is random."""
    rng = np.random.default_rng(4)
    x = rng.normal(size=(50, 5)).astype(np.float32)
    y = (x[:, 0] > 0).astype(np.int64)
    spec = _spec("fixed_noise")
    spec = LocalTrainingSpec(
        **{
            **spec.__dict__,
            "batch_size": 50,
            "learning_rate": 1.0,
            "momentum": 0.0,
            "noise_multiplier": sigma,
            "run_label": label,
        }
    )
    w0 = initial_weights(5, [8], 2, seed=100)
    result = train_local(w0, x, y, spec, client_id=0, dp_secret=7)
    return [w.astype(np.float64) - g.astype(np.float64) for w, g in zip(result.weights, w0, strict=True)]  # fmt: skip


def test_dp_noise_is_independent_across_conditions() -> None:
    """Reviewer's attack: with a shared noise vector z, updates u_s = a (g + s C z) of two
    conditions give a (g) = 2 u_1 - u_2 exactly (sigma 1 and 2): the noise cancels."""

    def estimate(label_1: str, label_2: str) -> float:
        noise_free = _one_step_update("main|domain|noise_free", 0.0)
        u1, u2 = _one_step_update(label_1, 1.0), _one_step_update(label_2, 2.0)
        return max(
            float(np.max(np.abs(2 * a - b - c))) for a, b, c in zip(u1, u2, noise_free, strict=True)
        )

    # Sanity of the test itself: if two runs DID share their randomness (same run label),
    # the cancellation recovers the noise-free update.
    assert estimate("main|domain|fedavg_dp_eps3", "main|domain|fedavg_dp_eps3") < 1e-5
    # Different conditions (and different splits) draw independent noise: no cancellation.
    assert estimate("main|domain|fedavg_dp_eps3", "main|domain|fedavg_dp_eps1") > 1e-2
    assert estimate("main|domain|fedavg_dp_eps3", "main|iid|fedavg_dp_eps3") > 1e-2


def test_released_quantities_table_and_guarantee_agree() -> None:
    from src.federated.privacy import RELEASED_QUANTITIES, released_quantities_markdown

    table = released_quantities_markdown()
    assert len(table.strip().splitlines()) == 2 + len(RELEASED_QUANTITIES)
    assert "for this run alone" in GUARANTEE and "add up" in GUARANTEE
    for name, covered, _ in RELEASED_QUANTITIES:
        assert f"`{name}`" in table
        if covered == "NOT covered":
            assert name in GUARANTEE, name
    assert "train-loss" not in {name for name, _, _ in RELEASED_QUANTITIES}


def test_dp_requires_a_client_secret() -> None:
    x = np.zeros((10, 5), dtype=np.float32)
    y = np.zeros(10, dtype=np.int64)
    with pytest.raises(ValueError, match="dp_secret"):
        train_local(initial_weights(5, [8], 2, seed=100), x, y, _spec("fixed_noise"), client_id=0)
