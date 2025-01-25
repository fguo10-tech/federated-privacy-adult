"""Aggregation rules on hand-made weights and equality with Flower's built-in FedAvg."""

from __future__ import annotations

import numpy as np
import pytest
from flwr.app import Array, ArrayRecord, Message, MetricRecord, RecordDict
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy.strategy_utils import aggregate_arrayrecords

from src.federated.aggregation import (
    aggregate,
    coordinate_median,
    trimmed_mean,
    unweighted_average,
    weighted_average,
)
from src.federated.strategy import ConfigurableFedAvg

# Two "layers" per client, values chosen by hand.
A = [np.array([1.0, 2.0]), np.array([[0.0]])]
B = [np.array([3.0, 6.0]), np.array([[4.0]])]
C = [np.array([5.0, 10.0]), np.array([[8.0]])]


def test_weighted_average_hand_computed() -> None:
    # Sizes 1, 3: weights 0.25, 0.75 -> 0.25*1 + 0.75*3 = 2.5 ; 0.25*2 + 0.75*6 = 5.0 ; 3.0
    result = weighted_average([A, B], [1, 3])
    np.testing.assert_allclose(result[0], [2.5, 5.0])
    np.testing.assert_allclose(result[1], [[3.0]])


def test_weighted_average_three_clients_hand_computed() -> None:
    # Sizes 2, 1, 1: weights 0.5, 0.25, 0.25 -> 0.5*1+0.25*3+0.25*5 = 2.5 ; layer2 = 0+1+2 = 3
    result = weighted_average([A, B, C], [2, 1, 1])
    np.testing.assert_allclose(result[0], [2.5, 5.0])
    np.testing.assert_allclose(result[1], [[3.0]])


def test_unweighted_is_plain_mean_and_differs_when_sizes_differ() -> None:
    plain = unweighted_average([A, B], [1, 3])
    np.testing.assert_allclose(plain[0], [2.0, 4.0])  # (1+3)/2, (2+6)/2
    weighted = weighted_average([A, B], [1, 3])
    assert not np.allclose(plain[0], weighted[0])


def test_unweighted_equals_weighted_for_equal_sizes() -> None:
    for a, b in zip(unweighted_average([A, B, C], [7, 7, 7]), weighted_average([A, B, C], [7, 7, 7]), strict=True):  # fmt: skip
        np.testing.assert_allclose(a, b)


def test_order_independence_with_client_ids() -> None:
    forward = weighted_average([A, B, C], [1, 2, 3], client_ids=[0, 1, 2])
    backward = weighted_average([C, B, A], [3, 2, 1], client_ids=[2, 1, 0])
    for a, b in zip(forward, backward, strict=True):
        assert np.array_equal(a, b)


def test_coordinate_median_and_trimmed_mean() -> None:
    outlier = [np.array([100.0, -100.0]), np.array([[1000.0]])]
    median = coordinate_median([A, B, C, outlier], [1, 1, 1, 1])
    np.testing.assert_allclose(median[0], [4.0, 4.0])  # medians of {1,3,5,100}, {2,6,10,-100}
    np.testing.assert_allclose(median[1], [[6.0]])  # median of {0,4,8,1000}
    trimmed = trimmed_mean([A, B, C, outlier], [1, 1, 1, 1], beta=0.25)
    np.testing.assert_allclose(trimmed[0], [4.0, 4.0])  # drop 1 low + 1 high, mean of middle 2
    np.testing.assert_allclose(trimmed_mean([A, B, C], [1, 5, 9], beta=0.0)[0], [3.0, 6.0])


def test_dtype_preserved_and_validation() -> None:
    a32 = [np.array([1.0, 2.0], dtype=np.float32)]
    b32 = [np.array([3.0, 4.0], dtype=np.float32)]
    assert weighted_average([a32, b32], [1, 1])[0].dtype == np.float32
    with pytest.raises(ValueError):
        weighted_average([A, [np.zeros(3), np.zeros((1, 1))]], [1, 1])
    with pytest.raises(ValueError):
        aggregate("bogus", [A], [1], beta=0.0)
    with pytest.raises(ValueError):
        trimmed_mean([A, B], [1, 1], beta=0.5)


def _reply(arrays: list[np.ndarray], n: int, pid: int) -> Message:
    record = ArrayRecord({f"p{i}": Array(a.astype(np.float32)) for i, a in enumerate(arrays)})
    metrics = MetricRecord({"num-examples": n, "partition-id": pid, "train-loss": float(pid)})
    return _make_reply(RecordDict({"arrays": record, "metrics": metrics}))


def _make_reply(content: RecordDict) -> Message:
    from flwr.app.metadata import Metadata

    metadata = Metadata(
        run_id=1,
        message_id="r",
        src_node_id=5,
        dst_node_id=1,
        reply_to_message_id="m",
        group_id="",
        created_at=0.0,
        ttl=60.0,
        message_type="train",
    )
    return Message(content=content, metadata=metadata)


@pytest.fixture(autouse=True)
def _task_identity():
    from flwr.supercore.task_identity import TaskIdentity

    previous = (TaskIdentity._task_id, TaskIdentity._run_id, TaskIdentity._node_id)
    TaskIdentity.task_id, TaskIdentity.run_id, TaskIdentity.node_id = 1, 1, 1
    yield
    TaskIdentity._task_id, TaskIdentity._run_id, TaskIdentity._node_id = previous


def test_strategy_weighted_equals_flower_builtin_fedavg() -> None:
    rng = np.random.default_rng(0)
    clients = [[rng.normal(size=(3, 2)), rng.normal(size=4)] for _ in range(3)]
    sizes = [10, 250, 37]
    replies = [_reply(arrays, n, pid) for pid, (arrays, n) in enumerate(zip(clients, sizes, strict=True))]  # fmt: skip
    ours = ConfigurableFedAvg(
        aggregation_rule="weighted",
        trimmed_mean_beta=0.0,
        num_clients=3,
        fraction_evaluate=0.0,
        fail_on_client_error=True,
    )
    our_arrays, our_metrics = ours.aggregate_train(1, replies)
    builtin_arrays, builtin_metrics = FedAvg().aggregate_train(1, replies)
    direct = aggregate_arrayrecords([m.content for m in replies], "num-examples")
    for key in builtin_arrays:
        np.testing.assert_allclose(our_arrays[key].numpy(), builtin_arrays[key].numpy(), rtol=1e-6, atol=1e-7)  # fmt: skip
        np.testing.assert_allclose(our_arrays[key].numpy(), direct[key].numpy(), rtol=1e-6, atol=1e-7)  # fmt: skip
    assert our_metrics["train-loss"] == pytest.approx(builtin_metrics["train-loss"])
    # Arrival order must not matter.
    shuffled, _ = ours.aggregate_train(2, list(reversed(replies)))
    for key in our_arrays:
        assert np.array_equal(shuffled[key].numpy(), our_arrays[key].numpy())


def test_strategy_unweighted_differs_from_builtin_when_sizes_differ() -> None:
    rng = np.random.default_rng(1)
    clients = [[rng.normal(size=(3, 2))] for _ in range(2)]
    replies = [_reply(c, n, pid) for pid, (c, n) in enumerate(zip(clients, [5, 500], strict=True))]
    plain = ConfigurableFedAvg(
        aggregation_rule="unweighted",
        trimmed_mean_beta=0.0,
        num_clients=2,
        fraction_evaluate=0.0,
        fail_on_client_error=True,
    )
    arrays, _ = plain.aggregate_train(1, replies)
    builtin, _ = FedAvg().aggregate_train(1, replies)
    assert not np.allclose(arrays["p0"].numpy(), builtin["p0"].numpy())
    expected = (clients[0][0] + clients[1][0]) / 2
    np.testing.assert_allclose(arrays["p0"].numpy(), expected, rtol=1e-6)
    assert plain.ledger.up[1] > 0


def test_baseline_registry_matches_configs() -> None:
    from src.federated import BASELINE_METHODS, BASELINE_REGISTRY, register_baseline

    assert BASELINE_METHODS == (
        "centralised_mlp",
        "centralised_mlp_client_rows",
        "svm",
        "random_forest",
        "decision_tree",
        "majority_class",
    )
    assert set(BASELINE_REGISTRY) == set(BASELINE_METHODS)
    with pytest.raises(ValueError):
        register_baseline("svm")(lambda config, dataset, seed: None)
