"""The real Flower path: ClientApp / ServerApp / strategy on the in-process Grid (and Ray).

Includes the proof that a client reply carries only model weights and aggregate metrics.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from flwr.app import ConfigRecord, Message, RecordDict
from flwr.serverapp import Grid

from src.common.config import from_dict
from src.federated.client import spec_to_config
from src.federated.client_app import (
    DP_REPLY_METRIC_KEYS,
    REPLY_METRIC_KEYS,
    ClientDataSource,
    make_client_app,
)
from src.federated.config import FederatedConfig, load_federated_config
from src.federated.datasets import load_dataset
from src.federated.inprocess import InProcessGrid, ray_available, run_inprocess, run_with_ray
from src.federated.model import (
    build_model,
    initial_weights,
    parameter_names,
    weights_to_arrayrecord,
)
from src.federated.runner import Task, run_task, training_spec
from src.federated.server_app import ServerResult, ServerSpec, make_server_app

TWO_CLIENTS = [
    "dataset.synthetic.domain_names=[alpha-like, beta-like]",
    "dataset.synthetic.train_rows=[400, 200]",
    "dataset.synthetic.test_rows=[200, 200]",
    "dataset.synthetic.positive_rates=[0.3, 0.4]",
    "dataset.synthetic.feature_shift=0.5",
    "training.rounds=5",
    "training.batch_size=32",
]


@pytest.fixture(scope="module")
def two_client(configs_dir: Path) -> tuple[FederatedConfig, dict]:
    return load_federated_config(configs_dir / "federated_fedavg.yaml", overrides=TWO_CLIENTS)


@pytest.fixture(autouse=True)
def _task_identity():
    from flwr.supercore.task_identity import TaskIdentity

    previous = (TaskIdentity._task_id, TaskIdentity._run_id, TaskIdentity._node_id)
    TaskIdentity.task_id, TaskIdentity.run_id, TaskIdentity.node_id = 1, 1, 1
    yield
    TaskIdentity._task_id, TaskIdentity._run_id, TaskIdentity._node_id = previous


def _apps(config: FederatedConfig, seed: int, dp=None):
    dataset = load_dataset(config.dataset, config.evaluation, seed)
    hidden = list(config.model.hidden_sizes)
    spec = training_spec(
        config,
        seed,
        dp,
        config.training.local_epochs,
        config.training.rounds,
        run_label=f"main|domain|{'fedavg' if dp is None else dp.name}",
    )
    train_config = spec_to_config(spec) | {"num-classes": dataset.num_classes}
    names = tuple(parameter_names(build_model(dataset.num_features, hidden, 2, 0)))
    source = ClientDataSource(
        config.dataset, config.evaluation, config.partitions, "domain", 2, seed
    )
    sink = ServerResult()
    server = make_server_app(
        ServerSpec(
            num_rounds=config.training.rounds,
            num_clients=2,
            aggregation_rule="weighted",
            trimmed_mean_beta=0.0,
            fraction_evaluate=config.federation.fraction_evaluate,
            train_config=train_config,
            evaluate_config={"hidden-sizes": hidden, "num-classes": 2},
            parameter_names=names,
        ),
        initial_weights(dataset.num_features, hidden, 2, seed),
        None,
        sink,
    )
    return server, make_client_app(source, None), sink, dataset, train_config, names


def _train_message(config: FederatedConfig, seed: int, node: int, dp=None, server_seed=None):
    _, client, _, dataset, train_config, names = _apps(config, seed, dp=dp)
    weights = initial_weights(dataset.num_features, list(config.model.hidden_sizes), 2, seed)
    if server_seed is not None:
        train_config = train_config | {"seed": server_seed}
    content = RecordDict(
        {
            "arrays": weights_to_arrayrecord(weights, names),
            "config": ConfigRecord(train_config | {"server-round": 1}),
        }
    )
    return client, Message(content, dst_node_id=node, message_type="train"), dataset, weights


def test_inprocess_grid_is_a_real_flower_grid(two_client) -> None:
    config, _ = two_client
    _, client, *_ = _apps(config, 100)
    grid = InProcessGrid(client, 2, run_id=1, secret_seed=100)
    assert isinstance(grid, Grid)
    assert list(grid.get_node_ids()) == [101, 102]


def test_client_reply_contains_no_raw_data(two_client) -> None:
    config, _ = two_client
    client, message, dataset, weights = _train_message(config, 100, node=101)
    grid = InProcessGrid(client, 2, run_id=1, secret_seed=100)
    (reply,) = list(grid.send_and_receive([message]))
    assert not reply.has_error()
    content = reply.content
    assert set(content.keys()) == {"arrays", "metrics"}
    assert len(content.config_records) == 0
    arrays = content["arrays"]
    assert list(arrays.keys()) == list(message.content["arrays"].keys())
    for array, reference in zip(arrays.to_numpy_ndarrays(), weights, strict=True):
        assert array.shape == reference.shape  # model-shaped, nothing else
    metrics = content["metrics"]
    assert set(metrics.keys()) == set(REPLY_METRIC_KEYS)
    assert all(isinstance(value, (int, float)) for value in metrics.values())
    rows = int((dataset.train.domain == 0).sum())
    assert metrics["num-examples"] == rows
    for array in arrays.to_numpy_ndarrays():
        assert rows not in array.shape  # no per-row array (features or labels)
    payload = sum(a.nbytes for a in weights)
    assert arrays.count_bytes() < payload + 200 * len(weights)  # weights plus small headers


def test_client_is_stateless_and_deterministic(two_client) -> None:
    config, _ = two_client
    client, message, _, _ = _train_message(config, 100, node=102)
    grid = InProcessGrid(client, 2, run_id=1, secret_seed=100)
    (first,) = list(grid.send_and_receive([message]))
    client2, message2, _, _ = _train_message(config, 100, node=102)
    (second,) = list(grid.send_and_receive([message2]))
    for a, b in zip(first.content["arrays"].to_numpy_ndarrays(), second.content["arrays"].to_numpy_ndarrays(), strict=True):  # fmt: skip
        assert np.array_equal(a, b)
    assert all(len(ctx.state) == 0 for ctx in grid._contexts.values())


def test_two_client_end_to_end_through_flower(two_client) -> None:
    config, raw = two_client
    task = Task("main", "federated", "domain", "fedavg", 100, "weighted", None, None, None, 0, ())
    output = run_task(raw, task)
    rounds = [r for r in output.per_round]
    assert [r["round"] for r in rounds] == list(range(config.training.rounds + 1))
    dataset = load_dataset(config.dataset, config.evaluation, 100)
    majority = max(np.mean(dataset.test.y == 0), np.mean(dataset.test.y == 1))
    assert rounds[-1]["accuracy"] > majority + 0.05
    assert rounds[-1]["train_loss"] < rounds[1]["train_loss"]
    # Training traffic only (client evaluation is off): one model down and one up per client.
    assert rounds[-1]["bytes_up_cum"] > 0 and rounds[-1]["bytes_down_cum"] == rounds[-1]["bytes_up_cum"]  # fmt: skip
    clients = [r for r in output.per_client if r["round"] == config.training.rounds]
    assert {r["client"] for r in clients} == {"alpha-like", "beta-like"}
    assert all(r["local_eval_accuracy"] == "" for r in clients)  # client evaluation is off


def test_same_seed_gives_identical_final_weights(two_client) -> None:
    config, _ = two_client
    finals = []
    for _ in range(2):
        server, client, sink, *_ = _apps(config, 101)
        run_inprocess(server, client, 2, run_id=5, secret_seed=5)
        finals.append(sink.final_weights)
    server, client, sink, *_ = _apps(config, 102)
    run_inprocess(server, client, 2, run_id=5, secret_seed=5)
    assert all(np.array_equal(a, b) for a, b in zip(finals[0], finals[1], strict=True))
    assert not all(np.array_equal(a, b) for a, b in zip(finals[0], sink.final_weights, strict=True))


def test_client_error_fails_loudly(two_client) -> None:
    from src.federated.strategy import ClientFailure

    config, raw = two_client
    broken = from_dict(FederatedConfig, raw)
    server, _, sink, *_ = _apps(broken, 100)
    _, good_client, *_ = _apps(broken, 100)
    with pytest.raises(ClientFailure):
        run_inprocess(
            server, good_client, 3, run_id=9, secret_seed=9
        )  # 3 nodes, but the split has 2 clients


@pytest.mark.flower_sim
@pytest.mark.slow
@pytest.mark.skipif(
    not ray_available() or os.environ.get("FEDERATED_RUN_RAY_TEST") != "1",
    reason="opt-in (FEDERATED_RUN_RAY_TEST=1) and needs flwr[simulation]; UNTESTED: on the "
    "development machine Ray workers crashed at start-up and the run hung",
)
def test_ray_engine_matches_inprocess(two_client) -> None:
    config, _ = two_client
    server, client, sink, *_ = _apps(config, 103)
    run_inprocess(server, client, 2, run_id=11, secret_seed=11)
    server_ray, client_ray, sink_ray, *_ = _apps(config, 103)
    run_with_ray(server_ray, client_ray, 2)
    assert sink_ray.final_weights is not None
    for a, b in zip(sink.final_weights, sink_ray.final_weights, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


class _RecordingGrid(InProcessGrid):
    """In-process grid that keeps every pushed message and every reply."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pushed_types: list[str] = []
        self.pushed: list[Message] = []
        self.replies: list[Message] = []

    def push_messages(self, messages):
        messages = list(messages)
        self.pushed_types.extend(m.metadata.message_type for m in messages)
        self.pushed.extend(messages)
        return super().push_messages(messages)

    def pull_messages(self, reply_to_message_ids=None, *, message_ids=None):
        replies = list(super().pull_messages(reply_to_message_ids, message_ids=message_ids))
        self.replies.extend(replies)
        return replies


def test_dp_run_releases_only_the_documented_quantities(two_client) -> None:
    from flwr.app import Context

    from src.federated.config import DPSweepEntry
    from src.federated.privacy import RELEASED_QUANTITIES

    config, _ = two_client
    assert config.federation.fraction_evaluate == 0.0
    entry = DPSweepEntry("fedavg_dp_eps3", "target_epsilon", 3.0, None, 1.0)
    server, client, sink, *_ = _apps(config, 104, dp=entry)
    grid = _RecordingGrid(client, 2, run_id=12, secret_seed=12)
    server(grid, Context(run_id=12, node_id=1, node_config={}, state=RecordDict(), run_config={}))
    assert set(grid.pushed_types) == {"train"}  # no evaluate message is ever sent
    assert len(grid.replies) == 2 * config.training.rounds
    documented = {name for name, _, _ in RELEASED_QUANTITIES} - {"model weights"}
    assert set(DP_REPLY_METRIC_KEYS) == documented
    for reply in grid.replies:
        assert set(reply.content.keys()) == {"arrays", "metrics"}
        assert set(reply.content["metrics"].keys()) == set(DP_REPLY_METRIC_KEYS)
        assert "train-loss" not in reply.content["metrics"]


def _dp_entry():
    from src.federated.config import DPSweepEntry

    return DPSweepEntry("fedavg_dp_eps3", "target_epsilon", 3.0, None, 1.0)


def _dp_update(config, grid_secret: int, server_seed: int, drop_secret: bool = False):
    client, message, _, weights = _train_message(
        config, 100, node=101, dp=_dp_entry(), server_seed=server_seed
    )
    grid = InProcessGrid(client, 2, run_id=1, secret_seed=grid_secret)
    if drop_secret:
        for context in grid._contexts.values():
            del context.node_config["client-secret-seed"]
    (reply,) = list(grid.send_and_receive([message]))
    assert not reply.has_error()
    return reply.content["arrays"].to_numpy_ndarrays()


def _same(a, b) -> bool:
    return all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def test_dp_noise_ignores_the_server_seed(two_client) -> None:
    config, _ = two_client
    first = _dp_update(config, grid_secret=100, server_seed=100)
    other_server_seed = _dp_update(config, grid_secret=100, server_seed=987654)
    assert _same(first, other_server_seed)  # (a) the server's seed has no influence


def test_dp_noise_depends_on_the_client_secret(two_client) -> None:
    config, _ = two_client
    first = _dp_update(config, grid_secret=100, server_seed=100)
    other_secret = _dp_update(config, grid_secret=200, server_seed=100)
    assert not _same(first, other_secret)  # (b) same message, different client secret


def test_client_secret_never_travels(two_client) -> None:
    from flwr.app import Context

    config, _ = two_client
    server, client, *_ = _apps(config, 105, dp=_dp_entry())
    grid = _RecordingGrid(client, 2, run_id=13, secret_seed=105)
    server(grid, Context(run_id=13, node_id=1, node_config={}, state=RecordDict(), run_config={}))
    secrets_ = {int(ctx.node_config["client-secret-seed"]) for ctx in grid._contexts.values()}
    assert len(secrets_) == 2
    for message in [*grid.pushed, *grid.replies]:
        for record in [*message.content.config_records.values(), *message.content.metric_records.values()]:  # fmt: skip
            assert not any("secret" in str(key) for key in record.keys())
            for value in record.values():
                values = value if isinstance(value, list) else [value]
                assert not any(isinstance(v, (int, float)) and v in secrets_ for v in values)


def test_dp_without_a_configured_secret_uses_os_entropy(two_client) -> None:
    config, _ = two_client
    first = _dp_update(config, grid_secret=100, server_seed=100, drop_secret=True)
    second = _dp_update(config, grid_secret=100, server_seed=100, drop_secret=True)
    assert all(np.isfinite(w).all() for w in first)
    assert not _same(first, second)  # (d) fresh entropy each run: not reproducible
