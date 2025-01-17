"""Flower ``ClientApp`` (message API, flwr 1.44).

The client reads its partition id from ``context.node_config["partition-id"]`` (the key
Flower's simulation engine provides; ``"num-partitions"`` is checked against the data
source), loads **only its own rows** through :class:`ClientDataSource`, trains with
:func:`src.federated.client.train_local`, and replies with exactly one ``ArrayRecord`` (the
updated weights, keyed by parameter name) and one ``MetricRecord`` (:data:`REPLY_METRIC_KEYS`
without DP, :data:`DP_REPLY_METRIC_KEYS` with DP; all scalars). Raw rows, features and labels are never put into a ``Message``;
``tests/test_federated_flower.py`` inspects a reply to prove it.

The client is **stateless and deterministic**: ``context.state`` is never read or written,
and the reply is a pure function of (global weights, config record, partition, client-held
secret). Under DP the noise and Poisson sampling come only from the client-held secret
``context.node_config["client-secret-seed"]`` (:func:`client_dp_secret`), never from anything
the server sends, so an honest-but-curious server cannot regenerate the noise.

In this simulation every client derives its rows from the same public files and the run
seed (the partition is a pure function of both); in a real deployment each client would
simply hold its own rows.
"""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from flwr.app import Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from src.common.config import from_dict, to_dict
from src.common.logging_utils import get_logger

from .attacks import DATA_ATTACKS, apply_update_attack, label_flip
from .client import evaluate_local, spec_from_config, train_local
from .config import DatasetConfig, EvaluationConfig, PartitionConfig
from .datasets import load_dataset
from .model import arrayrecord_to_weights, weights_to_arrayrecord
from .partition import make_partition

LOGGER = get_logger(__name__)

ARRAYS_KEY = "arrays"
CONFIG_KEY = "config"
METRICS_KEY = "metrics"

REPLY_METRIC_KEYS = (
    "partition-id",
    "num-examples",
    "train-loss",
    "steps",
    "sample-rate",
    "noise-multiplier",
    "epsilon",
    "delta",
    "update-norm",
    "empty-batches",
)
"""Keys of a training reply's ``MetricRecord`` WITHOUT differential privacy (aggregate scalars)."""

DP_REPLY_METRIC_KEYS = tuple(
    key for key in REPLY_METRIC_KEYS if key not in ("train-loss", "empty-batches")
)
"""Keys of a training reply under DP. ``train-loss`` and ``empty-batches`` are un-noised
statistics of the private rows and are not sent; every remaining key is listed, with its
privacy status, in :data:`src.federated.privacy.RELEASED_QUANTITIES`."""

EVAL_METRIC_KEYS = ("partition-id", "num-examples", "eval-loss", "eval-accuracy")
"""Keys of an evaluation reply's ``MetricRecord``."""


@dataclass(frozen=True)
class ClientDataSource:
    """Recipe by which a client finds its own rows: dataset, split, client count and seed."""

    dataset: DatasetConfig
    evaluation: EvaluationConfig
    partitions: PartitionConfig
    split: str
    num_clients: int
    seed: int

    def key(self) -> str:
        """Stable JSON key (used for per-process caching of the loaded rows)."""
        return json.dumps(to_dict(self), sort_keys=True)

    def client_rows(self, partition_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Features and labels of one client's training rows."""
        return _client_rows(self.key(), int(partition_id))


@lru_cache(maxsize=64)
def _client_rows(source_key: str, partition_id: int) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(source_key)
    dataset_cfg = from_dict(DatasetConfig, payload["dataset"])
    evaluation_cfg = from_dict(EvaluationConfig, payload["evaluation"])
    partition_cfg = from_dict(PartitionConfig, payload["partitions"])
    seed = int(payload["seed"])
    dataset = load_dataset(dataset_cfg, evaluation_cfg, seed)
    natural = payload["split"] == dataset.natural_split_name
    partition = make_partition(
        payload["split"],
        dataset,
        partition_cfg,
        seed,
        num_clients=None if natural else int(payload["num_clients"]),
    )
    if partition.num_clients != int(payload["num_clients"]):
        raise ValueError("partition size disagrees with num_clients")
    index = partition.indices[partition_id]
    x = dataset.train.x[index]
    y = dataset.train.y[index]
    x.setflags(write=False)
    y.setflags(write=False)
    return x, y


@dataclass(frozen=True)
class AttackSpec:
    """Byzantine behaviour (simulation only): which clients attack and how."""

    kind: str
    byzantine_ids: tuple[int, ...]
    scale_factor: float
    noise_std: float


SECRET_KEY = "client-secret-seed"
"""``Context.node_config`` key of the client-held DP secret. It lives only in the client's own
node configuration and is never put into any ``Message``."""

_WARNED_NO_SECRET = False


def client_dp_secret(context: Context) -> int:
    """The client's secret seed for DP randomness (noise and Poisson sampling).

    Read from ``context.node_config["client-secret-seed"]``, which the in-process harness sets
    for reproducibility. If it is absent (another engine, a real deployment) a fresh secret is
    drawn from operating-system entropy for every call and a one-time warning says the run is
    not reproducible. The server-sent ``seed`` is never used as a fallback.
    """
    global _WARNED_NO_SECRET
    if SECRET_KEY in context.node_config:
        return int(context.node_config[SECRET_KEY])
    if not _WARNED_NO_SECRET:
        LOGGER.warning(
            "No %r in node_config: drawing the DP secret from OS entropy; this DP run is "
            "therefore not reproducible.",
            SECRET_KEY,
        )
        _WARNED_NO_SECRET = True
    return secrets.randbits(32)


def make_client_app(source: ClientDataSource, attack: AttackSpec | None) -> ClientApp:
    """Build the ``ClientApp`` for one federated condition."""
    app = ClientApp()

    def _check_partition(context: Context) -> int:
        partition_id = int(context.node_config["partition-id"])
        num_partitions = int(context.node_config["num-partitions"])
        if num_partitions != source.num_clients:
            raise ValueError(
                f"node_config num-partitions={num_partitions} but the split has "
                f"{source.num_clients} clients"
            )
        return partition_id

    @app.train()
    def train(msg: Message, context: Context) -> Message:
        partition_id = _check_partition(context)
        spec = spec_from_config(msg.content[CONFIG_KEY])
        received = msg.content[ARRAYS_KEY]
        keys = list(received.keys())
        global_weights = arrayrecord_to_weights(received)
        x, y = source.client_rows(partition_id)
        attacking = attack is not None and partition_id in attack.byzantine_ids
        if attacking and attack.kind in DATA_ATTACKS:
            y = label_flip(y, spec.num_classes)
        dp_secret = client_dp_secret(context) if spec.dp_mode != "none" else None
        result = train_local(global_weights, x, y, spec, partition_id, dp_secret=dp_secret)
        weights = result.weights
        update_norm = result.update_norm
        if attacking:
            weights = apply_update_attack(
                attack.kind,
                weights,
                global_weights,
                attack.scale_factor,
                attack.noise_std,
                spec.seed,
                partition_id,
                spec.server_round,
            )
            update_norm = float(
                np.sqrt(sum(float(np.sum((w - g).astype(np.float64) ** 2)) for w, g in zip(weights, global_weights, strict=True)))  # fmt: skip
            )
        values = {
            "partition-id": partition_id,
            "num-examples": int(result.num_examples),
            "train-loss": float(result.train_loss),
            "steps": int(result.steps),
            "sample-rate": float(result.sample_rate),
            "noise-multiplier": float(result.noise_multiplier),
            "epsilon": float(result.epsilon),
            "delta": float(result.delta),
            "update-norm": float(update_norm),
            "empty-batches": int(result.empty_batches),
        }
        allowed = REPLY_METRIC_KEYS if spec.dp_mode == "none" else DP_REPLY_METRIC_KEYS
        metrics = MetricRecord({key: values[key] for key in allowed})
        content = RecordDict(
            {ARRAYS_KEY: weights_to_arrayrecord(weights, keys), METRICS_KEY: metrics}
        )
        return Message(content, reply_to=msg)

    @app.evaluate()
    def evaluate(msg: Message, context: Context) -> Message:
        # Evaluates the global model on the client's OWN training rows: an un-noised statistic
        # of private rows. It only runs if federation.fraction_evaluate > 0 (default 0).
        partition_id = _check_partition(context)
        config = msg.content[CONFIG_KEY]
        weights = arrayrecord_to_weights(msg.content[ARRAYS_KEY])
        x, y = source.client_rows(partition_id)
        scores = evaluate_local(
            weights, x, y, [int(h) for h in config["hidden-sizes"]], int(config["num-classes"])
        )
        metrics = MetricRecord(
            {
                "partition-id": partition_id,
                "num-examples": int(scores["num_examples"]),
                "eval-loss": float(scores["loss"]),
                "eval-accuracy": float(scores["accuracy"]),
            }
        )
        return Message(RecordDict({METRICS_KEY: metrics}), reply_to=msg)

    return app
