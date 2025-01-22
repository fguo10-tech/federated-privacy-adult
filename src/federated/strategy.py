"""Flower strategy whose aggregation calls :mod:`src.federated.aggregation`.

:class:`ConfigurableFedAvg` derives from Flower's message-API ``FedAvg``
(``flwr.serverapp.strategy.FedAvg``, flwr 1.44). It keeps Flower's node sampling, message
construction and reply validation, and replaces only the arithmetic of ``aggregate_train``:

* replies are **sorted by partition id** before aggregation, so the result does not depend on
  arrival order (Flower's ``sample_nodes`` draws with the global ``random`` module);
* the parameter arrays are combined by the function named in ``federation.aggregation_rule``
  — ``weighted`` (sample-count-weighted FedAvg, the default; equal to Flower's built-in
  FedAvg aggregation), ``unweighted`` (the unweighted mean, kept as an ablation),
  ``coordinate_median`` or ``trimmed_mean``;
* client-side evaluation is off by default (``federation.fraction_evaluate: 0``): Flower's
  ``configure_evaluate`` then returns no message and ``aggregate_evaluate`` returns ``None``;
* per-client training metrics (only aggregates: ``num-examples``, loss (non-DP only), ε, noise
  multiplier, sample rate, update norm) are kept in :attr:`client_history` for the result files, and the
  bytes of every message carrying model weights are counted in :attr:`ledger`.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable

from flwr.app import Array, ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg

from src.common.logging_utils import get_logger

from .aggregation import aggregate
from .communication import CommunicationLedger

LOGGER = get_logger(__name__)

PARTITION_KEY = "partition-id"
NUM_EXAMPLES_KEY = "num-examples"


class ClientFailure(RuntimeError):
    """A client replied with an error; experiments fail loudly instead of silently shrinking."""


class ConfigurableFedAvg(FedAvg):
    """Flower ``FedAvg`` with a configurable, order-independent aggregation rule."""

    def __init__(
        self,
        *,
        aggregation_rule: str,
        trimmed_mean_beta: float,
        num_clients: int,
        fraction_evaluate: float,
        fail_on_client_error: bool,
    ) -> None:
        super().__init__(
            fraction_train=1.0,
            fraction_evaluate=fraction_evaluate,
            min_train_nodes=num_clients,
            min_evaluate_nodes=num_clients if fraction_evaluate > 0 else 0,
            min_available_nodes=num_clients,
            weighted_by_key=NUM_EXAMPLES_KEY,
        )
        self.aggregation_rule = aggregation_rule
        self.trimmed_mean_beta = float(trimmed_mean_beta)
        self.fail_on_client_error = fail_on_client_error
        self.client_history: dict[int, list[dict[str, float]]] = {}
        self.client_eval_history: dict[int, list[dict[str, float]]] = {}
        self.ledger = CommunicationLedger()

    # ---- downlink accounting -------------------------------------------------------------
    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Flower's sampling and messages; additionally count the bytes sent."""
        messages = list(super().configure_train(server_round, arrays, config, grid))
        if messages:
            self.ledger.record_down(server_round, messages[0].content, len(messages))
        return messages

    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Flower's evaluation messages; the global model they carry is counted as downlink."""
        messages = list(super().configure_evaluate(server_round, arrays, config, grid))
        if messages:
            self.ledger.record_down(server_round, messages[0].content, len(messages))
        return messages

    # ---- aggregation ---------------------------------------------------------------------
    def _valid_sorted(self, replies: Iterable[Message], is_train: bool) -> list[Message]:
        valid, errors = self._check_and_log_replies(list(replies), is_train=is_train)
        if errors and self.fail_on_client_error:
            reasons = "; ".join(str(msg.error.reason) for msg in errors)
            raise ClientFailure(f"{len(errors)} client(s) failed: {reasons}")
        return sorted(valid, key=_partition_id)

    def aggregate_train(
        self, server_round: int, replies: Iterable[Message]
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """Aggregate with ``aggregation_rule`` after sorting replies by partition id."""
        valid = self._valid_sorted(replies, is_train=True)
        if not valid:
            return None, None
        contents = [msg.content for msg in valid]
        records = [next(iter(content.array_records.values())) for content in contents]
        keys = list(records[0].keys())
        updates = [record.to_numpy_ndarrays() for record in records]
        metrics = [next(iter(content.metric_records.values())) for content in contents]
        counts = [int(metric[NUM_EXAMPLES_KEY]) for metric in metrics]
        ids = [int(metric[PARTITION_KEY]) for metric in metrics]
        combined = aggregate(
            self.aggregation_rule, updates, counts, ids, beta=self.trimmed_mean_beta
        )
        arrays = ArrayRecord(
            OrderedDict((key, Array(array)) for key, array in zip(keys, combined, strict=True))
        )
        for content in contents:
            self.ledger.record_up(server_round, content)
        self.client_history[server_round] = [
            {key: _scalar(value) for key, value in metric.items()} for metric in metrics
        ]
        aggregated_metrics = self.train_metrics_aggr_fn(contents, self.weighted_by_key)
        return arrays, aggregated_metrics

    def aggregate_evaluate(
        self, server_round: int, replies: Iterable[Message]
    ) -> MetricRecord | None:
        """Flower's weighted metric average; per-client values are kept as well."""
        valid = self._valid_sorted(replies, is_train=False)
        if not valid:
            return None
        contents = [msg.content for msg in valid]
        metrics = [next(iter(content.metric_records.values())) for content in contents]
        self.client_eval_history[server_round] = [
            {key: _scalar(value) for key, value in metric.items()} for metric in metrics
        ]
        return self.evaluate_metrics_aggr_fn(contents, self.weighted_by_key)


def _partition_id(message: Message) -> int:
    metric = next(iter(message.content.metric_records.values()))
    return int(metric[PARTITION_KEY])


def _scalar(value) -> float:
    if isinstance(value, list):
        raise TypeError("per-client metrics must be scalars")
    return float(value)
