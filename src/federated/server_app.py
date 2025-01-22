"""Flower ``ServerApp`` (message API, flwr 1.44).

``@app.main()`` builds a :class:`~src.federated.strategy.ConfigurableFedAvg`, sends the
initial global model and the training configuration through the ``Grid`` it is given, and
calls ``strategy.start``. The same app runs unchanged on the in-process grid
(:mod:`src.federated.inprocess`, the default) and on Flower's Ray simulation engine.

Server-side evaluation (``evaluate_fn``) is the experimenter's measurement harness: after
every round it evaluates the global model on the shared evaluation set (the public test rows,
or validation rows when tuning). These per-round test numbers are logged for curves only and
never drive a choice. Results are handed back through a :class:`ServerResult` object captured
by the app (this process only), not through any Flower message.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

from .model import arrayrecord_to_weights, weights_to_arrayrecord
from .strategy import ConfigurableFedAvg

EvaluateFn = Callable[[int, list[np.ndarray]], dict[str, float]]


@dataclass(frozen=True)
class ServerSpec:
    """Everything the ServerApp needs; all values come from YAML."""

    num_rounds: int
    num_clients: int
    aggregation_rule: str
    trimmed_mean_beta: float
    fraction_evaluate: float
    train_config: dict[str, Any]
    evaluate_config: dict[str, Any]
    parameter_names: tuple[str, ...]


@dataclass
class ServerResult:
    """Filled in by the ServerApp when ``strategy.start`` returns."""

    final_weights: list[np.ndarray] | None = None
    strategy: ConfigurableFedAvg | None = None
    evaluations: dict[int, dict[str, float]] = field(default_factory=dict)
    train_metrics: dict[int, dict[str, float]] = field(default_factory=dict)


def make_server_app(
    spec: ServerSpec,
    initial_weights: list[np.ndarray],
    evaluate_fn: EvaluateFn | None,
    sink: ServerResult,
) -> ServerApp:
    """Build the ``ServerApp`` for one federated condition."""
    app = ServerApp()

    @app.main()
    def main(grid: Grid, context: Context) -> None:
        strategy = ConfigurableFedAvg(
            aggregation_rule=spec.aggregation_rule,
            trimmed_mean_beta=spec.trimmed_mean_beta,
            num_clients=spec.num_clients,
            fraction_evaluate=spec.fraction_evaluate,
            fail_on_client_error=True,
        )

        def server_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord | None:
            if evaluate_fn is None:
                return None
            scores = evaluate_fn(server_round, arrayrecord_to_weights(arrays))
            sink.evaluations[server_round] = dict(scores)
            return MetricRecord({k: float(v) for k, v in scores.items() if _is_number(v)})

        result = strategy.start(
            grid=grid,
            initial_arrays=weights_to_arrayrecord(initial_weights, spec.parameter_names),
            num_rounds=spec.num_rounds,
            train_config=ConfigRecord(dict(spec.train_config)),
            evaluate_config=ConfigRecord(dict(spec.evaluate_config)),
            evaluate_fn=server_evaluate,
        )
        sink.final_weights = arrayrecord_to_weights(result.arrays)
        sink.strategy = strategy
        sink.train_metrics = {
            int(r): {k: float(v) for k, v in record.items() if _is_number(v)}
            for r, record in result.train_metrics_clientapp.items()
        }

    return app


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
