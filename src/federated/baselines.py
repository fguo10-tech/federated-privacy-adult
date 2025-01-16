"""Baselines: centralised MLPs, local-only training, scikit-learn models, majority class.

Every baseline uses the same preprocessing and is evaluated on the same shared evaluation
set as the federated conditions (:func:`src.federated.evaluation.evaluate_predictions`).
All baseline settings are in YAML (``baselines.*``) and are
``[Assumption — not in paper]``.

Centralised baselines live in :data:`BASELINE_REGISTRY` (name -> builder). A builder has the
signature ``builder(config, dataset, seed) -> (predictions, train_loss, train_rows)``; adding a
baseline is one decorated function here plus its name in ``baselines.methods`` in YAML (and,
if it has settings, a block in the config dataclasses).

``local_only`` is not a registry entry: it depends on the client split and writes per-round
and per-client rows, so it runs as its own task (:func:`run_local_only`).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from .client import train_local
from .config import FederatedConfig
from .datasets import FederatedDataset, load_dataset
from .evaluation import Evaluation, evaluate_predictions
from .model import initial_weights
from .partition import make_partition
from .runner import (
    Task,
    TaskOutput,
    _domain_columns,
    _final_row,
    _group_rows,
    _predict,
    _round_metrics,
    _with,
    run_label,
    training_spec,
)

BaselineResult = tuple[np.ndarray, float, int]
BaselineBuilder = Callable[[FederatedConfig, FederatedDataset, int], BaselineResult]

BASELINE_REGISTRY: dict[str, BaselineBuilder] = {}
"""Centralised baselines by name, in registration order."""


def register_baseline(name: str) -> Callable[[BaselineBuilder], BaselineBuilder]:
    """Decorator adding a builder to :data:`BASELINE_REGISTRY` under ``name``."""

    def decorator(builder: BaselineBuilder) -> BaselineBuilder:
        if name in BASELINE_REGISTRY:
            raise ValueError(f"baseline {name!r} is already registered")
        BASELINE_REGISTRY[name] = builder
        return builder

    return decorator


# --------------------------------------------------------------------------------------
# local-only training
# --------------------------------------------------------------------------------------


def run_local_only(config: FederatedConfig, task: Task) -> TaskOutput:
    """Each client trains alone for the same rounds × local epochs; no federation.

    Client ``k`` uses the same initial model and the same per-round seeds
    (``derive_seed(seed, "train", k, round)``) as in the federated run, and its optimizer is
    rebuilt every round exactly as a stateless federated client's is.
    """
    dataset = load_dataset(config.dataset, config.evaluation, task.seed)
    natural = task.split == dataset.natural_split_name
    partition = make_partition(
        task.split, dataset, config.partitions, task.seed, None if natural else task.num_clients
    )
    hidden = list(config.model.hidden_sizes)
    rounds = config.training.rounds
    spec = _with(
        training_spec(
            config,
            task.seed,
            None,
            config.training.local_epochs,
            rounds,
            run_label=run_label(task),
        ),
        num_classes=dataset.num_classes,
    )
    weights0 = initial_weights(dataset.num_features, hidden, dataset.num_classes, task.seed)
    groups = config.evaluation.group_attributes
    per_round: list[dict[str, Any]] = []
    per_client: list[dict[str, Any]] = []
    final: list[dict[str, Any]] = []
    per_group: list[dict[str, Any]] = []
    initial_eval = evaluate_predictions(dataset, _predict(weights0, dataset, hidden), groups)
    round_evals: dict[int, list[Evaluation]] = {0: [initial_eval] * partition.num_clients}
    round_losses: dict[int, list[float]] = {}
    client_sizes = partition.sizes()
    for k, index in enumerate(partition.indices):
        weights = weights0
        x, y = dataset.train.x[index], dataset.train.y[index]
        name = partition.client_names[k]
        domain = dataset.domain_names[k] if k < len(dataset.domain_names) else ""
        last_loss = math.nan
        for server_round in range(1, rounds + 1):
            result = train_local(weights, x, y, _with(spec, server_round=server_round), k)
            weights = result.weights
            last_loss = result.train_loss
            evaluation = evaluate_predictions(dataset, _predict(weights, dataset, hidden), groups)
            round_evals.setdefault(server_round, []).append(evaluation)
            round_losses.setdefault(server_round, []).append(result.train_loss)
            per_round.append(
                {
                    "suite": task.suite,
                    "split": task.split,
                    "condition": f"local_only:{name}",
                    "seed": task.seed,
                    "round": server_round,
                    **_round_metrics(evaluation),
                    "train_loss": result.train_loss,
                    "bytes_up_cum": 0,
                    "bytes_down_cum": 0,
                    "epsilon_max": math.inf,
                    "delta": "",
                }
            )
            per_client.append(
                {
                    "suite": task.suite,
                    "split": task.split,
                    "condition": "local_only",
                    "seed": task.seed,
                    "round": server_round,
                    "client": name,
                    "domain": domain,
                    "n_train": result.num_examples,
                    "sample_rate": result.sample_rate,
                    "noise_multiplier": 0.0,
                    "epsilon": math.inf,
                    "update_norm": result.update_norm,
                    "train_loss": result.train_loss,
                    "local_eval_accuracy": "",
                    **_domain_columns(evaluation, domain),
                }
            )
        final_eval = round_evals[rounds][-1]
        final.append(
            _final_row(
                task,
                task.split,
                f"local_only:{name}",
                final_eval,
                train_loss=last_loss,
                rounds=rounds,
                bytes_total=0,
                bytes_up=0,
                bytes_down=0,
                epsilon_max=math.inf,
                delta="",
                train_rows=client_sizes[k],
                num_clients=1,
            )
        )
        per_group.extend(
            _group_rows(
                {"suite": task.suite, "split": task.split, "method": f"local_only:{name}", "seed": task.seed},  # fmt: skip
                final_eval,
            )
        )
    for server_round in range(0, rounds + 1):
        evals = round_evals[server_round]
        row = {
            "suite": task.suite,
            "split": task.split,
            "condition": "local_only",
            "seed": task.seed,
            "round": server_round,
        }
        for key in ("accuracy", "precision", "recall", "f1", "f1_macro"):
            row[key] = float(np.mean([e.overall[key] for e in evals]))
        losses = round_losses.get(server_round)
        row.update(
            train_loss=float(np.mean(losses)) if losses else "",
            bytes_up_cum=0,
            bytes_down_cum=0,
            epsilon_max=math.inf,
            delta="",
        )
        per_round.append(row)
    mean_row: dict[str, Any] = {
        "suite": task.suite,
        "split": task.split,
        "method": "local_only",
        "seed": task.seed,
    }
    numeric = ("accuracy", "precision", "recall", "f1", "f1_macro", "train_loss", "fairness_gap_accuracy", "fairness_gap_f1")  # fmt: skip
    for key in numeric:
        mean_row[key] = float(np.mean([row[key] for row in final]))
    mean_row.update(
        rounds=rounds,
        bytes_total=0,
        bytes_up=0,
        bytes_down=0,
        epsilon_max=math.inf,
        delta="",
        train_rows=int(np.mean(client_sizes)),
        num_clients=1,
    )
    final.append(mean_row)
    return TaskOutput(per_round, per_client, final, per_group, [])


# --------------------------------------------------------------------------------------
# centralised baselines
# --------------------------------------------------------------------------------------


def _centralised_mlp(
    config: FederatedConfig, dataset: FederatedDataset, seed: int, index: np.ndarray
) -> BaselineResult:
    """The federated network trained centrally on ``index`` rows for ``centralised_epochs``.

    One optimiser for all epochs, so SGD momentum carries across epochs; federated and
    local-only clients instead rebuild their optimiser (and momentum) every round.
    """
    train = dataset.train
    hidden = list(config.model.hidden_sizes)
    spec = _with(
        training_spec(config, seed, None, config.training.centralised_epochs, 1, run_label=""),
        num_classes=dataset.num_classes,
        server_round=1,
    )
    weights0 = initial_weights(dataset.num_features, hidden, dataset.num_classes, seed)
    result = train_local(weights0, train.x[index], train.y[index], spec, client_id=-1)
    return _predict(result.weights, dataset, hidden), result.train_loss, int(len(index))


@register_baseline("centralised_mlp")
def centralised_mlp(
    config: FederatedConfig, dataset: FederatedDataset, seed: int
) -> BaselineResult:
    """Same MLP, all training rows pooled, including the rows excluded from the client split.

    Excluded rows (Adult workclass ``?``, ``Without-pay``, ``Never-worked``) belong to no
    client; they are used only by the baselines trained on all rows (this one, svm,
    random_forest, decision_tree, majority_class) and, on the test side, in the shared test set.
    """
    return _centralised_mlp(config, dataset, seed, np.arange(len(dataset.train)))


@register_baseline("centralised_mlp_client_rows")
def centralised_mlp_client_rows(
    config: FederatedConfig, dataset: FederatedDataset, seed: int
) -> BaselineResult:
    """Same MLP, only the rows that belong to some client (isolates the excluded rows)."""
    return _centralised_mlp(config, dataset, seed, np.flatnonzero(dataset.train.domain >= 0))


def _fit_predict(model: Any, dataset: FederatedDataset) -> BaselineResult:
    model.fit(dataset.train.x, dataset.train.y)
    predictions = np.asarray(model.predict(dataset.test.x), dtype=np.int64)
    return predictions, math.nan, len(dataset.train)


@register_baseline("svm")
def svm(config: FederatedConfig, dataset: FederatedDataset, seed: int) -> BaselineResult:
    """scikit-learn ``SVC`` on all training rows (``baselines.svm``)."""
    from sklearn.svm import SVC

    settings = config.baselines.svm
    model = SVC(
        kernel=settings.kernel,
        C=settings.C,
        gamma=settings.gamma,
        cache_size=settings.cache_size_mb,
        random_state=seed,
    )
    return _fit_predict(model, dataset)


@register_baseline("random_forest")
def random_forest(config: FederatedConfig, dataset: FederatedDataset, seed: int) -> BaselineResult:
    """scikit-learn ``RandomForestClassifier`` on all training rows (single-threaded)."""
    from sklearn.ensemble import RandomForestClassifier

    settings = config.baselines.random_forest
    model = RandomForestClassifier(
        n_estimators=settings.n_estimators,
        max_depth=settings.max_depth,
        min_samples_leaf=settings.min_samples_leaf,
        random_state=seed,
        n_jobs=1,
    )
    return _fit_predict(model, dataset)


@register_baseline("decision_tree")
def decision_tree(config: FederatedConfig, dataset: FederatedDataset, seed: int) -> BaselineResult:
    """scikit-learn ``DecisionTreeClassifier`` on all training rows."""
    from sklearn.tree import DecisionTreeClassifier

    settings = config.baselines.decision_tree
    model = DecisionTreeClassifier(
        max_depth=settings.max_depth,
        min_samples_leaf=settings.min_samples_leaf,
        random_state=seed,
    )
    return _fit_predict(model, dataset)


@register_baseline("majority_class")
def majority_class(config: FederatedConfig, dataset: FederatedDataset, seed: int) -> BaselineResult:
    """Always predict the most frequent training label (the trivial reference)."""
    majority = int(np.bincount(dataset.train.y, minlength=dataset.num_classes).argmax())
    return np.full(len(dataset.test), majority, dtype=np.int64), math.nan, len(dataset.train)


BASELINE_METHODS = tuple(BASELINE_REGISTRY)
"""Names of all registered centralised baselines."""


def baseline_predictions(
    config: FederatedConfig, dataset: FederatedDataset, method: str, seed: int
) -> BaselineResult:
    """(predictions on the evaluation set, training loss or NaN, training rows) of a baseline."""
    if method not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline {method!r}; registered: {BASELINE_METHODS}")
    return BASELINE_REGISTRY[method](config, dataset, seed)


def run_baselines(config: FederatedConfig, task: Task) -> TaskOutput:
    """All configured baselines for one seed; rows are repeated for every reported split.

    Baselines do not depend on the client split, so they are trained once per seed and their
    rows are written again under each split of the main suite (identical values) to make every
    split's table complete. Every such row carries ``shared_across_splits`` (e.g.
    ``workclass;iid``) so a reader can tell the copies are one run.
    """
    dataset = load_dataset(config.dataset, config.evaluation, task.seed)
    shared = ";".join(task.report_splits)  # one run, copied under every split (SHARED_RUN_NOTE)
    final, per_group, per_client, timing = [], [], [], []
    for method in config.baselines.methods:
        if method not in BASELINE_REGISTRY:
            raise ValueError(f"Unknown baseline {method!r}; registered: {BASELINE_METHODS}")
        started = time.perf_counter()
        predictions, loss, rows = baseline_predictions(config, dataset, method, task.seed)
        evaluation = evaluate_predictions(dataset, predictions, config.evaluation.group_attributes)
        elapsed = time.perf_counter() - started
        for split in task.report_splits:
            timing.append({"suite": task.suite, "split": split, "condition": method, "seed": task.seed, "wall_time_s": elapsed, "shared_across_splits": shared})  # fmt: skip
            final.append(
                _final_row(
                    task,
                    split,
                    method,
                    evaluation,
                    train_loss=loss,
                    rounds=0,
                    bytes_total=0,
                    bytes_up=0,
                    bytes_down=0,
                    epsilon_max=math.inf,
                    delta="",
                    train_rows=rows,
                    num_clients=1,
                    shared_across_splits=shared,
                )
            )
            per_group.extend(
                _group_rows(
                    {"suite": task.suite, "split": split, "method": method, "seed": task.seed, "shared_across_splits": shared},  # fmt: skip
                    evaluation,
                )
            )
            for domain in dataset.domain_names:
                per_client.append(
                    {
                        "suite": task.suite,
                        "split": split,
                        "condition": method,
                        "seed": task.seed,
                        "round": 0,
                        "client": "",
                        "domain": domain,
                        "n_train": "",
                        "sample_rate": "",
                        "noise_multiplier": "",
                        "epsilon": "",
                        "update_norm": "",
                        "train_loss": "",
                        "local_eval_accuracy": "",
                        **_domain_columns(evaluation, domain),
                        "shared_across_splits": shared,
                    }
                )
    return TaskOutput([], per_client, final, per_group, timing)
