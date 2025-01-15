"""Federated learning with differential privacy.

The method follows the federated learning approach with differential privacy described by
Zhang et al. (2025), arXiv 2504.00282. This package fixes the dataset, client count, ε, model
and hyperparameters itself and labels every such choice ``[Assumption — not in paper]``:

* primary data: UCI Adult / Census Income (1994 U.S. Census extraction), three workclass
  clients (Private / Government / Self-employed, simulated domains of one public dataset)
  plus an IID control — :mod:`.datasets.adult`, :mod:`.partition`;
* Flower 1.44 message API: :mod:`.client_app`, :mod:`.server_app`, :mod:`.strategy`, run in
  one process by :mod:`.inprocess` (a real ``flwr.serverapp.Grid`` subclass, no Ray);
  raw rows never enter a Flower ``Message``;
* sample-count-weighted FedAvg (the default; the unweighted mean is kept as an ablation)
  and robust rules — :mod:`.aggregation`;
* per-client record-level DP-SGD with Opacus and RDP accounting over all rounds —
  :mod:`.privacy` (read its docstring for exactly what the (ε, δ) covers);
* a secure-aggregation masking SIMULATION that is not cryptographically secure —
  :mod:`.secure_agg`; Byzantine attacks — :mod:`.attacks`;
* baselines behind a name -> builder registry (:mod:`.baselines`), evaluation,
  communication accounting and the benchmark runner.
"""

from __future__ import annotations

from .aggregation import (
    aggregate,
    coordinate_median,
    trimmed_mean,
    unweighted_average,
    weighted_average,
)
from .baselines import (
    BASELINE_METHODS,
    BASELINE_REGISTRY,
    baseline_predictions,
    register_baseline,
)
from .client import LocalTrainingSpec, LocalTrainResult, evaluate_local, train_local
from .config import FederatedConfig, load_federated_config
from .datasets import FederatedDataset, TabularSplit, load_dataset
from .evaluation import Evaluation, evaluate_predictions
from .model import MLP, build_model, get_weights, initial_weights, set_weights
from .partition import Partition, make_partition, partition_statistics
from .privacy import GUARANTEE, calibrate_noise_multiplier, dp_schedule, epsilon_spent
from .secure_agg import IMPLEMENTED_VS_STUBBED, PairwiseMaskingSimulator, SecureAggregator

__all__ = [
    "BASELINE_METHODS",
    "BASELINE_REGISTRY",
    "GUARANTEE",
    "IMPLEMENTED_VS_STUBBED",
    "MLP",
    "Evaluation",
    "FederatedConfig",
    "FederatedDataset",
    "LocalTrainResult",
    "LocalTrainingSpec",
    "PairwiseMaskingSimulator",
    "Partition",
    "SecureAggregator",
    "TabularSplit",
    "aggregate",
    "baseline_predictions",
    "build_model",
    "calibrate_noise_multiplier",
    "coordinate_median",
    "dp_schedule",
    "epsilon_spent",
    "evaluate_local",
    "evaluate_predictions",
    "get_weights",
    "initial_weights",
    "load_dataset",
    "load_federated_config",
    "make_partition",
    "partition_statistics",
    "register_baseline",
    "set_weights",
    "train_local",
    "trimmed_mean",
    "unweighted_average",
    "weighted_average",
]
