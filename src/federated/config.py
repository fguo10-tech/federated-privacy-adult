"""Configuration dataclasses of the federated track and :func:`load_federated_config`.

Every field is required and has no default: values come only from the YAML files under
``configs/`` (``federated_adult.yaml``, ``federated_fedavg.yaml``, ``federated_dp.yaml``,
``optional/federated_femnist.yaml``), where each key is commented with its meaning, unit and
provenance. Nearly every value (dataset, client count, ε, model, hyperparameters) is
``[Assumption — not in paper]``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from src.common.config import from_dict, load_config
from src.common.logging_utils import get_logger

LOGGER = get_logger(__name__)

AggregationRule = Literal["weighted", "unweighted", "coordinate_median", "trimmed_mean"]
AttackKind = Literal["none", "sign_flip", "scale", "gaussian_noise", "label_flip"]


@dataclass(frozen=True)
class ExperimentConfig:
    """Run identity: name, seeds and default output folder."""

    name: str
    seeds: list[int]
    output_dir: str


@dataclass(frozen=True)
class ScalingRuleConfig:
    """``fixed_public`` transform of one continuous Adult column."""

    log1p: bool
    divisor: float


@dataclass(frozen=True)
class AdultConfig:
    """Where the UCI Adult files are and how they are preprocessed."""

    data_dir: str
    checksum_file: str
    verify_checksums: bool
    expect_full_size: bool
    scaling: Literal["fixed_public", "train_fitted"]
    fixed_public: dict[str, ScalingRuleConfig]
    age_bands: list[list[int]]


@dataclass(frozen=True)
class SyntheticDomainsConfig:
    """Toy three-domain generator (synthetic numbers only)."""

    domain_names: list[str]
    train_rows: list[int]
    test_rows: list[int]
    num_features: int
    feature_shift: float
    class_separation: float
    positive_rates: list[float]
    label_noise: float


@dataclass(frozen=True)
class FemnistConfig:
    """Optional LEAF-format FEMNIST reader settings."""

    data_dir: str
    max_writers: int
    min_samples_per_writer: int
    test_fraction: float


@dataclass(frozen=True)
class DatasetConfig:
    """Which dataset to use; exactly the block named by ``kind`` must be non-null."""

    kind: Literal["adult", "synthetic", "femnist"]
    adult: AdultConfig | None
    synthetic: SyntheticDomainsConfig | None
    femnist: FemnistConfig | None


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation set and fairness breakdown."""

    eval_on: Literal["test", "validation"]
    validation_fraction: float
    group_attributes: list[str]


@dataclass(frozen=True)
class ModelConfig:
    """MLP architecture (ReLU, no BatchNorm)."""

    hidden_sizes: list[int]


@dataclass(frozen=True)
class TrainingConfig:
    """Local and centralised SGD settings."""

    rounds: int
    local_epochs: int
    batch_size: int
    learning_rate: float
    momentum: float
    weight_decay: float
    centralised_epochs: int


@dataclass(frozen=True)
class FederationConfig:
    """Server-side settings."""

    aggregation_rule: AggregationRule
    trimmed_mean_beta: float
    fraction_evaluate: float
    engine: Literal["inprocess", "ray"]


@dataclass(frozen=True)
class DPSweepEntry:
    """One differential-privacy condition of the sweep."""

    name: str
    mode: Literal["target_epsilon", "fixed_noise"]
    target_epsilon: float | None
    noise_multiplier: float | None
    max_grad_norm: float


@dataclass(frozen=True)
class DPConfig:
    """DP-SGD accounting settings and the sweep of DP conditions."""

    delta: float
    accountant: Literal["rdp"]
    epsilon_tolerance: float
    sweep: list[DPSweepEntry]


@dataclass(frozen=True)
class QuantitySkewConfig:
    """Client-size distribution of the ``quantity_skew`` partition."""

    kind: Literal["dirichlet", "power_law"]
    alpha: float
    exponent: float


@dataclass(frozen=True)
class PartitionConfig:
    """Client splits of the main suite and of the partition ablation."""

    splits: list[str]
    ablations: list[str]
    label_skew_alpha: float
    quantity_skew: QuantitySkewConfig
    min_client_rows: int
    max_resample_attempts: int


@dataclass(frozen=True)
class SvmConfig:
    """scikit-learn ``SVC`` settings."""

    kernel: str
    C: float
    gamma: str
    cache_size_mb: float


@dataclass(frozen=True)
class ForestConfig:
    """scikit-learn ``RandomForestClassifier`` settings."""

    n_estimators: int
    max_depth: int | None
    min_samples_leaf: int


@dataclass(frozen=True)
class TreeConfig:
    """scikit-learn ``DecisionTreeClassifier`` settings."""

    max_depth: int | None
    min_samples_leaf: int


@dataclass(frozen=True)
class BaselinesConfig:
    """Which centralised baselines run in the ``main`` suite and their settings."""

    methods: list[str]
    svm: SvmConfig
    random_forest: ForestConfig
    decision_tree: TreeConfig


@dataclass(frozen=True)
class AttackConfig:
    """Byzantine behaviour of the attacking clients."""

    kind: AttackKind
    scale_factor: float
    noise_std: float


@dataclass(frozen=True)
class RobustnessConfig:
    """The ``robustness`` suite: IID clients, some Byzantine, several aggregation rules."""

    num_clients: int
    num_byzantine: int
    attack: AttackConfig
    aggregation_rules: list[str]


@dataclass(frozen=True)
class AggregationSuiteConfig:
    """The ``aggregation`` suite: sample-weighted FedAvg versus the unweighted mean."""

    split: str
    rules: list[str]


@dataclass(frozen=True)
class RuntimeConfig:
    """Process-level settings (not model hyperparameters)."""

    workers: int
    torch_threads: int


@dataclass(frozen=True)
class FederatedConfig:
    """Complete configuration of one federated benchmark."""

    experiment: ExperimentConfig
    dataset: DatasetConfig
    evaluation: EvaluationConfig
    model: ModelConfig
    training: TrainingConfig
    federation: FederationConfig
    dp: DPConfig
    partitions: PartitionConfig
    baselines: BaselinesConfig
    robustness: RobustnessConfig
    aggregation_suite: AggregationSuiteConfig
    runtime: RuntimeConfig


def validate(config: FederatedConfig) -> None:
    """Cross-field checks that a type annotation cannot express."""
    kind = config.dataset.kind
    if getattr(config.dataset, kind) is None:
        raise ValueError(f"dataset.kind is {kind!r} but dataset.{kind} is null")
    for entry in config.dp.sweep:
        if entry.mode == "target_epsilon" and (entry.target_epsilon or 0) <= 0:
            raise ValueError(f"dp.sweep {entry.name}: target_epsilon must be > 0")
        if entry.mode == "fixed_noise" and (entry.noise_multiplier or 0) <= 0:
            raise ValueError(f"dp.sweep {entry.name}: noise_multiplier must be > 0")
    if not 0.0 <= config.federation.trimmed_mean_beta < 0.5:
        raise ValueError("federation.trimmed_mean_beta must be in [0, 0.5)")
    if config.robustness.num_byzantine >= config.robustness.num_clients:
        raise ValueError("robustness.num_byzantine must be smaller than num_clients")
    rob = config.robustness
    if "trimmed_mean" in rob.aggregation_rules and rob.num_byzantine > 0:
        trimmed = int(config.federation.trimmed_mean_beta * rob.num_clients)
        if trimmed < rob.num_byzantine:
            LOGGER.warning(
                "federation.trimmed_mean_beta=%g trims %d value(s) per side among %d clients, "
                "fewer than the %d Byzantine clients: trimmed_mean cannot remove every attacker "
                "by construction. Set beta >= num_byzantine / num_clients = %g.",
                config.federation.trimmed_mean_beta,
                trimmed,
                rob.num_clients,
                rob.num_byzantine,
                rob.num_byzantine / rob.num_clients,
            )
    names = [entry.name for entry in config.dp.sweep]
    if len(set(names)) != len(names):
        raise ValueError("dp.sweep names must be unique")


def load_federated_config(
    path: str | Path,
    overrides: Iterable[str] | None = None,
    smoke: bool = False,
) -> tuple[FederatedConfig, dict[str, Any]]:
    """Load, merge and strictly validate a federated config.

    Returns the dataclass and the merged plain dict (saved as ``config_used.yaml``).
    """
    raw = load_config(path, overrides=overrides, smoke=smoke)
    config = from_dict(FederatedConfig, raw)
    validate(config)
    return config, raw
