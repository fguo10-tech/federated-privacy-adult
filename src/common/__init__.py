"""Shared utilities: configuration, seeding, logging, result files, metrics, plotting.

Nothing in this package knows about federated learning; ``src.federated`` and the experiment
scripts depend on it, never the other way round.
"""

from .config import (
    ConfigError,
    config_hash,
    deep_update,
    from_dict,
    get_path,
    load_config,
    load_yaml,
    parse_overrides,
    save_yaml,
    to_dict,
)
from .io import (
    ensure_dir,
    environment_info,
    read_csv,
    read_json,
    sanitise_command,
    save_run_metadata,
    write_csv,
    write_json,
)
from .labels import (
    ASSUMPTION,
    NOT_COMPARABLE,
    PAPER_REPORTED,
    REPRODUCED,
    TARGET_TBD,
    TBD,
)
from .logging_utils import get_logger, setup_logging
from .metrics import (
    classification_metrics,
    confusion_matrix,
    constant_predictor_weighted_scores,
    fairness_gap,
    format_mean_std,
    mean_std,
    summarise_by_seed,
)
from .seeding import derive_seed, rng_from_seed, seed_everything

__all__ = [
    "ASSUMPTION",
    "NOT_COMPARABLE",
    "PAPER_REPORTED",
    "REPRODUCED",
    "TARGET_TBD",
    "TBD",
    "ConfigError",
    "classification_metrics",
    "config_hash",
    "confusion_matrix",
    "constant_predictor_weighted_scores",
    "deep_update",
    "derive_seed",
    "ensure_dir",
    "environment_info",
    "fairness_gap",
    "format_mean_std",
    "from_dict",
    "get_logger",
    "get_path",
    "load_config",
    "load_yaml",
    "mean_std",
    "parse_overrides",
    "read_csv",
    "read_json",
    "rng_from_seed",
    "sanitise_command",
    "save_run_metadata",
    "save_yaml",
    "seed_everything",
    "setup_logging",
    "summarise_by_seed",
    "to_dict",
    "write_csv",
    "write_json",
]
