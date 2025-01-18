"""UCI Adult / Census Income: download, checksum pinning, parsing and preprocessing.

Dataset
-------
Adult, also known as Census Income, UCI Machine Learning Repository. Extracted by Barry
Becker from the **1994** U.S. Census database; donated on 1996-04-30. Cite as: Becker, B. &
Kohavi, R. (1996). Adult [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5XW20. Licence CC BY 4.0.
Canonical dataset page: http://archive.ics.uci.edu/ml/datasets/Adult

48,842 rows: 32,561 training rows (``adult.data``) and 16,281 test rows (``adult.test``).
Binary target: income ``>50K`` (positive class, label 1) versus ``<=50K``; about 24% of rows
are positive, so accuracy alone is misleading and F1 / precision / recall of the positive
class are always reported next to it. The use of this dataset is
``[Assumption — not in paper]``.

File quirks handled here: ``adult.test`` starts with a comment line (``|1x3 Cross
validator``) and its labels end with a period (``>50K.``); every field has a leading space;
missing values are ``?``.

Preprocessing (the dataset card repeats these choices; see :data:`ATTRIBUTE_ROLES`)
------------------------------------------------------------------------------------
* ``age``, ``race`` and ``sex`` are **evaluation-only protected attributes**. They are never
  model inputs, for any baseline or federated run, and are kept next to the features only for
  the per-group fairness analysis. ``relationship`` (Husband / Wife) and ``marital-status``
  remain inputs and are **proxies for sex**, so removing ``sex`` from the inputs does not
  make the models sex-blind.
* ``fnlwgt`` (a census sampling weight) is not an input. ``education`` (string) is dropped as
  redundant with ``education-num``, its ordinal encoding.
* Categorical inputs are one-hot encoded with a fixed vocabulary copied from the published
  schema ``adult.names`` (:data:`CATEGORY_VOCABULARY`) plus an explicit ``Unknown`` column for
  ``?``. Nothing is imputed, and encoding needs no statistic from any client.
* Continuous inputs are scaled in one of two modes (``dataset.adult.scaling``):
  ``fixed_public`` applies data-independent transforms with constants from YAML (optional
  ``log1p``, then division by a constant); nothing is fitted, so nothing can leak from the
  test set or across the client boundary. ``train_fitted`` standardises with mean / std of the
  pooled training rows; in a real federation that needs one extra aggregate release that the
  DP accounting in :mod:`src.federated.privacy` does **not** cover.

Clients built from this data (see :mod:`src.federated.partition`) are simulated domains
derived from one public census dataset, not real employers, hospitals or banks.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.common.logging_utils import get_logger, setup_logging

if TYPE_CHECKING:
    from src.federated.config import AdultConfig, EvaluationConfig

    from . import FederatedDataset

LOGGER = get_logger(__name__)

CANONICAL_PAGE = "http://archive.ics.uci.edu/ml/datasets/Adult"
DOI = "https://doi.org/10.24432/C5XW20"
CITATION = (
    "Becker, B. & Kohavi, R. (1996). Adult [Dataset]. UCI Machine Learning Repository. "
    "https://doi.org/10.24432/C5XW20"
)
EXTRACTION_YEAR = 1994
BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
FILE_NAMES = ("adult.data", "adult.test", "adult.names")
DOWNLOAD_TIMEOUT_S = 60.0
"""Network timeout per file. An operational setting of the download script, not a model
hyperparameter, so it lives here rather than in YAML."""

EXPECTED_TRAIN_ROWS = 32_561
EXPECTED_TEST_ROWS = 16_281

COLUMNS = (
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
)
"""Column order of ``adult.data`` and ``adult.test`` (14 attributes + the income label)."""

MISSING = "?"
UNKNOWN = "Unknown"
POSITIVE_LABEL = ">50K"
NEGATIVE_LABEL = "<=50K"

CATEGORY_VOCABULARY: dict[str, tuple[str, ...]] = {
    "workclass": (
        "Private",
        "Self-emp-not-inc",
        "Self-emp-inc",
        "Federal-gov",
        "Local-gov",
        "State-gov",
        "Without-pay",
        "Never-worked",
    ),
    "education": (
        "Bachelors",
        "Some-college",
        "11th",
        "HS-grad",
        "Prof-school",
        "Assoc-acdm",
        "Assoc-voc",
        "9th",
        "7th-8th",
        "12th",
        "Masters",
        "1st-4th",
        "10th",
        "Doctorate",
        "5th-6th",
        "Preschool",
    ),
    "marital-status": (
        "Married-civ-spouse",
        "Divorced",
        "Never-married",
        "Separated",
        "Widowed",
        "Married-spouse-absent",
        "Married-AF-spouse",
    ),
    "occupation": (
        "Tech-support",
        "Craft-repair",
        "Other-service",
        "Sales",
        "Exec-managerial",
        "Prof-specialty",
        "Handlers-cleaners",
        "Machine-op-inspct",
        "Adm-clerical",
        "Farming-fishing",
        "Transport-moving",
        "Priv-house-serv",
        "Protective-serv",
        "Armed-Forces",
    ),
    "relationship": (
        "Wife",
        "Own-child",
        "Husband",
        "Not-in-family",
        "Other-relative",
        "Unmarried",
    ),
    "race": ("White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"),
    "sex": ("Female", "Male"),
    "native-country": (
        "United-States",
        "Cambodia",
        "England",
        "Puerto-Rico",
        "Canada",
        "Germany",
        "Outlying-US(Guam-USVI-etc)",
        "India",
        "Japan",
        "Greece",
        "South",
        "China",
        "Cuba",
        "Iran",
        "Honduras",
        "Philippines",
        "Italy",
        "Poland",
        "Jamaica",
        "Vietnam",
        "Mexico",
        "Portugal",
        "Ireland",
        "France",
        "Dominican-Republic",
        "Laos",
        "Ecuador",
        "Taiwan",
        "Haiti",
        "Columbia",
        "Hungary",
        "Guatemala",
        "Nicaragua",
        "Scotland",
        "Thailand",
        "Yugoslavia",
        "El-Salvador",
        "Trinadad&Tobago",
        "Peru",
        "Hong",
        "Holand-Netherlands",
    ),
}
"""Category values exactly as listed in the published schema ``adult.names`` (spelling and
order included, e.g. ``Trinadad&Tobago``). ``tests/test_federated_adult.py`` checks this
against the downloaded ``adult.names``."""

CATEGORICAL_INPUTS = ("workclass", "marital-status", "occupation", "relationship", "native-country")
CONTINUOUS_INPUTS = ("education-num", "capital-gain", "capital-loss", "hours-per-week")
PROTECTED_ATTRIBUTES = ("age", "race", "sex")

ATTRIBUTE_ROLES: dict[str, tuple[str, str]] = {
    "age": ("evaluation-only", "protected attribute; used only for per-age-band fairness"),
    "workclass": (
        "input",
        "one-hot, published vocabulary + Unknown; also defines the three client domains",
    ),
    "fnlwgt": ("excluded", "census sampling weight, not a property of the person"),
    "education": ("excluded", "redundant with education-num (its ordinal encoding)"),
    "education-num": ("input", "continuous, scaled"),
    "marital-status": (
        "input",
        "one-hot, published vocabulary + Unknown; proxy for sex (Married-civ-spouse)",
    ),
    "occupation": ("input", "one-hot, published vocabulary + Unknown"),
    "relationship": (
        "input",
        "one-hot, published vocabulary + Unknown; proxy for sex (Husband / Wife)",
    ),
    "race": ("evaluation-only", "protected attribute; used only for per-group fairness"),
    "sex": ("evaluation-only", "protected attribute; used only for per-group fairness"),
    "capital-gain": ("input", "continuous, scaled (log1p in fixed_public mode)"),
    "capital-loss": ("input", "continuous, scaled (log1p in fixed_public mode)"),
    "hours-per-week": ("input", "continuous, scaled"),
    "native-country": ("input", "one-hot, published vocabulary + Unknown"),
}
"""Role of each of the 14 attributes (input / evaluation-only / excluded) with the reason."""

TARGET = "income"

ATTRIBUTE_TYPES: dict[str, str] = {
    "age": "integer (years)",
    "workclass": "categorical",
    "fnlwgt": "integer (sampling weight)",
    "education": "categorical",
    "education-num": "integer (ordinal 1-16)",
    "marital-status": "categorical",
    "occupation": "categorical",
    "relationship": "categorical",
    "race": "categorical",
    "sex": "categorical",
    "capital-gain": "integer (US dollars)",
    "capital-loss": "integer (US dollars)",
    "hours-per-week": "integer (hours)",
    "native-country": "categorical",
    TARGET: "binary label",
}
"""Raw type of every column of ``adult.data`` / ``adult.test``."""

_ROLE_LABELS = {
    "input": "model input",
    "evaluation-only": "evaluation-only",
    "excluded": "excluded",
}


def _default_encoding(name: str) -> str:
    if name == TARGET:
        return "`>50K` -> 1 (positive class), `<=50K` -> 0; trailing period of adult.test removed"
    role = ATTRIBUTE_ROLES[name][0]
    if role == "excluded":
        return "dropped before preprocessing"
    if role == "evaluation-only":
        if name == "age":
            return "not a feature; grouped into the YAML age bands for fairness reports"
        return "not a feature; kept as a group label for fairness reports"
    if name in CATEGORICAL_INPUTS:
        size = len(CATEGORY_VOCABULARY[name])
        return f"one-hot over the {size} published values + `Unknown` for `?` ({size + 1} columns)"
    if name in ("capital-gain", "capital-loss"):
        return "fixed_public: log1p, then / divisor from YAML (no fitted statistic)"
    return "fixed_public: value / divisor from YAML (no fitted statistic)"


def attribute_roles_markdown() -> str:
    """GitHub Markdown table of the 14 Adult attributes plus the target, generated from code.

    Columns: attribute, type, role (model input / evaluation-only / excluded / target), how it
    is encoded or scaled in the default ``fixed_public`` mode, and the reason.
    """
    lines = [
        "| Attribute | Type | Role | Encoding in the default mode (`fixed_public`) | Reason |",
        "|---|---|---|---|---|",
    ]
    for name in COLUMNS:
        if name == TARGET:
            role, reason = "target", "binary income label (>50K is about 24% of rows)"
        else:
            raw_role, reason = ATTRIBUTE_ROLES[name]
            role = _ROLE_LABELS[raw_role]
        cells = [f"`{name}`", ATTRIBUTE_TYPES[name], role, _default_encoding(name), reason]
        lines.append("| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |")
    return "\n".join(lines) + "\n"


WORKCLASS_DOMAINS: dict[str, tuple[str, ...]] = {
    "Private": ("Private",),
    "Government": ("Federal-gov", "State-gov", "Local-gov"),
    "Self-employed": ("Self-emp-inc", "Self-emp-not-inc"),
}
"""Client domains of the workclass split, in client order. Rows whose workclass is ``?``,
``Without-pay`` or ``Never-worked`` belong to no domain and are excluded from the client
split (they stay in the shared test set and in the centralised baseline's training rows)."""

EXCLUDED_WORKCLASSES = (MISSING, "Without-pay", "Never-worked")


def feature_names() -> tuple[str, ...]:
    """Model input columns in the order of the feature matrix."""
    names: list[str] = []
    for column in CATEGORICAL_INPUTS:
        names.extend(f"{column}={value}" for value in CATEGORY_VOCABULARY[column])
        names.append(f"{column}={UNKNOWN}")
    names.extend(CONTINUOUS_INPUTS)
    return tuple(names)


FEATURE_NAMES = feature_names()
"""Feature order of every Adult feature matrix in this repository (85 columns)."""


# --------------------------------------------------------------------------------------
# Download and checksum pinning
# --------------------------------------------------------------------------------------


def sha256_of(path: str | Path) -> str:
    """SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_checksums(path: str | Path) -> dict[str, str]:
    """Read a ``sha256sum``-style file (``<hex>  <name>``; ``#`` starts a comment)."""
    checksums: dict[str, str] = {}
    path = Path(path)
    if not path.is_file():
        return checksums
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, name = line.split(maxsplit=1)
        checksums[name.strip().lstrip("*")] = digest.lower()
    return checksums


def write_checksums(path: str | Path, checksums: Mapping[str, str], date: str) -> Path:
    """Write the trust-on-first-use checksum pin file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# SHA-256 of the UCI Adult files, recorded by data/download_adult.py.",
        "# UCI publishes no checksums: this is a trust-on-first-use pin made on "
        f"{date} from {BASE_URL}",
        "# Later downloads and loads are verified against it.",
    ]
    lines.extend(f"{checksums[name]}  {name}" for name in sorted(checksums))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def verify_checksums(
    data_dir: str | Path, checksum_file: str | Path, names: Sequence[str] = FILE_NAMES
) -> dict[str, str]:
    """Compare the files ``names`` in ``data_dir`` with the pinned checksums; raise on mismatch.

    Returns the observed digests. Files without a pin are reported, not rejected. Only the
    named files are opened (validation mode never opens ``adult.test``).
    """
    data_dir = Path(data_dir)
    pinned = read_checksums(checksum_file)
    observed: dict[str, str] = {}
    for name in names:
        path = data_dir / name
        if not path.is_file():
            continue
        observed[name] = sha256_of(path)
        expected = pinned.get(name)
        if expected is None:
            LOGGER.warning("No pinned checksum for %s in %s", name, checksum_file)
        elif expected != observed[name]:
            raise ValueError(
                f"Checksum mismatch for {path}: expected {expected}, got {observed[name]}. "
                "The file differs from the one pinned on first download; delete it and "
                "download again, or investigate the change."
            )
    return observed


def download_adult(
    data_dir: str | Path,
    checksum_file: str | Path,
    force: bool = False,
    timeout_s: float = DOWNLOAD_TIMEOUT_S,
) -> dict[str, str]:
    """Download ``adult.data``, ``adult.test`` and ``adult.names`` from the UCI host.

    Only the canonical UCI host is used (no mirror, no scikit-learn / OpenML / Hugging Face
    copy) and only the standard library. If ``checksum_file`` already pins a digest, the
    downloaded file must match it; otherwise the digest is recorded (trust on first use).
    Returns the digests of the three files.
    """
    from datetime import UTC, datetime

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    pinned = read_checksums(checksum_file)
    digests: dict[str, str] = {}
    for name in FILE_NAMES:
        target = data_dir / name
        if target.is_file() and not force:
            LOGGER.info("%s already present; not downloading again", target)
        else:
            url = BASE_URL + name
            LOGGER.info("Downloading %s", url)
            try:
                with urllib.request.urlopen(url, timeout=timeout_s) as response:  # noqa: S310
                    payload = response.read()
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                raise RuntimeError(
                    f"Could not download {url} ({exc}). Check the network connection; the "
                    f"dataset page is {CANONICAL_PAGE}."
                ) from exc
            temporary = target.with_suffix(target.suffix + ".part")
            temporary.write_bytes(payload)
            temporary.replace(target)
        digests[name] = sha256_of(target)
        if name in pinned and pinned[name] != digests[name]:
            raise ValueError(
                f"Checksum mismatch for {target}: pinned {pinned[name]}, got {digests[name]}"
            )
    if any(name not in pinned for name in FILE_NAMES):
        merged = {**pinned, **digests}
        write_checksums(checksum_file, merged, datetime.now(UTC).strftime("%Y-%m-%d"))
        LOGGER.info("Recorded SHA-256 checksums in %s", checksum_file)
    return digests


# --------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------


def parse_label(raw: str) -> int:
    """Map ``>50K`` / ``>50K.`` to 1 and ``<=50K`` / ``<=50K.`` to 0 (trailing period allowed)."""
    value = raw.strip().rstrip(".")
    if value == POSITIVE_LABEL:
        return 1
    if value == NEGATIVE_LABEL:
        return 0
    raise ValueError(f"Unrecognised Adult income label: {raw!r}")


def read_adult_file(path: str | Path) -> pd.DataFrame:
    """Parse one Adult file into a DataFrame with :data:`COLUMNS` and an integer ``label``.

    Leading spaces are stripped, a leading ``|`` comment line (``adult.test``) and blank lines
    are skipped, ``?`` is kept as the literal string ``?`` (missing), and the trailing period of
    test labels is removed.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"UCI Adult file not found: {path}. Download it first with "
            "`python data/download_adult.py` (or `make data`)."
        )
    records: list[list[str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("|"):
                continue
            fields = [field.strip() for field in stripped.split(",")]
            if len(fields) != len(COLUMNS):
                raise ValueError(
                    f"{path}:{line_number}: expected {len(COLUMNS)} fields, got {len(fields)}"
                )
            records.append(fields)
    frame = pd.DataFrame(records, columns=list(COLUMNS))
    for column in ("age", "fnlwgt", "education-num", "capital-gain", "capital-loss"):
        frame[column] = frame[column].astype(np.int64)
    frame["hours-per-week"] = frame["hours-per-week"].astype(np.int64)
    frame["label"] = frame["income"].map(parse_label).astype(np.int64)
    for column, vocabulary in CATEGORY_VOCABULARY.items():
        unknown = set(frame[column].unique()) - set(vocabulary) - {MISSING}
        if unknown:
            raise ValueError(f"{path}: column {column!r} has values outside the schema: {unknown}")
    return frame


def parse_names_vocabulary(path: str | Path) -> dict[str, tuple[str, ...]]:
    """Read the categorical vocabularies from the published ``adult.names`` file."""
    vocabulary: dict[str, tuple[str, ...]] = {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line or line.startswith("|"):
            continue
        name, _, values = line.partition(":")
        name = name.strip()
        if name in CATEGORY_VOCABULARY:
            items = [value.strip().rstrip(".") for value in values.split(",")]
            vocabulary[name] = tuple(item for item in items if item)
    return vocabulary


# --------------------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ScalingRule:
    """Data-independent transform of one continuous column: optional ``log1p``, then divide."""

    log1p: bool
    divisor: float


def one_hot_categoricals(frame: pd.DataFrame) -> np.ndarray:
    """One-hot encode :data:`CATEGORICAL_INPUTS` with the published vocabulary + ``Unknown``."""
    blocks: list[np.ndarray] = []
    for column in CATEGORICAL_INPUTS:
        vocabulary = (*CATEGORY_VOCABULARY[column], UNKNOWN)
        index = {value: i for i, value in enumerate(vocabulary)}
        codes = frame[column].map(lambda v, idx=index: idx[UNKNOWN] if v == MISSING else idx[v])
        block = np.zeros((len(frame), len(vocabulary)), dtype=np.float32)
        block[np.arange(len(frame)), codes.to_numpy(dtype=np.int64)] = 1.0
        blocks.append(block)
    return np.concatenate(blocks, axis=1)


def scale_fixed_public(frame: pd.DataFrame, rules: Mapping[str, ScalingRule]) -> np.ndarray:
    """Scale :data:`CONTINUOUS_INPUTS` with data-independent constants (nothing is fitted)."""
    missing = set(CONTINUOUS_INPUTS) - set(rules)
    if missing:
        raise ValueError(f"fixed_public scaling has no rule for {sorted(missing)}")
    columns = []
    for column in CONTINUOUS_INPUTS:
        rule = rules[column]
        values = frame[column].to_numpy(dtype=np.float64)
        if rule.log1p:
            values = np.log1p(values)
        columns.append(values / rule.divisor)
    return np.stack(columns, axis=1).astype(np.float32)


def fit_standardiser(train_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation of the continuous inputs on training rows (``train_fitted``)."""
    values = train_frame[list(CONTINUOUS_INPUTS)].to_numpy(dtype=np.float64)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def scale_train_fitted(frame: pd.DataFrame, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Standardise the continuous inputs with statistics fitted on training rows."""
    values = frame[list(CONTINUOUS_INPUTS)].to_numpy(dtype=np.float64)
    return ((values - mean) / std).astype(np.float32)


def workclass_domain_codes(frame: pd.DataFrame) -> np.ndarray:
    """Client-domain index per row (0 Private, 1 Government, 2 Self-employed, -1 excluded)."""
    lookup: dict[str, int] = {}
    for code, members in enumerate(WORKCLASS_DOMAINS.values()):
        for member in members:
            lookup[member] = code
    return frame["workclass"].map(lambda v: lookup.get(v, -1)).to_numpy(dtype=np.int64)


def age_band_labels(ages: np.ndarray, bands: Sequence[Sequence[int]]) -> np.ndarray:
    """Label each age with its inclusive band ``"lo-hi"`` from YAML (``"other"`` if none)."""
    labels = np.full(len(ages), "other", dtype=object)
    for low, high in bands:
        mask = (ages >= low) & (ages <= high)
        labels[mask] = f"{low}-{high}"
    return labels


def preprocess(
    train_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    scaling: str,
    fixed_rules: Mapping[str, ScalingRule],
) -> tuple[np.ndarray, np.ndarray]:
    """Feature matrices for training and evaluation rows, in :data:`FEATURE_NAMES` order.

    ``scaling`` is ``"fixed_public"`` or ``"train_fitted"``; the same mode is applied to both
    frames, and in ``train_fitted`` mode the statistics come from ``train_frame`` only.
    Protected attributes, ``fnlwgt`` and ``education`` never enter the matrix.
    """
    if scaling == "fixed_public":
        train_cont = scale_fixed_public(train_frame, fixed_rules)
        eval_cont = scale_fixed_public(eval_frame, fixed_rules)
    elif scaling == "train_fitted":
        mean, std = fit_standardiser(train_frame)
        train_cont = scale_train_fitted(train_frame, mean, std)
        eval_cont = scale_train_fitted(eval_frame, mean, std)
    else:
        raise ValueError(f"Unknown scaling mode {scaling!r}")
    train_x = np.concatenate([one_hot_categoricals(train_frame), train_cont], axis=1)
    eval_x = np.concatenate([one_hot_categoricals(eval_frame), eval_cont], axis=1)
    return train_x.astype(np.float32), eval_x.astype(np.float32)


def load_adult_frames(
    data_dir: str | Path,
    checksum_file: str | Path | None,
    verify: bool,
    expect_full_size: bool,
    read_test: bool,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Read ``adult.data`` and, if ``read_test``, ``adult.test`` (optionally verifying checksums).

    With ``read_test=False`` (validation mode) ``adult.test`` is neither checked, hashed nor
    opened, and ``None`` is returned in its place. With ``expect_full_size`` the row counts must
    be exactly 32,561 (and 16,281); the synthetic fixture is loaded with
    ``expect_full_size=False``.
    """
    data_dir = Path(data_dir)
    names = ("adult.data", "adult.test") if read_test else ("adult.data",)
    for name in names:
        if not (data_dir / name).is_file():
            raise FileNotFoundError(
                f"UCI Adult file {data_dir / name} is missing. Run `make data` "
                "(or `python data/download_adult.py`) first."
            )
    if verify and checksum_file is not None:
        verify_checksums(data_dir, checksum_file, names)
    train = read_adult_file(data_dir / "adult.data")
    test = read_adult_file(data_dir / "adult.test") if read_test else None
    if expect_full_size:
        if len(train) != EXPECTED_TRAIN_ROWS:
            raise ValueError(f"Expected {EXPECTED_TRAIN_ROWS} training rows, got {len(train)}")
        if test is not None and len(test) != EXPECTED_TEST_ROWS:
            raise ValueError(f"Expected {EXPECTED_TEST_ROWS} test rows, got {len(test)}")
    return train, test


def _groups(frame: pd.DataFrame, age_bands: Sequence[Sequence[int]]) -> dict[str, np.ndarray]:
    return {
        "sex": frame["sex"].to_numpy(dtype=object),
        "race": frame["race"].to_numpy(dtype=object),
        "age_band": age_band_labels(frame["age"].to_numpy(dtype=np.int64), age_bands),
    }


def load_adult_dataset(
    adult_cfg: AdultConfig, evaluation: EvaluationConfig, seed: int
) -> FederatedDataset:
    """Build the :class:`~src.federated.datasets.FederatedDataset` for UCI Adult.

    With ``evaluation.eval_on == "test"`` the evaluation set is all 16,281 test rows. With
    ``"validation"`` the test file is not read at all: a stratified validation split is carved
    from the training rows (seeded by ``seed``) and serves as the evaluation set, and the
    ``train_fitted`` statistics are computed on the remaining training rows only.
    """
    from . import FederatedDataset, TabularSplit, resolve_path, validation_indices

    data_dir = resolve_path(adult_cfg.data_dir)
    checksum_file = resolve_path(adult_cfg.checksum_file)
    validation = evaluation.eval_on == "validation"
    train_frame, test_frame = load_adult_frames(
        data_dir,
        checksum_file,
        adult_cfg.verify_checksums,
        adult_cfg.expect_full_size,
        read_test=not validation,
    )
    rules = {
        name: ScalingRule(log1p=rule.log1p, divisor=rule.divisor)
        for name, rule in adult_cfg.fixed_public.items()
    }
    raw_stats = describe_frames(train_frame, test_frame)
    if validation:
        keep, held = validation_indices(
            workclass_domain_codes(train_frame), evaluation.validation_fraction, seed
        )
        eval_frame = train_frame.iloc[held].reset_index(drop=True)
        train_frame = train_frame.iloc[keep].reset_index(drop=True)
    else:
        assert test_frame is not None
        eval_frame = test_frame
    train_x, eval_x = preprocess(train_frame, eval_frame, adult_cfg.scaling, rules)
    splits = []
    for frame, x in ((train_frame, train_x), (eval_frame, eval_x)):
        splits.append(
            TabularSplit(
                x=x,
                y=frame["label"].to_numpy(dtype=np.int64),
                domain=workclass_domain_codes(frame),
                groups=_groups(frame, adult_cfg.age_bands),
                subdomain=frame["workclass"].to_numpy(dtype=object),
            )
        )
    return FederatedDataset(
        name="adult",
        train=splits[0],
        test=splits[1],
        domain_names=tuple(WORKCLASS_DOMAINS),
        natural_split_name="workclass",
        feature_names=FEATURE_NAMES,
        num_classes=2,
        positive_class=1,
        eval_on=evaluation.eval_on,
        info={"raw": raw_stats, "scaling": adult_cfg.scaling},
    )


def describe_frames(train: pd.DataFrame, test: pd.DataFrame | None) -> dict[str, float]:
    """Row counts and positive rates of the raw files (test entries only if it was read)."""
    stats = {
        "train_rows": int(len(train)),
        "train_positive": int(train["label"].sum()),
        "train_positive_rate": float(train["label"].mean()),
    }
    if test is not None:
        stats.update(
            test_rows=int(len(test)),
            test_positive=int(test["label"].sum()),
            test_positive_rate=float(test["label"].mean()),
        )
    return stats


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m src.federated.datasets.adult --download``: fetch and pin the Adult files."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--download", action="store_true", help="download from the UCI host")
    parser.add_argument("--data-dir", default="data/adult", help="target folder")
    parser.add_argument("--checksums", default="data/checksums/adult.sha256", help="pin file")
    parser.add_argument("--force", action="store_true", help="download even if present")
    args = parser.parse_args(argv)
    setup_logging()
    if args.download:
        download_adult(args.data_dir, args.checksums, force=args.force)
    train, test = load_adult_frames(args.data_dir, args.checksums, True, True, read_test=True)
    stats = describe_frames(train, test)
    LOGGER.info(
        "UCI Adult (extracted from the %d U.S. Census database): %d training rows, %d test "
        "rows; positive rate %.4f (train) and %.4f (test)",
        EXTRACTION_YEAR,
        stats["train_rows"],
        stats["test_rows"],
        stats["train_positive_rate"],
        stats["test_positive_rate"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
