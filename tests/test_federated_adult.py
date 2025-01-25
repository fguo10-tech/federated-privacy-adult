"""Adult parsing, preprocessing and partitions on a synthetic fixture, and on the real files.

The fixture under ``tests/fixtures/federated/adult`` is random rows in the Adult file format
(not census data). Tests marked ``adult`` use the downloaded files and skip without them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.common.config import from_dict
from src.federated.config import AdultConfig, EvaluationConfig, PartitionConfig
from src.federated.datasets import resolve_path
from src.federated.datasets.adult import (
    CATEGORY_VOCABULARY,
    EXPECTED_TEST_ROWS,
    EXPECTED_TRAIN_ROWS,
    FEATURE_NAMES,
    PROTECTED_ATTRIBUTES,
    load_adult_dataset,
    parse_label,
    parse_names_vocabulary,
    read_adult_file,
    verify_checksums,
)
from src.federated.partition import make_partition, partition_statistics

RULES = {
    "education-num": {"log1p": False, "divisor": 16.0},
    "capital-gain": {"log1p": True, "divisor": 12.0},
    "capital-loss": {"log1p": True, "divisor": 12.0},
    "hours-per-week": {"log1p": False, "divisor": 100.0},
}
PARTITIONS = {
    "splits": ["workclass", "iid"],
    "ablations": ["iid_size_matched", "label_skew", "quantity_skew"],
    "label_skew_alpha": 0.5,
    "quantity_skew": {"kind": "dirichlet", "alpha": 1.0, "exponent": 1.5},
    "min_client_rows": 3,
    "max_resample_attempts": 200,
}


def adult_cfg(data_dir: Path, scaling: str = "fixed_public", full: bool = False) -> AdultConfig:
    return from_dict(
        AdultConfig,
        {
            "data_dir": str(data_dir),
            "checksum_file": str(data_dir / "missing.sha256"),
            "verify_checksums": False,
            "expect_full_size": full,
            "scaling": scaling,
            "fixed_public": RULES,
            "age_bands": [[17, 34], [35, 54], [55, 90]],
        },
    )


def eval_cfg(eval_on: str = "test") -> EvaluationConfig:
    return from_dict(
        EvaluationConfig,
        {"eval_on": eval_on, "validation_fraction": 0.25, "group_attributes": ["sex"]},
    )


@pytest.fixture(scope="module")
def fixture_dir(repo_root: Path) -> Path:
    return repo_root / "tests" / "fixtures" / "federated" / "adult"


def test_label_parsing_with_and_without_period() -> None:
    assert parse_label(">50K") == 1 and parse_label(" >50K.") == 1
    assert parse_label("<=50K") == 0 and parse_label("<=50K.") == 0
    with pytest.raises(ValueError):
        parse_label("50K")


def test_fixture_parsing_handles_comment_spaces_and_missing(fixture_dir: Path) -> None:
    train = read_adult_file(fixture_dir / "adult.data")
    test = read_adult_file(fixture_dir / "adult.test")
    assert len(train) == 102 and len(test) == 52  # comment and blank lines skipped
    assert set(test["label"].unique()) <= {0, 1}
    assert (train["workclass"] == "?").sum() == 2
    assert not any(str(v).startswith(" ") for v in train["occupation"])
    assert "?" in set(train["occupation"])


def test_protected_attributes_are_not_features(fixture_dir: Path) -> None:
    for name in FEATURE_NAMES:
        column = name.split("=")[0]
        assert column not in PROTECTED_ATTRIBUTES
        assert column not in ("fnlwgt", "education")
    dataset = load_adult_dataset(adult_cfg(fixture_dir), eval_cfg(), seed=100)
    assert dataset.train.x.shape[1] == len(FEATURE_NAMES) == 85
    assert set(dataset.train.groups) == {"sex", "race", "age_band"}
    # Every categorical block is one-hot (exactly one 1), "?" goes to the Unknown column.
    occupation = [i for i, n in enumerate(FEATURE_NAMES) if n.startswith("occupation=")]
    block = dataset.train.x[:, occupation]
    assert np.all(block.sum(axis=1) == 1)
    unknown = FEATURE_NAMES.index("occupation=Unknown")
    raw = read_adult_file(fixture_dir / "adult.data")
    np.testing.assert_array_equal(dataset.train.x[:, unknown] == 1, raw["occupation"] == "?")


def test_fixed_public_is_data_independent_and_train_fitted_is_not(fixture_dir: Path) -> None:
    fixed = load_adult_dataset(adult_cfg(fixture_dir), eval_cfg(), seed=100)
    raw = read_adult_file(fixture_dir / "adult.data")
    hours = FEATURE_NAMES.index("hours-per-week")
    np.testing.assert_allclose(fixed.train.x[:, hours], raw["hours-per-week"] / 100.0, rtol=1e-6)
    gain = FEATURE_NAMES.index("capital-gain")
    np.testing.assert_allclose(fixed.train.x[:, gain], np.log1p(raw["capital-gain"]) / 12.0, rtol=1e-6)  # fmt: skip
    fitted = load_adult_dataset(adult_cfg(fixture_dir, "train_fitted"), eval_cfg(), seed=100)
    assert abs(float(fitted.train.x[:, hours].mean())) < 1e-5


def test_exclusion_rule_and_domains(fixture_dir: Path) -> None:
    dataset = load_adult_dataset(adult_cfg(fixture_dir), eval_cfg(), seed=100)
    excluded = dataset.train.domain == -1
    assert set(dataset.train.subdomain[excluded]) == {"?", "Without-pay", "Never-worked"}
    partition = make_partition("workclass", dataset, from_dict(PartitionConfig, PARTITIONS), 100)
    assert partition.client_names == ("Private", "Government", "Self-employed")
    assert sum(partition.sizes()) == int((~excluded).sum())
    gov = set(dataset.train.subdomain[partition.indices[1]])
    assert gov <= {"Federal-gov", "State-gov", "Local-gov"}
    stats = partition_statistics(partition, dataset, 100)
    assert stats[-1]["client"] == "(excluded)" and stats[-1]["rows"] == 4


def test_all_partitions_sizes_and_determinism(fixture_dir: Path) -> None:
    dataset = load_adult_dataset(adult_cfg(fixture_dir), eval_cfg(), seed=100)
    cfg = from_dict(PartitionConfig, PARTITIONS)
    natural = make_partition("workclass", dataset, cfg, 100)
    pool = np.sort(np.concatenate(natural.indices))
    iid = make_partition("iid", dataset, cfg, 100)
    sizes = iid.sizes()
    assert max(sizes) - min(sizes) <= 1
    assert np.array_equal(np.sort(np.concatenate(iid.indices)), pool)
    matched = make_partition("iid_size_matched", dataset, cfg, 100)
    assert matched.sizes() == natural.sizes()
    for name in ("label_skew", "quantity_skew"):
        part = make_partition(name, dataset, cfg, 100)
        assert np.array_equal(np.sort(np.concatenate(part.indices)), pool)
        assert min(part.sizes()) >= cfg.min_client_rows
    again = make_partition("iid", dataset, cfg, 100)
    other = make_partition("iid", dataset, cfg, 101)
    assert all(np.array_equal(a, b) for a, b in zip(iid.indices, again.indices, strict=True))
    assert not all(np.array_equal(a, b) for a, b in zip(iid.indices, other.indices, strict=True))


def test_validation_mode_never_reads_test_rows(
    fixture_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import builtins
    import io
    import shutil

    from src.federated.datasets.adult import sha256_of, write_checksums

    data = tmp_path / "adult"
    shutil.copytree(fixture_dir, data)
    pins = {name: sha256_of(data / name) for name in ("adult.data", "adult.test")}
    write_checksums(data / "pins.sha256", pins, "test")
    cfg = from_dict(
        AdultConfig,
        {**adult_cfg(data).__dict__, "checksum_file": str(data / "pins.sha256"),
         "verify_checksums": True, "fixed_public": RULES},
    )  # fmt: skip
    opened: list[str] = []
    real_open = io.open

    def spy(file, *args, **kwargs):
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(io, "open", spy)
    monkeypatch.setattr(builtins, "open", spy)
    dataset = load_adult_dataset(cfg, eval_cfg("validation"), seed=100)
    monkeypatch.undo()
    assert any(path.endswith("adult.data") for path in opened)  # the spy does see opens
    assert not any(path.endswith("adult.test") for path in opened)
    assert "test_rows" not in dataset.info["raw"]
    assert len(dataset.train) + len(dataset.test) == 102
    assert dataset.eval_on == "validation"
    # Test mode, by contrast, does open (and checksum) adult.test.
    opened.clear()
    monkeypatch.setattr(io, "open", spy)
    load_adult_dataset(cfg, eval_cfg("test"), seed=100)
    monkeypatch.undo()
    assert any(path.endswith("adult.test") for path in opened)


def test_missing_files_give_clear_message(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make data"):
        load_adult_dataset(adult_cfg(tmp_path), eval_cfg(), seed=100)


# ---------------------------------------------------------------- real UCI Adult files

REAL_DIR = resolve_path("data/adult")
needs_adult = pytest.mark.skipif(
    not (REAL_DIR / "adult.data").is_file() or not (REAL_DIR / "adult.test").is_file(),
    reason="UCI Adult files not downloaded (run `make data`)",
)


@pytest.mark.adult
@needs_adult
def test_real_adult_counts_clients_and_schema() -> None:
    verify_checksums(REAL_DIR, resolve_path("data/checksums/adult.sha256"))
    cfg = adult_cfg(REAL_DIR, full=True)
    dataset = load_adult_dataset(cfg, eval_cfg(), seed=0)
    assert len(dataset.train) == EXPECTED_TRAIN_ROWS == 32_561
    assert len(dataset.test) == EXPECTED_TEST_ROWS == 16_281
    partition = make_partition("workclass", dataset, from_dict(PartitionConfig, PARTITIONS), 0)
    assert partition.sizes() == [22_696, 4_351, 3_657]
    assert int((dataset.train.domain == -1).sum()) == 1_857
    vocabulary = parse_names_vocabulary(REAL_DIR / "adult.names")
    for column, values in CATEGORY_VOCABULARY.items():
        assert vocabulary[column] == values


def test_attribute_roles_markdown_table() -> None:
    from src.federated.datasets.adult import attribute_roles_markdown

    table = attribute_roles_markdown()
    rows = [line for line in table.strip().splitlines()[2:]]
    assert len(rows) == 15  # 14 attributes + the income target
    by_name = {row.split("|")[1].strip().strip("`"): row.split("|")[3].strip() for row in rows}
    for name in ("age", "race", "sex"):
        assert by_name[name] == "evaluation-only"
    assert by_name["fnlwgt"] == by_name["education"] == "excluded"
    assert by_name["income"] == "target"
    assert sum(role == "model input" for role in by_name.values()) == 9
