"""Config strictness and documentation of every federated YAML file."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.common.config import ConfigError
from src.common.labels import ASSUMPTION, PAPER_REPORTED, TARGET_TBD
from src.federated.config import load_federated_config

TAGS = (ASSUMPTION, PAPER_REPORTED, TARGET_TBD)


def federated_configs(configs_dir: Path) -> list[Path]:
    paths = sorted(configs_dir.glob("federated_*.yaml"))
    paths += sorted((configs_dir / "optional").glob("federated_*.yaml"))
    return paths


def test_expected_config_files_exist(configs_dir: Path) -> None:
    names = {path.name for path in federated_configs(configs_dir)}
    assert {"federated_adult.yaml", "federated_fedavg.yaml", "federated_dp.yaml", "federated_femnist.yaml"} <= names  # fmt: skip


@pytest.mark.parametrize("smoke", [False, True])
def test_every_config_loads_strictly(configs_dir: Path, smoke: bool) -> None:
    for path in federated_configs(configs_dir):
        config, raw = load_federated_config(path, smoke=smoke)
        assert config.training.rounds > 0
        assert "smoke_overrides" not in raw and "base_config" not in raw


def test_unknown_and_missing_keys_are_rejected(configs_dir: Path, tmp_path: Path) -> None:
    path = configs_dir / "federated_fedavg.yaml"
    with pytest.raises(ConfigError, match="unknown"):
        load_federated_config(path, overrides=["training.typo_rate=0.1"])
    text = path.read_text(encoding="utf-8").replace("  momentum: 0.9", "  # momentum removed")
    broken = tmp_path / "broken.yaml"
    broken.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="missing required key"):
        load_federated_config(broken)


def test_every_key_is_commented_with_a_provenance_tag(configs_dir: Path) -> None:
    for path in federated_configs(configs_dir):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            assert "#" in stripped, f"{path.name}:{number} has no comment: {line!r}"
            assert any(tag in stripped for tag in TAGS), f"{path.name}:{number} lacks a tag"


def test_locked_adult_protocol(configs_dir: Path) -> None:
    config, _ = load_federated_config(configs_dir / "federated_adult.yaml")
    assert config.training.rounds == 20
    assert config.experiment.seeds == [0, 1, 2, 3, 4]
    assert config.partitions.splits == ["workclass", "iid"]
    assert sorted(e.target_epsilon for e in config.dp.sweep) == [1.0, 3.0, 8.0]
    assert config.dp.delta < 1 / 3657
    assert config.dataset.adult.scaling == "fixed_public"
    assert config.evaluation.eval_on == "test"
    assert set(config.evaluation.group_attributes) == {"sex", "race", "age_band"}


def test_trimmed_mean_beta_principle(configs_dir: Path, caplog: pytest.LogCaptureFixture) -> None:
    for path in federated_configs(configs_dir):
        for smoke in (False, True):
            config, _ = load_federated_config(path, smoke=smoke)
            rob = config.robustness
            assert int(config.federation.trimmed_mean_beta * rob.num_clients) >= rob.num_byzantine
    caplog.set_level("WARNING")
    load_federated_config(
        configs_dir / "federated_adult.yaml", overrides=["federation.trimmed_mean_beta=0.1"]
    )
    assert "cannot remove every attacker" in caplog.text
    assert load_federated_config(configs_dir / "federated_adult.yaml")[0].federation.fraction_evaluate == 0.0  # fmt: skip
