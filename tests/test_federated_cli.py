"""Command-line smoke runs: benchmark, plots, worker-count determinism, missing-data message."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from experiments import plot_federated, run_federated_benchmark
from src.common.io import read_csv

EXPECTED_FILES = (
    "config_used.yaml",
    "metadata.json",
    "per_round.csv",
    "per_client.csv",
    "final.csv",
    "summary.csv",
    "per_group.csv",
    "partition_stats.csv",
    "dp_tradeoff.csv",
    "timings.csv",
)


@pytest.mark.slow
def test_toy_smoke_all_suites_and_plots(configs_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "run"
    started = time.perf_counter()
    code = run_federated_benchmark.main(
        ["--config", str(configs_dir / "federated_fedavg.yaml"), "--suite", "all", "--smoke",
         "--out", str(out), "--log-level", "WARNING"]  # fmt: skip
    )
    elapsed = time.perf_counter() - started
    assert code == 0
    # The documented budget for this run is 30 s on an idle laptop. The bound asserted here is
    # deliberately loose: a tight wall-clock assertion fails on a busy machine or CI runner
    # for reasons that have nothing to do with the code.
    assert elapsed < 300.0, f"toy smoke run took {elapsed:.1f} s"
    for name in EXPECTED_FILES:
        assert (out / name).is_file(), name
    final = read_csv(out / "final.csv")
    suites = {row["suite"] for row in final}
    assert suites == {"main", "partitions", "aggregation", "robustness"}
    methods = {row["method"] for row in final if row["suite"] == "main"}
    assert {"fedavg", "fedavg_dp_eps1", "local_only", "svm", "majority_class"} <= methods
    assert any(m.startswith("local_only:") for m in methods)
    baseline = next(r for r in final if r["method"] == "svm")
    assert baseline["bytes_total"] == "0" and baseline["epsilon_max"] == "inf"
    dp_rows = [r for r in final if r["method"] == "fedavg_dp_eps1"]
    assert all(0 < float(r["epsilon_max"]) <= 1.0 + 1e-6 for r in dp_rows)
    assert plot_federated.main(["--run-dir", str(out)]) == 0
    for name in ("dp_tradeoff", "accuracy_per_round", "per_client_accuracy", "communication_cost", "partition_distribution"):  # fmt: skip
        assert (out / "figures" / f"{name}.png").is_file()
        assert (out / "figures" / f"{name}.csv").is_file()


@pytest.mark.slow
def test_results_do_not_depend_on_worker_count(configs_dir: Path, tmp_path: Path) -> None:
    common = ["--config", str(configs_dir / "federated_fedavg.yaml"), "--suite", "main", "--smoke",
              "--log-level", "WARNING", "--set", "baselines.methods=[centralised_mlp, majority_class]"]  # fmt: skip
    assert run_federated_benchmark.main([*common, "--workers", "1", "--out", str(tmp_path / "w1")]) == 0  # fmt: skip
    assert run_federated_benchmark.main([*common, "--workers", "3", "--out", str(tmp_path / "w3")]) == 0  # fmt: skip
    for name in ("final.csv", "per_round.csv", "per_client.csv", "per_group.csv", "dp_tradeoff.csv"):  # fmt: skip
        assert read_csv(tmp_path / "w1" / name) == read_csv(tmp_path / "w3" / name), name


def test_adult_config_without_files_says_make_data(
    configs_dir: Path, tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    code = run_federated_benchmark.main(
        ["--config", str(configs_dir / "federated_adult.yaml"), "--smoke", "--out", str(tmp_path),
         "--set", f"dataset.adult.data_dir={tmp_path / 'nothing_here'}", "--log-level", "ERROR"]  # fmt: skip
    )
    assert code == 2
    assert "make data" in capfd.readouterr().err
