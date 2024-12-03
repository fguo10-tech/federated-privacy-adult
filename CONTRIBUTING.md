# Contributing

Thank you for looking at this repository. It is a reference implementation for researchers, and
it shows its contribution as a Before / After comparison measured under documented conditions.
The method follows the federated learning approach with differential privacy described by Zhang
et al. (2025). The benchmark is Flower-based, sample-weighted FedAvg with Opacus
differential privacy on the UCI Adult (Census Income) dataset, with three workclass-based
clients, an IID control, centralised and local-only baselines and an epsilon sweep.

This is draft research code (version 0.1.x). Interfaces may still change.

## Who this guide is for

- Researchers who want to reproduce a result, benchmark a new method against the existing
  baselines, or extend the benchmark (a new aggregation rule, attack, accountant, model or
  dataset).
- Researchers who ran the code independently and want to report what they found, whether it
  matched or not.
- Engineers who want to see how a Flower federation with DP-SGD is wired, message by message.
  The clients are simulated in one process and the secure-aggregation module is a simulation,
  not cryptography; see `SECURITY.md`.

Everyone taking part is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md). Security
problems, including privacy-accounting errors, go through [SECURITY.md](SECURITY.md), never a
public issue.

## Setup

Python 3.11 or 3.12 is required. A CPU is enough for everything.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
make install                       # editable install with every optional extra
make test                          # full test suite, no network
make smoke                         # tiny end-to-end run on synthetic toy data, no download
make data                          # download UCI Adult into data/adult (git-ignored)
```

`make test` skips the tests marked `adult` when `data/adult` is absent; run `make data` first if
you work on anything that touches the Adult loader, the partitions or the reported results. `make test-fast` leaves out the tests marked `slow`. Run `make help` to see
every target. Your own run outputs go to `experiments/results/runs/`, which is git-ignored.
`experiments/results/reference/` holds the run behind every `[Reproduced]` number in the
documentation, and only `make reference` writes there; contributors do not commit to it.

## Branch and pull-request workflow

1. Open or find an issue first for anything larger than a typo, so that work is not duplicated.
   Changes listed under [Research proposals (RFCs)](#research-proposals-rfcs) need an accepted
   proposal before a pull request.
2. Fork the repository and create a topic branch from `main` with a short descriptive name,
   for example `feature/krum-aggregation` or `fix/adult-loader-checksum`.
3. Keep each pull request to one logical change. Separate refactoring from behaviour changes.
4. Before pushing, run `make format`, `make lint`, `make test` and `make smoke`.
5. Open the pull request against `main` and fill in the template checklist. CI runs lint
   (ruff, black), the test suite on Python 3.11 and 3.12, the smoke run and `make docs-check`.
6. A maintainer reviews. Push new commits in response rather than force-pushing over review
   history.

## Code style

- **Formatting and lint.** ruff and black, line length 100, Python 3.11 syntax. `make format`
  fixes most findings; `make lint` must pass with no changes needed.
- **Type hints** on every public function; `from __future__ import annotations` at the top of
  each module.
- **Docstrings** on every public module, class and function.
- **Logging, not print.** Library code under `src/` logs through
  `src.common.logging_utils.get_logger(__name__)`. Command-line scripts call `setup_logging` and
  may log a short final summary.
- **No hyperparameter in Python.** Every hyperparameter and experimental setting lives in a YAML
  file under `configs/`. Config dataclasses declare no defaults and are built with
  `src.common.config.from_dict`, which rejects missing and unknown keys.
- **Every YAML key is commented** with its meaning, its unit where one applies, and a provenance
  tag. A setting chosen by this repository is tagged `[Assumption — not in paper]`.
- **Seeds.** Explicit integer seeds; child seeds come from `src.common.seeding.derive_seed`.
- **Reuse `src/common/`** (config, seeding, io, metrics, plotting, labels, Before / After
  tables) instead of re-implementing it.
- **Smoke variant.** Every config may carry a `smoke_overrides:` mapping; entry points accept
  `--smoke`, and the smoke variant of a benchmark must stay well under 30 seconds on a laptop CPU.

## Test requirements

Tests use pytest, must be deterministic and must not touch the network. Mark tests that take
more than about five seconds with `@pytest.mark.slow`; tests that need the downloaded dataset with
`@pytest.mark.adult`. Build paths from the `repo_root` and `configs_dir` fixtures.

What a new component must come with:

- **A new centralised or local-only baseline.** A test that it trains on the same feature
  matrix as the federated clients, a test that it never sees the evaluation-only attributes (age,
  race, sex), and a determinism test (same seed, same output).
- **A new aggregation rule.** A test that it reduces to plain weighted averaging where it should
  (for example identical client updates), a test of sample weighting, a test with an empty or
  zero-weight client, and, for robust rules, a test with a known outlier update.
- **A new dataset loader.** Tests against a tiny synthetic fixture committed under `tests/`
  (never the real data), a checksum check, a schema check, and a test that client splits are
  disjoint and cover every row exactly once.
- **A new attack for the robustness suite.** A test that it changes only the clients it is
  assigned to, and a test that its effect on a known update is the documented one.
- **A new message field.** A test that the reply still carries no rows, features or labels, and an
  entry in `RELEASED_QUANTITIES` in `src/federated/privacy.py` saying whether the (ε, δ)
  guarantee covers it.
- **A new privacy component.** A test of the reported epsilon against an independent calculation
  for at least one known setting, and a test that the noise multiplier and sampling rate are read
  from the config.

## Integrity rules for results

Reviewers check every pull request that touches results, tables, figures or documentation
against this list.

1. Every number in code, comments, configs, tables and prose carries exactly one source label:
   `[Paper-reported]` (printed in the source paper and shown for reference only, taken only from
   `src/common/paper_values.py`; never a target for this code and never compared numerically
   with a value from this code), `[Reproduced]` (obtained by running this repository's code under
   the documented conditions; the label says where a number comes from, not that a number
   printed in a paper was reproduced) or `[Target / TBD]` (specified but not run; the cell shows `TBD`, never a value).
2. A `[Paper-reported]` value never appears in a `[Reproduced]` cell, and a reproduced number is
   never labelled `[Paper-reported]`.
3. A `[Reproduced]` number comes from at least 5 seeds.
4. `mean ± std` means: average within a seed, then the mean and the sample standard deviation
   (ddof = 1) across seeds, computed with `src.common.metrics.summarise_by_seed`.
5. Every `[Reproduced]` number has a raw CSV, a `config_used.yaml` and a `metadata.json` behind
   it (written by `src.common.io.save_run_metadata`).
6. Nothing is tuned on the reporting seeds 0–4 or on test data. Use seeds 100 and above for
   development and tuning.
7. No real medical, financial or personal data is committed. Datasets are downloaded by script
   into git-ignored folders; only generators, download scripts, checksums and tiny synthetic
   fixtures are committed.
8. Regressions are reported as they are. If the method under study loses to a baseline, that is
   the result. The baselines, metrics, data split and privacy accounting are not changed to make
   one method look better.

## Generated documentation

Result tables in the documentation sit between markers:

```markdown
<!-- BEGIN GENERATED: before_after_summary -->
...
<!-- END GENERATED: before_after_summary -->
```

Everything between a pair of markers is written by `make results` from the reference run. Never
edit it by hand; your edit will be overwritten, and if it is not, the table no longer matches the
data. `make docs-check` fails CI when any generated block is stale. If you change how a table is
built, change the generating code and run `make results` so the docs and the code agree in the
same pull request. Prose outside the markers is edited normally.

## Research proposals (RFCs)

Open a "Research proposal" issue before writing code for any change to:

- the federation protocol (round structure, client sampling, what a reply message carries);
- the aggregation rule used for the Before / After comparison;
- metric definitions, or which metrics are reported;
- the dataset, preprocessing, or the client split (including the workclass grouping and the IID
  control);
- privacy accounting (accountant, delta, sampling, clipping or how epsilon is reported);
- public interfaces, such as the Flower app handlers, the config schema or the CSV column tuples.

The form asks for the question, the hypothesis, the design, the evidence that would change the
conclusion, data sources and compute needs. A proposal is accepted when a maintainer says so in
the issue; the pull request then links to it.

## Submitting a benchmark result

Use the "Benchmark submission" issue form. Run your method on the standard protocol (at least 5
seeds, 20 rounds, and epsilon in {1, 3, 8} where DP is involved) and attach:

- the summary CSV and the raw per-seed CSVs;
- `config_used.yaml` and `metadata.json` from each run directory;
- the exact command, including every `--set` override;
- the seeds;
- the hardware and the package versions recorded in `metadata.json`.

Confirm in the form that nothing was tuned on seeds 0–4 or on test data. Submitted numbers are
labelled `[Reproduced]` only after a maintainer has re-run them from your config; until then they
are listed as external submissions with your attachments.

## Submitting a replication report

Use the "Replication report" issue form whether or not your numbers matched. A failed
replication is as useful as a successful one. Attach the same items as for a benchmark result
(summary CSV, `config_used.yaml`, `metadata.json`, exact command, seeds, hardware), state the
commit you ran, and give the observed values next to the values from this repository's reference run that you
compared against, with their source labels.

## How external replications are credited

Replications are recorded in [docs/REPLICATIONS.md](docs/REPLICATIONS.md) and mentioned in the
release notes of the next release. The reporter's name and affiliation are recorded only with
their explicit consent; otherwise the entry is anonymous. A replication reporter is credited as a
replicator. They are not added as an author of the code or of any write-up, and a replication is
never presented as an endorsement of the project.

## Review criteria

A reviewer asks:

- Does the change do one thing, and is it described accurately?
- Do lint, tests, smoke runs and `make docs-check` pass?
- Does every new setting live in YAML with a comment and a provenance tag?
- Does every number carry the right source label, and do the integrity rules above hold?
- Do the tests cover the behaviour, including the failure cases?
- Is the change reproducible from the config and seeds alone?
- Does it keep the baselines, data split and metrics fair to every method?
- Are documentation and docstrings updated?

## Good first issues

- Add a Krum aggregation rule as an alternative to weighted FedAvg, with a known-outlier test.
- Add a comparison of the Opacus RDP accountant with the PRV accountant for the same runs.
- Add per-group fairness plots (per client and per sensitive attribute) from `per_group.csv`.
- Add a second client split for Adult (for example grouped by `native-country`), with a test
  that the clients are disjoint and cover every training row.
- Port the CI test matrix to macOS runners.
- Add a hierarchical or asynchronous federated variant behind a config switch.
- Add a schema check that every YAML key in `configs/` carries a provenance-tagged comment.
