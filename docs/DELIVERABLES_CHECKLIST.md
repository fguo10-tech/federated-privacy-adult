# Deliverables checklist

Checked on 2024-11-19. A box is ticked only if the item exists **and** the check named beside
it was run and passed on that date. Anything that was not run, cannot be checked from a local
machine, or deliberately differs from the original brief is listed in
[section 3](#3-not-done-not-verified-or-deliberately-different) and is not hidden behind a tick.

**Where the checks ran.** The original acceptance checks ran on macOS, CPU only, Python 3.12.14,
with package versions as in [`requirements-lock.txt`](../requirements-lock.txt). The repository
is public at https://github.com/fguo10-tech/federated-privacy-adult and has a commit history.
GitHub Actions has completed successfully on a published main revision
([run 44860925060](https://github.com/fguo10-tech/federated-privacy-adult/actions/runs/44860925060)).
No outside party has reviewed any part of this draft. "Committed" means tracked by Git.

**Where this repository comes from.** This repository and its companion project
`ddqn-os-scheduling` were first drafted together as one combined project. On 2024-11-19 the
project owner decided to publish them as two independent repositories under the same
direction. The code, configuration, tests and reference results of this topic were moved here
without change; the shared utility code was copied and trimmed to what this topic uses; the
results pipeline was divided; and the documents were rewritten to describe this repository
only. Every check in this file was run on this repository after that division.

**Positioning.** By the project owner's decision the documents focus on the problem and on
what this implementation does: the problem, the design choices with their reasons, the
Before / After comparison with its regressions, and the limits. The source paper appears only
as a background citation, as the tag `[Assumption — not in paper]` and as reference values in
one generated table
([section 3.2](#32-deliberate-differences-from-the-original-brief) lists what this changed
relative to the original brief).

## 1. Acceptance criteria

- [x] **The repository installs from scratch with the documented commands.**
  Checked twice. (a) A new virtual environment inside the repository, `pip install -e ".[all]"`,
  then `make lint`, the test suite, `make smoke` and the example script, with that environment
  active: all succeeded. (b) A copy of only the files that will be committed (no downloaded
  dataset, no earlier outputs) in an empty folder, a second new environment installed with the
  extras the CI workflow uses, then the workflow's own steps (ruff, black, pytest,
  `make smoke`, `make docs-check`) and the example: all succeeded. The public clone URL is now
  available; this original clean-room check did not include a fresh clone from GitHub. Python
  3.12 only; see section 3 for Python 3.11.
- [x] **`make test` passes, including the FedAvg tests.**
  With the Adult files downloaded: 138 passed, 1 skipped (the opt-in test of Flower's Ray
  engine). In the clean copy without the dataset, with the CI selection: 137 passed, 1 skipped
  (the test that needs the real Adult files), 1 deselected (the Ray test).
  `tests/test_federated_aggregation.py` checks the weighted aggregate against hand-computed toy
  weights and against the unweighted mean.
- [x] **The two-client federated smoke run completes at small scale without errors.**
  `make smoke` runs the benchmark end to end on the synthetic toy domains. The two-client run
  goes through the real Flower `ServerApp` and `ClientApp`
  (`tests/test_federated_flower.py::test_two_client_end_to_end_through_flower`), and
  `tests/test_federated_cli.py` checks that results do not depend on the number of worker
  processes.
- [x] **All configs are external YAML, and every run saves raw CSV files and a copy of its
  config.**
  The loader in `src/common/config.py` has no defaults in code: a missing or unknown key is an
  error. Every run folder holds `config_used.yaml`, `metadata.json` and the raw CSV files.
  `tests/test_config_comments.py` checks that every config key has a comment and that every
  setting carries a provenance tag. Every public function, method and class in `src/` has a
  docstring and full type hints (checked by script; the linter does not enforce this).
- [x] **`README.md` is complete.**
  The fifteen parts of the original brief in order (the title block and fourteen sections),
  about 8,800 words of hand-written prose (generated tables not counted), 14 FAQ answers, the status banner and the Proposed Endeavor
  paragraph verbatim (both checked by `tests/test_docs_integrity.py`).
- [x] **`docs/report/PROJECT_REPORT.md` is complete.**
  Abstract, executive summary, chapters 1 to 13 and five appendices; about 16,900 words of
  hand-written prose; more than six tables; four embedded Mermaid diagrams and seven figures.
- [x] **The topic guide is complete.**
  `docs/investigations/01-federated-privacy-cross-domain.md`, about 8,700 words: scope, setup,
  results, observations, applicability conditions, open questions, and suggested use and next
  experiments.
- [x] **All four architecture diagrams render as Mermaid in Markdown.**
  The four diagrams in `docs/architecture/` and their identical copies embedded in other
  documents rendered without error with Mermaid 11.4.1. Rendering on github.com itself has not
  been observed; see section 3.
- [x] **The notebook is present and runs in order.**
  `make notebooks` executed it top to bottom (a little over a minute). It is committed without
  outputs, every cell that prints a number also prints its config and seeds, and it ends with an
  "Interpretation and limits" cell.
- [x] **The primary dataset is UCI Adult / Census Income exactly as locked.**
  1994 Census extraction, donated 1996-04-30, CC BY 4.0, 48,842 rows, 32,561 training and
  16,281 test rows, target income >$50K, canonical UCI source URL in the dataset card, the
  README and the report. The loader refuses files whose row counts or SHA-256 checksums differ
  (`tests/test_federated_adult.py::test_real_adult_counts_clients_and_schema`). The benchmark
  uses the workclass three-client split plus the IID control, 20 rounds, 5 seeds and the DP
  sweep ε ∈ {1, 3, 8} plus no noise, each marked `[Assumption — not in paper]` in
  `configs/federated_adult.yaml`. FEMNIST appears only as the optional secondary benchmark.
  One deliberate difference in preprocessing is listed in section 3.2.
- [x] **Poster copy and presentation outline are complete.**
  The poster has the exact copy, two results panels (Before / After, and ε versus accuracy)
  with the dataset named on each, the privacy panel and a layout specification. The deck has
  15 slides; every slide has three to five full-sentence bullets, a named figure and 80 to 120
  words of speaker notes (counted by script).
- [x] **The Before / After comparison is complete.**
  The table and the grouped bar chart, plus the ε-versus-accuracy table and chart, appear in
  `RESULTS.md`, the README, the project report, the poster and the slides. All five placements
  are injected from one set of generated files, built from the raw CSV files of the reference
  run; `make docs-check` fails if any placement differs.
- [x] **Every Delta and Improvement % follows the stated formula and direction convention.**
  Computed in one place, `src/common/before_after.py`, and tested in
  `tests/test_before_after.py`: Delta is After minus Before; a positive percentage always means
  better after; a zero, near-zero, infinite or non-comparable baseline gives
  "N/A — baseline zero / not comparable"; a missing side gives TBD. No percentage mixes a
  paper-reported value with a value from this code. Regressions are shown.
- [x] **No fabricated result appears anywhere, and every TBD is marked.**
  Every `[Reproduced]` number in the documents sits inside a generated block built from
  `experiments/results/reference/`. Three checks stand behind those files. Before the
  division, the whole reference run, including the ablation suites and the extended privacy
  study, was generated a second time with a different number of worker processes and matched
  byte for byte. After the division, every reference file here has the same SHA-256 as its
  original, every generated results table is identical to the one built before the division
  (identifiers aside), and the benchmark with its ablation suites was run once more with this
  repository's own code and environment: all seven result files matched the committed ones
  byte for byte. Paper values come from `src/common/paper_values.py` and are labelled
  `[Paper-reported]`.
- [x] **The researcher profiles appear only as illustrative target users.**
  Five profiles are named, only in chapter 12 of the project report, each with the statement
  that the person is not a collaborator, co-author, endorser or advisor and has not been
  contacted, and each with one study shape labelled `[Target / TBD]`. No named scholar is in
  any author, collaborator or acknowledgement field. A test fails if any of those names appears
  in any other file, or if the name that belongs to the companion project appears at all.
- [x] **No real sensitive data, no secret and no large data file is committed.**
  The Adult files are downloaded by `make data` into a git-ignored folder. The test fixtures
  are synthetic rows in the Adult file format. The largest committed file is a result CSV of
  about 2 MB; the committed files total about 11 MB, most of it the reference run. A test scans
  every committed file for dataset files, secret patterns and local paths.
- [x] **The documents stay on this repository's topic.**
  A test fails on any leftover of the combined draft, on method vocabulary of the companion
  project, and on text that discusses the source paper instead of this implementation.

## 2. Deliverables by location

| Location | Status | Notes |
|---|---|---|
| `README.md`, `LICENSE` (Apache-2.0), `CITATION.cff`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md` | Created | Fan Guo is listed as author; repository URLs and README badges are configured |
| `pyproject.toml`, `requirements.txt`, `requirements-lock.txt`, `Makefile` | Created, used by the checks above | The lock file records the tested versions |
| `.github/workflows/ci.yml`, four issue forms, pull-request template | Created | GitHub Actions has successful runs; see the linked run above |
| GitHub Security settings | Configured | Private vulnerability reporting is enabled |
| `configs/` | Created | `federated_adult.yaml`, `federated_fedavg.yaml`, `federated_dp.yaml`, `optional/federated_femnist.yaml`, `before_after.yaml` |
| `src/federated/` | Created, tested | Flower apps and strategy, aggregation rules, DP with Opacus, partitions, Adult loader, baselines, attacks, communication accounting, secure-aggregation simulation |
| `src/common/` | Created, tested | Config, seeding, logging, metrics, labels, Before / After computation, plotting |
| `tests/` | Created | 14 test files and synthetic fixtures; the result of the full run is in section 1 |
| `data/README.md`, `data/download_adult.py`, `data/checksums/`, `data/synthetic/` | Created | Dataset cards, download script, checksums, toy-domain generator; no dataset is committed |
| `experiments/` | Created | Benchmark runner, plotting script, Adult description script, reference-run script, table builder, document sync and notebook runner |
| `experiments/results/RESULTS.md`, `templates/`, `reference/` | Created | Results document, CSV templates with headers only, and the reference run behind every `[Reproduced]` number |
| `notebooks/` | Created, executed | One notebook, committed without outputs |
| `docs/architecture/` | Created, rendered | Four diagrams, each with explanation, assumptions and alternatives |
| `docs/report/` | Created | Project report, a short record of the source paper and the values it reports (`SOURCE_PAPER_FACTS.md`), bibliography |
| `docs/investigations/` | Created | One topic guide (the folder name is kept from the first draft) |
| `docs/poster/`, `docs/slides/` | Created | Poster copy and layout specification; 15-slide deck |
| `docs/REPLICATIONS.md` | Created | Empty register: no external replication has been reported |
| `examples/` | Created, run | One quickstart script |

## 3. Not done, not verified, or deliberately different

### 3.1 Not done or not verified

- **The repository is publicly published** at https://github.com/fguo10-tech/federated-privacy-adult.
  The test counts above describe the original validation snapshot, not this later documentation
  cleanup.
- **GitHub Actions has passed** on a published main revision (see the run linked above). The
  badge in the README shows the current workflow status.
- **Python 3.11 is declared but untested.** `pyproject.toml` allows it and the CI matrix
  includes it; every check here used Python 3.12.
- **Linux, Windows and GPU are untested.** Results on other hardware may differ in the last
  digits.
- **Mermaid rendering has not been visually checked on github.com.** Local render checks passed;
  this visual check is separate.
- **The extended privacy study was not re-run after the division.** Its result files are
  byte-identical copies of files that were verified by a second run before the division; the
  main benchmark and its ablation suites were run again here.
- **Flower's Ray simulation engine is `[Target / TBD]`.** The benchmark uses an in-process
  Flower grid. The Ray path has never completed a run on the development machine; its test is
  opt-in and was skipped.
- **FEMNIST / LEAF is `[Target / TBD]`.** A reader, a config and a test on a synthetic fixture
  exist; the benchmark has not been run on FEMNIST.
- **Secure aggregation is a simulation and is not wired into training.** It shows the
  arithmetic of pairwise masking only. The implemented-versus-stubbed table is in the README
  and in `src/federated/secure_agg.py`.
- **No large-language-model component exists.** The Proposed Endeavor mentions one; this
  repository implements federated learning with privacy accounting and lists the rest as
  planned.
- **No integrated demonstration with the companion project exists.** It is planned, not
  implemented.
- **No external replication and no outside review.** The review passes described in the
  documents were internal ones made while the draft was written.
- **Poster and slides are Markdown copy,** as the original brief asks; no PDF or slide file was
  produced.

### 3.2 Deliberate differences from the original brief

Decided by the project owner after the first draft:

1. **Two repositories instead of one.** The original brief asked for a single repository
   covering two topics. This repository covers one. As a consequence it has four architecture
   diagrams where the brief listed six across both topics, one notebook of the three, one topic
   guide of the two, and five of the six researcher profiles; the others are in the companion
   project, and the short slide version for platform audiences belongs to the companion as
   well. The import package of both projects is called `src`, so the two must not be installed
   into the same environment: use one virtual environment per project.
2. **Focus on the problem, not on the source paper.** The original brief asked the documents
   to examine the source paper point by point: a chapter of the report on it, a comparison of
   its values with this code's values, and an investigation report closing with a decision
   summary. The documents do not do that. The README banner reads "Draft reference
   implementation — values printed in the source paper are shown for reference only; …"; the
   values printed in the paper appear in one generated reference table, in `RESULTS.md` and in
   chapter 8 of the report, and are never compared with values from this code; chapter 3 of
   the report is "Design choices and assumptions"; `docs/report/SOURCE_PAPER_FACTS.md` is a
   short record of the paper; the document in `docs/investigations/` is a topic guide that ends
   with suggested use and next experiments; `[Reproduced]` is explained everywhere as "obtained
   by running this repository's code under the documented conditions"; and comments, docstrings
   and table captions in the code describe what the code does. That last change was checked to
   alter no behaviour: the syntax trees of every Python file (docstrings and string text aside)
   and the parsed values of every YAML file are unchanged, and every result CSV and figure has
   the same SHA-256 as before. Two things went beyond wording. An appendix of the first draft
   that did arithmetic on the values printed in the paper was removed from the report. And one
   unit test (`tests/test_common_metrics.py`) no longer compares a helper's output with those
   printed values: five assertions were removed, the test was renamed, and its check of the
   helper against a direct computation was kept; a second test was renamed with identical
   statements.

Decided while the first draft was written:

3. **Scaling of the continuous Adult features is data-independent by default.** The brief asks
   for scaling fitted on training data. The default mode `fixed_public` divides by public
   constants from YAML and fits nothing, because a mean and standard deviation computed from
   client rows would be an extra release that the DP guarantee does not cover. The fitted
   variant is available as `dataset.adult.scaling: train_fitted`.
4. **The canonical Adult URL is written without the full stop that follows it in the brief.**
   The full stop was read as sentence punctuation; a link that ends in one does not resolve.
5. **The researcher profiles are named in one place only,** chapter 12 of the project report.
   The issue forms, the replication invitation and every other document describe the same
   archetypes without names. Two details given in the brief could not be confirmed from public
   pages and were left out; the profiles state only what could be confirmed.
6. **The reference run is committed:** about 9 MB of raw CSV files and figures under
   `experiments/results/reference/`. These are results, not datasets; your own runs go to the
   git-ignored `experiments/results/runs/`.

### 3.3 What a reader of the results should know

- **The Before / After comparison has gains and regressions.** Weighted FedAvg is more accurate
  than the clients' local-only models and less accurate than the centralised tree models; with
  differential privacy, accuracy changes little at ε = 8 and ε = 3 and drops more at ε = 1, and
  F1 is lower at every ε tested. The tables are in
  [`RESULTS.md`](../experiments/results/RESULTS.md).
- **Four defects in the privacy protocol of this code** were found and fixed during drafting
  and are described openly in the report.

## 4. Owner decisions remaining

- No affiliation was supplied; only Fan Guo is listed. Add an affiliation only if it is confirmed.
- Compare the Proposed Endeavor paragraph with the original exhibit, character by character;
  that exhibit is not included in this repository.
- Decide on the deliberate differences in section 3.2.
- No formal release has been published. The package remains `0.1.0.dev0`; `date-released` is
  omitted from `CITATION.cff` until a release is made.

## 5. Re-running the checks

Run these with the project's virtual environment active.

```bash
make install        # editable install with every optional extra
make lint           # ruff and black --check
make test           # full test suite, including the documentation integrity tests
make smoke          # tiny end-to-end run of the benchmark on synthetic toy data
make data           # download UCI Adult from the canonical UCI source and verify checksums
make notebooks      # execute the notebook top to bottom
make docs-check     # fail if any generated table in the documents is out of date
make reference      # re-create the reference run (overwrites it), then rebuild every table
```
