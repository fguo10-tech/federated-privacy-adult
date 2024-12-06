# Dataset cards

This folder holds the download script, the checksum pin and the toy-data generator for every
dataset the repository uses. It contains no dataset: downloaded and generated data are
git-ignored and are never committed. The benchmark of this repository trains sample-weighted
federated averaging with record-level differential privacy on simulated clients; the cards below
say where each dataset comes from, how it is turned into clients, and what results on it can and
cannot support.

| Dataset | Role | In the repository | Results |
|---|---|---|---|
| [UCI Adult / Census Income](#1-uci-adult--census-income-primary-dataset) | primary benchmark; every reported result | download script and checksum pin | `[Reproduced]`, see [RESULTS.md](../experiments/results/RESULTS.md) |
| [FEMNIST / LEAF](#2-femnist--leaf-optional-secondary-image-benchmark) | optional secondary image benchmark | a reader for LEAF-format files and a config | none; every result `[Target / TBD]` |
| [Synthetic toy federated domains](#3-synthetic-toy-federated-domains) | tests and smoke runs only | generator | none reported |

No real medical, financial or personal records are in this repository. UCI Adult is a named
public dataset that is downloaded by script from its canonical host.

---

## 1. UCI Adult / Census Income (primary dataset)

### Summary

Adult, also known as "Census Income", from the UCI Machine Learning Repository, extracted by
Barry Becker from the **1994** Census database. Each row describes one person by 14 attributes;
the task, as printed by UCI, is: "Prediction task is to determine whether a person's income is
over $50,000 a year." The target is binary, `>50K` versus `<=50K`, and about 24% of rows are
positive, so accuracy alone is misleading and this repository always reports precision, recall and
F1 of the `>50K` class next to it.

| Fact | Value |
|---|---|
| Name | Adult (Census Income), UCI Machine Learning Repository |
| Canonical page | [http://archive.ics.uci.edu/ml/datasets/Adult](http://archive.ics.uci.edu/ml/datasets/Adult) (redirects to https://archive.ics.uci.edu/dataset/2/adult) |
| Creators | Barry Becker and Ronny Kohavi |
| Extraction | from the 1994 Census database by Barry Becker, with the conditions `((AAGE>16) && (AGI>100) && (AFNLWGT>1) && (HRSWK>0))` |
| Donated to UCI | 1996-04-30 |
| Rows | 48,842 in total: 32,561 training rows (`adult.data`) and 16,281 test rows (`adult.test`) |
| Attributes | 14: age, workclass, fnlwgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country |
| UCI characteristics | Multivariate; Social Science; Classification; has missing values |
| Target | income >$50K a year: `>50K` (positive, label 1) versus `<=50K` (label 0); about 24% positive |

Using Adult, and everything about how its rows are turned into federated clients, is
`[Assumption — not in paper]`.

### Source and licence

- Citation, as printed by UCI: Becker, B. & Kohavi, R. (1996). Adult [Dataset]. UCI Machine
  Learning Repository. https://doi.org/10.24432/C5XW20.
- Licence: Creative Commons Attribution 4.0 International (CC BY 4.0). Attribution is given by the
  citation above; the files are not redistributed here.
- Bibliography key: `becker1996adult` in [references.bib](../docs/report/references.bib).

### How it is downloaded

```bash
make data                          # or, equivalently:
python data/download_adult.py      # = python -m src.federated.datasets.adult --download
python data/download_adult.py --force   # download again even if the files are present
```

The script ([download_adult.py](download_adult.py), implemented in
[src/federated/datasets/adult.py](../src/federated/datasets/adult.py)) fetches `adult.data`,
`adult.test` and `adult.names` from
`https://archive.ics.uci.edu/ml/machine-learning-databases/adult/` — the canonical UCI host
only, with the Python standard library. It uses no mirror and no copy from scikit-learn, OpenML or
Hugging Face. Files go to `data/adult/`, which is git-ignored. A file that is already present is
not downloaded again unless `--force` is given. After downloading, the script parses both data
files and checks that they contain exactly 32,561 and 16,281 rows.

**Checksums.** [checksums/adult.sha256](checksums/adult.sha256) records the SHA-256 digest of the
three files. UCI publishes no checksums, so this file is a **trust-on-first-use pin**: it records
what the canonical host served on the day the pin was made (the date is in the file's header).
Every later download is checked against it, and every load of the data by the benchmark
(`dataset.adult.verify_checksums: true` in
[configs/federated_adult.yaml](../configs/federated_adult.yaml)) refuses files whose digest
differs. What the pin proves: that your files are byte-for-byte the files this repository's
reference run used. What it does not prove: that those files are the ones UCI originally
published in 1996, that the host was not serving altered files on the day of the pin, or that the
data is free of errors. A file without a pin is reported with a warning, not rejected. When the
benchmark runs in validation mode (used for tuning), the test file is neither opened nor hashed.

### Preprocessing

Everything below is what [src/federated/datasets/adult.py](../src/federated/datasets/adult.py)
does; every choice is `[Assumption — not in paper]`.

**Parsing.** Leading spaces are stripped from every field; the first line of `adult.test`
(`|1x3 Cross validator`) and blank lines are skipped; the trailing period of the test labels
(`>50K.`) is removed. Any categorical value outside the published schema stops the load with an
error.

**Missing values.** Missing values appear as `?`. Nothing is imputed and no row is dropped: `?` becomes its own category, encoded as an explicit
`Unknown` one-hot column.

**Categorical inputs** (workclass, marital-status, occupation, relationship, native-country) are
one-hot encoded over a **fixed vocabulary copied from the published schema `adult.names`** (spelling
and order included), plus the `Unknown` column. Because the vocabulary is a constant, encoding needs
no statistic from any client or from the test set. A test checks the vocabulary against the
downloaded `adult.names`.

**Continuous inputs** (education-num, capital-gain, capital-loss, hours-per-week) are scaled in one
of two modes, chosen with `dataset.adult.scaling`:

| Mode | What it does | Fitted on | Status |
|---|---|---|---|
| `fixed_public` | optional `log1p` (capital-gain and capital-loss), then division by a constant from the YAML file | nothing | **default**, used by every reported run |
| `train_fitted` | standardisation with the mean and standard deviation of the pooled training rows | training rows only (the remaining training rows in validation mode) | available, not used for any reported result |

The default fits nothing. Nothing can therefore leak from the test set into the features, nothing
is shared across the client boundary, and the differential-privacy guarantee needs no additional
release. The `train_fitted` mode never touches test rows either, but in a real federation its
pooled statistics would have to be computed and released across clients, and that release is not
covered by the DP accounting.

**Excluded and evaluation-only attributes.** `fnlwgt` (a census sampling weight, not a property of
the person) and `education` (redundant with its ordinal encoding `education-num`) are dropped.
`age`, `race` and `sex` are **evaluation-only**: they never enter the feature matrix of any model —
baseline, local-only or federated — and are kept next to the features only for the per-group
breakdown (age grouped into the bands set in the config). The resulting feature matrix has 85
columns. The role of each attribute, generated from the code:

<!-- BEGIN GENERATED: adult_attribute_roles -->
| Attribute | Type | Role | Encoding in the default mode (`fixed_public`) | Reason |
|---|---|---|---|---|
| `age` | integer (years) | evaluation-only | not a feature; grouped into the YAML age bands for fairness reports | protected attribute; used only for per-age-band fairness |
| `workclass` | categorical | model input | one-hot over the 8 published values + `Unknown` for `?` (9 columns) | one-hot, published vocabulary + Unknown; also defines the three client domains |
| `fnlwgt` | integer (sampling weight) | excluded | dropped before preprocessing | census sampling weight, not a property of the person |
| `education` | categorical | excluded | dropped before preprocessing | redundant with education-num (its ordinal encoding) |
| `education-num` | integer (ordinal 1-16) | model input | fixed_public: value / divisor from YAML (no fitted statistic) | continuous, scaled |
| `marital-status` | categorical | model input | one-hot over the 7 published values + `Unknown` for `?` (8 columns) | one-hot, published vocabulary + Unknown; proxy for sex (Married-civ-spouse) |
| `occupation` | categorical | model input | one-hot over the 14 published values + `Unknown` for `?` (15 columns) | one-hot, published vocabulary + Unknown |
| `relationship` | categorical | model input | one-hot over the 6 published values + `Unknown` for `?` (7 columns) | one-hot, published vocabulary + Unknown; proxy for sex (Husband / Wife) |
| `race` | categorical | evaluation-only | not a feature; kept as a group label for fairness reports | protected attribute; used only for per-group fairness |
| `sex` | categorical | evaluation-only | not a feature; kept as a group label for fairness reports | protected attribute; used only for per-group fairness |
| `capital-gain` | integer (US dollars) | model input | fixed_public: log1p, then / divisor from YAML (no fitted statistic) | continuous, scaled (log1p in fixed_public mode) |
| `capital-loss` | integer (US dollars) | model input | fixed_public: log1p, then / divisor from YAML (no fitted statistic) | continuous, scaled (log1p in fixed_public mode) |
| `hours-per-week` | integer (hours) | model input | fixed_public: value / divisor from YAML (no fitted statistic) | continuous, scaled |
| `native-country` | categorical | model input | one-hot over the 41 published values + `Unknown` for `?` (42 columns) | one-hot, published vocabulary + Unknown |
| `income` | binary label | target | `>50K` -> 1 (positive class), `<=50K` -> 0; trailing period of adult.test removed | binary income label (>50K is about 24% of rows) |
<!-- END GENERATED: adult_attribute_roles -->

**Fairness note.** Removing `sex` from the inputs does not make the models sex-blind:
`relationship` (Husband / Wife) and `marital-status` (Married-civ-spouse) remain inputs and are
proxies for sex. The per-group table in RESULTS.md is descriptive and is not a fairness
certification.

### Splits (IID and non-IID)

**Train and test.** The published split is kept: models train on rows of `adult.data` and every
reported metric is computed on the 16,281 rows of `adult.test` (by the server for federated runs;
per-client and per-group metrics use subsets of those rows). For tuning, a
validation split stratified by client domain is carved from the training rows instead, and the
test file is not read at all.

**Clients are simulated.** The clients below are simulated domains derived from one public census
dataset. They are not real employers, hospitals or banks, no organisation took part, and the whole
federated design — the use of Adult, the workclass clients, the IID control, the ablation
partitions, the 20-round / 5-seed protocol and the ε sweep — is `[Assumption — not in paper]`.

**Workclass split (primary, non-IID).** Built from training rows only, the same for every seed:

| Client | Workclass values |
|---|---|
| Private | `Private` |
| Government | `Federal-gov`, `State-gov`, `Local-gov` |
| Self-employed | `Self-emp-inc`, `Self-emp-not-inc` |

**Exclusion rule.** Training rows whose workclass is `?`, `Without-pay` or `Never-worked` belong to
no client. They are not discarded: they remain in the training rows of the centralised baselines
(which therefore see slightly more data than the federation; a "client rows only" centralised MLP
is reported as a control), and test rows with those values remain in the shared test set.

**IID control.** The union of the three clients' rows, shuffled with the run seed and dealt into
three clients of equal size (sizes differ by at most one row). It has the same rows as the
workclass split, so any difference between the two is caused by how the rows are grouped.

Per-client row counts, positive counts and label rates, computed from the downloaded `adult.data`:

<!-- BEGIN GENERATED: adult_partition_stats -->
Computed from the downloaded `adult.data` (training rows only); source CSV `experiments/results/reference/federated_adult/partition_stats.csv`. [Reproduced]

| Split | Client | Training rows | Positives (>50K) | Positive rate | Share of client rows | Workclass composition |
|---|---|---|---|---|---|---|
| Workclass split (primary, non-IID) | Private | 22,696 | 4,963 | 0.2187 | 0.7392 | Private 22696 |
| Workclass split (primary, non-IID) | Government | 4,351 | 1,341 | 0.3082 | 0.1417 | Federal-gov 960, Local-gov 2093, State-gov 1298 |
| Workclass split (primary, non-IID) | Self-employed | 3,657 | 1,346 | 0.3681 | 0.1191 | Self-emp-inc 1116, Self-emp-not-inc 2541 |
| IID control, seed 0 | iid_0 | 10,235 | 2,619 | 0.2559 | 0.3333 | Federal-gov 338, Local-gov 735, Private 7525, Self-emp-inc 377, Self-emp-not-inc 855, State-gov 405 |
| IID control, seed 0 | iid_1 | 10,235 | 2,500 | 0.2443 | 0.3333 | Federal-gov 322, Local-gov 678, Private 7582, Self-emp-inc 369, Self-emp-not-inc 842, State-gov 442 |
| IID control, seed 0 | iid_2 | 10,234 | 2,531 | 0.2473 | 0.3333 | Federal-gov 300, Local-gov 680, Private 7589, Self-emp-inc 370, Self-emp-not-inc 844, State-gov 451 |

Excluded from every client split: 1,857 training rows (? 1836, Never-worked 7, Without-pay 14; positive rate 0.1029). The workclass split is the same for every seed; the IID control is reshuffled with each run seed, so its per-client positive rates vary slightly between seeds while its client sizes stay fixed.
<!-- END GENERATED: adult_partition_stats -->

The table shows the two partitions of the main benchmark: the workclass split (the same for every
seed) and the three-client IID control for the first reporting seed. The ablation partitions below
are not in it.

**Ablation partitions** (same client rows; settings in the `partitions:` and `robustness:` sections
of [configs/federated_adult.yaml](../configs/federated_adult.yaml); all `[Assumption — not in
paper]`):

| Partition | Construction |
|---|---|
| `iid_size_matched` | shuffled like the IID control, but with the workclass clients' sizes |
| `label_skew` | for each label, client shares drawn from a Dirichlet distribution (`label_skew_alpha`) |
| `quantity_skew` | shuffled rows dealt into clients whose sizes follow a Dirichlet draw (or a power law) |
| robustness suite | an IID partition into more clients (`robustness.num_clients`), some of which attack |

Dirichlet-drawn skewed partitions are redrawn (at most `max_resample_attempts` times) until every
client holds at least `min_client_rows` rows. Every
partition is a deterministic function of the data, the settings and the seed. To print the
partition statistics yourself:

```bash
python -m experiments.describe_adult --out experiments/results/runs/adult_description
```

### Privacy note

Adult is a public dataset; this repository adds no personal data to it and
commits none of it. The federated benchmark treats each simulated client's rows as if they were
private and protects them with record-level (ε, δ) differential privacy (DP-SGD per client). That
guarantee covers the model updates a client sends; it does not cover the row count each client
reports in the clear or the quantities derived from it, client-level privacy, hyperparameter tuning, `train_fitted` scaling
statistics, or the combined release of several runs on the same rows (see "Released quantities
under DP" in [RESULTS.md](../experiments/results/RESULTS.md)). DP noise is drawn from a
client-held secret seed that is never transmitted; the benchmark fixes that secret for
reproducibility, which is acceptable for research on public data and must not be done in a
deployment. Because the data is public, publishing many DP runs on the same rows harms no one here;
the same practice on private data would spend the privacy budget many times over.

### Claims this dataset can and cannot support

Results on Adult **support** claims about the federated method on a public U.S. census benchmark:
how weighted FedAvg compares with local-only training, with a centralised model of the same
architecture and with classical baselines; what record-level DP costs at a given ε; and how client
heterogeneity built from one attribute affects training.

They **do not support** claims about any present-day employer, hospital, bank or user population
(Ding et al., 2021), about real cross-organisation federations, about any data modality other than
tabular data, or about fairness in deployment.

### Known limitations

- **Dated population.** The rows describe the 1994 U.S. population and labour market. Income levels,
  occupations and their relationship to the label have changed since; distributions are dated.
- **Proxy target.** Income above a fixed threshold is a proxy target, not a measure of merit, and
  the threshold is not adjusted for inflation or region.
- **Non-random missing values.** Missing values are not missing at random: the partition table
  shows that the training rows excluded from the client split, most of which have a missing
  workclass, have a different positive rate from the client rows. Encoding `?` as its own category
  lets the models use that information; it also means the excluded group is not a random sample.
- **Simulated clients from one attribute.** Splitting by workclass creates heterogeneity in one
  dimension only; real federations differ in feature definitions, collection processes and time.
  Three clients is a very small federation.
- **Sampling weight ignored.** `fnlwgt` is dropped, so results describe the extracted rows, not the
  weighted U.S. population.
- **Well-known critique.** Ding et al. (2021) discuss the limitations of Adult as a benchmark and
  propose newer census-derived datasets as replacements.

---

## 2. FEMNIST / LEAF (optional secondary image benchmark)

### Summary

FEMNIST is the federated version of handwritten-character data in LEAF (Caldas et al., 2018,
"LEAF: A Benchmark for Federated Settings", arXiv:1812.01097). As stated on https://leaf.cmu.edu/ it
has 805,263 samples and 3,550 users (writers); the site prints the sample count as "80,5263" and the
LEAF GitHub README rounds the user count to 3500. It has 62 classes (10 digits, 26 lowercase and 26
uppercase letters) and 28×28 images, per the LEAF GitHub README. One writer is one natural client.

In this repository FEMNIST is **optional**. It is never the primary benchmark, nothing is
downloaded, and every FEMNIST result is `[Target / TBD]`. The classical baselines (SVM,
Random Forest, Decision Tree) are not run on it.

### Source and licence

LEAF, https://github.com/TalwalkarLab/leaf , licence BSD 2-Clause. FEMNIST is built by LEAF's
scripts from NIST Special Database 19. Bibliography key: `caldas2018leaf`.

### How it is generated or downloaded

This repository does not download it. Prepare it yourself with the LEAF scripts; the LEAF README's
command for a small non-IID sample is:

```bash
./preprocess.sh -s niid --sf 0.05 -k 0 -t sample
```

Place the resulting LEAF-format JSON shards in `data/femnist/train/` (git-ignored) and run the
optional config [configs/optional/federated_femnist.yaml](../configs/optional/federated_femnist.yaml)
(the command is in Section 4 of [RESULTS.md](../experiments/results/RESULTS.md)).

### Preprocessing

The reader in [src/federated/datasets/femnist.py](../src/federated/datasets/femnist.py) loads every
shard, checks that each image has 784 values and each label lies in 0–61, flattens each image to a
784-value vector for the same MLP code, skips writers with fewer samples than
`min_samples_per_writer`, and keeps the first `max_writers` eligible writers in sorted order. All
settings are `[Assumption — not in paper]`.

### Splits (IID and non-IID)

The natural split is one client per writer (non-IID); the IID control shuffles the same samples into
equal clients. A per-writer fraction of samples (`test_fraction`) forms the shared test set.

### Privacy note

FEMNIST holds handwriting samples written by people. None of it is in this repository. The reader is tested only on a fabricated fixture of random pixels.

### Claims this dataset can and cannot support

No claim is supported, because nothing has been run. Once run, results would support claims about
the federated method on a public handwriting benchmark with many small natural clients, and nothing
about Adult.

### Known limitations

Not exercised at any real scale; the MLP ignores image structure; the delta in the optional config
must stay below one over the smallest writer's sample count.

---

## 3. Synthetic toy federated domains

### Summary

Three seeded synthetic domains that share one binary classification task, used only by unit tests,
smoke runs and [configs/federated_fedavg.yaml](../configs/federated_fedavg.yaml). They are never
the primary benchmark and no result from them is reported.

### Source and licence

Generated by [src/federated/datasets/synthetic_domains.py](../src/federated/datasets/synthetic_domains.py);
same licence as the code.

### How it is generated

```bash
python data/synthetic/generate_federated_toy.py --seed 100 --out experiments/results/runs/federated_toy_data
```

The benchmark itself does not read these CSV files: `make smoke` and `make federated-toy` run
[configs/federated_fedavg.yaml](../configs/federated_fedavg.yaml), whose `dataset.kind` is
`synthetic`, and generate the domains in memory from each run seed. The script above only writes
the same rows to disk so that they can be inspected.

One shared weight vector defines the task. Each domain draws Gaussian features shifted along its own
random direction (covariate shift), labels rows by a linear score with a domain-specific intercept
chosen to give a configured positive rate (label-prior shift), and flips a small fraction of labels.
Domain sizes differ on purpose so that weighted and unweighted averaging give different results. All
settings are in the `dataset.synthetic` section of the config and are `[Assumption — not in paper]`.

### Preprocessing

None; the features are drawn with unit variance by construction (each domain's mean is shifted on
purpose).

### Splits (IID and non-IID)

The natural split is one client per domain (non-IID); the IID control shuffles the same rows into
equal clients.

### Privacy note

The domain names `healthcare-like`, `finance-like` and `user-behaviour-like` are labels on synthetic
numbers and nothing more; the data is not about patients, accounts or users.

### Claims this dataset can and cannot support

They support only checks that the code runs and behaves as specified (for example that weighted and
unweighted aggregation differ when client sizes differ). They support no claim about any real domain.

### Known limitations

A linear task with Gaussian features is far easier and far more regular than real data.

---

## What is committed and what is not

| Committed | Not committed (git-ignored) |
|---|---|
| [download_adult.py](download_adult.py) and the toy-data generator in [synthetic/](synthetic/) | `data/adult/` — the downloaded Adult files |
| [checksums/adult.sha256](checksums/adult.sha256) — the trust-on-first-use pin | `data/femnist/` — any FEMNIST shards you prepare |
| Tiny fabricated test fixtures in [tests/fixtures/federated/](../tests/fixtures/federated/README.md) | `data/synthetic/*.csv` and `*.npz` — generated datasets |
| The reference results (tables, CSVs and charts, no raw data) | `data/cache/`, your own runs under `experiments/results/runs/` |

The Adult fixtures under `tests/fixtures/federated/adult/` are **fabricated rows in the Adult file
format** — leading spaces, `?` for missing values, the `|1x3 Cross validator` comment line and
`>50K.` labels with a trailing period in the test file — drawn from the published schema vocabulary
with a fixed seed. They are not census records and contain no row of the real dataset. The LEAF
fixture holds three fake writers with random pixel values and random labels.

## References

- Becker, B., and Kohavi, R. (1996). Adult [Dataset]. UCI Machine Learning Repository.
  https://doi.org/10.24432/C5XW20
- Caldas, S., Duddu, S. M. K., Wu, P., Li, T., Konečný, J., McMahan, H. B., Smith, V., and
  Talwalkar, A. (2018). LEAF: A Benchmark for Federated Settings. arXiv:1812.01097.
- Ding, F., Hardt, M., Miller, J., and Schmidt, L. (2021). Retiring Adult: New Datasets for Fair
  Machine Learning. *Advances in Neural Information Processing Systems 34 (NeurIPS)*.
