`configs/federated_adult.yaml` (verbatim; comments carry the provenance tag of every key):

```yaml
# PRIMARY federated protocol: UCI Adult / Census Income with three workclass-based clients
# and an IID control.
#
# Dataset: Adult (Census Income), UCI Machine Learning Repository, extracted by Barry Becker
# from the 1994 U.S. Census database; donated 1996-04-30. Becker, B. & Kohavi, R. (1996).
# Adult [Dataset]. https://doi.org/10.24432/C5XW20. Licence CC BY 4.0.
# Canonical page: http://archive.ics.uci.edu/ml/datasets/Adult
# Download with `python data/download_adult.py` (files are git-ignored).
#
# Provenance: every value below is [Assumption — not in paper]: it is chosen by this
# repository.
# Hyperparameters were chosen on a validation split of the training rows with development
# seeds >= 100 (evaluation.eval_on: validation); never on the test set or on seeds 0-4.

experiment:                           # run identity: name, seeds, output folder. [Assumption — not in paper]
  name: federated_adult               # run name written to metadata. [Assumption — not in paper]
  seeds: [0, 1, 2, 3, 4]              # REPORTING seeds of the reference benchmark (run once, not for tuning; develop with --seeds 100 101). [Assumption — not in paper]
  output_dir: experiments/results/runs/federated_adult   # default --out folder (relative to the repo root). [Assumption — not in paper]

dataset:                              # which dataset and how it is read and preprocessed. [Assumption — not in paper]
  kind: adult                         # which block below is used: adult | synthetic | femnist. [Assumption — not in paper]
  adult:                              # UCI Adult settings. [Assumption — not in paper]
    data_dir: data/adult              # folder holding adult.data / adult.test / adult.names (relative to the repo root). [Assumption — not in paper]
    checksum_file: data/checksums/adult.sha256   # trust-on-first-use SHA-256 pins of the three files. [Assumption — not in paper]
    verify_checksums: true            # refuse to load files whose SHA-256 differs from the pin. [Assumption — not in paper]
    expect_full_size: true            # assert exactly 32,561 training and 16,281 test rows. [Assumption — not in paper]
    scaling: fixed_public             # fixed_public (data-independent constants below; nothing fitted, nothing leaks) | train_fitted (mean/std of pooled training rows; needs an extra release NOT covered by the DP accounting). [Assumption — not in paper]
    fixed_public:                     # fixed_public transform per continuous input: optional log1p, then divide by divisor. [Assumption — not in paper]
      education-num: {log1p: false, divisor: 16.0}    # years-of-education code 1..16 -> (0, 1]. [Assumption — not in paper]
      capital-gain: {log1p: true, divisor: 12.0}      # log1p(US dollars) / 12 (log1p of the schema's 99,999 cap is 11.5). [Assumption — not in paper]
      capital-loss: {log1p: true, divisor: 12.0}      # log1p(US dollars) / 12. [Assumption — not in paper]
      hours-per-week: {log1p: false, divisor: 100.0}  # hours per week / 100. [Assumption — not in paper]
    age_bands: [[17, 24], [25, 34], [35, 44], [45, 54], [55, 64], [65, 90]]   # inclusive age bands (years) of the evaluation-only age attribute. [Assumption — not in paper]
  synthetic: null                     # unused. [Assumption — not in paper]
  femnist: null                       # unused. [Assumption — not in paper]

evaluation:                           # shared evaluation set and fairness breakdown. [Assumption — not in paper]
  eval_on: test                       # test = all 16,281 test rows (benchmark) | validation = stratified split of the training rows (tuning; the test file is not read). [Assumption — not in paper]
  validation_fraction: 0.2            # fraction of each client's (and the excluded group's) training rows held out when eval_on is validation. [Assumption — not in paper]
  group_attributes: [sex, race, age_band]   # evaluation-only protected attributes for the per-group breakdown (never model inputs). [Assumption — not in paper]

model:                                # network architecture. [Assumption — not in paper]
  hidden_sizes: [128, 64]             # MLP hidden layer widths (units); ReLU, no BatchNorm (Opacus needs per-sample gradients). Chosen from {[64,32],[128,64]} on validation, seeds 100-101. [Assumption — not in paper]

training:                             # local and centralised SGD settings. [Assumption — not in paper]
  rounds: 20                          # communication rounds (locked protocol). [Assumption — not in paper]
  local_epochs: 1                     # local passes over a client's rows per round. [Assumption — not in paper]
  batch_size: 64                      # minibatch size (rows); with DP the expected Poisson batch size. Chosen from {64, 256} on validation, seeds 100-101. [Assumption — not in paper]
  learning_rate: 0.01                 # SGD learning rate. Chosen from {0.005, 0.01, 0.03, 0.1} by mean validation F1 of centralised_mlp + fedavg (both splits), seeds 100-101; differences were < 0.01. [Assumption — not in paper]
  momentum: 0.9                       # SGD momentum (rebuilt every round: clients are stateless). [Assumption — not in paper]
  weight_decay: 0.0                   # L2 penalty coefficient. [Assumption — not in paper]
  centralised_epochs: 20              # epochs of the centralised MLP baselines (= rounds x local_epochs, equal number of passes). The centralised MLP keeps its SGD momentum across all its epochs (one optimiser), while federated and local-only clients rebuild the optimiser every round. [Assumption — not in paper]

federation:                           # server-side settings. [Assumption — not in paper]
  aggregation_rule: weighted          # weighted (sample-weighted FedAvg, n_k / sum n; the default) | unweighted (the unweighted mean (1/N) sum theta_i, kept as an ablation) | coordinate_median | trimmed_mean. [Assumption — not in paper]
  trimmed_mean_beta: 0.2             # fraction trimmed at each end by trimmed_mean (0 <= beta < 0.5); = num_byzantine / num_clients of the robustness suite (2 / 10): set by principle so trimmed_mean can drop every attacker per side, NOT tuned; a smaller value triggers a warning. [Assumption — not in paper]
  fraction_evaluate: 0.0              # client-side evaluation OFF for every condition: evaluating the global model on a client's own training rows releases an un-noised statistic of those private rows. All reported metrics come from the server-side evaluation on the public test set (or the validation rows when tuning). With 0, communication cost counts training traffic only. [Assumption — not in paper]
  engine: inprocess                   # inprocess (this repo's Flower Grid, no Ray; the default and the only tested engine). The value ray (flwr.simulation, needs flwr[simulation]) is [Target / TBD]: it has never completed a run; Ray workers failed to register on the development machine. [Assumption — not in paper]

dp:                                   # differential-privacy settings (DP-SGD via Opacus). [Assumption — not in paper]
  delta: 1.0e-5                       # DP delta per client; below 1 / smallest client size (Self-employed, about 3,657 rows). [Assumption — not in paper]
  accountant: rdp                     # Opacus RDP accountant for noise calibration and epsilon reporting. [Assumption — not in paper]
  epsilon_tolerance: 0.01             # precision of Opacus's noise-multiplier binary search (epsilon units). [Assumption — not in paper]
  sweep:                              # DP conditions of the main suite: weighted FedAvg + per-client record-level DP-SGD. [Assumption — not in paper]
    - name: fedavg_dp_eps8            # condition name. [Assumption — not in paper]
      mode: target_epsilon            # calibrate each client's noise so it spends at most target_epsilon over all 20 rounds. [Assumption — not in paper]
      target_epsilon: 8.0             # per-client epsilon after the last round. [Assumption — not in paper]
      noise_multiplier: null          # unused in target_epsilon mode. [Assumption — not in paper]
      max_grad_norm: 2.0              # per-sample gradient clipping norm (L2); chosen from {0.5, 1, 2, 4} on validation F1 at eps 3 and 1, seeds 100-101. [Assumption — not in paper]
    - name: fedavg_dp_eps3            # condition name. [Assumption — not in paper]
      mode: target_epsilon            # see above. [Assumption — not in paper]
      target_epsilon: 3.0             # per-client epsilon after the last round. [Assumption — not in paper]
      noise_multiplier: null          # unused in target_epsilon mode. [Assumption — not in paper]
      max_grad_norm: 2.0              # per-sample gradient clipping norm (L2); chosen from {0.5, 1, 2, 4} on validation F1 at eps 3 and 1, seeds 100-101. [Assumption — not in paper]
    - name: fedavg_dp_eps1            # condition name. [Assumption — not in paper]
      mode: target_epsilon            # see above. [Assumption — not in paper]
      target_epsilon: 1.0             # per-client epsilon after the last round. [Assumption — not in paper]
      noise_multiplier: null          # unused in target_epsilon mode. [Assumption — not in paper]
      max_grad_norm: 2.0              # per-sample gradient clipping norm (L2); chosen from {0.5, 1, 2, 4} on validation F1 at eps 3 and 1, seeds 100-101. [Assumption — not in paper]

partitions:                           # client splits. [Assumption — not in paper]
  splits: [workclass, iid]            # main-suite splits: workclass (Private / Government / Self-employed, natural non-IID) and the IID control (same rows, 3 equal shuffled clients). Rows with workclass ?, Without-pay or Never-worked (1,857 training rows) belong to NO client; they are used only by the centralised baselines trained on all rows (centralised_mlp, svm, random_forest, decision_tree, majority_class) and, on the test side, in the shared test set. [Assumption — not in paper]
  ablations: [iid_size_matched, label_skew, quantity_skew]   # partitions suite (same client rows). [Assumption — not in paper]
  label_skew_alpha: 0.5               # Dirichlet concentration over the two labels (smaller = more skew). [Assumption — not in paper]
  quantity_skew:                      # client-size distribution of the quantity_skew partition. [Assumption — not in paper]
    kind: dirichlet                   # dirichlet | power_law client sizes. [Assumption — not in paper]
    alpha: 1.0                        # Dirichlet concentration of client sizes. [Assumption — not in paper]
    exponent: 1.5                     # power-law exponent (used when kind is power_law). [Assumption — not in paper]
  min_client_rows: 500                # every skewed client must hold at least this many rows (redraw otherwise). [Assumption — not in paper]
  max_resample_attempts: 100          # maximum redraws before giving up. [Assumption — not in paper]

baselines:                            # centralised baselines. [Assumption — not in paper]
  methods: [centralised_mlp, centralised_mlp_client_rows, svm, random_forest, decision_tree, majority_class]   # all trained centrally with the same preprocessing and evaluated on the same test set. [Assumption — not in paper]
  svm:                                # scikit-learn SVC settings. [Assumption — not in paper]
    kernel: rbf                       # scikit-learn SVC kernel (RBF; dominates the baseline run time). [Assumption — not in paper]
    C: 1.0                            # SVC regularisation strength. [Assumption — not in paper]
    gamma: scale                      # RBF width rule of scikit-learn (1 / (n_features * Var(X))). [Assumption — not in paper]
    cache_size_mb: 1000.0             # SVC kernel cache in MB (speed only). [Assumption — not in paper]
  random_forest:                      # scikit-learn RandomForestClassifier settings. [Assumption — not in paper]
    n_estimators: 200                 # number of trees. [Assumption — not in paper]
    max_depth: null                   # maximum tree depth (null = unlimited). [Assumption — not in paper]
    min_samples_leaf: 2               # minimum rows per leaf. [Assumption — not in paper]
  decision_tree:                      # scikit-learn DecisionTreeClassifier settings. [Assumption — not in paper]
    max_depth: 10                     # maximum depth. [Assumption — not in paper]
    min_samples_leaf: 5               # minimum rows per leaf. [Assumption — not in paper]

robustness:                           # robustness suite: IID clients, some Byzantine. [Assumption — not in paper]
  num_clients: 10                     # IID clients of the robustness suite (union of the client rows). [Assumption — not in paper]
  num_byzantine: 2                    # how many of them attack (the last client ids). [Assumption — not in paper]
  attack:                             # behaviour of the Byzantine clients. [Assumption — not in paper]
    kind: sign_flip                   # none | sign_flip | scale | gaussian_noise | label_flip. [Assumption — not in paper]
    scale_factor: 10.0                # update multiplier of the scale attack. [Assumption — not in paper]
    noise_std: 1.0                    # per-coordinate std of the gaussian_noise attack. [Assumption — not in paper]
  aggregation_rules: [weighted, coordinate_median, trimmed_mean]   # rules compared under attack. [Assumption — not in paper]

aggregation_suite:                    # aggregation suite: sample-weighted FedAvg vs the unweighted mean. [Assumption — not in paper]
  split: workclass                    # split on which weighted and unweighted FedAvg are compared (sizes differ, so they differ). [Assumption — not in paper]
  rules: [weighted, unweighted]       # weighted = sample-weighted FedAvg (default); unweighted = the unweighted mean, kept as an ablation. [Assumption — not in paper]

runtime:                              # process-level settings (not model hyperparameters). [Assumption — not in paper]
  workers: 8                          # default number of worker processes (--workers overrides); results do not depend on it. [Assumption — not in paper]
  torch_threads: 1                    # torch intra-op threads per process (1 keeps results identical across worker counts). [Assumption — not in paper]

smoke_overrides:                      # merged with --smoke: a quick end-to-end check on the real files (still needs data/adult). [Assumption — not in paper]
  experiment:                         # run identity: name, seeds, output folder. [Assumption — not in paper]
    seeds: [100]                      # one development seed. [Assumption — not in paper]
  training:                           # local and centralised SGD settings. [Assumption — not in paper]
    rounds: 2                         # two rounds only. [Assumption — not in paper]
    centralised_epochs: 2             # = rounds x local_epochs. [Assumption — not in paper]
  baselines:                          # centralised baselines. [Assumption — not in paper]
    methods: [centralised_mlp, decision_tree, majority_class]   # skip the slow RBF SVM and the forest in smoke runs. [Assumption — not in paper]
  runtime:                            # process-level settings (not model hyperparameters). [Assumption — not in paper]
    workers: 4                        # four worker processes. [Assumption — not in paper]
```
