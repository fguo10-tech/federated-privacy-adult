# Security policy

## Status

This is draft research code, version 0.1.x. It is maintained on a best-effort basis. There is no
promised response time and no supported-release window; fixes land on `main`.

## How to report privately

Use GitHub private vulnerability reporting: open the repository's **Security** tab and choose
**Report a vulnerability**. Never report a security issue in a public issue, pull request or
discussion. The same private channel is used for Code of Conduct reports.

Please include what you found, the commit you looked at, the steps or the config that show the
problem, and what you think the impact is. Give the maintainers a reasonable chance to fix the
problem before you disclose it publicly.

## What counts as a security issue here

- Ordinary code vulnerabilities, for example unsafe deserialisation, path traversal in the run
  and data scripts, or a download that skips checksum verification.
- Errors in differential-privacy accounting: a reported epsilon that is lower than the true
  guarantee, a wrong sampling rate or delta, noise or clipping that is not applied as configured,
  or any path that trains without the privacy engine when the config asks for it.
- Anything that lets raw client rows cross the client boundary, for example in messages, logs,
  metrics, checkpoints or result files.
- A quantity that leaves a client but is missing from `RELEASED_QUANTITIES` in
  `src/federated/privacy.py`, or is listed there as covered by the (ε, δ) guarantee when it is
  not.
- A path by which the server could learn or predict a client's DP noise, for example through
  the run seed or any other field of a message.
- Committed secrets (keys, tokens, credentials) or committed sensitive data, including any copy
  of a real dataset or a non-synthetic record.

## Out of scope

- Weaknesses that follow from the stated threat model of a simulated component (see the warning
  below), unless the documentation claims a protection the code does not give.
- Vulnerabilities in third-party packages that are not triggered by this repository's code;
  please report those upstream.
- Results that differ from this repository's reference run. These are research questions, not
  security issues; to report a difference from the reference run, use the "Replication report"
  issue form.
- Denial of service by running very large configs on your own machine.

## Warning

- Do not use this code with real medical, financial or personal data without an independent
  privacy and security review.
- The secure-aggregation component is a simulation of the protocol's data flow. It is not
  cryptography and gives no confidentiality guarantee.
- The clients are simulated in one process from one public dataset. In the benchmark the harness
  sets each client's DP secret deterministically from the run seed so that runs are reproducible;
  a deployment must not do this.
- The (ε, δ) guarantee is record-level, against an honest-but-curious server, for one run. It does
  not cover the row count each client sends for weighted FedAvg or the quantities derived from
  it; see [docs/architecture/03-trust-boundary.md](docs/architecture/03-trust-boundary.md).
