# Architecture diagrams

Four diagrams of the draft research code, each drawn from the code as it is. Every file has one
Mermaid diagram followed by an explanation that names the implementing source files, the
assumptions, the design alternatives considered and what the diagram does not show. Other
documents embed these diagrams through the generated blocks `diagram_*`, which
`experiments/build_results.py` extracts from the first Mermaid block of each file, so the file
names below are fixed.

| File | Diagram |
|---|---|
| [01-system-overview.md](01-system-overview.md) | The federated module, the shared `src/common` package and the experiment and reporting layer, with the edges between them. |
| [02-federated-detail.md](02-federated-detail.md) | One Flower round in one process: client-side local training and DP-SGD, the reply message, the in-process grid, weighted FedAvg and server-side evaluation. |
| [03-trust-boundary.md](03-trust-boundary.md) | The client boundary that raw data never crosses, every artefact that does cross it in each direction, and whether the (ε, δ) guarantee covers it. |
| [04-experiment-flow.md](04-experiment-flow.md) | Configuration, run, raw CSV, summary, Before / After CSV and chart, generated block, and the documents that include it. |

Diagrams 1 and 4 describe the repository as a whole; diagrams 2 and 3 zoom into the federated
round and its privacy boundary. A reader who wants to know where a number in the documentation
comes from should start with diagram 4; a reader who wants to know what leaves a client should
start with diagram 3.

None of the diagrams shows a large-language-model component or an integrated demo, because this
draft implements neither. None of them shows a result; reproduced values appear only in the
generated blocks.
