```mermaid
flowchart TD
    Cfg["Configs, YAML"]
    Ref["run_reference"]
    RunF["run_federated_benchmark"]
    Own["Own runs, git-ignored"]
    Raw["Raw CSVs per round,<br/>client, group"]
    Summary["summary.csv<br/>mean and std over seeds"]
    Plot["plot_federated"]
    Figs["Per-run figures"]
    BACfg["before_after.yaml"]
    Paper["paper_values.py"]
    Arch["Architecture diagrams"]
    Build["build_results"]
    BAOut["Before / After CSVs<br/>and charts"]
    Gen["Generated Markdown blocks"]
    Sync["sync_docs"]
    Docs["README, RESULTS, report,<br/>poster, slides"]
    Check["Stale-block check<br/>in CI and tests"]

    Cfg --> RunF
    Ref -->|"reporting seeds"| RunF
    RunF -.->|"make federated"| Own
    RunF --> Raw
    Raw --> Summary
    Raw --> Plot
    Plot --> Figs
    Summary --> Build
    BACfg --> Build
    Paper --> Build
    Cfg -->|"verbatim listings"| Build
    Arch -->|"first mermaid block"| Build
    Build --> BAOut
    Build --> Gen
    Gen --> Sync
    Sync --> Docs
    Docs --> Check
    Gen --> Check
```
