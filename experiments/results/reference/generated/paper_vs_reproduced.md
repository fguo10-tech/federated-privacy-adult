Dataset: UCI Adult / Census Income (1994 U.S. Census extraction; 48,842 rows; workclass clients Private / Government / Self-employed plus IID control)

| Method | Metric | Paper-reported | Reproduced (mean ± std) | Notes |
|---|---|---|---|---|
| SVM (centralised) | Accuracy | 0.3124 [Paper-reported] | 0.8401 ± 0.0000 [Reproduced] | Centralised on pooled Adult training rows. |
| SVM (centralised) | Precision | 0.1032 [Paper-reported] | 0.6983 ± 0.0000 [Reproduced] | Centralised on pooled Adult training rows. |
| SVM (centralised) | Recall | 0.2987 [Paper-reported] | 0.5686 ± 0.0000 [Reproduced] | Centralised on pooled Adult training rows. |
| SVM (centralised) | F1 | 0.1543 [Paper-reported] | 0.6268 ± 0.0000 [Reproduced] | Centralised on pooled Adult training rows. |
| Random Forest (centralised) | Accuracy | 0.3278 [Paper-reported] | 0.8650 ± 0.0003 [Reproduced] | Centralised on pooled Adult training rows. |
| Random Forest (centralised) | Precision | 0.1147 [Paper-reported] | 0.7726 ± 0.0020 [Reproduced] | Centralised on pooled Adult training rows. |
| Random Forest (centralised) | Recall | 0.3156 [Paper-reported] | 0.6072 ± 0.0032 [Reproduced] | Centralised on pooled Adult training rows. |
| Random Forest (centralised) | F1 | 0.1679 [Paper-reported] | 0.6800 ± 0.0014 [Reproduced] | Centralised on pooled Adult training rows. |
| Decision Tree (centralised) | Accuracy | 0.2945 [Paper-reported] | 0.8598 ± 0.0001 [Reproduced] | Centralised on pooled Adult training rows. |
| Decision Tree (centralised) | Precision | 0.0978 [Paper-reported] | 0.7587 ± 0.0003 [Reproduced] | Centralised on pooled Adult training rows. |
| Decision Tree (centralised) | Recall | 0.2764 [Paper-reported] | 0.5964 ± 0.0005 [Reproduced] | Centralised on pooled Adult training rows. |
| Decision Tree (centralised) | F1 | 0.1432 [Paper-reported] | 0.6678 ± 0.0004 [Reproduced] | Centralised on pooled Adult training rows. |
| Weighted FedAvg (no DP) | Accuracy | 0.3500 [Paper-reported] | 0.8372 ± 0.0018 [Reproduced] | Weighted FedAvg, workclass clients, no DP, 20 rounds. |
| Weighted FedAvg (no DP) | Precision | 0.1255 [Paper-reported] | 0.6626 ± 0.0154 [Reproduced] | Weighted FedAvg, workclass clients, no DP, 20 rounds. |
| Weighted FedAvg (no DP) | Recall | 0.3500 [Paper-reported] | 0.6361 ± 0.0300 [Reproduced] | Weighted FedAvg, workclass clients, no DP, 20 rounds. |
| Weighted FedAvg (no DP) | F1 | 0.1815 [Paper-reported] | 0.6485 ± 0.0089 [Reproduced] | Weighted FedAvg, workclass clients, no DP, 20 rounds. |
| Majority class | Accuracy | not reported | 0.7638 ± 0.0000 [Reproduced] | Always predicts <=50K; shown because accuracy alone is misleading. |
| Majority class | Precision | not reported | 0.0000 ± 0.0000 [Reproduced] | Always predicts <=50K; shown because accuracy alone is misleading. |
| Majority class | Recall | not reported | 0.0000 ± 0.0000 [Reproduced] | Always predicts <=50K; shown because accuracy alone is misleading. |
| Majority class | F1 | not reported | 0.0000 ± 0.0000 [Reproduced] | Always predicts <=50K; shown because accuracy alone is misleading. |

The Paper-reported column is shown for reference only: it comes from a different setup, while the Reproduced column is measured on UCI Adult (binary; precision, recall and F1 are those of the >50K class) with the workclass clients. No difference between the two columns is computed anywhere in this repository.
