# Source paper

This file is a short record of the source paper of this repository: its citation, its method
in one paragraph, and the values it reports.

## Citation

Zhang, Y., Liu, J., Wang, J., Dai, L., Guo, F., and Cai, G. (2025). Federated Learning for
Cross-Domain Data Privacy: A Distributed Approach to Secure Collaboration. In *2025 8th
International Symposium on Big Data and Applied Statistics (ISBDAS)*, pp. 824–828. IEEE.

| Field | Value |
|---|---|
| Authors (in order) | Yiwei Zhang, Jie Liu, Jiawei Wang, Lu Dai, Fan Guo, Guohui Cai |
| Title | Federated Learning for Cross-Domain Data Privacy: A Distributed Approach to Secure Collaboration |
| Venue | 2025 8th International Symposium on Big Data and Applied Statistics (ISBDAS), pp. 824–828, IEEE |
| DOI | [10.1109/ISBDAS64762.2025.11116917](https://doi.org/10.1109/ISBDAS64762.2025.11116917) |
| IEEE Xplore document | [11116917](https://ieeexplore.ieee.org/document/11116917/) |
| arXiv | [2504.00282](https://arxiv.org/abs/2504.00282) |
| BibTeX key | `zhang2025federated` in [references.bib](references.bib) |

## Method

Clients in medical, financial and user-data domains each hold a local dataset and train a
shared model locally by gradient descent on their own loss. Only model parameters are sent to a
central server, which combines the client updates into a global model by averaging them, and
noise is added to the clients' gradient updates for differential privacy. Homomorphic
encryption and secure multi-party computation are named as further techniques for protecting
the updates. The method is compared with support vector machine, random forest and decision
tree baselines on accuracy, precision, recall and F1.

## Reported values

The values below are printed in Table 1 of the paper and are `[Paper-reported]`. They come from
a different setup, are shown for reference only and are not compared with values from this code.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| SVM | 0.3124 [Paper-reported] | 0.1032 [Paper-reported] | 0.2987 [Paper-reported] | 0.1543 [Paper-reported] |
| Random Forest | 0.3278 [Paper-reported] | 0.1147 [Paper-reported] | 0.3156 [Paper-reported] | 0.1679 [Paper-reported] |
| Decision Tree | 0.2945 [Paper-reported] | 0.0978 [Paper-reported] | 0.2764 [Paper-reported] | 0.1432 [Paper-reported] |
| The paper's method | 0.35 [Paper-reported] | 0.1255 [Paper-reported] | 0.35 [Paper-reported] | 0.1815 [Paper-reported] |
