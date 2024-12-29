Dataset: UCI Adult / Census Income (1994 U.S. Census extraction; 48,842 rows; workclass clients Private / Government / Self-employed plus IID control)

Source label: [Reproduced]. Extended privacy study from `configs/federated_dp.yaml` (it inherits the locked Adult protocol and adds settings); mean ± sample standard deviation across seeds on the shared test set; source `experiments/results/reference/federated_adult_dp/summary.csv`. The locked sweep ε in {1, 3, 8, no noise}} is reported separately; this table shows where utility actually starts to fall.

**Target ε grid**

| Clients | Setting | Achieved ε (max over clients) | Accuracy | F1 (class >50K) | Precision | Recall |
|---|---|---|---|---|---|---|
| workclass clients | target ε = 0.1 | 0.094 ± 0.000 | 0.7514 ± 0.0229 | 0.4600 ± 0.1138 | 0.4674 ± 0.0595 | 0.4746 ± 0.1968 |
| workclass clients | target ε = 0.3 | 0.298 ± 0.000 | 0.8053 ± 0.0062 | 0.5466 ± 0.0272 | 0.6103 ± 0.0280 | 0.5000 ± 0.0658 |
| workclass clients | target ε = 0.5 | 0.496 ± 0.000 | 0.8200 ± 0.0094 | 0.6062 ± 0.0077 | 0.6296 ± 0.0328 | 0.5863 ± 0.0248 |
| workclass clients | target ε = 1 | 0.997 ± 0.000 | 0.8225 ± 0.0056 | 0.6236 ± 0.0082 | 0.6252 ± 0.0140 | 0.6222 ± 0.0034 |
| workclass clients | target ε = 2 | 1.999 ± 0.000 | 0.8393 ± 0.0023 | 0.6299 ± 0.0063 | 0.6908 ± 0.0069 | 0.5789 ± 0.0098 |
| workclass clients | target ε = 3 | 2.996 ± 0.000 | 0.8360 ± 0.0054 | 0.6364 ± 0.0090 | 0.6696 ± 0.0247 | 0.6081 ± 0.0326 |
| workclass clients | target ε = 5 | 5.000 ± 0.000 | 0.8379 ± 0.0034 | 0.6361 ± 0.0147 | 0.6769 ± 0.0025 | 0.6003 ± 0.0251 |
| workclass clients | target ε = 8 | 7.999 ± 0.000 | 0.8355 ± 0.0041 | 0.6272 ± 0.0079 | 0.6751 ± 0.0129 | 0.5857 ± 0.0092 |
| workclass clients | target ε = 16 | 16.000 ± 0.000 | 0.8359 ± 0.0038 | 0.6325 ± 0.0086 | 0.6723 ± 0.0185 | 0.5981 ± 0.0238 |
| workclass clients | no noise (ε = ∞) | ∞ | 0.8372 ± 0.0018 | 0.6485 ± 0.0089 | 0.6626 ± 0.0154 | 0.6361 ± 0.0300 |
| IID control | target ε = 0.1 | 0.096 ± 0.000 | 0.7602 ± 0.0194 | 0.4731 ± 0.0755 | 0.4973 ± 0.0524 | 0.4713 ± 0.1482 |
| IID control | target ε = 0.3 | 0.293 ± 0.000 | 0.8194 ± 0.0116 | 0.5867 ± 0.0151 | 0.6416 ± 0.0383 | 0.5425 ± 0.0299 |
| IID control | target ε = 0.5 | 0.496 ± 0.000 | 0.8382 ± 0.0026 | 0.6245 ± 0.0098 | 0.6910 ± 0.0058 | 0.5698 ± 0.0159 |
| IID control | target ε = 1 | 0.992 ± 0.000 | 0.8376 ± 0.0030 | 0.6271 ± 0.0083 | 0.6865 ± 0.0224 | 0.5785 ± 0.0284 |
| IID control | target ε = 2 | 1.994 ± 0.000 | 0.8409 ± 0.0015 | 0.6432 ± 0.0045 | 0.6842 ± 0.0093 | 0.6070 ± 0.0140 |
| IID control | target ε = 3 | 2.991 ± 0.000 | 0.8406 ± 0.0017 | 0.6314 ± 0.0051 | 0.6960 ± 0.0113 | 0.5781 ± 0.0144 |
| IID control | target ε = 5 | 4.992 ± 0.000 | 0.8406 ± 0.0021 | 0.6377 ± 0.0015 | 0.6890 ± 0.0091 | 0.5936 ± 0.0055 |
| IID control | target ε = 8 | 7.995 ± 0.000 | 0.8414 ± 0.0010 | 0.6337 ± 0.0029 | 0.6973 ± 0.0069 | 0.5809 ± 0.0088 |
| IID control | target ε = 16 | 15.998 ± 0.000 | 0.8413 ± 0.0016 | 0.6348 ± 0.0036 | 0.6958 ± 0.0082 | 0.5838 ± 0.0094 |
| IID control | no noise (ε = ∞) | ∞ | 0.8459 ± 0.0012 | 0.6402 ± 0.0114 | 0.7140 ± 0.0150 | 0.5812 ± 0.0278 |

**Clipping norm at a fixed target ε** (the default clipping norm is in the config)

| Clients | Setting | Achieved ε (max over clients) | Accuracy | F1 (class >50K) | Precision | Recall |
|---|---|---|---|---|---|---|
| workclass clients | target ε = 3, clipping norm 0.5 | 2.996 ± 0.000 | 0.8330 ± 0.0014 | 0.6136 ± 0.0045 | 0.6765 ± 0.0087 | 0.5617 ± 0.0120 |
| workclass clients | target ε = 3, clipping norm 1 | 2.996 ± 0.000 | 0.8338 ± 0.0018 | 0.6245 ± 0.0049 | 0.6697 ± 0.0076 | 0.5852 ± 0.0112 |
| workclass clients | target ε = 3, clipping norm 4 | 2.996 ± 0.000 | 0.8337 ± 0.0067 | 0.6284 ± 0.0074 | 0.6670 ± 0.0285 | 0.5954 ± 0.0234 |
| IID control | target ε = 3, clipping norm 0.5 | 2.991 ± 0.000 | 0.8338 ± 0.0015 | 0.5974 ± 0.0089 | 0.6982 ± 0.0062 | 0.5223 ± 0.0157 |
| IID control | target ε = 3, clipping norm 1 | 2.991 ± 0.000 | 0.8370 ± 0.0007 | 0.6268 ± 0.0056 | 0.6827 ± 0.0045 | 0.5796 ± 0.0122 |
| IID control | target ε = 3, clipping norm 4 | 2.991 ± 0.000 | 0.8401 ± 0.0042 | 0.6391 ± 0.0109 | 0.6858 ± 0.0245 | 0.6003 ± 0.0357 |

**Fixed noise multiplier** (ε is whatever results)

| Clients | Setting | Achieved ε (max over clients) | Accuracy | F1 (class >50K) | Precision | Recall |
|---|---|---|---|---|---|---|
| workclass clients | noise multiplier 0.5 | 25.753 ± 0.000 | 0.8357 ± 0.0030 | 0.6251 ± 0.0094 | 0.6790 ± 0.0201 | 0.5803 ± 0.0290 |
| workclass clients | noise multiplier 1 | 3.959 ± 0.000 | 0.8339 ± 0.0046 | 0.6276 ± 0.0111 | 0.6696 ± 0.0308 | 0.5934 ± 0.0410 |
| workclass clients | noise multiplier 2 | 1.341 ± 0.000 | 0.8122 ± 0.0133 | 0.5914 ± 0.0280 | 0.6104 ± 0.0368 | 0.5762 ± 0.0465 |
| IID control | noise multiplier 0.5 | 16.819 ± 0.000 | 0.8412 ± 0.0014 | 0.6369 ± 0.0046 | 0.6929 ± 0.0092 | 0.5894 ± 0.0131 |
| IID control | noise multiplier 1 | 2.179 ± 0.000 | 0.8390 ± 0.0019 | 0.6311 ± 0.0015 | 0.6880 ± 0.0107 | 0.5831 ± 0.0097 |
| IID control | noise multiplier 2 | 0.761 ± 0.000 | 0.8367 ± 0.0043 | 0.6301 ± 0.0042 | 0.6785 ± 0.0175 | 0.5887 ± 0.0121 |
