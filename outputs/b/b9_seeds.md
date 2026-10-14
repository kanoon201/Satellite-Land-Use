# B9 over training seeds

Mean ± SD (sample SD) over the seeds listed. Only the training seed changes; data, split and settings are fixed.
Thai scores: 4 classes. WorldCover = 3,023 pure patches; clean = 199 hand-labelled patches against the vote.

| Model | Test data | Fix | Seeds | Acc (WorldCover) | Macro-F1 (WorldCover) | Macro-F1 (clean) |
|---|---|---|---|---|---|---|
| ResNet-18 (12 bands) | eurosat-test | — | 1, 2, 42 | 0.992 ± 0.001 | 0.992 ± 0.001 | — |
| ResNet-18 (12 bands) | thai-L1C | — | 1, 2, 42 | 0.675 ± 0.036 | 0.831 ± 0.027 | 0.809 ± 0.024 |
| ResNet-18 (12 bands) | thai-L1C | AdaBN | 1, 2, 42 | 0.818 ± 0.030 | 0.915 ± 0.015 | 0.885 ± 0.012 |
| ResNet-18 (12 bands) | thai-L1C | AdaBN + grassland removed | 1, 2, 42 | 0.956 ± 0.018 | 0.946 ± 0.018 | 0.941 ± 0.012 |
| ResNet-18 (12 bands) | thai-L1C | Thai normalisation | 1, 2, 42 | 0.742 ± 0.030 | 0.823 ± 0.021 | 0.813 ± 0.015 |
| ResNet-18 (12 bands) | thai-L1C | Thai normalisation + grassland removed | 1, 2, 42 | 0.912 ± 0.026 | 0.866 ± 0.035 | 0.891 ± 0.011 |
| ResNet-18 (12 bands) | thai-L1C | grassland removed | 1, 2, 42 | 0.857 ± 0.045 | 0.862 ± 0.038 | 0.886 ± 0.022 |
| ResNet-18 (12 bands) | thai-L2A | — | 1, 2, 42 | 0.281 ± 0.097 | 0.396 ± 0.045 | 0.394 ± 0.022 |
| ResNet-18 (12 bands) | thai-L2A | B9 set to EuroSAT mean | 1, 2, 42 | 0.284 ± 0.062 | 0.504 ± 0.036 | 0.519 ± 0.055 |
| ResNet-18 (12 bands) | thai-L2A | B9 set to EuroSAT mean + grassland removed | 1, 2, 42 | 0.523 ± 0.099 | 0.667 ± 0.071 | 0.722 ± 0.049 |
| ResNet-18 (12 bands) | thai-L2A | grassland removed | 1, 2, 42 | 0.797 ± 0.094 | 0.806 ± 0.044 | 0.803 ± 0.023 |
