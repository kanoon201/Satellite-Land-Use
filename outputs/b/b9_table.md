# B9: comparison of all experiments

Seed 42. Thai scores: 4 classes, grassland predictions count as errors. WorldCover = 3,023 pure patches; clean = 199 hand-labelled patches against the vote.

| Model | Test data | Fix | Acc (WorldCover) | Macro-F1 (WorldCover) | Macro-F1 (clean) | Predicted grassland | Forest recall Nan / Kanchanaburi |
|---|---|---|---|---|---|---|---|
| Random Forest | eurosat-test | — | 0.944 | 0.949 | — | — | — |
| ResNet-18 (12 bands) | eurosat-test | — | 0.992 | 0.991 | — | — | — |
| Random Forest | thai-L1C | — | 0.605 | 0.707 | 0.782 | 4.4% | 0.29 / 0.43 |
| Random Forest | thai-L1C | grassland removed | 0.610 | 0.700 | 0.779 | 0.0% | 0.31 / 0.43 |
| Random Forest | thai-L2A | — | 0.351 | 0.467 | 0.491 | 64.3% | 0.18 / 0.36 |
| Random Forest | thai-L2A | grassland removed | 0.830 | 0.767 | 0.790 | 0.0% | 0.92 / 0.92 |
| ResNet-18 (12 bands) | thai-L1C | — | 0.705 | 0.849 | 0.831 | 24.8% | 0.53 / 0.64 |
| ResNet-18 (12 bands) | thai-L1C | AdaBN | 0.787 | 0.897 | 0.874 | 18.0% | 0.67 / 0.74 |
| ResNet-18 (12 bands) | thai-L1C | AdaBN + grassland removed | 0.940 | 0.930 | 0.934 | 0.0% | 0.99 / 0.92 |
| ResNet-18 (12 bands) | thai-L1C | Thai normalisation | 0.760 | 0.844 | 0.810 | 19.9% | 0.69 / 0.75 |
| ResNet-18 (12 bands) | thai-L1C | Thai normalisation + grassland removed | 0.941 | 0.903 | 0.903 | 0.0% | 0.97 / 0.98 |
| ResNet-18 (12 bands) | thai-L1C | grassland removed | 0.886 | 0.885 | 0.894 | 0.0% | 0.80 / 0.87 |
| ResNet-18 (12 bands) | thai-L2A | — | 0.221 | 0.344 | 0.371 | 75.8% | 0.00 / 0.16 |
| ResNet-18 (12 bands) | thai-L2A | B9 set to EuroSAT mean | 0.290 | 0.493 | 0.506 | 53.4% | 0.00 / 0.02 |
| ResNet-18 (12 bands) | thai-L2A | B9 set to EuroSAT mean + grassland removed | 0.422 | 0.589 | 0.666 | 0.0% | 0.01 / 0.06 |
| ResNet-18 (12 bands) | thai-L2A | grassland removed | 0.689 | 0.755 | 0.776 | 0.0% | 0.44 / 0.68 |
