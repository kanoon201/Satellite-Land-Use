# B7: fine-tuning on a few Thai patches

Test: 1540 pure Thai patches in test blocks (WorldCover labels) and 106 hand-labelled patches in test blocks (vote). Fine-tuning patches come from train blocks only. Mean ± SD over draws 0-2. Macro-F1 over 4 Thai classes.

| Run | Thai patches per class | Macro-F1 (test blocks, WorldCover) | Macro-F1 (hand-labelled) | Forest recall Nan / Kanchanaburi | Predicted grassland | EuroSAT test macro-F1 |
|---|---|---|---|---|---|---|
| zero-shot | 0 | 0.847 | 0.839 | 0.58 / 0.58 | 23.3% | 0.992 |
| zero-shot (grassland removed) | 0 | 0.829 | 0.872 | 0.72 / 0.70 | 0.0% | 0.992 |
| AdaBN | 0 | 0.894 | 0.888 | 0.75 / 0.66 | 19.5% | 0.874 |
| AdaBN (grassland removed) | 0 | 0.912 | 0.932 | 0.99 / 0.85 | 0.0% | 0.874 |
| finetune | 10 | 0.969 ± 0.009 | 0.968 ± 0.006 | 0.94 / 0.96 | 0.1% | 0.881 ± 0.024 |
| finetune | 25 | 0.977 ± 0.014 | 0.987 ± 0.003 | 0.93 / 0.98 | 0.1% | 0.850 ± 0.021 |
| finetune | 50 | 0.979 ± 0.015 | 0.990 ± 0.010 | 0.94 / 0.99 | 0.0% | 0.800 ± 0.034 |
| adabn+finetune | 10 | 0.954 ± 0.016 | 0.951 ± 0.010 | 0.98 / 0.90 | 0.3% | 0.712 ± 0.031 |
| adabn+finetune | 25 | 0.968 ± 0.022 | 0.954 ± 0.015 | 0.99 / 0.92 | 0.3% | 0.707 ± 0.036 |
| adabn+finetune | 50 | 0.989 ± 0.008 | 0.974 ± 0.015 | 0.99 / 0.99 | 0.1% | 0.701 ± 0.035 |
