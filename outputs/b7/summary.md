# B7: fine-tuning on a few labelled Thai patches

Made with `src/finetune_resnet.py`. Question: how much does a small number of labelled Thai patches add to the ResNet-18 trained on EuroSAT?

## Set-up

- **Base model.** ResNet-18 on 12 bands, trained on EuroSAT with `src/train_resnet.py` (seed 42, the team's code and settings).
  The team's checkpoint is not in git, so it was trained again on this machine: EuroSAT validation macro-F1 0.990, best epoch 17
  (`base_model_training_log.csv`). Its zero-shot macro-F1 on all 3,023 pure Thai patches is 0.860 (team checkpoint: 0.849; the SD over
  three seeds in part B is 0.027). The team's prediction files and training logs were not changed.
- **Split.** Each study area is cut into 36 blocks of 8 x 8 patches (5.1 km); half of the blocks of each area are train blocks, half test
  blocks (seed 42, `split.csv`). Fine-tuning patches come only from the inside of train blocks, so every training patch is at least one patch
  (640 m) away from every test patch (checked: closest distance 2 patches). The 199 hand-labelled patches never train.
- **Test.** 1,540 pure patches in test blocks with WorldCover labels (forest 932, crop 260, water 216, built-up 132), and the 106
  hand-labelled patches that lie in test blocks, with the vote. Baselines (zero-shot, AdaBN) are scored on exactly these patches.
- **Fine-tuning.** 10, 25 or 50 WorldCover-labelled pure patches per Thai class (forest half from Nan, half from Kanchanaburi), three random
  draws each. All weights trained, AdamW lr 1e-4, 30 epochs, flips and 90-degree turns, BatchNorm statistics frozen. Two variants: starting from the
  EuroSAT checkpoint (`finetune`), or from the AdaBN-adapted model (`adabn+finetune`). The settings were fixed in advance and not tuned
  (there is no Thai validation set), so the test patches did not influence them.

## Results

Mean ± SD over the three draws (`b7_table.md`, `results.csv`, `b7_finetune.png`). Macro-F1 over the four Thai classes.

| Run | Thai patches per class | Macro-F1, test blocks (WorldCover) | Macro-F1, hand-labelled | Forest recall Nan / Kanchanaburi | EuroSAT test macro-F1 |
|---|---|---|---|---|---|
| zero-shot | 0 | 0.847 | 0.839 | 0.58 / 0.58 | 0.992 |
| AdaBN | 0 | 0.894 | 0.888 | 0.75 / 0.66 | 0.874 |
| AdaBN, grassland removed | 0 | 0.912 | 0.932 | 0.99 / 0.85 | 0.874 |
| finetune | 10 | 0.969 ± 0.009 | 0.968 ± 0.006 | 0.94 / 0.96 | 0.881 ± 0.024 |
| finetune | 25 | 0.977 ± 0.014 | 0.987 ± 0.003 | 0.93 / 0.98 | 0.850 ± 0.021 |
| finetune | 50 | 0.979 ± 0.015 | 0.990 ± 0.010 | 0.94 / 0.99 | 0.800 ± 0.034 |
| adabn+finetune | 10 | 0.954 ± 0.016 | 0.951 ± 0.010 | 0.98 / 0.90 | 0.712 ± 0.031 |
| adabn+finetune | 25 | 0.968 ± 0.022 | 0.954 ± 0.015 | 0.99 / 0.92 | 0.707 ± 0.036 |
| adabn+finetune | 50 | 0.989 ± 0.008 | 0.974 ± 0.015 | 0.99 / 0.99 | 0.701 ± 0.035 |

## Findings

1. **A few labelled Thai patches help more than any change without labels.** 10 patches per class (40 in total) raise macro-F1 on the test
   blocks from 0.847 (zero-shot) to 0.969 ± 0.009, above the best fix without labels (AdaBN with grassland removed, 0.912). The hand-labelled
   patches show the same (0.839 to 0.968).
2. **More patches add little.** 25 and 50 patches per class give 0.977 and 0.979; the spread between draws (0.01 to 0.02) is about as large as
   the gain. Most of the benefit comes from the first 10 patches per class.
3. **The dry-forest mistake of A7 disappears.** Forest recall goes from 0.58 to about 0.95 in both study areas, and the share of patches predicted as
   grassland from 23% to 0%. This fits the explanation in A7 that the model had never seen dry-season forest: a few examples are enough to show it.
4. **AdaBN before fine-tuning does not help.** `adabn+finetune` is not better than `finetune` (lower at 10 and 25 patches, within the spread at 50).
5. **The fine-tuned model forgets part of Europe.** EuroSAT test macro-F1 falls from 0.992 to 0.881 (10 patches) and 0.800 (50 patches): the model
   specialises in the Thai study areas. (For the AdaBN variants the EuroSAT score is low mainly because their BatchNorm statistics are Thai.)

## Limits

- Training and test patches come from the **same study areas, the same dates and the same composite**. They are at least 640 m apart, but the
  landscape is the same. The numbers show how well the model adapts **within these study areas**; a new region of Thailand would likely score lower.
  Each class still comes from one area (see `DATA_NOTES.md`).
- The test set is the 1,540 pure patches of the test blocks, not the 3,023 of part B (those include the training patches), so the scores here are
  not in the B9 table. Compare runs only within this table.
- Fine-tuning labels are WorldCover labels; A6 found them 98% in agreement with the human vote on pure patches.
- One base model (seed 42, trained on this machine) and three draws of training patches.

## Files

| file | content |
|---|---|
| `b7_table.md`, `results.csv` | scores of every run (results.csv: also per-class F1, both test sets) |
| `b7_finetune.png` | macro-F1 against Thai patches per class, and EuroSAT test after fine-tuning |
| `split.csv` | block and side of every Thai patch, and whether it may train |
| `training_patches.csv` | the Thai patches used by each run |
| `predictions/` | Thai predictions of every run, same columns as `outputs/predictions/`; do not score them on all pure patches |
| `base_model_training_log.csv` | training log of the base model trained on this machine |
