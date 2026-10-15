# A6: how much the Thai scores depend on the labels and on the sample

Made with `notebooks/a6_label_and_sample_effects.ipynb`. It compares the three scorings of `src/evaluate.py` for the 14 seed 42 runs of part B
on the Thai patches:

| scoring | patches | labels |
|---|---|---|
| (a) `worldcover` | 3,023 pure patches | WorldCover |
| (b) `clean-vote` | 199 hand-labelled patches | vote of three people (step A1) |
| (c) `clean-wc` | the same 199 patches | WorldCover |

The scores were recomputed and agree with `outputs/b/b9_results.csv` (all 42). Intervals come from 2,000 bootstrap draws (seed 42).
Nothing was trained or predicted for this step. All findings are measured; the one guess is marked.

## Findings

### 1. The labels hardly matter for pure patches: (b) against (c)

The vote and WorldCover differ on 4 of the 199 patches (all in the Nan study area: 3 WorldCover forest that people call crop, 1 the other way).
Changing the labels moves macro-F1 by at most 0.021, and in 13 of 14 runs the 95% interval of the change includes zero
(`label_effect.csv`). Both label sets rank the 14 runs in exactly the same order (Spearman 1.000).

For pure patches, WorldCover equals the human vote on 195 of 199 patches: **98.0% (95% interval 94.9% to 99.2%)**, and on 188 of 189 where
nobody looked at the WorldCover label: 99.5% (97.1% to 99.9%) (`worldcover_reliability.csv`). This is far better than the about 75% quoted for
WorldCover as a whole, because pure patches are the easy case: large areas of one class, no edges and no mixed pixels. Mixed patches were not checked.

**So:** the 3,023 WorldCover-labelled patches can be used as the main Thai test set; the noisy labels do not distort the scores of pure patches.

### 2. Accuracy depends on the class mix, macro-F1 much less: (c) against (a)

The 3,023 patches are 59% forest; the 199 are 25% of each class. Forest is the hardest Thai class, so it pulls the accuracy of the large set down.
Drawing 199-patch samples from the 3,023 with exactly the class-and-area mix of the clean set changes the accuracy by -0.03 to +0.32 depending on the run
(ResNet-18 on L1C without a fix: 0.705 on all 3,023, 0.814 for the clean-set mix). Macro-F1 moves much less (0.849 against 0.867).

**So:** accuracies on the 3,023 and on the 199 patches must not be compared with each other. Compare macro-F1, as the team rules already say.

### 3. The 199 patches happen to be a slightly hard sample (measured; cause checked)

Against 2,000 random samples with the same class-and-area mix, the macro-F1 of the clean set (with WorldCover labels) is below the median in
10 of 14 runs. For the main L1C models it lies at the low edge: lower than 97% of the draws for ResNet-18 without a fix, 99% with Thai normalisation,
97% with grassland removed, and 82% with AdaBN (`sample_effect.csv`, `sample_effect_main_model.png`).
The patches were drawn at random by `src/make_label_sample.py` and their purity within each class equals that of all pure patches, so this is
the chance of one small draw, not a biased selection. The 14 runs are scored on the same 199 patches, so they are not 14 independent signs.

**So:** for the size of an effect, quote the 3,023-patch scores. Use the 199 patches to check the labels, not as the headline number.

### 4. A score on 199 patches is only precise to about +-0.04 macro-F1

The 95% interval of macro-F1 on the clean set is 0.07 to 0.12 wide (`clean_set_intervals.csv`). Paired comparisons on the 199 patches
(`method_differences_on_clean_set.csv`):

| comparison (macro-F1, vote labels) | on 199 | 95% interval | on 3,023 | clear on 199? |
|---|---|---|---|---|
| AdaBN minus no fix (ResNet-18, L1C) | +0.043 | +0.004 to +0.082 | +0.048 | yes |
| Thai normalisation minus no fix | -0.021 | -0.064 to +0.022 | -0.005 | no |
| AdaBN minus Thai normalisation | +0.064 | +0.025 to +0.104 | +0.053 | yes |
| ResNet-18 minus Random Forest (L1C) | +0.049 | -0.001 to +0.100 | +0.143 | no (just) |
| L1C minus L2A (ResNet-18) | +0.459 | +0.395 to +0.522 | +0.505 | yes |

**So:** the main claims (L1C much better than L2A; AdaBN better than no fix and better than Thai normalisation) hold on the hand-labelled patches too.
"Thai normalisation does not help" cannot be confirmed or rejected on 199 patches.

### 5. Both test sets rank the methods the same way

Spearman rank correlation of macro-F1 over the 14 runs: 0.982 between (a) and (b), 0.982 between (a) and (c), 1.000 between (b) and (c)
(`three_scorings.png`). Which method is best does not depend on the test set or the labels.

## For the slides and the report

- Main Thai number: macro-F1 on the 3,023 pure patches with WorldCover labels. Next to it the 199-patch score, as a check with human labels.
- One sentence on the labels: on pure patches WorldCover agrees with three human labellers 98% of the time (95% interval 95% to 99%), so the
  noisy labels do not change the scores or the order of the methods.
- Do not compare accuracies between the two test sets; they have different class mixes.

## Files in this folder

| file | content |
|---|---|
| `three_scorings.png` | macro-F1 of every run under the three scorings, with the 95% interval of the 199-patch score |
| `sample_effect_main_model.png` | what chance gives for a 199-patch set with the clean-set mix, main model without and with AdaBN |
| `label_effect.csv` | (b) minus (c) for every run, with paired bootstrap interval |
| `sample_effect.csv` | (a), (c) and the resampled range for every run |
| `clean_set_intervals.csv` | 95% interval of the 199-patch macro-F1 for every run |
| `method_differences_on_clean_set.csv` | paired comparisons of the main methods on the 199 patches |
| `worldcover_reliability.csv` | agreement of WorldCover with the human vote, with Wilson interval |
