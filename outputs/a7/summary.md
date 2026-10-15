# A7: error analysis of the Thai predictions

Made with `notebooks/a7_error_analysis.ipynb`. Model: ResNet-18 on 12 bands, trained on EuroSAT, seed 42, applied to the
3,023 pure Thai **L1C** patches without any change ("zero-shot"). The AdaBN run and the seed 1 / seed 2 models are used for comparison.
No model was trained or run for this step; all predictions come from `outputs/predictions/`.

Each finding says whether it is **measured** (computed from all patches) or a **likely explanation** (fits the numbers and the
pictures, but not proven: no image of the same places in another season was checked). Results describe the study areas,
not Thailand as a whole (see `DATA_NOTES.md`).

## Findings

### 1. Almost all mistakes are vegetation classes taken for something else (measured)

891 of 3,023 patches are wrong.

| true class -> predicted | wrong patches | share of all mistakes |
|---|---|---|
| forest -> grassland | 618 | 69% |
| crop -> grassland | 132 | 15% |
| forest -> crop | 93 | 10% |
| crop -> built-up | 43 | 5% |
| other | 5 | 1% |

Built-up (Bangkok study area, 280 patches) and water (Kanchanaburi study area, 431 patches) have no mistake at all.
Forest is wrong in 36% of the Kanchanaburi patches and 47% of the Nan patches; crop in 32% of the Suphan Buri patches.
Figure: `error_pairs.png`; tables: `error_pairs.csv`, `errors_by_class_and_area.csv`.

### 2. Forest is taken for grassland or crop when it is less green and drier (measured; cause likely)

Inside the forest class, the patches predicted as grassland or crop have more red light and more short-wave infrared, and a lower NDVI,
than the forest patches predicted right. Short-wave infrared alone separates them well:

| forest patches, median | NDVI | B11 (SWIR) | B12 (SWIR) |
|---|---|---|---|
| Kanchanaburi, predicted right | 0.585 | 1,581 | 691 |
| Kanchanaburi, predicted grassland | 0.433 | 1,891 | 948 |
| Kanchanaburi, predicted crop | 0.343 | 2,079 | 1,161 |
| EuroSAT Forest (reference) | 0.724 | 1,372 | 540 |
| EuroSAT Pasture (reference) | 0.667 | 1,958 | 922 |

AUC for telling *forest -> grassland* from *forest, right*: B12 0.97, B11 0.95, NDVI 0.88. For *forest -> crop*: B12 0.997, NDVI 0.94.
The SWIR of the forest patches taken for grassland is close to EuroSAT Pasture and far from EuroSAT Forest.
In the pictures (`examples_forest_as_grassland.png`, `examples_forest_as_crop.png`) these patches show a mottled canopy with brown and
pink areas; the forest patches predicted right are an even dark green.

**Likely explanation:** the Thai images are a January-February dry-season composite. Part of the forest in these areas loses leaves or
dries out then, so the ground and dry litter show through. EuroSAT Forest is a dense green canopy, so the model has never seen a forest
that looks dry, and it picks the European class with the closest look.

On the map (`error_maps.png`) the forest mistakes gather on the lower hills next to the valleys and fields (Nan) and on the north-west
shore of the reservoir (Kanchanaburi), while the dense forest on the eastern mountains of Nan is predicted right. This is a visual reading
of the map, not a measured result.

### 3. Crop is taken for grassland when it is green, not when it is bare (measured; corrects our earlier guess)

We expected harvested, bare rice fields to be taken for grassland. The data say the opposite: the crop patches predicted as grassland are
**greener** than those predicted right (median NDVI 0.534 against 0.443 in Suphan Buri), and no single band separates them well (AUC 0.58 to 0.62).
In the pictures (`examples_crop_as_grassland.png`) they are mostly fields of young green rice in one even colour, while the crop patches
predicted right are a mix of brown and green fields.

**Likely explanation:** in EuroSAT, AnnualCrop and PermanentCrop are mostly brown or mixed fields (median NDVI 0.34), while Pasture is an even
green field (NDVI 0.67). An even green paddy looks more like a European pasture than like a European crop field.

### 4. Crop is taken for built-up when straight lines cross the fields (visual; cause likely)

The 43 crop patches predicted as built-up (`examples_crop_as_built-up.png`) are green paddies crossed by straight roads, canals or
embankments. Their colours are close to the crop patches predicted right (AUC 0.55 to 0.58 for every band), so the colour does not explain
the mistake. **Likely explanation:** EuroSAT built-up includes the class Highway, where a straight road through fields is the main feature,
so the model links straight lines with built-up.

### 5. The mistakes are systematic and confident (measured)

- 743 patches are wrong for all three trained models (seeds 42, 1, 2); 1,240 are wrong for at least one. Most mistakes come from the data,
  not from the chance of one training run.
- 44% of the wrong predictions have a confidence above 0.95 (right predictions: 79%). The confidence cannot be used to flag the mistakes
  (`confidence_right_vs_wrong.png`).

### 6. The mistakes are the model's, not WorldCover's (measured, small sample)

On the 199 hand-labelled patches, the zero-shot model disagrees with WorldCover on 45. The three labellers side with WorldCover on 41,
with the model on 1, and with neither on 3. So a "wrong" prediction on a pure patch is almost always a real mistake of the model.
This holds for pure patches only; mixed patches were not labelled by hand.

### 7. AdaBN fixes many mistakes but not the dry forest (measured)

AdaBN turns 321 wrong patches right and 75 right patches wrong (net 246 fewer mistakes).
It fixes forest -> grassland 196 times, crop -> grassland 74, forest -> crop 25 and crop -> built-up 24.
Forest -> grassland is still the largest mistake (484). AdaBN adds new mistakes that the zero-shot model did not make:
crop -> forest 22, crop -> water 6, crop -> grassland 18, built-up -> water 9 (Bangkok study area), forest -> grassland 15.

## What this means for the project

- The domain shift between Europe and Thailand is mostly a shift of **what vegetation looks like in the season of the images**, not a shift
  of cities or water.
- Normalising the statistics (AdaBN) helps but cannot teach the model a dry forest it has never seen. Ideas that follow from this, **not tested**:
  fine-tune on a few Thai forest and crop patches (keep the 199 hand-labelled patches out), train on images of the same season, or use
  composites of more than one season.
- Grassland is the class the model falls back on for vegetation it does not recognise; removing it from the output (the `_nograss` runs of part B)
  helps for that reason, not because the patches look like Thai grassland.

## Files in this folder

| file | content |
|---|---|
| `error_pairs.png`, `error_pairs.csv` | mistakes by true and predicted class, zero-shot and AdaBN |
| `errors_by_class_and_area.csv` | error rate per class and study area |
| `examples_*.png` | wrong patches, right patches of the same class, and EuroSAT patches of the predicted class |
| `ndvi_swir_right_vs_wrong.png`, `feature_medians.csv`, `feature_auc.csv` | NDVI and band values of right and wrong patches |
| `error_maps.png` | where the mistakes are in each study area |
| `confidence_right_vs_wrong.png` | confidence of right and wrong predictions |
| `patch_features.csv` | one row per pure patch: predictions, confidence, NDVI and band means |
