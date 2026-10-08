# CLAUDE.md

Course project (Data Mining, Kasetsart University, group assignment on complex data): classify Thai land use from Sentinel-2
images with models trained on European data (EuroSAT) and measure the domain shift. Train in Europe, test in Thailand.
Deadline for slides, report and video: 19 October. Read `README.md` for status, data and findings, `DATA_NOTES.md` (Thai) for the limits of the Thai test set.

## Working with the team

- The team writes to you in Thai. Answer in Thai unless asked otherwise. Code, comments, notebooks and docs are in English.
- Do not commit or push unless asked.
- Do not delete or edit data files, the labellers' answer files (`outputs/manual_labels/<name>.csv`) or anything under `data/` unless asked.
- Say what was run and what was only read from code. Mark guesses about causes as guesses. Do not report model numbers that were not computed.

## Rules for results (they come from the data, see DATA_NOTES.md)

- Report macro-F1 next to accuracy, per-class precision/recall/F1 and a row-normalised confusion matrix. Thai test patches are imbalanced (forest 1,776 against built-up 281).
- Thai evaluation uses 4 classes (`config.THAI_EVAL_CLASSES`: forest, crop, built-up, water); the model predicts 5. No pure grassland patch exists in Thailand.
- Each Thai class comes almost entirely from one study area. Never write "X% on built-up in Thailand"; write "in the Bangkok study area". Call the areas "30.72 x 30.72 km study areas", not provinces. Report forest separately for Nan and Kanchanaburi.
- Thai labels come from ESA WorldCover (a model, noisy). The clean test set is `outputs/a1/clean_test_set.csv` (199 patches, vote of three people). Report both test sets.
- Both Thai sets are test sets (3,023 pure patches with WorldCover labels; the 199 clean patches are a subset with a second, human label). Models train on EuroSAT only; keep the 199 out of any fine-tuning data.
- The 199 clean patches support "WorldCover labels of pure patches are mostly reliable" (about 2% differ), not more: mixed patches were not checked, the sample is small, no confidence interval yet.
- A6 scores three ways from the same predictions: 3,023 vs WorldCover, 199 vs vote, the same 199 vs WorldCover. Use macro-F1 and per-class scores.
- Main Thai test images are L1C (`thailand_X_L1C.npy`), because EuroSAT is L1C. L2A is the comparison. B9 of L2A is abnormal; check before using it.
- Thai images are a January-February median composite, EuroSAT patches are single dates from any season: say so when explaining differences.

## Data (not in git)

`data/eurosat/` (from `prepare_eurosat.py`), `data/thailand/<area>/` (GeoTIFFs), `data/patches/` (Thai patches), see README for sizes and sources.
The arrays are large (EuroSAT 2.65 GB, each Thai patch array 0.9 GB): open them with `np.load(..., mmap_mode="r")` and never print or read them whole.

## Conventions and pitfalls

- Paths, band list, class mappings, seed (`SEED = 42`), purity threshold (`PURITY_THRESHOLD = 0.90`) are in `src/config.py`. Use them; do not hard-code.
- Bands: `config.BANDS` has 12 (B10 removed because L2A has none). Arrays are (patches, 12, 64, 64), reflectance x 10000, uint16. Class order: `config.CLASSES` (forest, crop, grassland, built-up, water).
- **Read `thailand_patches.csv` with `pd.read_csv(..., keep_default_na=False)`.** The area `nan` (Nan province) is otherwise read as a missing value and the whole area disappears.
- Exploration notebooks (a2 to a5) draw the same pixels: 100,000 per class and source with seed 42, EuroSAT training patches only, Thai pure patches only. Keep the sampling code identical if you add a notebook, so figures stay comparable.
- The labelling logic lives in `src/label_tool.py`, not in the notebook (VS Code may keep an old copy of an open notebook). Change the module, not the notebook cells.
- On this Windows setup, writing a file that was written a moment ago can fail with `OSError: Invalid argument`. The scripts retry; do the same in new code that saves figures.
- Charts: light surface `#fcfcfb`, categorical colours in the fixed order blue `#2a78d6`, orange `#eb6834`, aqua `#1baf7a` (EuroSAT, Thai L1C, Thai L2A in a2 to a5), thin lines, legend always present.

## Commands

```bash
python src/prepare_eurosat.py        # needs data/eurosat/EuroSAT_MS.zip
python src/make_thai_patches.py      # needs data/thailand/
python src/make_label_sample.py      # the 200 patches for step A1
python src/make_label_reference.py   # reference figures for the labellers
python src/merge_labels.py           # votes over outputs/manual_labels/*.csv -> outputs/a1/
```

## Open work

Models (Random Forest on band statistics and NDVI, ResNet on 12 bands and on RGB), normalisation experiments and the evaluation are not in the repo yet.
Steps A6 (compare predictions with WorldCover and the clean test set) and A7 (error analysis with example images) wait for a predictions file:
proposed format, one CSV per model and test set with columns `patch_id`, `predicted_label` (class names of `config.CLASSES`).
Slides and report are written after all analysis is done.
