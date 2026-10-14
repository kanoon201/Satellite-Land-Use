"""Shared loading and saving for Part B (models and evaluation).

Every Part B script loads data through here, so band order, class order, normalisation
and the prediction-file format stay the same across models.

    load_eurosat()        uint16 (27000, 12, 64, 64), class 0-4, split 0/1/2
    load_thai(level)      uint16 (9216, 12, 64, 64) for "L1C" or "L2A", plus thailand_patches.csv
    eurosat_band_stats()  per-band mean / std of the EuroSAT training split, shape (12,)
    save_predictions()    outputs/predictions/predictions_<model>_<data>.csv
"""
import json
import time

import numpy as np
import pandas as pd

import config as c

PRED_DIR = c.OUTPUT_DIR / "predictions"
B_DIR = c.OUTPUT_DIR / "b"                 # B9 table, per-class scores, confusion/
TRAIN_DIR = B_DIR / "training"             # training logs and summaries of each model
MODEL_DIR = c.DATA_DIR / "models"          # inside data/, so checkpoints never reach git
FEATURE_DIR = c.DATA_DIR / "features"

SPLITS = {"train": 0, "val": 1, "test": 2}
EVAL_IDS = [c.CLASSES.index(name) for name in c.THAI_EVAL_CLASSES]


def check_patches(x, n=None):
    assert x.ndim == 4 and x.shape[1:] == (len(c.BANDS), c.PATCH_SIZE, c.PATCH_SIZE), x.shape
    assert x.dtype == np.uint16, x.dtype
    if n is not None:
        assert len(x) == n, (len(x), n)


def load_meta():
    meta = json.loads((c.EUROSAT_DIR / "eurosat_meta.json").read_text())
    assert meta["bands"] == c.BANDS, "eurosat_meta.json band order differs from config.BANDS"
    assert meta["classes"] == c.CLASSES, "eurosat_meta.json class order differs from config.CLASSES"
    return meta


def load_eurosat(mmap=True):
    """Images, target class (0-4) and split (0 train, 1 val, 2 test)."""
    load_meta()
    x = np.load(c.EUROSAT_DIR / "eurosat_X.npy", mmap_mode="r" if mmap else None)
    y = np.load(c.EUROSAT_DIR / "eurosat_y.npy")
    split = np.load(c.EUROSAT_DIR / "eurosat_split.npy")
    check_patches(x, len(y))
    assert len(split) == len(y) and set(np.unique(y)) <= set(range(len(c.CLASSES)))
    return x, y, split


def load_thai(level, mmap=True):
    """Images of all 9,216 Thai patches and their table. Filter on table["pure"] when scoring."""
    assert level in ("L1C", "L2A"), level
    x = np.load(c.PATCH_DIR / f"thailand_X_{level}.npy", mmap_mode="r" if mmap else None)
    # keep_default_na=False: otherwise the area "nan" (Nan province) is read as a missing value
    table = pd.read_csv(c.PATCH_DIR / "thailand_patches.csv", keep_default_na=False)
    check_patches(x, len(table))
    assert (table["patch_id"].to_numpy() == np.arange(len(table))).all(), "patch_id must be the row index"
    assert set(table["area"]) == set(c.AOIS), set(table["area"])
    table["pure"] = table["pure"].astype(str) == "True"
    return x, table


def eurosat_band_stats():
    """Per-band mean and std of the EuroSAT training split, in config.BANDS order."""
    meta = load_meta()
    mean = np.array([meta["train_band_mean"][b] for b in c.BANDS], dtype="float32")
    std = np.array([meta["train_band_std"][b] for b in c.BANDS], dtype="float32")
    return mean, std


def without_grassland(proba):
    """Thai runs with the "_nograss" suffix: set p_grassland to 0 and renormalise the other four classes.

    Justified by prior knowledge, not by the test labels: only 0.04% of Thailand is pure WorldCover grassland
    at patch scale (see config.THAI_EVAL_CLASSES). Returns the new probabilities and the number of patches
    whose whole probability was on grassland (those are spread evenly over the other classes).
    """
    p = np.array(proba, dtype="float64")
    p[:, c.CLASSES.index("grassland")] = 0.0
    total = p.sum(1, keepdims=True)
    empty = (total[:, 0] == 0)
    p[empty] = 1.0
    p[empty, c.CLASSES.index("grassland")] = 0.0
    return p / p.sum(1, keepdims=True), int(empty.sum())


def write_file(path, write):
    """Call write(path). Windows sometimes refuses to open a file written a moment ago, so retry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(5):
        try:
            write(path)
            return
        except OSError:
            if attempt == 4:
                raise
            time.sleep(1)


def write_text(path, text):
    write_file(path, lambda p: p.write_text(text, encoding="utf-8"))


def write_table(path, table):
    write_file(path, lambda p: table.to_csv(p, index=False))


def save_predictions(model, data, patch_ids, proba):
    """One row per patch: patch_id, predicted_label, confidence, p_<class> for every class.

    patch_ids index the source arrays (eurosat_X.npy or thailand_X_<level>.npy).
    """
    proba = np.asarray(proba, dtype="float64")
    assert proba.shape == (len(patch_ids), len(c.CLASSES)), proba.shape
    table = pd.DataFrame({
        "patch_id": patch_ids,
        "predicted_label": np.array(c.CLASSES)[proba.argmax(1)],
        "confidence": proba.max(1).round(4),
    })
    for i, name in enumerate(c.CLASSES):
        table[f"p_{name}"] = proba[:, i].round(4)
    path = PRED_DIR / f"predictions_{model}_{data}.csv"
    write_table(path, table)
    return path
