"""B1: Random Forest baseline on per-band statistics and spectral indices.

Features of one patch (63 numbers), computed from raw reflectance x 10000 (no normalisation needed):
    for each of the 12 bands: mean, std, 10th / 50th / 90th percentile
    mean NDVI (B8-B4)/(B8+B4), NDWI (B3-B8)/(B3+B8), NDBI (B11-B8)/(B11+B8)

The forest is trained on the EuroSAT training split. The few settings tried are compared on the
validation split only; EuroSAT test and the Thai patches are predicted once, with the chosen model.

Writes:
    data/features/<source>.npy                  cached features (eurosat, thailand_L1C, thailand_L2A)
    data/models/rf.joblib                       the trained forest
    outputs/predictions/predictions_rf_<data>.csv   eurosat-test, thai-L1C, thai-L2A (all 9,216 Thai patches),
                                                    and thai-*_nograss (grassland removed, see b_data.without_grassland)
    outputs/b/training/rf_feature_importance.csv
    outputs/b/training/rf_summary.txt                    settings, validation scores and a first look at the scores

The scores printed here are a quick check. The full evaluation (per-class scores, confusion matrices,
the 199 clean patches, forest by area) is done by evaluate.py.

Usage:  python src/train_rf.py            (add --refresh to recompute the cached features)
"""
import argparse
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

import b_data as d
import config as c

STATS = ["mean", "std", "p10", "p50", "p90"]
INDICES = {"NDVI": ("B8", "B4"), "NDWI": ("B3", "B8"), "NDBI": ("B11", "B8")}
FEATURE_NAMES = [f"{b}_{s}" for b in c.BANDS for s in STATS] + list(INDICES)

# small grid, scored on the EuroSAT validation split
SETTINGS = [
    {"min_samples_leaf": 1, "max_features": "sqrt"},
    {"min_samples_leaf": 3, "max_features": "sqrt"},
    {"min_samples_leaf": 1, "max_features": 0.3},
    {"min_samples_leaf": 3, "max_features": 0.3},
]
N_TREES = 300


def features(x, chunk=1000):
    """(N, 12, 64, 64) uint16 -> (N, 63) float32, processed in chunks to keep memory small."""
    out = np.empty((len(x), len(FEATURE_NAMES)), dtype="float32")
    band = {b: i for i, b in enumerate(c.BANDS)}
    for start in range(0, len(x), chunk):
        p = np.asarray(x[start:start + chunk], dtype="float32").reshape(-1, len(c.BANDS), c.PATCH_SIZE ** 2)
        q10, q50, q90 = np.percentile(p, [10, 50, 90], axis=2)
        per_band = np.stack([p.mean(2), p.std(2), q10, q50, q90], axis=2).reshape(len(p), -1)
        idx = []
        for a, b in INDICES.values():
            pa, pb = p[:, band[a]], p[:, band[b]]
            idx.append(((pa - pb) / np.maximum(pa + pb, 1.0)).mean(1))
        out[start:start + len(p)] = np.concatenate([per_band, np.stack(idx, 1)], 1)
    assert np.isfinite(out).all()
    return out


def cached_features(name, x, refresh):
    path = d.FEATURE_DIR / f"{name}.npy"
    if path.exists() and not refresh:
        f = np.load(path)
        if f.shape == (len(x), len(FEATURE_NAMES)):
            return f
    t = time.time()
    f = features(x)
    d.FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, f)
    print(f"  {name}: {f.shape} features in {time.time() - t:.0f} s")
    return f


def scores(y_true, y_pred, labels=None):
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="recompute the cached features")
    args = parser.parse_args()
    lines = [f"B1 Random Forest, seed {c.SEED}, {N_TREES} trees, class_weight=balanced", ""]

    def say(text=""):
        print(text)
        lines.append(text)

    print("features")
    x_eu, y_eu, split = d.load_eurosat()
    f_eu = cached_features("eurosat", x_eu, args.refresh)
    thai = {}
    for level in ("L1C", "L2A"):
        x_th, table = d.load_thai(level)
        thai[level] = cached_features(f"thailand_{level}", x_th, args.refresh)

    train, val, test = (split == d.SPLITS[s] for s in ("train", "val", "test"))

    say("validation split (EuroSAT, 5 classes)")
    best, best_f1 = None, -1.0
    for setting in SETTINGS:
        rf = RandomForestClassifier(n_estimators=N_TREES, class_weight="balanced", n_jobs=-1,
                                    random_state=c.SEED, **setting)
        rf.fit(f_eu[train], y_eu[train])
        acc, f1 = scores(y_eu[val], rf.predict(f_eu[val]))
        say(f"  {setting}: accuracy {acc:.4f}, macro-F1 {f1:.4f}")
        if f1 > best_f1:
            best, best_f1, model = setting, f1, rf
    say(f"chosen: {best}")
    say()

    d.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, d.MODEL_DIR / "rf.joblib", compress=3)

    ids = np.flatnonzero(test)
    proba = model.predict_proba(f_eu[ids])
    d.save_predictions("rf", "eurosat-test", ids, proba)
    acc, f1 = scores(y_eu[ids], proba.argmax(1))
    say(f"EuroSAT test (5 classes): accuracy {acc:.4f}, macro-F1 {f1:.4f}")

    say()
    say("Thai pure patches against WorldCover (4 eval classes; first look, see evaluate.py)")
    pure = table["pure"].to_numpy()
    y_true = table["label_id"].to_numpy()[pure]
    for level, f_th in thai.items():
        proba = model.predict_proba(f_th)
        nograss, empty = d.without_grassland(proba)
        for run, p in ((f"thai-{level}", proba), (f"thai-{level}_nograss", nograss)):
            d.save_predictions("rf", run, table["patch_id"].to_numpy(), p)
            y_pred = p.argmax(1)[pure]
            acc, f1 = scores(y_true, y_pred, labels=d.EVAL_IDS)
            grass = (y_pred == c.CLASSES.index("grassland")).mean()
            say(f"  {run:20s} accuracy {acc:.4f}, macro-F1 {f1:.4f}, predicted grassland {100 * grass:.1f}% of {pure.sum()}")
        if empty:
            say(f"    ({empty} patches had p_grassland = 1; spread evenly over the other classes, so argmax gives forest)")

    importance = pd.DataFrame({"feature": FEATURE_NAMES, "importance": model.feature_importances_})
    importance = importance.sort_values("importance", ascending=False)
    d.write_table(d.TRAIN_DIR / "rf_feature_importance.csv", importance.round(5))
    say()
    say("top 10 features: " + ", ".join(importance["feature"].head(10)))

    d.write_text(d.TRAIN_DIR / "rf_summary.txt", "\n".join(lines) + "\n")
    print(f"\nsaved predictions in {d.PRED_DIR}, summary in {d.TRAIN_DIR}")


if __name__ == "__main__":
    main()
