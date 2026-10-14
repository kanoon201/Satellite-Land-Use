"""B8-B9: score every prediction file in outputs/predictions/ and build the comparison table.

File names are predictions_<model>_<run>.csv, where <run> is eurosat-test or thai-<L1C|L2A>[_<fix>...].

EuroSAT test is scored on its 5 classes. Every Thai run is scored three ways from the same predictions
(DATA_NOTES.md, README "Next"):
    worldcover   the 3,023 pure patches against WorldCover
    clean-vote   the 199 hand-labelled patches (outputs/a1/clean_test_set.csv) against the vote
    clean-wc     the same 199 patches against WorldCover
(clean-vote vs clean-wc shows the effect of the labels, clean-wc vs worldcover the effect of the sample.)

Thai scores use the 4 eval classes (config.THAI_EVAL_CLASSES); a grassland prediction counts as an error.
Also reported: share of patches predicted as grassland, and forest recall in Nan and Kanchanaburi separately
(on the 3,023 patches).

Writes into outputs/b/:
    b9_results.csv     one row per model, run and scoring
    b9_table.md        the B9 table for the slides (WorldCover and clean-vote side by side), seed 42 runs only
    b9_seeds.md / .csv mean and SD over the training seeds, for models trained with more than one seed
                       (train_resnet.py --seed n writes resnet18_<bands>_seed<n>; seed 42 has no suffix)
    per_class.csv      precision / recall / F1 / support per class
    confusion/<model>_<run>.png   (seed 42 only) row-normalised confusion matrix (true class x predicted class), on the 3,023
                                  pure patches for Thai runs and on the test split for EuroSAT

Usage:  python src/evaluate.py
"""
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

import b_data as d
import config as c

CONF_DIR = d.B_DIR / "confusion"
FIX_NAMES = {
    "adabn": "AdaBN",
    "thainorm": "Thai normalisation",
    "b9mean": "B9 set to EuroSAT mean",
    "nograss": "grassland removed",
}
MODEL_NAMES = {"rf": "Random Forest", "resnet18_12band": "ResNet-18 (12 bands)", "resnet18_rgb": "ResNet-18 (RGB)"}
SURFACE, INK, BLUE = "#fcfcfb", "#1f1f1f", "#2a78d6"
CMAP = LinearSegmentedColormap.from_list("seq_blue", [SURFACE, "#a9c9ee", BLUE, "#123f78"])


def parse(path):
    """predictions_<model>_<run>.csv -> (model, test data, list of fixes)."""
    m = re.fullmatch(r"predictions_(.+?)_(eurosat-test|thai-L1C|thai-L2A)((?:_[a-z0-9]+)*)\.csv", path.name)
    assert m, f"unexpected prediction file name: {path.name}"
    fixes = [f for f in m.group(3).split("_") if f]
    assert all(f in FIX_NAMES for f in fixes), fixes
    return m.group(1), m.group(2), fixes


def split_seed(model):
    """resnet18_12band_seed1 -> ("resnet18_12band", 1); a name without suffix used config.SEED."""
    m = re.fullmatch(r"(.+)_seed(\d+)", model)
    return (m.group(1), int(m.group(2))) if m else (model, c.SEED)


def load_predictions(path, n):
    p = pd.read_csv(path)
    assert len(p) == n and (p["patch_id"].diff().dropna() > 0).all(), f"{path.name}: expected {n} sorted rows"
    assert set(p["predicted_label"]) <= set(c.CLASSES), set(p["predicted_label"])
    return p.set_index("patch_id")


def score(y_true, y_pred, labels):
    return {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "pred_grassland_pct": 100 * np.mean(np.asarray(y_pred) == "grassland"),
    }


def per_class(y_true, y_pred, labels):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return [{"class": k, "precision": p[i], "recall": r[i], "f1": f[i], "support": s[i]} for i, k in enumerate(labels)]


def plot_confusion(y_true, y_pred, rows, title, path):
    """Rows = true classes, columns = all 5 predicted classes, each row sums to 1."""
    m = confusion_matrix(y_true, y_pred, labels=c.CLASSES).astype(float)
    m = m[[c.CLASSES.index(r) for r in rows]]
    m = m / np.maximum(m.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.2, 0.75 * len(rows) + 1.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.imshow(m, cmap=CMAP, vmin=0, vmax=1, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=10,
                    color="white" if m[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(c.CLASSES)), c.CLASSES, color=INK)
    ax.set_yticks(range(len(rows)), rows, color=INK)
    ax.set_xlabel("predicted class", color=INK)
    ax.set_ylabel("true class", color=INK)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, m.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, m.shape[0]), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)   # 2px gap between cells
    ax.tick_params(which="minor", length=0)
    fig.tight_layout()
    d.write_file(path, lambda p: fig.savefig(p, dpi=130, facecolor=SURFACE))
    plt.close(fig)


def fmt(v):
    return "—" if pd.isna(v) else f"{v:.3f}"


def write_seed_table(results):
    """Mean and SD over seeds for every model trained with more than one seed."""
    seeds = results.groupby("model")["seed"].nunique()
    multi = results[results["model"].isin(seeds[seeds > 1].index)]
    if multi.empty:
        return
    keys = ["model", "test_data", "fix", "run", "scoring"]
    agg = multi.groupby(keys).agg(seeds=("seed", lambda v: ", ".join(map(str, sorted(v)))),
                                  accuracy_mean=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
                                  macro_f1_mean=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std")).reset_index()
    d.write_table(d.B_DIR / "b9_seeds.csv", agg.round(4))

    def cell(run_rows, scoring, metric):
        r = run_rows[run_rows["scoring"] == scoring]
        return "—" if r.empty else f"{r[metric + '_mean'].iloc[0]:.3f} ± {r[metric + '_sd'].iloc[0]:.3f}"

    lines = ["# B9 over training seeds", "",
             "Mean ± SD (sample SD) over the seeds listed. Only the training seed changes; data, split and settings are fixed.",
             "Thai scores: 4 classes. WorldCover = 3,023 pure patches; clean = 199 hand-labelled patches against the vote.", "",
             "| Model | Test data | Fix | Seeds | Acc (WorldCover) | Macro-F1 (WorldCover) | Macro-F1 (clean) |",
             "|---|---|---|---|---|---|---|"]
    runs = agg[keys[:-1]].drop_duplicates()
    order = sorted(runs.itertuples(index=False), key=lambda k: (k.test_data != "eurosat-test", k.model, k.test_data, k.fix != "—", k.fix))
    for k in order:
        rows = agg[(agg["model"] == k.model) & (agg["run"] == k.run)]
        main = "eurosat-test" if k.test_data == "eurosat-test" else "worldcover"
        lines.append(f"| {MODEL_NAMES.get(k.model, k.model)} | {k.test_data} | {k.fix} | {rows['seeds'].iloc[0]} | "
                     f"{cell(rows, main, 'accuracy')} | {cell(rows, main, 'macro_f1')} | {cell(rows, 'clean-vote', 'macro_f1')} |")
    d.write_text(d.B_DIR / "b9_seeds.md", "\n".join(lines) + "\n")
    print("\n".join(lines) + "\n")


def main():
    _, y_eu, split = d.load_eurosat()
    test_ids = np.flatnonzero(split == d.SPLITS["test"])
    y_eu_test = np.array(c.CLASSES)[y_eu[test_ids]]

    table = pd.read_csv(c.PATCH_DIR / "thailand_patches.csv", keep_default_na=False)
    pure = table[table["pure"].astype(str) == "True"].set_index("patch_id")
    clean = pd.read_csv(c.OUTPUT_DIR / "a1" / "clean_test_set.csv", keep_default_na=False).set_index("patch_id")
    assert len(pure) == 3023 and len(clean) == 199 and clean.index.isin(pure.index).all()
    assert (clean["worldcover"] == pure.loc[clean.index, "label"]).all(), "clean_test_set worldcover column disagrees"
    scorings = {
        "worldcover": (pure.index, pure["label"]),
        "clean-vote": (clean.index, clean["consensus"]),
        "clean-wc": (clean.index, clean["worldcover"]),
    }

    files = sorted(d.PRED_DIR.glob("predictions_*.csv"))
    assert files, f"no prediction files in {d.PRED_DIR}"
    CONF_DIR.mkdir(parents=True, exist_ok=True)
    results, classes = [], []
    for path in files:
        full_model, data, fixes = parse(path)
        run = path.name[len(f"predictions_{full_model}_"):-4]
        model, seed = split_seed(full_model)
        base = {"model": model, "seed": seed, "train_data": "EuroSAT", "test_data": data,
                "fix": " + ".join(FIX_NAMES[f] for f in fixes) or "—", "run": run}
        name = f"{MODEL_NAMES.get(model, model)}, {data}" + (f", {base['fix']}" if fixes else "")
        draw = seed == c.SEED

        if data == "eurosat-test":
            pred = load_predictions(path, len(test_ids))
            assert (pred.index.to_numpy() == test_ids).all()
            y_pred = pred["predicted_label"].to_numpy()
            results.append({**base, "scoring": "eurosat-test", **score(y_eu_test, y_pred, c.CLASSES)})
            classes += [{**base, "scoring": "eurosat-test", **r} for r in per_class(y_eu_test, y_pred, c.CLASSES)]
            if draw:
                plot_confusion(y_eu_test, y_pred, c.CLASSES, name, CONF_DIR / f"{model}_{run}.png")
            continue

        pred = load_predictions(path, len(table))["predicted_label"]
        for scoring, (ids, y_true) in scorings.items():
            y_true, y_pred = y_true.to_numpy(), pred.loc[ids].to_numpy()
            row = {**base, "scoring": scoring, **score(y_true, y_pred, c.THAI_EVAL_CLASSES)}
            if scoring == "worldcover":
                for area in ("nan", "kanchanaburi"):
                    forest = (pure["label"] == "forest") & (pure["area"] == area)
                    row[f"forest_recall_{area}"] = np.mean(pred.loc[pure.index[forest]] == "forest")
            results.append(row)
            classes += [{**base, "scoring": scoring, **r} for r in per_class(y_true, y_pred, c.THAI_EVAL_CLASSES)]
        if draw:
            plot_confusion(pure["label"].to_numpy(), pred.loc[pure.index].to_numpy(), c.THAI_EVAL_CLASSES,
                           f"{name}\n3,023 pure patches vs WorldCover, rows sum to 1", CONF_DIR / f"{model}_{run}.png")

    results = pd.DataFrame(results)
    d.write_table(d.B_DIR / "b9_results.csv", results.round(4))
    d.write_table(d.B_DIR / "per_class.csv", pd.DataFrame(classes).round(4))

    write_seed_table(results)

    # B9 table for the slides: one row per model and run, seed 42
    main_seed = results[results["seed"] == c.SEED]
    wide = main_seed.pivot_table(index=["model", "test_data", "fix", "run"], columns="scoring",
                               values=["accuracy", "macro_f1"], aggfunc="first")
    main_rows = main_seed[main_seed["scoring"] == "worldcover"].set_index(["model", "run"])
    lines = ["# B9: comparison of all experiments", "",
             f"Seed {c.SEED}. Thai scores: 4 classes, grassland predictions count as errors. "
             "WorldCover = 3,023 pure patches; clean = 199 hand-labelled patches against the vote.", "",
             "| Model | Test data | Fix | Acc (WorldCover) | Macro-F1 (WorldCover) | Macro-F1 (clean) | Predicted grassland | Forest recall Nan / Kanchanaburi |",
             "|---|---|---|---|---|---|---|---|"]
    order = sorted(wide.index, key=lambda k: (k[1] != "eurosat-test", k[0] != "rf", k[1], k[2] != "—", k[2]))
    for model, data, fix, run in order:
        r = wide.loc[(model, data, fix, run)]
        if data == "eurosat-test":
            acc, f1, f1c, gr, fr = r[("accuracy", "eurosat-test")], r[("macro_f1", "eurosat-test")], np.nan, np.nan, "—"
        else:
            acc, f1, f1c = r[("accuracy", "worldcover")], r[("macro_f1", "worldcover")], r[("macro_f1", "clean-vote")]
            g = main_rows.loc[(model, run)]
            gr = g["pred_grassland_pct"]
            fr = f"{g['forest_recall_nan']:.2f} / {g['forest_recall_kanchanaburi']:.2f}"
        lines.append(f"| {MODEL_NAMES.get(model, model)} | {data} | {fix} | {fmt(acc)} | {fmt(f1)} | {fmt(f1c)} | "
                     f"{'—' if pd.isna(gr) else f'{gr:.1f}%'} | {fr} |")
    d.write_text(d.B_DIR / "b9_table.md", "\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nsaved b9_results.csv, b9_table.md, b9_seeds.md, per_class.csv and confusion matrices in {d.B_DIR}")


if __name__ == "__main__":
    main()
