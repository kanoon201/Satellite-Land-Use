"""B7: fine-tune the EuroSAT ResNet-18 on a few labelled Thai patches, and score it fairly.

Question: how much does a small number of labelled Thai patches add to the model trained in Europe?

Train / test split of the Thai patches (fixed, seed 42):
    Every study area is cut into 6 x 6 blocks of 8 x 8 patches (5.1 km). Half of the blocks of each area are
    "train" blocks, the other half "test" blocks. Patches next to each other look almost the same, so a random
    split patch by patch would let the model see its test patches; whole blocks keep them apart.
    - Fine-tuning patches come only from the inside of train blocks (the outer ring of patches is left out),
      so at least one patch (640 m) separates any training patch from any test patch.
    - The 199 hand-labelled patches never train. Of them, the ones lying in test blocks form the clean test set here.
    - Labels for fine-tuning: WorldCover, pure patches only (A6: 98% agree with the human vote on pure patches).

Runs (each repeated for draws 0, 1, 2 of the training patches):
    finetune         start from the EuroSAT checkpoint; BatchNorm statistics kept from EuroSAT
    adabn+finetune   first AdaBN (BatchNorm statistics of the non-pure Thai patches, as in predict.py), then fine-tune
    n per class      10, 25, 50 Thai patches of each of the four Thai classes (forest half Nan, half Kanchanaburi)
All weights are trained (AdamW, lr 1e-4, 30 epochs, flips and 90-degree turns); BatchNorm statistics are frozen
during fine-tuning, so only what the labels teach changes. The loss uses the 5-class head; grassland gets no examples.

Baselines (zero-shot and AdaBN, with and without grassland) are scored on exactly the same test patches.
The full test set of part B cannot be used here: it contains the training patches.

Writes into outputs/b7/ (not outputs/predictions/, which evaluate.py scores on all 3,023 pure patches):
    split.csv                 patch_id, block, side (train / test), in_train_pool
    training_patches.csv      the Thai patches used by every run
    results.csv               one row per run and scoring
    b7_table.md               mean and SD over the three draws
    predictions/<run>.csv     Thai predictions of every run, same columns as outputs/predictions/
    b7_finetune.png           macro-F1 against the number of Thai patches, and what is lost on EuroSAT

Needs data/models/resnet18_12band.pt (python src/train_resnet.py).
Usage:  python src/finetune_resnet.py
        python src/finetune_resnet.py --plot-only     (redraw the figure from results.csv)
"""
import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, recall_score

import b_data as d
import config as c
import predict as pr
import train_resnet as tr

OUT = c.OUTPUT_DIR / "b7"
BLOCK = 8                    # patches per block side (8 x 64 px x 10 m = 5.1 km)
SIZES = [10, 25, 50]         # Thai patches per class
DRAWS = [0, 1, 2]
EPOCHS, LR, BATCH = 30, 1e-4, 32


def make_split(table, clean_ids):
    """Blocks of 8 x 8 patches, half of each area's blocks for training (seed 42)."""
    split = table[["patch_id", "area", "row", "col", "label", "label_id", "pure"]].copy()
    split["block"] = split["area"] + "_" + (split["row"] // BLOCK).astype(str) + "_" + (split["col"] // BLOCK).astype(str)
    rng = np.random.default_rng(c.SEED)
    train_blocks = set()
    for area in c.AOIS:
        blocks = np.array(sorted(split.loc[split["area"] == area, "block"].unique()))
        rng.shuffle(blocks)
        train_blocks |= set(blocks[: len(blocks) // 2])
    split["side"] = np.where(split["block"].isin(train_blocks), "train", "test")
    inner = split["row"].mod(BLOCK).between(1, BLOCK - 2) & split["col"].mod(BLOCK).between(1, BLOCK - 2)
    split["in_train_pool"] = (split["side"] == "train") & inner & split["pure"] & ~split["patch_id"].isin(clean_ids)
    return split


def draw_training_patches(split, n, draw):
    """n patches per Thai class from the train pool; forest half from Nan, half from Kanchanaburi."""
    rng = np.random.default_rng(c.SEED + draw)
    pool = split[split["in_train_pool"]]
    picked = []
    for cls in c.THAI_EVAL_CLASSES:
        part = pool[pool["label"] == cls]
        if cls == "forest":
            for area, k in (("nan", n // 2), ("kanchanaburi", n - n // 2)):
                ids = part.loc[part["area"] == area, "patch_id"].to_numpy()
                picked += list(rng.choice(ids, k, replace=False))
        else:
            picked += list(rng.choice(part["patch_id"].to_numpy(), n, replace=False))
    return np.sort(np.array(picked))


def fine_tune(model, x, y, mean, std, device, seed):
    """Train all weights on (x, y); BatchNorm layers stay in eval mode, so their statistics do not move."""
    tr.set_seed(seed)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    for _ in range(EPOCHS):
        model.train()
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()
        order = rng.permutation(len(x))
        for start in range(0, len(order), BATCH):
            ids = order[start:start + BATCH]
            xb = tr.augment(tr.to_input(x[ids], mean, std, device))
            loss = F.cross_entropy(model(xb), torch.from_numpy(y[ids]).to(device))
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
    model.eval()
    return model


def score_thai(name, proba, y_true, ids, table):
    """Scores over the four Thai classes; a grassland prediction counts as wrong (same rule as evaluate.py)."""
    pred = proba.argmax(1)
    row = {"accuracy": accuracy_score(y_true, pred),
           "macro_f1": f1_score(y_true, pred, labels=d.EVAL_IDS, average="macro", zero_division=0),
           "pred_grassland_pct": 100 * np.mean(pred == c.CLASSES.index("grassland")), "n": len(ids)}
    for k, cls in zip(d.EVAL_IDS, c.THAI_EVAL_CLASSES):
        row[f"f1_{cls}"] = f1_score(y_true, pred, labels=[k], average="macro", zero_division=0)
    forest = y_true == c.CLASSES.index("forest")
    for area in ("nan", "kanchanaburi"):
        sel = forest & (table.loc[ids, "area"].to_numpy() == area)
        row[f"forest_recall_{area}"] = recall_score(y_true[sel], pred[sel], labels=[c.CLASSES.index("forest")],
                                                    average="macro", zero_division=0) if sel.any() else np.nan
    return {"run": name, **row}


def plot(res):
    """Left: Thai macro-F1 against Thai patches per class. Right: EuroSAT test macro-F1 (what is forgotten)."""
    surface, ink, ink2, grid, axis = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
    colours = {"finetune": "#eb6834", "adabn+finetune": "#2a78d6"}
    base_style = {"zero-shot": ("#898781", ":"), "AdaBN": ("#1baf7a", "--"), "AdaBN (grassland removed)": ("#1baf7a", "-.")}
    w = res[res["scoring"] == "test blocks, WorldCover"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=surface)
    for ax, metric, title in [(axes[0], "macro_f1", "Thai test blocks: macro-F1 (4 classes, WorldCover labels)"),
                              (axes[1], "eurosat_test_macro_f1", "EuroSAT test: macro-F1 after fine-tuning")]:
        ax.set_facecolor(surface)
        for name, (col, style) in base_style.items():
            ax.axhline(w.loc[w["variant"] == name, metric].mean(), color=col, linestyle=style, linewidth=1.8, label=name)
        for variant, col in colours.items():
            g = w[w["variant"] == variant].groupby("n_per_class")[metric]
            ax.errorbar(g.mean().index, g.mean(), yerr=g.std(), color=col, marker="o", linewidth=2, capsize=4,
                        label=f"{variant} (mean +- SD of 3 draws)")
        ax.set_xticks(SIZES)
        ax.set_xlabel("labelled Thai patches per class used for fine-tuning")
        ax.set_title(title, loc="left", fontsize=11, color=ink)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(axis)
        ax.grid(axis="y", color=grid, linewidth=0.8)
        ax.tick_params(colors=ink2)
    axes[0].set_ylabel("macro-F1")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    d.write_file(OUT / "b7_finetune.png", lambda path: fig.savefig(path, dpi=130, bbox_inches="tight"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet18_12band")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        plot(pd.read_csv(OUT / "results.csv"))
        return
    t0 = time.time()
    device = tr.pick_device()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "predictions").mkdir(exist_ok=True)

    x_th, table = d.load_thai("L1C", mmap=False)
    clean = pd.read_csv(c.OUTPUT_DIR / "a1" / "clean_test_set.csv", keep_default_na=False).set_index("patch_id")
    split = make_split(table, clean.index)
    d.write_table(OUT / "split.csv", split[["patch_id", "block", "side", "in_train_pool"]])

    test_ids = split.loc[(split["side"] == "test") & split["pure"], "patch_id"].to_numpy()
    clean_test = clean[clean.index.isin(split.loc[split["side"] == "test", "patch_id"])]
    clean_ids = clean_test.index.to_numpy()
    y_test = table.loc[test_ids, "label_id"].to_numpy()
    y_clean = np.array([c.CLASSES.index(v) for v in clean_test["consensus"]])
    unlabeled = np.flatnonzero(~table["pure"].to_numpy())
    pool = split[split["in_train_pool"]]
    print(f"test blocks: {len(test_ids)} pure patches {pd.Series(table.loc[test_ids, 'label']).value_counts().to_dict()}; "
          f"{len(clean_ids)} hand-labelled patches")
    print(f"train pool: {len(pool)} patches {pool['label'].value_counts().to_dict()}")
    assert not set(pool["patch_id"]) & set(clean.index), "hand-labelled patches must never train"
    assert not set(pool["patch_id"]) & set(test_ids), "training and test patches overlap"

    x_eu, y_eu, eu_split = d.load_eurosat()
    eu_ids = np.flatnonzero(eu_split == d.SPLITS["test"])
    x_eu_test = np.asarray(x_eu[eu_ids])

    base, ck = pr.load_checkpoint(args.model, device)
    mean, std = pr.tensor(ck["mean"], device), pr.tensor(ck["std"], device)
    print(f"checkpoint: epoch {ck['epoch']}, EuroSAT val macro-F1 {ck['val_macro_f1']:.4f}")

    results, used = [], []

    def evaluate(model, name, variant, n, draw):
        proba = tr.predict_proba(model, x_th, mean, std, device)
        eu_pred = tr.predict_proba(model, x_eu_test, mean, std, device).argmax(1)
        eu_f1 = f1_score(y_eu[eu_ids], eu_pred, average="macro")
        versions = [(name, proba)]
        if n == 0:                                    # baselines: also with grassland removed, as in part B
            versions.append((f"{name}_nograss", d.without_grassland(proba)[0]))
        for run, p in versions:
            out = pd.DataFrame({"patch_id": table["patch_id"], "predicted_label": np.array(c.CLASSES)[p.argmax(1)],
                                "confidence": p.max(1).round(4), **{f"p_{k}": p[:, i].round(4) for i, k in enumerate(c.CLASSES)}})
            d.write_table(OUT / "predictions" / f"{run}.csv", out)
            base_info = {"variant": variant + (" (grassland removed)" if run.endswith("_nograss") else ""),
                         "n_per_class": n, "draw": draw, "eurosat_test_macro_f1": eu_f1}
            results.append({**score_thai(run, p[test_ids], y_test, test_ids, table), **base_info, "scoring": "test blocks, WorldCover"})
            results.append({**score_thai(run, p[clean_ids], y_clean, clean_ids, table), **base_info, "scoring": "test blocks, hand-labelled"})
            last = results[-2]
            print(f"  {run:32s} macro-F1 {last['macro_f1']:.3f}  (hand-labelled {results[-1]['macro_f1']:.3f}), "
                  f"forest recall Nan {last['forest_recall_nan']:.2f} / Kanchanaburi {last['forest_recall_kanchanaburi']:.2f}, "
                  f"grassland {last['pred_grassland_pct']:.1f}%, EuroSAT test {eu_f1:.3f}  [{time.time() - t0:.0f}s]", flush=True)

    print("\nbaselines")
    evaluate(base, "zero-shot", "zero-shot", 0, 0)
    adabn, _ = pr.load_checkpoint(args.model, device)
    pr.adapt_batchnorm(adabn, x_th[unlabeled], mean, std, device)
    evaluate(adabn, "adabn", "AdaBN", 0, 0)
    adabn_state = {k: v.clone() for k, v in adabn.state_dict().items()}

    for variant in ("finetune", "adabn+finetune"):
        print(f"\n{variant}")
        for n in SIZES:
            for draw in DRAWS:
                ids = draw_training_patches(split, n, draw)
                used += [{"run": f"{variant}_n{n}_draw{draw}", "patch_id": int(i)} for i in ids]
                model, _ = pr.load_checkpoint(args.model, device)
                if variant == "adabn+finetune":
                    model.load_state_dict(adabn_state)
                model = fine_tune(model, x_th[ids], table.loc[ids, "label_id"].to_numpy(), mean, std, device,
                                  seed=c.SEED + draw)
                evaluate(model, f"{variant}_n{n}_draw{draw}", variant, n, draw)

    res = pd.DataFrame(results)
    d.write_table(OUT / "results.csv", res.round(4))
    d.write_table(OUT / "training_patches.csv", pd.DataFrame(used))

    lines = ["# B7: fine-tuning on a few Thai patches", "",
             f"Test: {len(test_ids)} pure Thai patches in test blocks (WorldCover labels) and {len(clean_ids)} hand-labelled patches "
             "in test blocks (vote). Fine-tuning patches come from train blocks only. Mean ± SD over draws 0-2. Macro-F1 over 4 Thai classes.", "",
             "| Run | Thai patches per class | Macro-F1 (test blocks, WorldCover) | Macro-F1 (hand-labelled) | Forest recall Nan / Kanchanaburi | Predicted grassland | EuroSAT test macro-F1 |",
             "|---|---|---|---|---|---|---|"]
    for (variant, n), g in res.groupby(["variant", "n_per_class"], sort=False):
        w, h = g[g["scoring"] == "test blocks, WorldCover"], g[g["scoring"] == "test blocks, hand-labelled"]
        ms = lambda s: f"{s.mean():.3f}" if len(s) == 1 else f"{s.mean():.3f} ± {s.std():.3f}"
        lines.append(f"| {variant} | {n} | {ms(w['macro_f1'])} | {ms(h['macro_f1'])} | "
                     f"{w['forest_recall_nan'].mean():.2f} / {w['forest_recall_kanchanaburi'].mean():.2f} | "
                     f"{w['pred_grassland_pct'].mean():.1f}% | {ms(w['eurosat_test_macro_f1'])} |")
    d.write_text(OUT / "b7_table.md", "\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nsaved to {OUT} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
