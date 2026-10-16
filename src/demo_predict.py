"""Demo: predict the Thai L1C patches with ResNet-18 + AdaBN and show what is right and wrong.

Same steps as notebooks/demo_predict.ipynb, as a script:

    1. load the ResNet-18 trained on EuroSAT only (data/models/<model>.pt, made by train_resnet.py)
    2. AdaBN: re-estimate the BatchNorm statistics on the 6,193 Thai patches that are not pure (no labels)
    3. predict every Thai patch

then score the 3,023 pure patches against WorldCover and save one figure: one right patch per class (top row)
and four forest patches predicted as grassland (bottom row), drawn with seed 42.

Without torch or the checkpoint the script reads outputs/predictions/predictions_<model>_thai-L1C_adabn.csv, the file
predict.py wrote for the same run, and says so. The pictures and scores are then the same; only step 1-3 are skipped.

Where this sits in Part B (the model code itself is not repeated here, it is imported from predict.py and train_resnet.py):

    B2    train_resnet.py   trains ResNet-18 on EuroSAT and writes the checkpoint            (done before, not run here)
    B3-B6 predict.py        predicts EuroSAT test and every Thai variant for ALL models/runs; the run used here is
                            B6b "thai-L1C_adabn" and its file is outputs/predictions/predictions_<model>_thai-L1C_adabn.csv
    B8-B9 evaluate.py       the full scoring (three test sets, tables, confusion matrices)
    A6/A7 notebooks         comparison of the test sets and the error analysis with example images
    this  demo_predict.py   repeats only the B6b steps for one model, with a quick score and a picture; it writes no
                            prediction file and does not replace evaluate.py (its numbers are a first look, one seed)

What each block of the script does:

    predict_with_model()  steps 1-3 above: checkpoint -> AdaBN on the non-pure patches -> probabilities of all patches
    true_colour()         one patch (12 bands, uint16) -> a true-colour image from B4, B3, B2
    draw()                the figure: one right patch per Thai class (green) and four forest patches predicted as
                          grassland (orange), chosen with seed 42, so the picture is the same on every run
    main()                load the Thai data -> split pure / not pure -> get predictions (run the model, or read the
                          saved file) -> score the pure patches -> print the confusion table -> save the figure

Usage:  python src/demo_predict.py
        python src/demo_predict.py --model resnet18_12band --figure outputs/demo/demo_predict.png
        python src/demo_predict.py --saved        (always read the saved predictions)
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import b_data as d
import config as c


def predict_with_model(model_name, x, unlabeled):
    """Steps 1-3: load the checkpoint, AdaBN on the non-pure patches, predict all patches."""
    import predict
    import train_resnet as tr

    device = tr.pick_device()
    model, ck = predict.load_checkpoint(model_name, device)                  # 1. trained on EuroSAT only
    mean, std = predict.tensor(ck["mean"], device), predict.tensor(ck["std"], device)
    x = np.asarray(x)
    predict.adapt_batchnorm(model, x[unlabeled], mean, std, device)           # 2. AdaBN, no labels
    proba = tr.predict_proba(model, x, mean, std, device)                     # 3. predict every patch
    return proba


def true_colour(x, patch_id):
    iR, iG, iB = (c.BANDS.index(b) for b in ("B4", "B3", "B2"))
    a = np.asarray(x[patch_id], dtype="float32")
    return np.clip(np.stack([a[iR], a[iG], a[iB]], -1) / 2200.0, 0, 1) ** (1 / 1.6)


def draw(x, res, test, right, path):
    rng = np.random.default_rng(c.SEED)
    ok, bad = test[right], test[~right]
    good_ids = [rng.choice(ok[ok["label"] == k].index) for k in c.THAI_EVAL_CLASSES]
    forest_as_grass = bad[(bad["label"] == "forest") & (bad["predicted_label"] == "grassland")]
    wrong_ids = rng.choice(forest_as_grass.index, 4, replace=False)

    fig, axs = plt.subplots(2, 4, figsize=(13, 7.2), facecolor="#fcfcfb")
    for row, (ids, color) in enumerate(((good_ids, "#1baf7a"), (wrong_ids, "#eb6834"))):
        for ax, i in zip(axs[row], ids):
            r = res.loc[i]
            ax.imshow(true_colour(x, int(r["patch_id"])), interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor(color)
                s.set_linewidth(3)
            ax.set_title(f"true: {r['label']}\nmodel: {r['predicted_label']} ({r['confidence']:.2f})",
                         fontsize=12, color=color)
    fig.suptitle("Thai L1C patches, ResNet-18 + AdaBN: right (top), wrong (bottom)", fontsize=14)
    fig.tight_layout(h_pad=3.0, rect=(0, 0, 1, 0.96))
    d.write_file(path, lambda p: fig.savefig(p, dpi=150, facecolor=fig.get_facecolor()))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet18_12band")
    parser.add_argument("--figure", default=str(c.OUTPUT_DIR / "demo" / "demo_predict.png"),
                        help="where to save the picture")
    parser.add_argument("--saved", action="store_true", help="read the saved predictions, do not run the model")
    args = parser.parse_args()

    # Block 1: data. Thai L1C patches and their table; split into the test set (pure) and the patches that only
    # AdaBN may use (not pure, no label is read from them).
    x, table = d.load_thai("L1C")
    pure = table["pure"].to_numpy()
    unlabeled = np.flatnonzero(~pure)
    print(f"{len(table):,} Thai patches: {pure.sum():,} pure (test set), {len(unlabeled):,} not pure (AdaBN only)")

    # Block 2: predictions. Run the model if its checkpoint (and torch) are here, otherwise read the file that
    # predict.py saved for the same run. The printed "source:" line says which one happened.
    use_model = not args.saved and (d.MODEL_DIR / f"{args.model}.pt").exists()
    if use_model:
        try:
            proba = predict_with_model(args.model, x, unlabeled)
        except ImportError as error:
            print(f"cannot run the model ({error}); reading the saved predictions instead")
            use_model = False
    if use_model:
        pred = pd.DataFrame({"patch_id": table["patch_id"],
                             "predicted_label": np.array(c.CLASSES)[proba.argmax(1)],
                             "confidence": proba.max(1)})
        print("source: the model ran here (AdaBN + prediction)")
    else:
        pred = pd.read_csv(d.PRED_DIR / f"predictions_{args.model}_thai-L1C_adabn.csv")
        print("source: saved predictions of the same run (outputs/predictions/), the model did not run")

    # Block 3: scores on the 3,023 pure patches (WorldCover labels): right and wrong, accuracy, macro-F1 over the
    # four Thai classes, share predicted as grassland, and the table of true label against model answer.
    res = table.merge(pred, on="patch_id")
    test = res[res["pure"]]
    right = test["predicted_label"] == test["label"]
    macro = f1_score(test["label"], test["predicted_label"], labels=c.THAI_EVAL_CLASSES,
                     average="macro", zero_division=0)
    grass = (test["predicted_label"] == "grassland").mean()
    print(f"\nright {right.sum():,}   wrong {(~right).sum():,}")
    print(f"accuracy {accuracy_score(test['label'], test['predicted_label']):.3f}   "
          f"macro-F1 (4 classes) {macro:.3f}   predicted grassland {100 * grass:.1f}%")
    print("\nrows: true label (WorldCover), columns: model answer")
    print(pd.crosstab(test["label"], test["predicted_label"])
          .reindex(index=c.THAI_EVAL_CLASSES, columns=c.CLASSES, fill_value=0).to_string())

    # Block 4: the picture (see draw()).
    draw(x, res, test, right, Path(args.figure))
    print(f"\nsaved {args.figure}")


if __name__ == "__main__":
    main()
