"""B3-B6: predict EuroSAT test and the Thai patches with a trained ResNet checkpoint.

All runs reuse one checkpoint (data/models/<model>.pt from train_resnet.py); only the input or the
normalisation changes:

    eurosat-test          B3  EuroSAT test split, EuroSAT training statistics
    thai-L1C              B4  Thai L1C, EuroSAT training statistics (main Thai result)
    thai-L2A              B5  Thai L2A, EuroSAT training statistics
    thai-L2A_b9mean       B5  as above, but B9 set to the EuroSAT mean: L2A B9 is abnormal (mean ~2,500
                              against ~450 in L1C and ~730 in EuroSAT), this shows how much of B5 comes from it
    thai-L1C_thainorm     B6a Thai L1C normalised with per-band mean / std of the Thai NON-pure patches
    thai-L1C_adabn        B6b EuroSAT statistics, but the BatchNorm statistics of the network re-estimated
                              on the Thai NON-pure patches (AdaBN)

Every Thai run is also saved with the suffix "_nograss": the same probabilities with grassland removed and
the other four classes renormalised (b_data.without_grassland). Grassland is almost absent from Thailand at
patch scale, so this uses prior knowledge, not the test labels. Keep reporting the plain runs too: the share
of Thai patches predicted as grassland is itself a result.

B6 uses only the 6,193 non-pure Thai patches: same sensor, areas and dates as the test patches, but never
scored and used without labels. The 3,023 pure patches (the test set) and the 199 hand-labelled patches
never enter any statistic.

Thai predictions cover all 9,216 patches; filter on `pure` when scoring. Files go to
outputs/predictions/predictions_<model>_<run>.csv. The scores printed here are a quick check against
WorldCover; the full evaluation is done by evaluate.py.

Usage:  python src/predict.py                         (model resnet18_12band)
        python src/predict.py --model resnet18_rgb
"""
import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

import b_data as d
import config as c
import train_resnet as tr


def load_checkpoint(name, device):
    ck = torch.load(d.MODEL_DIR / f"{name}.pt", map_location="cpu", weights_only=False)
    assert ck["classes"] == c.CLASSES, ck["classes"]
    model = tr.build_model(len(ck["bands"]), pretrained=False)
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck


def band_stats(x, chunk=1000):
    """Per-band mean and std over all pixels of uint16 patches (N, bands, 64, 64)."""
    s1 = np.zeros(x.shape[1])
    s2 = np.zeros(x.shape[1])
    for start in range(0, len(x), chunk):
        p = np.asarray(x[start:start + chunk], dtype="float64")
        s1 += p.mean((2, 3)).sum(0)
        s2 += (p ** 2).mean((2, 3)).sum(0)
    mean = s1 / len(x)
    return mean.astype("float32"), np.sqrt(s2 / len(x) - mean ** 2).astype("float32")


@torch.no_grad()
def adapt_batchnorm(model, x, mean, std, device, batch=256):
    """AdaBN: replace the running mean / variance of every BatchNorm layer by those of x. No weights change."""
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None            # cumulative average over all batches
    model.train()
    for start in range(0, len(x), batch):
        model(tr.to_input(x[start:start + batch], mean, std, device))
    model.eval()


def tensor(v, device):
    return torch.tensor(v, device=device)[None, :, None, None]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet18_12band")
    args = parser.parse_args()

    device = tr.pick_device()
    model, ck = load_checkpoint(args.model, device)
    band_idx = [c.BANDS.index(b) for b in ck["bands"]]
    eu_mean, eu_std = tensor(ck["mean"], device), tensor(ck["std"], device)
    lines = [f"{args.model}: checkpoint epoch {ck['epoch']}, val macro-F1 {ck['val_macro_f1']:.4f}, "
             f"seed {ck['seed']}, device {device}", ""]

    def say(text=""):
        print(text)
        lines.append(text)

    def select(x):
        return x if len(band_idx) == len(c.BANDS) else x[:, band_idx]

    # B3: EuroSAT test
    x_eu, y_eu, split = d.load_eurosat()
    ids = np.flatnonzero(split == d.SPLITS["test"])
    proba = tr.predict_proba(model, select(x_eu[ids]), eu_mean, eu_std, device)
    d.save_predictions(args.model, "eurosat-test", ids, proba)
    pred = proba.argmax(1)
    say(f"eurosat-test (5 classes): accuracy {accuracy_score(y_eu[ids], pred):.4f}, "
        f"macro-F1 {f1_score(y_eu[ids], pred, average='macro'):.4f}")
    say()
    say("Thai pure patches against WorldCover (4 eval classes; first look, see evaluate.py)")

    x_l1c, table = d.load_thai("L1C", mmap=False)
    x_l2a, _ = d.load_thai("L2A", mmap=False)
    x_l1c, x_l2a = select(x_l1c), select(x_l2a)
    pure = table["pure"].to_numpy()
    unlabeled = np.flatnonzero(~pure)
    clean = pd.read_csv(c.OUTPUT_DIR / "a1" / "clean_test_set.csv", keep_default_na=False)["patch_id"]
    assert not set(unlabeled) & set(clean), "hand-labelled patches must not enter the B6 statistics"

    x_l2a_b9 = None
    if "B9" in ck["bands"]:
        x_l2a_b9 = x_l2a.copy()
        x_l2a_b9[:, ck["bands"].index("B9")] = np.uint16(round(float(ck["mean"][ck["bands"].index("B9")])))

    th_mean, th_std = band_stats(x_l1c[unlabeled])
    say(f"  (B6a statistics from {len(unlabeled)} non-pure L1C patches)")

    runs = [
        ("thai-L1C", x_l1c, eu_mean, eu_std, False),
        ("thai-L2A", x_l2a, eu_mean, eu_std, False),
        ("thai-L2A_b9mean", x_l2a_b9, eu_mean, eu_std, False),
        ("thai-L1C_thainorm", x_l1c, tensor(th_mean, device), tensor(th_std, device), False),
        ("thai-L1C_adabn", x_l1c, eu_mean, eu_std, True),
    ]
    for run, x, mean, std, adabn in runs:
        if x is None:
            continue
        if adabn:
            model, _ = load_checkpoint(args.model, device)   # fresh copy, so no other run is affected
            adapt_batchnorm(model, x[unlabeled], mean, std, device)
        proba = tr.predict_proba(model, x, mean, std, device)
        nograss, _ = d.without_grassland(proba)   # softmax is never exactly 1, so no patch is left empty
        for name, p in ((run, proba), (f"{run}_nograss", nograss)):
            d.save_predictions(args.model, name, table["patch_id"].to_numpy(), p)
            y_true, y_pred = table["label_id"].to_numpy()[pure], p.argmax(1)[pure]
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, labels=d.EVAL_IDS, average="macro", zero_division=0)
            grass = (y_pred == c.CLASSES.index("grassland")).mean()
            say(f"  {name:26s} accuracy {acc:.4f}, macro-F1 {f1:.4f}, predicted grassland {100 * grass:.1f}%")

    d.write_text(d.TRAIN_DIR / f"{args.model}_predict_summary.txt", "\n".join(lines) + "\n")
    print(f"\nsaved predictions in {d.PRED_DIR}")


if __name__ == "__main__":
    main()
