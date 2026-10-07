"""Draw reference examples of every class for the people who label Thai patches by eye (step A1).

Writes two figures into outputs/manual_labels/:
    reference_eurosat.png  EuroSAT training patches (clean labels, but European). Thai patches look
                           different, so use them to learn what each class means, not as a template.
    reference_thai.png     Thai patches whose WorldCover label is very pure. WorldCover is only about 75%
                           accurate, so these are typical examples, not guaranteed correct ones.

Every example is shown twice: true colour (B4, B3, B2) and false colour (B8, B4, B3), the same pictures the
labelling notebook shows. Patches that are in the labelling sample are never used, so no answer is given away.
The area of a Thai example is not printed, for the same reason as in the labelling notebook.

Usage:  python src/make_label_reference.py
"""
import io
import time
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as c

OUT_DIR = c.OUTPUT_DIR / "manual_labels"
TRUE = [c.BANDS.index(b) for b in ("B4", "B3", "B2")]
FALSE = [c.BANDS.index(b) for b in ("B8", "B4", "B3")]

# EuroSAT folders shown for every target class (one entry per example)
EUROSAT_PLAN = {
    "forest": ["Forest", "Forest", "Forest"],
    "crop": ["AnnualCrop", "AnnualCrop", "PermanentCrop"],
    "grassland": ["HerbaceousVegetation", "HerbaceousVegetation", "Pasture"],
    "built-up": ["Residential", "Industrial", "Highway"],
    "water": ["River", "SeaLake", "SeaLake"],
}
THAI_PURITY = 0.98      # Thai examples must be at least this pure


def save(fig, path, dpi=100):
    """Write the figure to a png; retry a few times, because Windows sometimes refuses to open a file that was just written."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi)
    for attempt in range(5):
        try:
            path.write_bytes(buffer.getvalue())
            break
        except OSError:
            if attempt == 4:
                raise
            time.sleep(1)
    plt.close(fig)
    print("saved", path)


def draw(rows, title, path):
    """rows: list of (class name, [(patch, caption), ...]); every example takes a true and a false colour panel."""
    n_pairs = max(len(examples) for _, examples in rows)
    fig, axes = plt.subplots(len(rows), 2 * n_pairs, figsize=(2.1 * 2 * n_pairs, 2.3 * len(rows)), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for r, (name, examples) in enumerate(rows):
        for k, (patch, caption) in enumerate(examples):
            patch = patch.astype("float32")
            for j, (bands, scale) in enumerate(((TRUE, 3000), (FALSE, 5000))):
                ax = axes[r, 2 * k + j]
                ax.axis("on")
                ax.imshow(np.clip(patch[bands].transpose(1, 2, 0) / scale, 0, 1), interpolation="bilinear")
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    ax.set_title("true colour" if j == 0 else "false colour", fontsize=8)
                if j == 0 and caption:
                    ax.set_xlabel(caption, fontsize=8)
        axes[r, 0].set_ylabel(name, fontsize=12, fontweight="bold")
    fig.suptitle(title)
    fig.tight_layout()
    save(fig, path)


def eurosat_figure():
    X = np.load(c.EUROSAT_DIR / "eurosat_X.npy", mmap_mode="r")
    y10 = np.load(c.EUROSAT_DIR / "eurosat_y10.npy")
    split = np.load(c.EUROSAT_DIR / "eurosat_split.npy")
    rng = np.random.default_rng(c.SEED)
    rows = []
    for name, folders in EUROSAT_PLAN.items():
        examples = []
        for folder, count in Counter(folders).items():
            pool = np.flatnonzero((y10 == c.EUROSAT_CLASSES.index(folder)) & (split == 0))   # training patches only
            examples += [(X[i], folder) for i in rng.choice(pool, size=count, replace=False)]
        rows.append((name, examples))
    draw(rows, "EuroSAT examples (Europe, clean labels): folder name under each patch", OUT_DIR / "reference_eurosat.png")
    return rows


def thai_figure():
    table = pd.read_csv(c.PATCH_DIR / "thailand_patches.csv", keep_default_na=False)
    X = np.load(c.PATCH_DIR / "thailand_X_L2A.npy", mmap_mode="r")
    in_sample = set(pd.read_csv(OUT_DIR / "label_sample.csv")["patch_id"])
    pool = table[table["pure"] & (table["purity"] >= THAI_PURITY) & ~table["patch_id"].isin(in_sample)]
    rng = np.random.default_rng(c.SEED)

    rows = []
    for name in c.THAI_EVAL_CLASSES:
        part = pool[pool["label"] == name]
        if name == "forest":    # two forest areas, so the examples show both
            ids = [rng.choice(part[part["area"] == a].index, size=2, replace=False) for a in ("nan", "kanchanaburi")]
            ids = np.concatenate(ids)
        else:
            ids = rng.choice(part.index, size=min(4, len(part)), replace=False)
        rows.append((name, [(X[table.loc[i, "patch_id"]], "") for i in ids]))
        print(f"  {name}: {len(part)} candidate patches")
    draw(rows, "Thai examples (WorldCover label, purity >= 98%; WorldCover is about 75% accurate)", OUT_DIR / "reference_thai.png")
    return rows


def compact_figure(eurosat_rows, thai_rows):
    """One small figure with two EuroSAT and two Thai examples per class, shown above the patch in the labelling notebook."""
    thai = dict(thai_rows)
    fig, axes = plt.subplots(len(c.CLASSES), 8, figsize=(8, 5.4), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for r, (name, examples) in enumerate(eurosat_rows):
        eu = [examples[0], examples[2]]                          # different EuroSAT folders where there are several
        th = thai.get(name, [])
        th = [th[0], th[2]] if name == "forest" and len(th) == 4 else th[:2]   # forest: one example from each area
        for k, (patch, caption) in enumerate(eu + th):
            patch = patch.astype("float32")
            for j, (bands, scale) in enumerate(((TRUE, 3000), (FALSE, 5000))):
                ax = axes[r, 2 * k + j]
                ax.axis("on")
                ax.imshow(np.clip(patch[bands].transpose(1, 2, 0) / scale, 0, 1), interpolation="bilinear")
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    ax.set_title("true" if j == 0 else "false", fontsize=7)
                if j == 0 and caption:
                    ax.set_xlabel(caption.replace("HerbaceousVegetation", "Herbaceous"), fontsize=7)
        axes[r, 0].set_ylabel(name, fontsize=10, fontweight="bold")
    axes[c.CLASSES.index("grassland"), 4].annotate(
        "no Thai example", xy=(1.0, 0.5), xycoords="axes fraction", ha="center", fontsize=8)
    fig.subplots_adjust(left=0.08, right=0.995, top=0.90, bottom=0.03, wspace=0.06, hspace=0.45)
    fig.text(0.31, 0.955, "EuroSAT (Europe, clean labels)", ha="center", fontsize=10)
    fig.text(0.77, 0.955, "Thai (WorldCover label, may be wrong)", ha="center", fontsize=10)
    save(fig, OUT_DIR / "reference_compact.png", dpi=170)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    eurosat_rows = eurosat_figure()
    thai_rows = thai_figure()
    compact_figure(eurosat_rows, thai_rows)


if __name__ == "__main__":
    main()
