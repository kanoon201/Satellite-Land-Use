"""Prepare EuroSAT for training: 12 shared bands, 5 target classes, fixed train/val/test split.

Reads the 27,000 GeoTIFF patches straight from EuroSAT_MS.zip and writes into data/eurosat/:
    eurosat_X.npy      (27000, 12, 64, 64) uint16, bands in config.BANDS order
    eurosat_y.npy      target class index 0-4, see config.CLASSES
    eurosat_y10.npy    original class index 0-9, see config.EUROSAT_CLASSES
    eurosat_split.npy  0 = train, 1 = validation, 2 = test
    eurosat_meta.json  class counts and per-band mean / std of the training set

Usage:  python src/prepare_eurosat.py
"""
import io
import json
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from sklearn.model_selection import train_test_split

import config as c


def load_patches():
    """Step 1: read every patch and its original class from the zip."""
    with zipfile.ZipFile(c.EUROSAT_ZIP) as z:
        names = sorted(n for n in z.namelist() if n.endswith(".tif"))
        x = np.empty((len(names), c.PATCH_SIZE, c.PATCH_SIZE, len(c.EUROSAT_BAND_ORDER)), dtype="uint16")
        y10 = np.empty(len(names), dtype="int64")
        for i, name in enumerate(names):
            x[i] = tifffile.imread(io.BytesIO(z.read(name)))
            y10[i] = c.EUROSAT_CLASSES.index(name.split("/")[-2])
            if (i + 1) % 3000 == 0:
                print(f"  read {i + 1}/{len(names)}", flush=True)
    return x, y10


def check_band_order(x):
    """The files carry no band names, so confirm the documented order from the values.

    B10 (cirrus) sees almost no ground signal, and B8A measures nearly the same
    wavelength as B7/B8, so both are easy to recognise.
    """
    sample = x[::20].reshape(-1, x.shape[-1]).astype("float32")
    means = sample.mean(0)
    corr = np.corrcoef(sample[::20].T)
    order = c.EUROSAT_BAND_ORDER
    b10, b8a, b7 = order.index("B10"), order.index("B8A"), order.index("B7")
    assert means.argmin() == b10, "lowest-valued channel should be B10"
    assert corr[b8a, b7] > 0.99, "B8A should track B7 almost exactly"
    print(f"  B10 is channel {b10} (mean {means[b10]:.0f}), B8A is channel {b8a} (r with B7 = {corr[b8a, b7]:.3f})")


def select_bands(x):
    """Step 2: drop B10, reorder to config.BANDS, and move bands to the front for PyTorch."""
    keep = [c.EUROSAT_BAND_ORDER.index(b) for b in c.BANDS]
    return np.ascontiguousarray(x[..., keep].transpose(0, 3, 1, 2))


def merge_classes(y10):
    """Step 3: map the 10 EuroSAT classes onto the 5 target classes."""
    lookup = np.array([c.CLASSES.index(c.EUROSAT_TO_TARGET[name]) for name in c.EUROSAT_CLASSES])
    return lookup[y10]


def make_split(y10):
    """Step 4: stratified split, so every original class keeps its share in each part."""
    idx = np.arange(len(y10))
    rest = c.SPLIT_FRACTIONS["val"] + c.SPLIT_FRACTIONS["test"]
    train, other = train_test_split(idx, test_size=rest, stratify=y10, random_state=c.SEED)
    val, test = train_test_split(
        other, test_size=c.SPLIT_FRACTIONS["test"] / rest, stratify=y10[other], random_state=c.SEED
    )
    split = np.empty(len(y10), dtype="int8")
    split[train], split[val], split[test] = 0, 1, 2
    return split


def save_samples_figure(x, y, y10):
    """One example of each original class in true colour, titled with its target class."""
    rgb = [c.BANDS.index(b) for b in ("B4", "B3", "B2")]
    fig, axes = plt.subplots(2, 5, figsize=(15, 7.4))
    for k, ax in enumerate(axes.ravel()):
        i = np.flatnonzero(y10 == k)[0]
        img = x[i][rgb].transpose(1, 2, 0).astype("float32")
        ax.imshow(np.clip(img / 3000, 0, 1))
        ax.set_title(f"{c.EUROSAT_CLASSES[k]}\n-> {c.CLASSES[y[i]]}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("EuroSAT: 10 original classes and the target class each one maps to")
    fig.tight_layout(h_pad=2.5)
    fig.savefig(c.OUTPUT_DIR / "data_prep" / "eurosat_samples.png", dpi=110)


def main():
    print("Step 1: reading patches from the zip")
    x13, y10 = load_patches()
    print(f"  {x13.shape[0]} patches, {x13.shape[1]} x {x13.shape[2]} px, {x13.shape[3]} bands, {x13.dtype}")
    check_band_order(x13)

    print("Step 2: selecting the 12 shared bands")
    x = select_bands(x13)
    del x13
    print(f"  array is now {x.shape}, bands {c.BANDS}")

    print("Step 3: merging 10 classes into 5")
    y = merge_classes(y10)
    for k, name in enumerate(c.CLASSES):
        parts = [f"{n} {np.sum(y10 == j)}" for j, n in enumerate(c.EUROSAT_CLASSES) if c.EUROSAT_TO_TARGET[n] == name]
        print(f"  {name:10s} {np.sum(y == k):5d}  = " + " + ".join(parts))

    print("Step 4: splitting into train / validation / test")
    split = make_split(y10)
    names = ["train", "val", "test"]
    for s, part in enumerate(names):
        counts = [int(np.sum((split == s) & (y == k))) for k in range(len(c.CLASSES))]
        print(f"  {part:5s} {int(np.sum(split == s)):5d}  " + "  ".join(f"{n} {v}" for n, v in zip(c.CLASSES, counts)))

    train = x[split == 0].astype("float32")
    mean = train.mean(axis=(0, 2, 3))
    std = train.std(axis=(0, 2, 3))
    del train

    np.save(c.EUROSAT_DIR / "eurosat_X.npy", x)
    np.save(c.EUROSAT_DIR / "eurosat_y.npy", y)
    np.save(c.EUROSAT_DIR / "eurosat_y10.npy", y10)
    np.save(c.EUROSAT_DIR / "eurosat_split.npy", split)
    meta = {
        "bands": c.BANDS,
        "classes": c.CLASSES,
        "original_classes": c.EUROSAT_CLASSES,
        "seed": c.SEED,
        "split_sizes": {part: int(np.sum(split == s)) for s, part in enumerate(names)},
        "class_counts": {name: int(np.sum(y == k)) for k, name in enumerate(c.CLASSES)},
        "train_band_mean": dict(zip(c.BANDS, mean.round(2).tolist())),
        "train_band_std": dict(zip(c.BANDS, std.round(2).tolist())),
    }
    (c.EUROSAT_DIR / "eurosat_meta.json").write_text(json.dumps(meta, indent=2))
    save_samples_figure(x, y, y10)
    print("saved arrays to", c.EUROSAT_DIR)
    print("saved figure to", c.OUTPUT_DIR / "data_prep" / "eurosat_samples.png")


if __name__ == "__main__":
    main()
