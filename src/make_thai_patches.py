"""Cut the Thai rasters into 64 x 64 patches and give each one a label from WorldCover.

Every area is split into a 48 x 48 grid of non-overlapping patches, the same size
as EuroSAT. All patches are kept; the table says which ones are pure enough to use.

Writes into data/patches/:
    thailand_X_L1C.npy    (N, 12, 64, 64) uint16, top-of-atmosphere, bands in config.BANDS order
    thailand_X_L2A.npy    (N, 12, 64, 64) uint16, surface reflectance, same patches and order
    thailand_y.npy        majority target class index 0-4 of every patch
    thailand_patches.csv  one row per patch: area, position, label, purity, class shares

Usage:  python src/make_thai_patches.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

import config as c

P = c.PATCH_SIZE
OTHER = len(c.CLASSES)  # index used for WorldCover classes outside the five targets


def cut(array):
    """(bands, H, W) -> (patches, bands, 64, 64), patches ordered row by row."""
    bands, height, width = array.shape
    rows, cols = height // P, width // P
    array = array[:, : rows * P, : cols * P]
    return array.reshape(bands, rows, P, cols, P).transpose(1, 3, 0, 2, 4).reshape(rows * cols, bands, P, P)


def read(area, product):
    with rasterio.open(c.thai_path(area, f"{product}.tif")) as src:
        return src.read(), src.transform, list(src.descriptions)


def class_shares(worldcover_patches):
    """Share of each target class (plus 'other') among the 4,096 pixels of every patch."""
    lookup = np.full(256, OTHER, dtype="int64")
    for code, name in c.WORLDCOVER_TO_TARGET.items():
        lookup[code] = c.CLASSES.index(name)
    target = lookup[worldcover_patches.reshape(len(worldcover_patches), -1)]
    return np.stack([(target == k).mean(axis=1) for k in range(OTHER + 1)], axis=1)


def process_area(area):
    l1c, transform, l1c_bands = read(area, "S2_L1C")
    l2a, _, l2a_bands = read(area, "S2_L2A")
    worldcover, _, _ = read(area, "worldcover")

    # same 12 bands in the same order for both products (this drops B10 from L1C)
    x_l1c = cut(l1c[[l1c_bands.index(b) for b in c.BANDS]])
    x_l2a = cut(l2a[[l2a_bands.index(b) for b in c.BANDS]])
    shares = class_shares(cut(worldcover))

    grid = l1c.shape[1] // P
    row, col = np.divmod(np.arange(grid * grid), grid)
    # centre of each patch in map coordinates, then in longitude / latitude
    x, y = transform * (col * P + P / 2, row * P + P / 2)
    lon, lat = Transformer.from_crs(c.THAI_CRS, "EPSG:4326", always_xy=True).transform(x, y)

    label_id = shares[:, :OTHER].argmax(axis=1)
    purity = shares[:, :OTHER].max(axis=1)
    table = pd.DataFrame({
        "area": area, "row": row, "col": col, "lon": np.round(lon, 6), "lat": np.round(lat, 6),
        "label": np.array(c.CLASSES)[label_id], "label_id": label_id, "purity": np.round(purity, 4),
        **{f"share_{name}": np.round(shares[:, k], 4) for k, name in enumerate(c.CLASSES)},
        "share_other": np.round(shares[:, OTHER], 4),
        "has_missing_pixels": (x_l1c == 0).all(axis=1).any(axis=(1, 2)) | (x_l2a == 0).all(axis=1).any(axis=(1, 2)),
    })
    table["pure"] = (table["purity"] >= c.PURITY_THRESHOLD) & ~table["has_missing_pixels"]
    return x_l1c, x_l2a, table


def save_samples_figure(x, table, per_class=8):
    """Random pure patches of every class in true colour, for a visual check of the labels."""
    rgb = [c.BANDS.index(b) for b in ("B4", "B3", "B2")]
    rng = np.random.default_rng(c.SEED)
    fig, axes = plt.subplots(len(c.THAI_EVAL_CLASSES), per_class, figsize=(2 * per_class, 2.25 * len(c.THAI_EVAL_CLASSES)))
    for r, name in enumerate(c.THAI_EVAL_CLASSES):
        ids = table.index[table["pure"] & (table["label"] == name)].to_numpy()
        chosen = rng.choice(ids, size=min(per_class, len(ids)), replace=False) if len(ids) else []
        for k, ax in enumerate(axes[r]):
            ax.set_xticks([])
            ax.set_yticks([])
            if k < len(chosen):
                img = x[chosen[k]][rgb].transpose(1, 2, 0).astype("float32")
                ax.imshow(np.clip(img / 3000, 0, 1))
                ax.set_title(table.loc[chosen[k], "area"], fontsize=9)
            else:
                ax.axis("off")
        axes[r, 0].set_ylabel(f"{name}\n({len(ids)} pure)", fontsize=11)
    fig.suptitle(f"Thai patches labelled by WorldCover (purity >= {c.PURITY_THRESHOLD:.0%}), true colour")
    fig.tight_layout()
    fig.savefig(c.OUTPUT_DIR / "data_prep" / "thailand_patch_samples.png", dpi=110)


def main():
    parts = [process_area(area) for area in c.AOIS]
    x_l1c = np.concatenate([p[0] for p in parts])
    x_l2a = np.concatenate([p[1] for p in parts])
    table = pd.concat([p[2] for p in parts], ignore_index=True)
    table.insert(0, "patch_id", np.arange(len(table)))

    c.PATCH_DIR.mkdir(parents=True, exist_ok=True)
    np.save(c.PATCH_DIR / "thailand_X_L1C.npy", x_l1c)
    np.save(c.PATCH_DIR / "thailand_X_L2A.npy", x_l2a)
    np.save(c.PATCH_DIR / "thailand_y.npy", table["label_id"].to_numpy())
    table.to_csv(c.PATCH_DIR / "thailand_patches.csv", index=False)

    print(f"patches: {len(table)} in total, array shape {x_l1c.shape} {x_l1c.dtype}")
    print(f"patches with missing pixels: {int(table['has_missing_pixels'].sum())}")

    print(f"\npure patches (purity >= {c.PURITY_THRESHOLD:.0%}) by class and area:")
    pure = table[table["pure"]]
    counts = pd.crosstab(pure["label"], pure["area"]).reindex(index=c.CLASSES, columns=list(c.AOIS), fill_value=0)
    counts["total"] = counts.sum(axis=1)
    counts.loc["all classes"] = counts.sum()
    print(counts.to_string())
    print(f"\nkept {len(pure)} of {len(table)} patches ({len(pure) / len(table):.1%})")

    print("\npure patches per class at other thresholds:")
    rows = {}
    for t in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0):
        ok = table[(table["purity"] >= t) & ~table["has_missing_pixels"]]
        rows[f">= {t:.0%}"] = ok["label"].value_counts().reindex(c.CLASSES, fill_value=0)
    summary = pd.DataFrame(rows).T
    summary["total"] = summary.sum(axis=1)
    print(summary.to_string())

    save_samples_figure(x_l2a, table)
    print("\nsaved arrays and table to", c.PATCH_DIR)
    print("saved figure to", c.OUTPUT_DIR / "data_prep" / "thailand_patch_samples.png")


if __name__ == "__main__":
    main()
