"""Check the downloaded Thai rasters and draw a quick-look figure.

Prints, for every area: grid alignment, missing pixels, band means, cloud-free
observation counts and the WorldCover class shares. Saves outputs/thailand_quicklook.png.

Usage:  python src/check_thailand.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

import config as c


def read(name, product):
    with rasterio.open(c.thai_path(name, f"{product}.tif")) as src:
        return src.read(), src.transform, src.crs, src.descriptions


def stretch(rgb):
    """Scale reflectance to 0-1 between the 2nd and 98th percentile for display."""
    lo, hi = np.percentile(rgb[rgb > 0], (2, 98))
    return np.clip((rgb - lo) / (hi - lo), 0, 1)


def main():
    codes = sorted(c.WORLDCOVER_CLASSES)
    cmap = ListedColormap([c.WORLDCOVER_CLASSES[k][1] for k in codes])
    norm = BoundaryNorm(codes + [codes[-1] + 1], cmap.N)

    areas = list(c.AOIS)
    fig, axes = plt.subplots(len(areas), 4, figsize=(18, 4.6 * len(areas)))

    for row, name in enumerate(areas):
        l2a, transform, crs, bands = read(name, "S2_L2A")
        l1c, t1, _, bands1 = read(name, "S2_L1C")
        wc, t2, _, _ = read(name, "worldcover")
        obs, t3, _, _ = read(name, "clearobs")

        aligned = transform == t1 == t2 == t3 and l2a.shape[1:] == l1c.shape[1:] == wc.shape[1:] == obs.shape[1:]
        print(f"\n[{name}] {l2a.shape[2]} x {l2a.shape[1]} px, {crs}, all rasters on one grid: {aligned}")
        print(f"  L2A bands {len(bands)}, L1C bands {len(bands1)}")
        print(f"  pixels with no cloud-free observation: {np.mean(obs[0] == 0) * 100:.3f}%")
        print(f"  cloud-free observations per pixel: min {obs.min()}, median {int(np.median(obs))}, max {obs.max()}")
        print("  L2A band means:", {b: int(l2a[i].mean()) for i, b in enumerate(bands)})
        print("  L1C band means:", {b: int(l1c[i].mean()) for i, b in enumerate(bands1)})
        values, counts = np.unique(wc, return_counts=True)
        print("  WorldCover shares:")
        for v, k in sorted(zip(values, counts), key=lambda p: -p[1]):
            label = c.WORLDCOVER_CLASSES.get(int(v), ("no data",))[0]
            target = c.WORLDCOVER_TO_TARGET.get(int(v), "discarded")
            print(f"    {int(v):3d} {label:26s} {k / wc.size * 100:5.1f}%  -> {target}")

        idx = [bands.index(b) for b in ("B4", "B3", "B2")]
        axes[row, 0].imshow(stretch(l2a[idx].astype("float32").transpose(1, 2, 0)))
        axes[row, 0].set_title(f"{name}: true colour (B4, B3, B2)")
        idx = [bands.index(b) for b in ("B8", "B4", "B3")]
        axes[row, 1].imshow(stretch(l2a[idx].astype("float32").transpose(1, 2, 0)))
        axes[row, 1].set_title("false colour (B8, B4, B3): vegetation is red")
        axes[row, 2].imshow(wc[0], cmap=cmap, norm=norm, interpolation="nearest")
        axes[row, 2].set_title("ESA WorldCover 2021 labels")
        im = axes[row, 3].imshow(obs[0], cmap="viridis")
        axes[row, 3].set_title("cloud-free observations per pixel")
        fig.colorbar(im, ax=axes[row, 3], fraction=0.046)
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

    present = [Patch(color=c.WORLDCOVER_CLASSES[k][1], label=c.WORLDCOVER_CLASSES[k][0]) for k in codes]
    fig.legend(handles=present, loc="lower center", ncol=6, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = c.OUTPUT_DIR / "thailand_quicklook.png"
    fig.savefig(out, dpi=110)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
