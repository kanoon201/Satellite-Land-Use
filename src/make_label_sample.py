"""Pick the Thai patches that are labelled by eye (step A1) and split them between three labellers.

200 patches are drawn from the pure patches: 50 per class, with forest split between Nan and
Kanchanaburi so the two areas can be compared. Within every class, 5 patches (20 in total) go to
group 0, which every labeller sees, to measure how often people agree. The remaining 45 per class
are dealt out to groups 1, 2 and 3, so each labeller gets the same class mix.

Writes outputs/manual_labels/label_sample.csv with patch_id, group and order only. The area and the
WorldCover label are left out on purpose, so the labeller cannot be steered by them.

Usage:  python src/make_label_sample.py
"""
import numpy as np
import pandas as pd

import config as c

# (class, area or None for all areas, number of patches)
PLAN = [
    ("forest", "nan", 25),
    ("forest", "kanchanaburi", 25),
    ("crop", None, 50),
    ("built-up", None, 50),
    ("water", None, 50),
]
OVERLAP_PER_CLASS = 5   # patches per class that every labeller sees
N_GROUPS = 3
OUT_DIR = c.OUTPUT_DIR / "manual_labels"


def main():
    # keep_default_na=False: the area name "nan" must not be read as a missing value
    table = pd.read_csv(c.PATCH_DIR / "thailand_patches.csv", keep_default_na=False)
    pure = table[table["pure"]]

    picked = []
    for label, area, n in PLAN:
        pool = pure[pure["label"] == label]
        if area is not None:
            pool = pool[pool["area"] == area]
        picked.append(pool.sample(n=min(n, len(pool)), random_state=c.SEED))
    sample = pd.concat(picked).sample(frac=1, random_state=c.SEED)   # shuffle, so classes are mixed

    sample["group"] = 0
    for _, rows in sample.groupby("label"):
        rest = rows.index.to_numpy()[OVERLAP_PER_CLASS:]
        sample.loc[rest, "group"] = np.arange(len(rest)) % N_GROUPS + 1
    sample["order"] = np.arange(len(sample))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sample[["patch_id", "group", "order"]].to_csv(OUT_DIR / "label_sample.csv", index=False)

    print(f"{len(sample)} patches, saved to {OUT_DIR / 'label_sample.csv'}")
    print("\npatches per class and group (group 0 is seen by everyone):")
    print(pd.crosstab(sample["label"], sample["group"]).reindex(c.THAI_EVAL_CLASSES).to_string())
    print("\nforest by area:")
    print(sample[sample["label"] == "forest"]["area"].value_counts().to_string())


if __name__ == "__main__":
    main()
