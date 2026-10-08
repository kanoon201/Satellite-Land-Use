"""Merge the answers of the labellers (step A1) into one vote per patch.

Reads every outputs/manual_labels/<name>.csv (the file name is the labeller; label_sample.csv is skipped)
and writes into outputs/a1/:
    votes.csv          one row per patch: every labeller's answer, the consensus and how it came about
    clean_test_set.csv the patches that have a consensus label (the clean test set)
    summary.txt        the same tables as printed below

Consensus of a patch: "unsure" answers do not vote. The label with the most votes wins if it has at least two votes and
more than any other label. Status is "unanimous" (everybody gave that label), "majority", "split" (a tie, or
every label has one vote) or "undecided" (fewer than two labellers gave a class).

The agreement with WorldCover is also given for the blind answers only: a labeller who pressed "show WorldCover label"
saw the label first (peeked = True), so such answers are not an independent check.

Usage:  python src/merge_labels.py
"""
import time
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd

import config as c

LABEL_DIR = c.OUTPUT_DIR / "manual_labels"
OUT_DIR = c.OUTPUT_DIR / "a1"
ANSWERS = c.CLASSES + ["unsure"]


def load_answers():
    frames = {}
    for path in sorted(LABEL_DIR.glob("*.csv")):
        if path.name == "label_sample.csv":
            continue
        # keep_default_na=False: nothing here may be turned into an empty value
        answers = pd.read_csv(path, keep_default_na=False).drop_duplicates("patch_id", keep="last")
        frames[path.stem] = answers.set_index("patch_id")
    return frames


def consensus(answers):
    """answers: the labels given to one patch (one per labeller). Returns (label, status)."""
    valid = [a for a in answers if a != "unsure"]
    if len(valid) < 2:
        return "", "undecided"
    ranked = Counter(valid).most_common()
    top, votes = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0
    if votes >= 2 and votes > second:
        return top, "unanimous" if votes == len(answers) else "majority"
    return "", "split"


def cohen_kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    observed = (a == b).mean()
    expected = sum((a == k).mean() * (b == k).mean() for k in set(a) | set(b))
    return (observed - expected) / (1 - expected) if expected < 1 else float("nan")


def fleiss_kappa(table):
    """table: patches x answers, every cell the number of labellers who gave that answer (same total in every row)."""
    n = table.sum(axis=1)[0]
    p_class = table.sum(axis=0) / table.sum()
    p_item = ((table ** 2).sum(axis=1) - n) / (n * (n - 1))
    expected = (p_class ** 2).sum()
    return (p_item.mean() - expected) / (1 - expected)


def write(path, text=None, table=None):
    # Windows sometimes refuses to open a file that was written a moment ago, so retry a few times
    for attempt in range(5):
        try:
            if table is not None:
                table.to_csv(path, index=False)
            else:
                path.write_text(text, encoding="utf-8")
            return
        except OSError:
            if attempt == 4:
                raise
            time.sleep(1)


def pct(x):
    return f"{100 * x:.1f}%"


def agreement_by(frame, key):
    """Number of patches and share where the consensus equals the WorldCover label, for every value of `key`."""
    rows = {value: {"patches": len(g), "equal": pct((g["consensus"] == g["worldcover"]).mean())} for value, g in frame.groupby(key)}
    return pd.DataFrame(rows).T


def main():
    frames = load_answers()
    names = list(frames)
    if len(names) < 2:
        raise SystemExit(f"need at least two answer files in {LABEL_DIR}, found {names}")

    patches = pd.read_csv(c.PATCH_DIR / "thailand_patches.csv", keep_default_na=False).set_index("patch_id")
    ids = sorted(set().union(*[set(f.index) for f in frames.values()]))
    votes = pd.DataFrame(index=pd.Index(ids, name="patch_id"))
    for n in names:
        votes[n] = frames[n]["manual_label"]
        votes[f"{n}_peeked"] = frames[n]["peeked"].astype(str).str.lower().eq("true") if "peeked" in frames[n] else False
        votes[f"{n}_peeked"] = votes[f"{n}_peeked"].fillna(False).astype(bool)
    votes["n_answers"] = votes[names].notna().sum(axis=1)
    results = [consensus([a for a in row if isinstance(a, str)]) for row in votes[names].itertuples(index=False)]
    votes["consensus"] = [r[0] for r in results]
    votes["status"] = [r[1] for r in results]
    votes["any_peeked"] = votes[[f"{n}_peeked" for n in names]].any(axis=1)
    votes["worldcover"] = patches.loc[votes.index, "label"]
    votes["worldcover_purity"] = patches.loc[votes.index, "purity"]
    votes["area"] = patches.loc[votes.index, "area"]

    lines = []

    def say(text=""):
        lines.append(text)
        print(text)

    say(f"labellers: {', '.join(names)}")
    say(f"patches with at least one answer: {len(votes)}; answered by all {len(names)}: {int((votes['n_answers'] == len(names)).sum())}")

    say("\n== What every labeller answered (number of patches) ==")
    counts = pd.DataFrame({n: frames[n]["manual_label"].value_counts() for n in names}).reindex(ANSWERS).fillna(0).astype(int)
    counts.loc["answers with WorldCover shown (peeked)"] = [int(votes[f"{n}_peeked"].sum()) for n in names]
    say(counts.to_string())

    say("\n== Agreement between labellers ==")
    pairs = []
    for a, b in combinations(names, 2):
        both = votes[[a, b]].dropna()
        pairs.append({"pair": f"{a} - {b}", "patches": len(both), "same answer": pct((both[a] == both[b]).mean()),
                      "Cohen kappa": round(cohen_kappa(both[a], both[b]), 3)})
    say(pd.DataFrame(pairs).set_index("pair").to_string())
    full = votes[votes["n_answers"] == len(names)]
    table = np.array([[sum(row[n] == k for n in names) for k in ANSWERS] for _, row in full.iterrows()])
    say(f"Fleiss kappa over the {len(full)} patches everybody answered: {fleiss_kappa(table):.3f}")
    say("(kappa: 1 = perfect, 0 = no better than chance; above 0.6 is usually called good, above 0.8 very good)")

    say("\n== Consensus ==")
    say(votes["status"].value_counts().reindex(["unanimous", "majority", "split", "undecided"]).fillna(0).astype(int).to_string())
    has = votes[votes["consensus"] != ""]
    say(f"{len(has)} of {len(votes)} patches have a consensus label ({pct(len(has) / len(votes))}); they form the clean test set")
    say("\nconsensus label by WorldCover label (patches with a consensus):")
    say(pd.crosstab(has["worldcover"], has["consensus"]).reindex(columns=ANSWERS[:-1]).fillna(0).astype(int).to_string())

    say("\n== How often does the human answer equal the WorldCover label? ==")
    rows = []
    for n in names:
        for label, subset in (("all answers", votes[votes[n].notna() & (votes[n] != "unsure")]),
                              ("blind answers only", votes[votes[n].notna() & (votes[n] != "unsure") & ~votes[f"{n}_peeked"]])):
            rows.append({"who": n, "answers": label, "patches": len(subset), "equal to WorldCover": pct((subset[n] == subset["worldcover"]).mean())})
    blind = has[~has["any_peeked"]]
    rows.append({"who": "consensus", "answers": "all consensus patches", "patches": len(has), "equal to WorldCover": pct((has["consensus"] == has["worldcover"]).mean())})
    rows.append({"who": "consensus", "answers": "nobody peeked", "patches": len(blind), "equal to WorldCover": pct((blind["consensus"] == blind["worldcover"]).mean())})
    say(pd.DataFrame(rows).to_string(index=False))

    say("\nconsensus equal to WorldCover, by WorldCover class (nobody peeked):")
    say(agreement_by(blind, "worldcover").to_string())
    forest = blind[blind["worldcover"] == "forest"]
    say("\nforest, by area (nobody peeked):")
    say(agreement_by(forest, "area").to_string())
    say("\nThese 200 patches are pure WorldCover patches, so this is the agreement in the easy case. Mixed patches are not in the sample.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write(OUT_DIR / "votes.csv", table=votes.reset_index())
    clean = has[["area", "consensus", "status", "worldcover", "any_peeked"]].reset_index()
    clean = clean.join(patches[["row", "col"]], on="patch_id")
    write(OUT_DIR / "clean_test_set.csv", table=clean)
    write(OUT_DIR / "summary.txt", text="\n".join(lines) + "\n")
    print(f"\nsaved votes.csv, clean_test_set.csv and summary.txt in {OUT_DIR}")


if __name__ == "__main__":
    main()
