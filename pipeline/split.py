"""Stage 2 - spatial split and imbalanced training set.

1. Every patch is assigned to a BLOCK_DEG x BLOCK_DEG geographic block.
2. StratifiedGroupKFold keeps each block whole while keeping class mixes similar across folds:
   fold 0 -> test, fold 1 -> validation, the other folds -> training pool.
3. The training pool is subsampled so class counts fall geometrically from CLASSES[0] down to
   CLASSES[-1], with the largest class IMBALANCE_RATIO times the smallest. Validation and test
   are left untouched.

Outputs:
    artifacts/splits.csv     idx, block, split (train/val/test/unused), train_full (bool)
    reports/split_counts.csv and reports/split_map.png, reports/class_counts.png

Run:  python -m pipeline.split
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import BallTree

from pipeline.config import BLOCK_DEG, CLASSES, IMBALANCE_RATIO, N_FOLDS, REPORT_DIR, SEED, WORK_DIR


def assign_blocks(meta):
    row = np.floor(meta["lat"] / BLOCK_DEG).astype(int)
    col = np.floor(meta["lon"] / BLOCK_DEG).astype(int)
    return row.astype(str) + "_" + col.astype(str)


def imbalanced_targets(available):
    """Geometric class sizes n_k = n_top * ratio^(-k/(K-1)), with n_top as large as the data allows."""
    k = np.arange(len(CLASSES))
    decay = IMBALANCE_RATIO ** (-k / (len(CLASSES) - 1))
    n_top = min(available[c] / decay[i] for i, c in enumerate(CLASSES))
    return {c: int(round(n_top * decay[i])) for i, c in enumerate(CLASSES)}


def proximity_check(meta):
    """How close is each test patch to its nearest training-pool patch, spatial vs random split?
    A patch is 640 m wide, so a centre-to-centre distance under ~0.7 km means the two touch."""
    coords = np.radians(meta[["lat", "lon"]].to_numpy())
    test = (meta["split"] == "test").to_numpy()
    train = meta["train_full"].to_numpy()
    perm = np.random.default_rng(SEED).permutation(len(meta))
    rand_test, rand_train = perm[:test.sum()], perm[test.sum():][:train.sum()]
    rows = []
    for name, te, tr in [("spatial", coords[test], coords[train]), ("random", coords[rand_test], coords[rand_train])]:
        d = BallTree(tr, metric="haversine").query(te)[0][:, 0] * 6371.0
        rows.append({"split": name, "median_km_to_nearest_train": np.median(d),
                     "share_within_1km": np.mean(d < 1.0), "share_adjacent_0.7km": np.mean(d < 0.7)})
    return pd.DataFrame(rows)


def main():
    meta = pd.read_csv(WORK_DIR / "metadata.csv")
    meta["block"] = assign_blocks(meta)
    print(f"{meta['block'].nunique()} geographic blocks of {BLOCK_DEG} deg")

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = np.empty(len(meta), dtype=int)
    for f, (_, idx) in enumerate(sgkf.split(meta, meta["label"], groups=meta["block"])):
        folds[idx] = f
    meta["split"] = np.select([folds == 0, folds == 1], ["test", "val"], "train")
    meta["train_full"] = meta["split"] == "train"         # balanced reference training set

    # Imbalance the training pool only
    pool = meta[meta["split"] == "train"]
    targets = imbalanced_targets(pool["class_name"].value_counts())
    rng = np.random.default_rng(SEED)
    keep = []
    for c, n in targets.items():
        idx = pool.index[pool["class_name"] == c].to_numpy()
        keep.extend(rng.choice(idx, size=n, replace=False))
    meta.loc[(meta["split"] == "train") & ~meta.index.isin(keep), "split"] = "unused"

    # Sanity checks: no block in two evaluation splits, untouched val/test
    used = meta[meta["split"] != "unused"]
    assert used.groupby("block")["split"].nunique().max() == 1, "a block leaks across splits"

    counts = meta.groupby(["class_name", "split"]).size().unstack(fill_value=0)
    counts["train_full"] = meta[meta["train_full"]].groupby("class_name").size()
    counts = counts.reindex(CLASSES)[["train_full", "train", "val", "test"]]
    counts.loc["TOTAL"] = counts.sum()
    print(counts.to_string())
    ratio = counts.loc[CLASSES, "train"].max() / counts.loc[CLASSES, "train"].min()
    print(f"Training imbalance ratio: {ratio:.1f}")

    WORK_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    leakage = proximity_check(meta)
    print(leakage.to_string(index=False, float_format="%.3f"))
    leakage.to_csv(REPORT_DIR / "leakage_check.csv", index=False)
    meta[["idx", "block", "split", "train_full"]].to_csv(WORK_DIR / "splits.csv", index=False)
    counts.to_csv(REPORT_DIR / "split_counts.csv")

    # Figure: where each split lives on the map
    fig, ax = plt.subplots(figsize=(8, 7))
    for s, colour in [("unused", "#cccccc"), ("train", "#2a6fdb"), ("val", "#f2a900"), ("test", "#d1495b")]:
        d = meta[meta["split"] == s]
        ax.scatter(d["lon"], d["lat"], s=2, c=colour, label=f"{s} ({len(d)})", alpha=0.7)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Spatial split by {BLOCK_DEG}° blocks")
    ax.legend(markerscale=5)
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "split_map.png", dpi=130)
    plt.close(fig)

    # Figure: class counts per split
    fig, ax = plt.subplots(figsize=(10, 4.5))
    counts.loc[CLASSES, ["train_full", "train", "val", "test"]].plot.bar(ax=ax, width=0.8)
    ax.set_ylabel("Patches")
    ax.set_title(f"Class counts: imbalanced training set (ratio {ratio:.0f}:1) vs untouched val/test")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "class_counts.png", dpi=130)
    plt.close(fig)
    print(f"Saved splits.csv to {WORK_DIR} and figures to {REPORT_DIR}")


if __name__ == "__main__":
    main()
