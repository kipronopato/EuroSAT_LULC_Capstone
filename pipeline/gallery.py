"""Picture of the data: a few true-colour training patches per class, enlarged.

Output: reports/sample_gallery.png
Run:    python -m pipeline.gallery
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.config import CLASSES, REPORT_DIR, SEED, WORK_DIR
from pipeline.data import to_rgb8

PER_CLASS = 8


def main():
    meta = pd.read_csv(WORK_DIR / "metadata.csv")
    patches = np.load(WORK_DIR / "patches.npy", mmap_mode="r")
    picks = meta.groupby("class_name").sample(PER_CLASS, random_state=SEED)

    fig, axes = plt.subplots(len(CLASSES), PER_CLASS, figsize=(PER_CLASS * 1.3, len(CLASSES) * 1.35))
    for r, c in enumerate(CLASSES):
        for k, idx in enumerate(picks.loc[picks["class_name"] == c, "idx"]):
            ax = axes[r, k]
            ax.imshow(to_rgb8(np.asarray(patches[idx])), interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if k == 0:
                ax.set_ylabel(c, rotation=0, ha="right", va="center", fontsize=9)
    fig.suptitle("EuroSAT: true-colour Sentinel-2 patches (64 x 64 px = 640 m x 640 m)", y=0.995)
    fig.tight_layout()
    REPORT_DIR.mkdir(exist_ok=True)
    fig.savefig(REPORT_DIR / "sample_gallery.png", dpi=110)
    print(f"Saved {REPORT_DIR / 'sample_gallery.png'}")


if __name__ == "__main__":
    main()
