"""Copy one TEST-split patch per class into samples/ (as the original 13-band GeoTIFF and as an
8-bit RGB PNG), for the API tests, the Streamlit demo and the /docs page.

Run:  python -m pipeline.make_samples
"""
import zipfile

import numpy as np
import pandas as pd
from PIL import Image

from pipeline.config import HERE, SEED, WORK_DIR, ZIP_PATH
from pipeline.data import to_rgb8

SAMPLE_DIR = HERE / "samples"


def main():
    meta = pd.read_csv(WORK_DIR / "metadata.csv").merge(pd.read_csv(WORK_DIR / "splits.csv"), on="idx")
    picks = meta[meta["split"] == "test"].groupby("class_name").sample(1, random_state=SEED)
    patches = np.load(WORK_DIR / "patches.npy", mmap_mode="r")
    SAMPLE_DIR.mkdir(exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as z:
        for _, r in picks.iterrows():
            (SAMPLE_DIR / f"{r['class_name']}.tif").write_bytes(z.read(r["file"]))
            Image.fromarray(to_rgb8(np.asarray(patches[r["idx"]]))).save(SAMPLE_DIR / f"{r['class_name']}.png")
    print(f"Saved {len(picks)} GeoTIFF + PNG samples to {SAMPLE_DIR}")


if __name__ == "__main__":
    main()
