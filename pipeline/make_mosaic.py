"""Build a demo 'landscape' for the land-cover map: TEST-split patches stitched into a 6 x 8 grid (384 x 512 px,
3.84 km x 5.12 km at 10 m), saved as a 13-band GeoTIFF, an RGB PNG and a ground-truth class map.

It is a synthetic mosaic (neighbouring tiles come from different places, so edges are abrupt), but because
every tile's class is known it measures how well the sliding-window map recovers regions.

Run:  python -m pipeline.make_mosaic
"""
import numpy as np
import pandas as pd
from PIL import Image

from pipeline.config import CLASSES, HERE, SEED, WORK_DIR
from pipeline.data import to_rgb8

SAMPLE_DIR = HERE / "samples"
CODES = {"A": "AnnualCrop", "F": "Forest", "V": "HerbaceousVegetation", "P": "Pasture", "C": "PermanentCrop",
         "S": "SeaLake", "E": "Residential", "W": "River", "I": "Industrial", "H": "Highway"}
LAYOUT = [
    "FFFVVAAA",
    "FFFVVAAA",
    "WWWWWWWW",
    "SSPPEEII",
    "SSPPEEII",
    "CCCHHHHH",
]


def main():
    import rasterio
    from rasterio.transform import from_origin

    meta = pd.read_csv(WORK_DIR / "metadata.csv").merge(pd.read_csv(WORK_DIR / "splits.csv"), on="idx")
    test = meta[meta["split"] == "test"]
    patches = np.load(WORK_DIR / "patches.npy", mmap_mode="r")
    rng = np.random.default_rng(SEED)

    need = {c: sum(row.count(k) for row in LAYOUT) for k, c in CODES.items()}
    pools = {c: list(rng.choice(test.loc[test["class_name"] == c, "idx"].to_numpy(), n, replace=False))
             for c, n in need.items()}

    rows, truth = [], []
    for line in LAYOUT:
        tiles = [np.asarray(patches[pools[CODES[k]].pop()]) for k in line]
        rows.append(np.concatenate(tiles, axis=1))
        truth.append(np.concatenate([np.full((64, 64), CLASSES.index(CODES[k]), np.uint8) for k in line], axis=1))
    mosaic, truth = np.concatenate(rows, axis=0), np.concatenate(truth, axis=0)

    SAMPLE_DIR.mkdir(exist_ok=True)
    h, w, b = mosaic.shape
    with rasterio.open(SAMPLE_DIR / "mosaic_demo.tif", "w", driver="GTiff", width=w, height=h, count=b,
                       dtype="uint16", crs="EPSG:32632", transform=from_origin(500000, 5300000, 10, 10),
                       compress="deflate") as dst:
        dst.write(np.moveaxis(mosaic, -1, 0))
    Image.fromarray(to_rgb8(mosaic)).save(SAMPLE_DIR / "mosaic_demo.png")
    Image.fromarray(truth).save(SAMPLE_DIR / "mosaic_truth.png")
    print(f"Saved {h}x{w} mosaic ({len(LAYOUT) * len(LAYOUT[0])} test patches) to {SAMPLE_DIR}")


if __name__ == "__main__":
    main()
