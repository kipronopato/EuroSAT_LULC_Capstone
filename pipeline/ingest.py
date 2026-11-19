"""Stage 1 - ingest EuroSAT_MS.

Reads every 13-band GeoTIFF straight out of the zip (no extraction needed), records its
class and centre coordinate (lat/lon), and packs all pixels into one uint16 array on disk.

Outputs (in artifacts/):
    metadata.csv   one row per patch: idx, file, label, class_name, lat, lon
    patches.npy    uint16 array of shape (N, 64, 64, 13), row i matches metadata idx i

Run:  python -m pipeline.ingest
"""
import importlib.util
import os
import urllib.request
import zipfile
from pathlib import Path

# Use rasterio's own PROJ database, not one left in PROJ_LIB by another tool (e.g. PostGIS)
_proj = Path(importlib.util.find_spec("rasterio").origin).parent / "proj_data"
if _proj.exists():
    os.environ["PROJ_DATA"] = os.environ["PROJ_LIB"] = str(_proj)

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as warp_transform

from pipeline.config import CLASSES, WORK_DIR, ZIP_PATH, ZIP_URL


def list_tifs(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        names = sorted(n for n in z.namelist() if n.lower().endswith(".tif"))
    rows = []
    for n in names:
        class_name = n.split("/")[-2]
        if class_name in CLASSES:
            rows.append({"file": n, "class_name": class_name, "label": CLASSES.index(class_name)})
    return pd.DataFrame(rows)


def main():
    if not ZIP_PATH.exists():
        print(f"Downloading EuroSAT (2.07 GB) from {ZIP_URL}")
        ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    meta = list_tifs(ZIP_PATH)
    print(f"Found {len(meta)} patches in {ZIP_PATH.name}")
    print(meta["class_name"].value_counts().to_string())

    patches = np.lib.format.open_memmap(WORK_DIR / "patches.npy", mode="w+",
                                        dtype=np.uint16, shape=(len(meta), 64, 64, 13))
    lats, lons = [], []
    archive = zipfile.ZipFile(ZIP_PATH)
    for i, f in enumerate(meta["file"]):
        with rasterio.MemoryFile(archive.read(f)) as mem, mem.open() as src:
            patches[i] = np.moveaxis(src.read(), 0, -1)          # (bands, H, W) -> (H, W, bands)
            cx = (src.bounds.left + src.bounds.right) / 2
            cy = (src.bounds.bottom + src.bounds.top) / 2
            lon, lat = warp_transform(src.crs, "EPSG:4326", [cx], [cy])   # each tile has its own UTM zone
        lats.append(lat[0])
        lons.append(lon[0])
        if (i + 1) % 2000 == 0:
            print(f"  read {i + 1}/{len(meta)}")
    patches.flush()

    meta.insert(0, "idx", np.arange(len(meta)))
    meta["lat"], meta["lon"] = lats, lons
    meta.to_csv(WORK_DIR / "metadata.csv", index=False)
    print(f"Saved metadata.csv and patches.npy {patches.shape} to {WORK_DIR}")
    print(f"Latitude {meta.lat.min():.2f}..{meta.lat.max():.2f}, longitude {meta.lon.min():.2f}..{meta.lon.max():.2f}")


if __name__ == "__main__":
    main()
