"""Paths and constants shared by every pipeline stage."""
import os
from pathlib import Path

HERE       = Path(__file__).resolve().parent.parent          # project root
DATA_DIR   = HERE / "data" / "EuroSAT"                        # raw download (git-ignored)
ZIP_PATH   = DATA_DIR / "EuroSATallBands.zip"                 # 13-band GeoTIFFs, one folder per class
ZIP_URL    = "https://huggingface.co/datasets/torchgeo/eurosat/resolve/main/EuroSATallBands.zip"
WORK_DIR   = HERE / "artifacts"                               # metadata, splits, packed arrays
RESEARCH_MODEL_DIR = WORK_DIR / "research_champion"           # independent of the packaged serving models
REPORT_DIR = HERE / "reports"                                 # figures and tables for the write-up

TRACKING   = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{(HERE / 'mlflow.db').as_posix()}")
EXPERIMENT = "capstone_eurosat_lulc"
REG_NAME   = "eurosat_lulc_classifier"

SEED = 42

# EuroSAT classes, ordered from the class kept most common to the one made rarest in the
# imbalanced training set (cropland and forest dominate, highways/industry/rivers are rare).
CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Pasture", "PermanentCrop",
    "SeaLake", "Residential", "River", "Industrial", "Highway",
]
IMBALANCE_RATIO = 20        # largest training class / smallest training class

# Sentinel-2 band order inside every EuroSAT_MS GeoTIFF
BANDS   = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B08A", "B09", "B10", "B11", "B12"]
RGB_IDX = [3, 2, 1]         # B04 (red), B03 (green), B02 (blue)
RED, NIR = 3, 7             # B04, B08 -> NDVI

# Spatial split: patches are grouped into square blocks of this size (degrees) and whole
# blocks are assigned to train / validation / test, so adjacent patches never straddle splits.
BLOCK_DEG = 0.1
N_FOLDS   = 7               # StratifiedGroupKFold: 1 fold test, 1 fold validation, 5 folds train

# Reflectance -> 8-bit true colour. Shared by training and the API so an RGB PNG means the same thing.
RGB_MAX_REFLECTANCE = 3000.0
