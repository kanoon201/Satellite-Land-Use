"""Shared settings. Works unchanged on the local machine and on Google Colab."""
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    PROJECT_DIR = Path("/content/drive/MyDrive/DataMining_Project_AT")
else:
    PROJECT_DIR = Path(__file__).resolve().parents[1]  # folder that contains src/

DATA_DIR = PROJECT_DIR / "data"

EUROSAT_DIR = DATA_DIR / "eurosat"
THAILAND_DIR = DATA_DIR / "thailand"
PATCH_DIR = DATA_DIR / "patches"
OUTPUT_DIR = PROJECT_DIR / "outputs"

EE_PROJECT = "satellite-land-use-510508"

# Sentinel-2 bands shared by EuroSAT (L1C, 13 bands) and GEE L2A (12 bands).
# B10 (cirrus) does not exist in L2A, so it is dropped from EuroSAT as well.
BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
EUROSAT_BAND_ORDER = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11", "B12", "B8A"]

PATCH_SIZE = 64  # pixels at 10 m = 640 m, same as EuroSAT

CLASSES = ["forest", "crop", "grassland", "built-up", "water"]

EUROSAT_TO_TARGET = {
    "Forest": "forest",
    "AnnualCrop": "crop",
    "PermanentCrop": "crop",
    "Pasture": "grassland",
    "HerbaceousVegetation": "grassland",
    "Residential": "built-up",
    "Industrial": "built-up",
    "Highway": "built-up",
    "River": "water",
    "SeaLake": "water",
}

# ESA WorldCover pixel codes -> target class. Codes not listed are discarded.
WORLDCOVER_TO_TARGET = {
    10: "forest",     # Tree cover
    40: "crop",       # Cropland
    30: "grassland",  # Grassland
    50: "built-up",   # Built-up
    80: "water",      # Permanent water bodies
}
