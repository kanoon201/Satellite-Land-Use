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

# ---- Thai study areas -------------------------------------------------------
# All three areas lie in UTM zone 47N, the native grid of their Sentinel-2 tiles.
THAI_CRS = "EPSG:32647"
PIXEL_SIZE = 10          # metres
AOI_SIZE_PX = 3072       # 48 x 48 patches of 64 px = 30.72 km per side

# name -> (lon, lat) of the area centre
AOIS = {
    "bangkok": (100.55, 13.80),      # dense city, river, suburbs
    "suphanburi": (100.12, 14.47),   # irrigated rice plain
    "nan": (100.78, 18.78),          # forested mountains, upland crops
}

# Dry season of 2021: fewest clouds, and the same year as WorldCover v200.
S2_START = "2021-01-01"
S2_END = "2021-03-01"

S2_COLLECTIONS = {
    "L2A": "COPERNICUS/S2_SR_HARMONIZED",  # surface reflectance, no B10
    "L1C": "COPERNICUS/S2_HARMONIZED",     # top-of-atmosphere, same level as EuroSAT
}
BANDS_L1C = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]

# Cloud Score+ "cs_cdf" runs from 0 (cloud or shadow) to 1 (clear); keep pixels at or above this.
CLOUD_SCORE_COLLECTION = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
CLEAR_THRESHOLD = 0.60

WORLDCOVER_COLLECTION = "ESA/WorldCover/v200"

# code -> (name, official map colour)
WORLDCOVER_CLASSES = {
    10: ("Tree cover", "#006400"),
    20: ("Shrubland", "#ffbb22"),
    30: ("Grassland", "#ffff4c"),
    40: ("Cropland", "#f096ff"),
    50: ("Built-up", "#fa0000"),
    60: ("Bare / sparse vegetation", "#b4b4b4"),
    70: ("Snow and ice", "#f0f0f0"),
    80: ("Permanent water bodies", "#0064c8"),
    90: ("Herbaceous wetland", "#0096a0"),
    95: ("Mangroves", "#00cf75"),
    100: ("Moss and lichen", "#fae6a0"),
}


def thai_path(area, product):
    """Path of one downloaded file, e.g. thai_path("nan", "S2_L2A.tif")."""
    return THAILAND_DIR / area / f"{area}_{product}"
