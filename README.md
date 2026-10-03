# Satellite Land-Use Classification: Europe to Thailand

Group Assignment #2 (Analytics on Complex Data), Data Mining, Kasetsart University.

Classify Thai land use (forest, crop, grassland, built-up, water) from Sentinel-2
imagery using models trained on European data (EuroSAT), and measure the domain
shift between the two regions.

## Data sources

| Source | Role | Labels |
|---|---|---|
| EuroSAT (Sentinel-2, 13 bands, 27,000 patches) | training | clean, one per patch, 10 classes |
| Sentinel-2 L2A over Thailand (Google Earth Engine) | test images | none |
| ESA WorldCover 10 m | test labels | noisy (~75% accurate), one per pixel, 11 classes |

## Layout

```
src/config.py     paths, band list, class mappings (works locally and on Colab)
notebooks/        analysis notebooks
outputs/          figures and metrics
slides_report/    presentation and report
data/             not tracked in git (eurosat/, thailand/, patches/)
```

## Setup

```bash
conda create -n landuse python=3.12 -y
conda activate landuse
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

On Google Colab, place the project at `MyDrive/DataMining_Project_AT`; `src/config.py`
detects Colab and switches paths automatically.
