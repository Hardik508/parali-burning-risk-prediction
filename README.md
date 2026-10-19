# Predictive System for Parali Burning Risk Detection

A deep-learning prototype that predicts, **one to five days in advance**, how likely a cropland location in the
**Hisar and Sirsa districts of Haryana, India** is to be burnt during the parali (paddy stubble) burning season.

Satellite fire maps show where fields have *already* been burnt. This system looks ahead instead: it combines
satellite imagery, weather data, geospatial analysis and the history of nearby fires to rank locations by burn risk,
so that district teams can plan field visits and machinery support before burning starts.

> Course project, Deep Learning (DSE3120), B.Tech CSE (Data Science), Manipal University Jaipur.

---

## How it works

| Input | Source | Model branch |
|---|---|---|
| 32 × 32 image chip (red, green, blue, near-infrared, 10 m) | Sentinel-2 surface reflectance | CNN (3 conv blocks) |
| 60-day series (12 × 5-day steps): NDVI, NDMI, temperature, max temperature, humidity, vapour-pressure deficit, soil moisture, wind, rain | Sentinel-2 + ERA5-Land | LSTM (64 units) |
| Fire history (4 counts of earlier fires within 1–5 km) + harvest timing (5 NDVI features) | NASA FIRMS VIIRS + NDVI series | Dense layer (16) |

The three branches are joined and passed through a small classifier head (dropout 0.2 → 64 → dropout 0.5 → 1).
The output is a burn-risk score between 0 and 1. The network has 52,657 parameters.

Labels are VIIRS 375 m active-fire detections (NASA FIRMS) on cropland, seasons **2021–2025**, 1 Oct – 30 Nov.
Every input is taken only from data **before** the prediction date, so the model never sees the fire it predicts.

## Results of the final model

All seasons were pooled and split by **0.1° grid cell** (70 / 15 / 15), so test locations are never seen in training.

| Set | Samples | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| Training | 8,388 | 0.863 | 0.779 | 0.851 | 0.813 | 0.943 | 0.901 |
| Validation | 1,818 | 0.852 | 0.789 | 0.825 | 0.806 | 0.926 | 0.889 |
| **Test** | **1,917** | **0.881** | **0.794** | **0.788** | **0.791** | **0.947** | **0.889** |

- Test PR-AUC 95% CI (1,000 bootstrap resamples): **0.866 – 0.908**; a model with no skill scores 0.286.
- Test confusion matrix at the validation-chosen threshold (0.553): TN 1,257 · FP 112 · FN 116 · TP 432.
- Flagging the riskiest **20%** of test locations catches **62.2%** of all burns with **89.0%** precision (3.11× better than random).

Exact values: [`results/final_model/final_model_metrics.json`](results/final_model/final_model_metrics.json).
Figures for the final model are in [`results/final_model/`](results/final_model/).

## Quick start (Windows)

```bat
:: 1. one-time setup (creates .venv and installs packages)
setup_env.bat

:: 2. run the demo: loads the trained model, evaluates it and opens the report and risk map
RUN_DEMO.bat
```

The demo writes `demo_output/demo_report.html` (all tables and graphs) and `demo_output/risk_map.html`
(interactive map). No training and no Google Earth Engine account are needed.

### Linux / macOS

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd code && python run_demo.py --out ../demo_output
```

### Retraining

```bash
cd code
python train.py                # pooled, location-grouped split; writes ../train_output/run_<date_time>/
python train.py --epochs 2     # quick check
python check_gpu.py            # is PyTorch using the GPU?
```

Each run writes a `SUMMARY.txt` and appends one line to `train_output/runs_log.csv`.
Retraining starts from random weights, so the numbers will differ slightly from the saved model.

### Rebuilding the dataset from scratch (optional)

`code/collect_gee.py` downloads FIRMS fires and extracts Sentinel-2 / ERA5-Land data from Google Earth Engine.
It needs a Google Cloud project with Earth Engine enabled and a free FIRMS map key:

```bash
python collect_gee.py --project YOUR_GCP_PROJECT --map_key YOUR_FIRMS_KEY --years 2021 2022 2023 2024 2025
python add_history.py           # adds the fire-history features
```

## Repository layout

```
code/
  train.py           model (CNN + LSTM + dense), data loading, split, training, evaluation
  run_demo.py        loads the saved model and produces the report, tables, graphs and risk map
  paper_results.py   full experiment used for the report figures
  make_map.py        interactive risk map (folium)
  collect_gee.py     data collection (FIRMS + Google Earth Engine)
  add_history.py     fire-history features
  gen_synthetic.py   small synthetic dataset for testing the code
  check_gpu.py       checks whether PyTorch can see the GPU
data/
  real_dataset_matched.npz   the dataset used to train the model (12,123 samples, stored with Git LFS)
  fire_history.csv           FIRMS VIIRS detections used for the fire-history features
  dataset.py                 exports the arrays to Excel
results/
  proposed_model.pt          trained weights of the final model
  final_model/               metrics and figures of the final model
  ...                        tables, figures and predictions from the full experiment
docs/                        code guide, VS Code guide, data sources
```

### Dataset format (`real_dataset_matched.npz`)

| Array | Shape | Meaning |
|---|---|---|
| `X_img` | 12123 × 4 × 32 × 32 | Sentinel-2 chip (R, G, B, NIR), reflectance 0–1 |
| `X_seq` | 12123 × 12 × 9 | 12 five-day steps × 9 features (`feature_names`) |
| `X_hist` | 12123 × 4 | fire-history counts, log(1 + x) (`hist_names`) |
| `y` | 12123 | 1 = burn detected, 0 = no burn |
| `year`, `doy`, `lat`, `lon` | 12123 | season, day of year, location |

```python
import numpy as np
d = np.load("data/real_dataset_matched.npz")
print(d.files, d["X_img"].shape, d["y"].mean())
```

## Limitations

- VIIRS misses small, short-lived and cloud-covered fires, so some "no burn" samples are probably undetected burns.
- The test set contains unseen **locations**, not an unseen **season**; forecasting a completely new season is harder.
- 34% of the samples are burns, more than the real share of burning fields, so precision in real use would be lower.
- Only the Hisar–Sirsa region of Haryana is covered.

## Data sources and credits

- Fire detections: NASA FIRMS, VIIRS S-NPP 375 m (NASA open data).
- Imagery: Copernicus Sentinel-2 (contains modified Copernicus Sentinel data 2021–2025).
- Weather: ERA5-Land, Copernicus Climate Change Service (contains modified Copernicus Climate Change Service information).
- Land cover: ESA WorldCover 10 m v200.
- Processing: Google Earth Engine.

## Team

| Name | Email |
|---|---|
| Member 1 | member1@muj.manipal.edu |
| Member 2 | member2@muj.manipal.edu |
| Member 3 | member3@muj.manipal.edu |

Department of Computer Science and Engineering, Manipal University Jaipur.
