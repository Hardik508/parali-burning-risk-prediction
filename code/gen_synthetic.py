"""
Synthetic data generator for the stubble-burning early-warning prototype.

PURPOSE: lets you test the WHOLE pipeline (training, evaluation, plots) before the
real satellite data is downloaded. The numbers you get from this data are NOT
real results and must never be reported as such.

Output: data/synthetic_dataset.npz with the SAME format the real collector produces:
    X_seq : (N, 12, 9)   12 time steps (5 days apart) x 9 features, in this order:
              satellite : ndvi (greenness), ndmi (moisture of plants/soil)
              heat      : temp (mean, C), temp_max (daily max, C)
              dryness   : rh (relative humidity, %), vpd (vapour pressure deficit, kPa),
                          soil (top-soil moisture, m3/m3)
              other     : wind (m/s), precip (rain, mm)
            NaN = missing (cloud gap) for the two satellite features
    X_img : (N, 4, 32, 32) satellite chip, bands = [Red, Green, Blue, NIR], reflectance 0..1
    y     : (N,) 1 = fire occurred within the next 5 days, 0 = no fire
    X_hist: (N, 4)       fire history, log(1+count) of earlier VIIRS detections near the field:
              hist_prev_1km : within ~1 km, previous 2 seasons
              hist_prev_5km : within ~5 km, previous 2 seasons
              hist_now_20d  : within ~5 km, this season, days t0-20 .. t0-6
              hist_ring_5d  : in the 1-5 km ring, this season, days t0-5 .. t0-1
    year, lat, lon, doy : metadata per sample;   feature_names, hist_names : the names above
"""
import argparse
import os
import numpy as np
from scipy.ndimage import gaussian_filter

T_STEPS, STEP_DAYS = 12, 5
LAT_RANGE, LON_RANGE = (28.8, 29.9), (74.6, 76.3)  # Hisar-Sirsa box, Haryana
FEATURES = ["ndvi", "ndmi", "temp", "temp_max", "rh", "vpd", "soil", "wind", "precip"]
# fire-history inputs (log(1+count) of earlier fire detections near the field; only fires BEFORE the observation date)
HIST_NAMES = ["hist_prev_1km", "hist_prev_5km", "hist_now_20d", "hist_ring_5d"]


def sat_vapour_pressure(t_c):
    """Saturation vapour pressure in hPa (Magnus formula)."""
    return 6.112 * np.exp(17.62 * t_c / (243.12 + t_c))


def make_ndvi(t_days, harvest_day, rng):
    """NDVI curve of a paddy field: green plateau, senescence, sharp harvest drop, bare stubble."""
    peak = rng.uniform(0.65, 0.82)
    stubble = rng.uniform(0.15, 0.27)
    ndvi = np.empty_like(t_days, dtype=float)
    for i, t in enumerate(t_days):
        if t < harvest_day:
            senescence = np.clip((t - (harvest_day - 20)) / 20.0, 0, 1)  # yellowing before harvest
            ndvi[i] = peak - 0.25 * senescence
        else:
            since = t - harvest_day
            drop = np.clip(since / 4.0, 0, 1)                          # drop within ~4 days
            level = (peak - 0.25) * (1 - drop) + stubble * drop
            regrow = np.clip((since - 15) / 20.0, 0, 1) * 0.15         # wheat sown ~15 days later
            ndvi[i] = level + regrow
    return ndvi + rng.normal(0, 0.03, size=ndvi.shape)


def make_weather(t_days, dry_spell, rng):
    """Seasonal weather. dry_spell=True makes the last days hotter, drier, calmer (fire-prone)."""
    n = len(t_days)
    temp = 28 - 0.15 * (t_days - 274) + rng.normal(0, 1.5, n)
    rh = 52 + rng.normal(0, 8, n)
    rain_prob = 0.08
    heat_shift = 0.0
    if dry_spell:
        rh = rh - np.linspace(0, 9, n)           # humidity falls towards the observation date
        heat_shift = 1.5
        rain_prob = 0.03
    temp = temp + heat_shift
    temp_max = temp + 8 + rng.normal(0, 1.5, n)
    rh = np.clip(rh, 15, 95)
    vpd = sat_vapour_pressure(temp) * (1 - rh / 100) / 10         # kPa, physically consistent
    precip = np.where(rng.random(n) < rain_prob, rng.exponential(6, n), 0.0)
    recent_rain = np.convolve(precip, np.ones(3), mode="full")[:n]  # rain in the last 3 steps
    soil = np.clip(0.17 + 0.012 * recent_rain + rng.normal(0, 0.015, n) - (0.02 if dry_spell else 0), 0.05, 0.40)
    wind = np.clip(rng.gamma(4, 0.6, n) - (0.4 if dry_spell else 0), 0, None)
    return temp, temp_max, rh, vpd, soil, wind, precip


def make_chip(ndvi_now, rng, size=32):
    """4-band chip whose colour depends on how green the field is at observation time."""
    green_frac = np.clip((ndvi_now - 0.2) / 0.55, 0, 1)
    green = np.array([0.05, 0.08, 0.04, 0.40])    # R, G, B, NIR of healthy crop
    straw = np.array([0.18, 0.17, 0.12, 0.25])    # R, G, B, NIR of harvested stubble
    base = green_frac * green + (1 - green_frac) * straw
    chip = np.empty((4, size, size), dtype=np.float32)
    rows = (np.arange(size) % 4 < 2).astype(float) * rng.uniform(0.0, 0.02)  # tractor-row stripes
    for b in range(4):
        texture = gaussian_filter(rng.normal(0, 1, (size, size)), sigma=3) * 0.03
        noise = rng.normal(0, 0.015, (size, size))
        chip[b] = base[b] + texture + noise + rows[:, None]
    return np.clip(chip, 0, 1)


def generate(n, pos_frac, seed):
    rng = np.random.default_rng(seed)
    X_seq = np.zeros((n, T_STEPS, len(FEATURES)), np.float32)
    X_img = np.zeros((n, 4, 32, 32), np.float32)
    y = np.zeros(n, np.int64)
    X_hist = np.zeros((n, len(HIST_NAMES)), np.float32)
    year = rng.integers(2021, 2026, n)
    lat = rng.uniform(*LAT_RANGE, n)
    lon = rng.uniform(*LON_RANGE, n)
    doy = rng.integers(274, 330, n)  # 1 Oct .. 25 Nov

    for i in range(n):
        d0 = doy[i]
        t_days = d0 - STEP_DAYS * np.arange(T_STEPS - 1, -1, -1)  # oldest .. now
        if rng.random() < pos_frac:
            y[i] = 1
            harvest = d0 - rng.uniform(0, 7)          # recently harvested, about to be burned
            dry_shift, dry_spell = -0.05, True        # drier straw + hot, dry, calm spell
        else:
            r = rng.random()
            if r < 0.45:
                harvest = d0 + rng.uniform(3, 25)     # crop still standing
            elif r < 0.75:
                harvest = d0 - rng.uniform(8, 30)     # harvested long ago
            else:
                harvest = d0 - rng.uniform(0, 7)      # HARD negative: just harvested, NOT burned
            dry_shift, dry_spell = 0.0, rng.random() < 0.15   # some negatives also have dry weather

        ndvi = make_ndvi(t_days, harvest, rng)
        ndmi = 0.6 * ndvi - 0.10 + dry_shift + rng.normal(0, 0.06, T_STEPS)
        temp, temp_max, rh, vpd, soil, wind, precip = make_weather(t_days, dry_spell, rng)

        seq = np.stack([ndvi, ndmi, temp, temp_max, rh, vpd, soil, wind, precip], axis=1).astype(np.float32)
        gaps = rng.random((T_STEPS, 2)) < 0.15       # cloud gaps in the optical features only
        seq[:, :2][gaps] = np.nan
        X_seq[i] = seq
        X_img[i] = make_chip(ndvi[-1], rng)
        # fire history: burned places tend to burn again, so positives get higher counts on average,
        # but with a lot of overlap (20% of negatives sit in "hot" areas) so history is helpful, not decisive.
        hot = y[i] == 1 or rng.random() < 0.20
        m = 1.0 if hot else 0.25
        c1 = rng.poisson(m * 1.0)                       # previous 2 seasons, ~1 km
        c5 = c1 + rng.poisson(m * 4.0)                  # ~5 km contains the 1 km count
        now20 = rng.poisson(m * 1.5)
        ring5 = rng.poisson(m * 1.2)
        X_hist[i] = np.log1p([c1, c5, now20, ring5])

    return dict(X_seq=X_seq, X_img=X_img, X_hist=X_hist, y=y, year=year, lat=lat, lon=lon, doy=doy,
                feature_names=np.array(FEATURES), hist_names=np.array(HIST_NAMES))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--pos_frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/synthetic_dataset.npz")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    d = generate(a.n, a.pos_frac, a.seed)
    np.savez_compressed(a.out, **d)
    print(f"saved {a.out}: seq {d['X_seq'].shape}, img {d['X_img'].shape}, "
          f"positives {d['y'].sum()} / {len(d['y'])}, features {[str(x) for x in d['feature_names']]}")
