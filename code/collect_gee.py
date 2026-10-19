"""
REAL data collection: NASA FIRMS fire points (labels) + Google Earth Engine (Sentinel-2, ERA5 weather).

Output: data/real_dataset.npz  (same format as gen_synthetic.py -> train.py works unchanged)

How a sample is built
---------------------
POSITIVE : a VIIRS fire detected on cropland on date D.  Observation date t0 = D - k, k random in 1..5.
           Inputs use ONLY satellite/weather data that ended before t0, so the model never sees the fire.
NEGATIVE : a random cropland point and random date in the season with NO fire within ~1 km during
           [t0-5, t0+10] days.
Inputs   : 12 steps x 5 days (60 days before t0) of 9 features  +  a 32x32 chip.
           satellite: NDVI, NDMI | heat: temp, temp_max | dryness: rh, vpd, soil | other: wind, precip
FIRE HISTORY: 4 extra inputs = counts of EARLIER fires near the field (previous 2 seasons within 1 km / 5 km,
           this season's recent fires within 5 km). Only fires dated before t0 are counted (see history_features).

Run order (do the smoke test FIRST, it takes a few minutes):
    python collect_gee.py --project YOUR_GCP_PROJECT --map_key YOUR_FIRMS_KEY --smoke
    python collect_gee.py --project YOUR_GCP_PROJECT --map_key YOUR_FIRMS_KEY --years 2021 2022 2023 2024 2025

Finished years are cached in data/shards/, so if Colab disconnects just re-run the same command.

KNOWN LIMITATION (write this in your report!): FIRMS misses many fires (that is the whole premise of
the project), so some "negative" samples are really undetected burns. Labels are therefore noisy and
reported recall is a lower-bound-style estimate, not ground truth.
"""
import argparse
import datetime as dt
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

BBOX = dict(west=74.6, south=28.8, east=76.3, north=29.9)  # Hisar + Sirsa, Haryana
T_STEPS, STEP_DAYS, N_STACK = 12, 5, 24                    # 24 five-day steps from 1 Aug cover to ~3 Dec
GRID_START = (8, 1)                                        # month, day: step 0 starts here
CHIP = 32
FEATURES = ["ndvi", "ndmi", "temp", "temp_max", "rh", "vpd", "soil", "wind", "precip"]  # same order as gen_synthetic.py
HIST_NAMES = ["hist_prev_1km", "hist_prev_5km", "hist_now_20d", "hist_ring_5d"]           # same order as gen_synthetic.py
R_NEAR, R_WIDE = 0.01, 0.05                                # ~1 km and ~5 km, in degrees


# ----------------------------------------------------------------------------- pure helpers (tested offline)
def grid_start(year):
    return dt.date(year, *GRID_START)


def last_step_index(t0, year):
    """Index of the last 5-day step that ENDS on or before t0 (so no data after t0 leaks in)."""
    days = (t0 - grid_start(year)).days
    return days // STEP_DAYS - 1


def window_indices(t0, year):
    last = last_step_index(t0, year)
    first = last - (T_STEPS - 1)
    if first < 0 or last >= N_STACK:
        return None
    return list(range(first, last + 1))


def season_bounds(year):
    return dt.date(year, 10, 1), dt.date(year, 11, 30)


def dedupe_fires(fires, cell=0.003):
    """Many VIIRS pixels hit one field on one day; keep one per ~300 m cell per date."""
    f = fires.copy()
    f["cy"], f["cx"] = (f.latitude / cell).round(), (f.longitude / cell).round()
    return f.drop_duplicates(["cy", "cx", "acq_date"]).drop(columns=["cy", "cx"]).reset_index(drop=True)


def fire_nearby(lat, lon, t0, fire_lat, fire_lon, fire_day, r_deg=0.01, before=5, after=10):
    d0 = t0.toordinal()
    m = (np.abs(fire_lat - lat) < r_deg) & (np.abs(fire_lon - lon) < r_deg) & \
        (fire_day >= d0 - before) & (fire_day <= d0 + after)
    return bool(m.any())


def history_features(lat, lon, t0, cur, prev):
    """Fire-history inputs for one sample. cur / prev = (fire_lat, fire_lon, fire_day_ordinal) arrays for the
    current season / the previous two seasons. EVERY count only uses fires dated before t0 (no leakage), and
    the recent-fire windows are chosen so they cannot be affected by how negatives were picked
    (negatives only exclude fires within ~1 km during t0-5 .. t0+10):
        hist_prev_1km : previous 2 seasons, within ~1 km
        hist_prev_5km : previous 2 seasons, within ~5 km
        hist_now_20d  : this season, within ~5 km, days t0-20 .. t0-6   (stops 6 days before t0)
        hist_ring_5d  : this season, ring 1-5 km (NOT inside 1 km), days t0-5 .. t0-1
    Returns log(1+count) for each, float32 array of 4.
    """
    pl, po, pd_ = prev
    dy, dx = np.abs(pl - lat), np.abs(po - lon)
    c1 = int(((dy < R_NEAR) & (dx < R_NEAR)).sum())
    c5 = int(((dy < R_WIDE) & (dx < R_WIDE)).sum())
    cl, co, cd = cur
    dy, dx = np.abs(cl - lat), np.abs(co - lon)
    d0 = t0.toordinal()
    wide = (dy < R_WIDE) & (dx < R_WIDE)
    now20 = int((wide & (cd >= d0 - 20) & (cd <= d0 - 6)).sum())
    inner = (dy < R_NEAR) & (dx < R_NEAR)
    ring5 = int((wide & ~inner & (cd >= d0 - 5) & (cd <= d0 - 1)).sum())
    return np.log1p(np.array([c1, c5, now20, ring5], dtype=np.float32))


def build_sequence(row, idxs):
    """row = dict/Series with columns like ndvi_7 ... ; returns (12, 5) float32 with NaN gaps."""
    seq = np.full((T_STEPS, len(FEATURES)), np.nan, np.float32)
    for t, j in enumerate(idxs):
        for f, name in enumerate(FEATURES):
            v = row.get(f"{name}_{j}", np.nan)
            seq[t, f] = np.nan if v is None else v
    return seq


def fix_chip(arr_list, size=CHIP):
    """List-of-lists (H x W) -> exactly size x size float array (crop or edge-pad)."""
    a = np.asarray(arr_list, dtype=np.float32)
    a = a[:size, :size]
    ph, pw = size - a.shape[0], size - a.shape[1]
    if ph > 0 or pw > 0:
        a = np.pad(a, ((0, max(ph, 0)), (0, max(pw, 0))), mode="edge")
    return a


# ----------------------------------------------------------------------------- FIRMS (fire labels)
def fetch_fires(year, map_key, product="VIIRS_SNPP_SP"):
    """Download VIIRS fire detections for Sep 25 - Dec 5 in 5-day windows (API limit-safe)."""
    frames, day = [], dt.date(year, 9, 25)
    end = dt.date(year, 12, 5)
    box = f"{BBOX['west']},{BBOX['south']},{BBOX['east']},{BBOX['north']}"
    while day < end:
        url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/{product}/{box}/5/{day}"
        for attempt in range(1, 6):          # FIRMS is slow/flaky at times: patient timeout, 5 tries
            try:
                r = requests.get(url, timeout=(15, 180))
                if r.status_code == 200 and r.text.startswith("latitude"):
                    frames.append(pd.read_csv(io.StringIO(r.text)))
                    break
                err = f"HTTP {r.status_code}: {r.text[:200]}"
            except requests.exceptions.RequestException as e:
                err = str(e)
            if attempt == 5:
                raise RuntimeError(f"FIRMS request failed: {err}\nURL: {url.replace(map_key, '***')}")
            print(f"  FIRMS retry {attempt}/5 in {attempt * 10}s ({err[:80]})")
            time.sleep(attempt * 10)
        day += dt.timedelta(days=5)
    df = pd.concat(frames, ignore_index=True)
    df = df[df.confidence.astype(str).str.lower() != "l"]            # drop low-confidence detections
    df["acq_date"] = pd.to_datetime(df.acq_date).dt.date
    return dedupe_fires(df).drop_duplicates(["latitude", "longitude", "acq_date"]).reset_index(drop=True)


def load_fires(year, map_key, product, cache_dir):
    """fetch_fires with a small CSV cache, so the previous-season fires are downloaded only once."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"fires_{year}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["acq_date"] = pd.to_datetime(df.acq_date).dt.date
        return df
    df = fetch_fires(year, map_key, product)
    df.to_csv(path, index=False)
    return df


def save_fire_history(fire_dir, out="data/fire_history.csv"):
    """Merge the downloaded per-season fire CSVs into ONE historical-burning dataset you can open, plot and cite."""
    files = sorted(f for f in os.listdir(fire_dir) if f.startswith("fires_"))
    df = pd.concat([pd.read_csv(os.path.join(fire_dir, f)).assign(season=int(f[6:10])) for f in files], ignore_index=True)
    keep = [c for c in ["season", "acq_date", "latitude", "longitude", "confidence", "frp", "daynight"] if c in df.columns]
    df[keep].to_csv(out, index=False)
    print(f"historical burning dataset -> {out}: {len(df)} fire detections, seasons {sorted(df.season.unique().tolist())}")
    return out


def fire_arrays(fires):
    return (fires.latitude.values, fires.longitude.values, np.array([d.toordinal() for d in fires.acq_date]))


# ----------------------------------------------------------------------------- Earth Engine
def ee_init(project):
    import ee
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)
    return ee


def build_stack(ee, year, geom):
    """One image with 24 steps x 9 features + a 'crop' band. Each step is a 5-day median composite."""
    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(geom)
    era = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR").filterBounds(geom)
    empty_opt = ee.Image.constant([0, 0]).rename(["ndvi", "ndmi"]).updateMask(ee.Image.constant(0))

    def mask_clouds(img):
        scl = img.select("SCL")
        good = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
        return img.updateMask(good)

    def indices(img):
        return img.normalizedDifference(["B8", "B4"]).rename("ndvi").addBands(
            img.normalizedDifference(["B8", "B11"]).rename("ndmi"))

    bands = []
    for j in range(N_STACK):
        a = grid_start(year) + dt.timedelta(days=STEP_DAYS * j)
        b = a + dt.timedelta(days=STEP_DAYS)
        col = s2.filterDate(str(a), str(b)).filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60)).map(mask_clouds)
        opt = ee.Image(ee.Algorithms.If(col.size().gt(0), col.map(indices).median(), empty_opt))
        w = era.filterDate(str(a), str(b))
        t_mean = w.select("temperature_2m").mean().subtract(273.15)                 # heat: mean temp, C
        t_dew = w.select("dewpoint_temperature_2m").mean().subtract(273.15)
        temp_max = w.select("temperature_2m_max").mean().subtract(273.15).rename("temp_max")  # heat: daily max, C
        magnus = "6.112 * exp(17.62 * T / (243.12 + T))"                            # saturation vapour pressure, hPa
        es = t_mean.expression(magnus, {"T": t_mean})
        e = t_dew.expression(magnus, {"T": t_dew})
        rh = e.divide(es).multiply(100).clamp(0, 100).rename("rh")                  # dryness: relative humidity, %
        vpd = es.subtract(e).divide(10).rename("vpd")                               # dryness: vapour pressure deficit, kPa
        soil = w.select("volumetric_soil_water_layer_1").mean().rename("soil")      # dryness: top-soil moisture
        temp = t_mean.rename("temp")
        wind = w.select("u_component_of_wind_10m").mean().hypot(w.select("v_component_of_wind_10m").mean()).rename("wind")
        precip = w.select("total_precipitation_sum").sum().multiply(1000).rename("precip")  # rain, mm
        bands.append(ee.Image.cat([opt, temp, temp_max, rh, vpd, soil, wind, precip])
                     .rename([f"{n}_{j}" for n in FEATURES]))
    crop = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").eq(40).rename("crop")
    return ee.Image.cat(bands + [crop]), crop


def sample_random_cropland(ee, crop, geom, n, seed):
    pts = crop.selfMask().sample(region=geom, scale=100, numPixels=n, seed=seed, geometries=True).getInfo()
    return [(f["geometry"]["coordinates"][1], f["geometry"]["coordinates"][0]) for f in pts["features"]]


def extract_series(ee, stack, samples, chunk=300):
    """samples: DataFrame[sid, lat, lon]. Returns DataFrame of per-step band means (50 m buffer) per sid."""
    out = []
    for i in range(0, len(samples), chunk):
        part = samples.iloc[i:i + chunk]
        fc = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]).buffer(50), {"sid": int(r.sid)})
            for r in part.itertuples()])
        for attempt in range(4):
            try:
                res = stack.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=10, tileScale=4).getInfo()
                break
            except Exception as e:  # transient quota / timeout errors
                if attempt == 3:
                    raise
                print(f"  retry after error: {str(e)[:100]}")
                time.sleep(10 * (attempt + 1))
        out += [f["properties"] for f in res["features"]]
        print(f"  series {min(i + chunk, len(samples))}/{len(samples)}")
    return pd.DataFrame(out)


def get_chip(ee, year, last_idx, lat, lon):
    """32x32 chip [Red, Green, Blue, NIR] from the last two complete steps (10 days) before t0."""
    a = grid_start(year) + dt.timedelta(days=STEP_DAYS * (last_idx - 1))
    b = grid_start(year) + dt.timedelta(days=STEP_DAYS * (last_idx + 1))
    pt = ee.Geometry.Point([lon, lat])
    region = pt.buffer(160).bounds()
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(pt).filterDate(str(a), str(b))
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60)))
    img = col.median().select(["B4", "B3", "B2", "B8"]).reproject("EPSG:32643", None, 10)
    for attempt in range(3):
        try:
            d = img.sampleRectangle(region=region, defaultValue=0).getInfo()["properties"]
            chip = np.stack([fix_chip(d[b_]) for b_ in ["B4", "B3", "B2", "B8"]]) / 10000.0
            return np.clip(chip, 0, 1).astype(np.float32)
        except Exception:
            time.sleep(3 * (attempt + 1))
    return None


# ----------------------------------------------------------------------------- one year
def process_year(ee, year, map_key, args, geom):
    shard = os.path.join(args.shard_dir, f"year_{year}.npz")
    if os.path.exists(shard):
        print(f"[{year}] cached -> {shard}")
        return
    rng = np.random.default_rng(year)
    print(f"[{year}] downloading FIRMS fire points ...")
    fires = load_fires(year, map_key, args.product, args.fire_dir)
    print(f"[{year}] {len(fires)} fire detections after cleaning")
    print(f"[{year}] downloading fire history of {year - 2} and {year - 1} (used as model input) ...")
    prev_fires = [fire_arrays(load_fires(y, map_key, "VIIRS_SNPP_SP", args.fire_dir)) for y in (year - 2, year - 1)]
    prev = tuple(np.concatenate([p[k] for p in prev_fires]) for k in range(3))
    cur = fire_arrays(fires)
    s_lo, s_hi = season_bounds(year)

    # ---- positives
    pos = fires.sample(frac=1.0, random_state=year).head(int(args.max_pos * 1.6))
    rows = []
    for r in pos.itertuples():
        t0 = r.acq_date - dt.timedelta(days=int(rng.integers(1, 6)))
        if s_lo <= t0 <= s_hi and window_indices(t0, year):
            rows.append(dict(lat=r.latitude, lon=r.longitude, t0=t0, y=1))

    # ---- negatives
    n_neg = int(len(rows) * args.neg_ratio)
    f_lat, f_lon = fires.latitude.values, fires.longitude.values
    f_day = np.array([d.toordinal() for d in fires.acq_date])
    stack, crop = build_stack(ee, year, geom)
    cand = sample_random_cropland(ee, crop, geom, n_neg * 2, seed=year)
    span = (s_hi - s_lo).days
    neg = 0
    for lat, lon in cand:
        if neg >= n_neg:
            break
        t0 = s_lo + dt.timedelta(days=int(rng.integers(0, span + 1)))
        if not fire_nearby(lat, lon, t0, f_lat, f_lon, f_day):
            rows.append(dict(lat=lat, lon=lon, t0=t0, y=0))
            neg += 1
    samples = pd.DataFrame(rows).reset_index(drop=True)
    samples["sid"] = samples.index
    print(f"[{year}] {int(samples.y.sum())} positives, {int((samples.y == 0).sum())} negatives before filtering")

    # ---- time series + cropland check
    series = extract_series(ee, stack, samples)
    df = samples.merge(series, on="sid", how="inner")
    df = df[df.crop.fillna(0) > 0.5].reset_index(drop=True)      # keep only cropland samples
    print(f"[{year}] {len(df)} samples on cropland")

    # ---- chips (parallel) + assemble
    def work(r):
        idxs = window_indices(r.t0, year)
        chip = get_chip(ee, year, idxs[-1], r.lat, r.lon)
        return r, idxs, chip

    seqs, imgs, hists, ys, lats, lons, doys = [], [], [], [], [], [], []
    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        for k, (r, idxs, chip) in enumerate(ex.map(work, df.itertuples()), 1):
            if chip is not None and (chip.sum(0) > 0).mean() > 0.5:   # drop fully cloudy chips
                seqs.append(build_sequence(df.loc[r.Index].to_dict(), idxs))
                hists.append(history_features(r.lat, r.lon, r.t0, cur, prev))
                imgs.append(chip); ys.append(r.y); lats.append(r.lat); lons.append(r.lon)
                doys.append(r.t0.timetuple().tm_yday)
            if k % 100 == 0:
                print(f"  chips {k}/{len(df)}")
    os.makedirs(args.shard_dir, exist_ok=True)
    np.savez_compressed(shard, X_seq=np.array(seqs), X_img=np.array(imgs), X_hist=np.array(hists), y=np.array(ys),
                        year=np.full(len(ys), year), lat=np.array(lats), lon=np.array(lons), doy=np.array(doys),
                        feature_names=np.array(FEATURES), hist_names=np.array(HIST_NAMES))
    print(f"[{year}] saved {len(ys)} samples ({int(sum(ys))} positive) -> {shard}")


def merge_shards(out, shard_dir):
    files = sorted(f for f in os.listdir(shard_dir) if f.startswith("year_"))
    parts = [np.load(os.path.join(shard_dir, f)) for f in files]
    if any("X_hist" not in p.files for p in parts):
        raise RuntimeError(f"Some files in {shard_dir} were made before fire history was added (no X_hist). "
                           f"Delete that folder and re-run so every year is rebuilt.")
    names = ("feature_names", "hist_names")
    merged = {k: np.concatenate([p[k] for p in parts]) for k in parts[0].files if k not in names}
    for k in names:
        merged[k] = parts[0][k]
    np.savez_compressed(out, **merged)
    print(f"merged {len(files)} years -> {out}: {len(merged['y'])} samples, {int(merged['y'].sum())} positive")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="your Google Cloud project id registered for Earth Engine")
    ap.add_argument("--map_key", default=os.environ.get("FIRMS_MAP_KEY"))
    ap.add_argument("--years", type=int, nargs="+", default=[2021, 2022, 2023, 2024, 2025])
    ap.add_argument("--max_pos", type=int, default=600, help="max positives per year")
    ap.add_argument("--neg_ratio", type=float, default=3.0)
    ap.add_argument("--product", default="VIIRS_SNPP_SP", help="use VIIRS_SNPP_NRT for the current year")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--out", default="data/real_dataset.npz")
    ap.add_argument("--shard_dir", default="data/shards")
    ap.add_argument("--fire_dir", default="data/fires", help="cache of downloaded FIRMS fire CSVs")
    ap.add_argument("--smoke", action="store_true", help="tiny test run: 1 year, ~20 positives")
    args = ap.parse_args()
    assert args.map_key, "give --map_key or set FIRMS_MAP_KEY (free: firms.modaps.eosdis.nasa.gov/api/)"
    if args.smoke:  # separate folder so a smoke test never touches real cached data
        args.years, args.max_pos = [args.years[-1]], 20
        args.out, args.shard_dir = "data/smoke_dataset.npz", "data/shards_smoke"
    ee = ee_init(args.project)
    geom = ee.Geometry.Rectangle([BBOX["west"], BBOX["south"], BBOX["east"], BBOX["north"]])
    for y in args.years:
        process_year(ee, y, args.map_key, args, geom)
    merge_shards(args.out, args.shard_dir)
    save_fire_history(args.fire_dir)
