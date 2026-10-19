"""
Add the fire-history inputs to data you ALREADY collected, without touching Earth Engine again.

Why this exists: shards made by an older collect_gee.py have no fire-history inputs. The history only needs
each sample's year, day-of-year and location (all saved in the shard) plus the FIRMS fire list (a few minutes
to download), so it can be added afterwards.

It also checks (and can fix) a sampling problem: if "no burn" dates are spread evenly over Oct-Nov while burns
cluster in late Oct / early Nov, the model can score well just by learning the CALENDAR (colder, later = burn).
That would not be real fire prediction. The check compares the two date distributions; --match_dates writes a
copy of the dataset where no-burn samples are thinned so both groups share the same dates.

Usage (from the folder that contains data/):
    python add_history.py --map_key YOUR_FIRMS_KEY                    # add X_hist to every shard, merge, check dates
    python add_history.py --map_key YOUR_FIRMS_KEY --match_dates      # also write data/real_dataset_matched.npz

Originals are copied to data/shards_backup/ first. Then train with:
    python train.py --data data/real_dataset.npz            (or data/real_dataset_matched.npz)
"""
import argparse
import datetime as dt
import os
import shutil

import numpy as np

import collect_gee as cg


def sample_dates(year, doy):
    return [dt.date(int(y), 1, 1) + dt.timedelta(days=int(d) - 1) for y, d in zip(year, doy)]


def add_history_to_shard(d, cur, prev):
    """d: dict of arrays from a shard. Returns X_hist (N, 4) using only fires before each sample's date."""
    dates = sample_dates(d["year"], d["doy"])
    return np.stack([cg.history_features(float(la), float(lo), t0, cur, prev)
                     for la, lo, t0 in zip(d["lat"], d["lon"], dates)]).astype(np.float32)


def week_hist(doy, edges):
    return np.histogram(doy, bins=edges)[0]


def date_report(doy, y, edges=None):
    """Share of burn vs no-burn samples per ~weekly bin, and the largest gap between the two shapes."""
    edges = edges if edges is not None else np.arange(270, 345, 7)
    pos, neg = week_hist(doy[y == 1], edges), week_hist(doy[y == 0], edges)
    ps, ns = pos / max(pos.sum(), 1), neg / max(neg.sum(), 1)
    return edges, ps, ns, float(0.5 * np.abs(ps - ns).sum())      # last = total variation distance, 0 (same) .. 1


def match_dates(d, seed=0, step=7):
    """Keep all burns; thin no-burns per year so their weekly date histogram follows the burns' (no new data made up)."""
    rng = np.random.default_rng(seed)
    keep = np.zeros(len(d["y"]), bool)
    keep[d["y"] == 1] = True
    for yr in np.unique(d["year"]):
        in_y = d["year"] == yr
        wk = (d["doy"] - 1) // step
        n_pos = ((d["y"] == 1) & in_y).sum()
        for w in np.unique(wk[in_y]):
            cell = in_y & (wk == w)
            npos_w = ((d["y"] == 1) & cell).sum()
            neg_idx = np.where(cell & (d["y"] == 0))[0]
            want = min(len(neg_idx), int(round(3 * npos_w)))           # up to 3 no-burns per burn, same weeks
            if want:
                keep[rng.choice(neg_idx, want, replace=False)] = True
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map_key", default=os.environ.get("FIRMS_MAP_KEY"))
    ap.add_argument("--shard_dir", default="data/shards")
    ap.add_argument("--fire_dir", default="data/fires")
    ap.add_argument("--out", default="data/real_dataset.npz")
    ap.add_argument("--product", default="VIIRS_SNPP_SP")
    ap.add_argument("--match_dates", action="store_true", help="also write a date-matched copy of the dataset")
    a = ap.parse_args()
    assert a.map_key, "give --map_key or set FIRMS_MAP_KEY"

    files = sorted(f for f in os.listdir(a.shard_dir) if f.startswith("year_"))
    backup = a.shard_dir.rstrip("/\\") + "_backup"
    if not os.path.exists(backup):
        shutil.copytree(a.shard_dir, backup)
        print(f"backup of the original shards -> {backup}")

    fires = {}
    def get(y):
        if y not in fires:
            print(f"fire list {y} ...")
            fires[y] = cg.fire_arrays(cg.load_fires(y, a.map_key, a.product, a.fire_dir))
        return fires[y]

    for f in files:
        path = os.path.join(a.shard_dir, f)
        d = dict(np.load(path))
        year = int(d["year"][0])
        prev_parts = [get(year - 2), get(year - 1)]
        prev = tuple(np.concatenate([p[k] for p in prev_parts]) for k in range(3))
        d["X_hist"] = add_history_to_shard(d, get(year), prev)
        d["hist_names"] = np.array(cg.HIST_NAMES)
        np.savez_compressed(path, **d)
        pos, neg = d["X_hist"][d["y"] == 1].mean(0), d["X_hist"][d["y"] == 0].mean(0)
        print(f"[{year}] history added to {len(d['y'])} samples | mean burn {pos.round(2)} vs no-burn {neg.round(2)}")

    cg.merge_shards(a.out, a.shard_dir)
    cg.save_fire_history(a.fire_dir, os.path.join(os.path.dirname(a.out) or ".", "fire_history.csv"))

    d = dict(np.load(a.out))
    edges, ps, ns, tv = date_report(d["doy"], d["y"])
    print("\nDATE CHECK (share of samples per week, day-of-year starting at the bin edge)")
    print("week_start  burn%  no-burn%")
    for e, p, n in zip(edges[:-1], ps, ns):
        print(f"{int(e):>9}  {p * 100:5.1f}  {n * 100:7.1f}")
    print(f"difference between the two date patterns: {tv:.2f}  (0 = identical, above ~0.15 = the calendar alone gives the model hints)")
    if a.match_dates:
        keep = match_dates(d)
        out = {k: (v[keep] if hasattr(v, "shape") and v.shape[:1] == keep.shape else v) for k, v in d.items()}
        m_path = os.path.join(os.path.dirname(a.out) or ".", "real_dataset_matched.npz")
        np.savez_compressed(m_path, **out)
        _, _, _, tv2 = date_report(out["doy"], out["y"])
        print(f"matched dataset -> {m_path}: {int(keep.sum())} samples ({int(out['y'].sum())} burn), date difference now {tv2:.2f}")


if __name__ == "__main__":
    main()
