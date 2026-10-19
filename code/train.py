"""
Train and evaluate the stubble-burning early-warning models.

Models compared (the comparison is what makes the project defensible):
    1. Random Forest baseline (classical ML, no deep learning; sees the same inputs incl. fire history)
    2. Fire history only   (small network on the 4 past-fire counts: how far does history alone get?)
    3. LSTM only   (NDVI/weather time series)
    4. CNN only    (satellite image chip)
    5. Hybrid CNN+LSTM, no extra inputs / + harvest timing / + fire history (each shows what that input adds)
    6. Hybrid CNN+LSTM + harvest timing + fire history  <- the proposed model
    7. Ablation: hybrid with satellite indices only

Split (final model): ALL YEARS ARE POOLED. Fields are grouped into 0.1-degree (about 11 km) location cells and the
cells are split 70% / 15% / 15% into train / validation / test, so no location is in more than one split. This tests
prediction at locations the model has never seen. (Splitting single fields at random would leak information, because
neighbouring fields share weather and look alike.) A stricter forecasting test is also available:
    set STUBBLE_SPLIT=year   -> train on 2021-2023, validate on 2024, test on 2025 (a season never seen).

Run it:   python train.py            (uses ..\\data\\real_dataset_matched.npz, writes to a new folder ..\\train_output\\run_<date_time>)
          python train.py --epochs 2  (quick test)
It never touches the saved final results in ..\\results.
"""
import argparse
import datetime
import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve,
                             precision_score, recall_score, f1_score, confusion_matrix)

SPLIT = os.environ.get("STUBBLE_SPLIT", "random")
CELL = float(os.environ.get("STUBBLE_CELL", "0.1"))
PATIENCE = int(os.environ.get("STUBBLE_PATIENCE", "8"))
HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------- data
def interpolate_gaps(X):
    """Fill cloud gaps by straight-line interpolation along time, using ONLY that sample's own values."""
    X = X.copy()
    n, t, f = X.shape
    idx = np.arange(t)
    for i in range(n):
        for j in range(f):
            col = X[i, :, j]
            bad = np.isnan(col)
            if bad.any() and not bad.all():
                col[bad] = np.interp(idx[bad], idx[~bad], col[~bad])
    return X


HARVEST_NAMES = ["harvest_drop", "harvest_days_since", "harvest_flag", "ndvi_below_peak", "ndvi_change_15d"]


def harvest_features(seq):
    """Harvest timing from the NDVI series (feature 0; cloud gaps already filled). Farmers burn straw soon after
    harvest, and harvest shows up as a sharp NDVI drop, so this is the 'when was it harvested' clue.
        harvest_drop       : biggest NDVI fall over 2 steps (10 days) in the window
        harvest_days_since : days between that fall and the observation date (60 if there was no clear fall)
        harvest_flag       : 1 if the fall is big enough (>= 0.10) to count as a harvest
        ndvi_below_peak    : how far the latest NDVI is below the window's peak, as a share of the peak
        ndvi_change_15d    : NDVI now minus NDVI 3 steps (15 days) ago
    All of it uses only steps that end before the observation date."""
    ndvi = seq[:, :, 0]
    drops = ndvi[:, :-2] - ndvi[:, 2:]                       # fall between step t and t+2
    k = drops.argmax(1)
    drop = np.clip(drops.max(1), 0, None)
    flag = (drop >= 0.10).astype(np.float32)
    days = np.where(flag > 0, (ndvi.shape[1] - 1 - (k + 2)) * 5.0, 60.0)
    peak = ndvi.max(1)
    below = (peak - ndvi[:, -1]) / (np.abs(peak) + 1e-6)
    change = ndvi[:, -1] - ndvi[:, -4]
    return np.stack([drop, days, flag, below, change], axis=1).astype(np.float32)


def load_and_split(path):
    d = np.load(path)
    names = [str(x) for x in d["feature_names"]] if "feature_names" in d.files else \
        [f"feature_{i}" for i in range(d["X_seq"].shape[2])]
    seq = interpolate_gaps(d["X_seq"].astype(np.float32))
    img, y, year = d["X_img"].astype(np.float32), d["y"].astype(np.int64), d["year"]
    hist = d["X_hist"].astype(np.float32) if "X_hist" in d.files else np.zeros((len(y), 0), np.float32)
    hist_names = [str(x) for x in d["hist_names"]] if "hist_names" in d.files else []
    years = sorted(np.unique(year))
    if len(years) < 3:
        raise ValueError(f"Need data from at least 3 different years (train / validation / test), found {len(years)}: "
                         f"{[int(x) for x in years]}. The smoke-test file has only one year; train on the full dataset.")
    test_y, val_y = years[-1], years[-2]
    tr, va, te = year < val_y, year == val_y, year == test_y
    if SPLIT == "random":
        cell = np.round(d["lat"] / CELL).astype(int) * 100000 + np.round(d["lon"] / CELL).astype(int)
        cells = np.unique(cell)
        r = np.random.default_rng(0).permutation(len(cells))
        grp = np.empty(len(cells), int)
        grp[r[:int(0.70 * len(cells))]] = 0
        grp[r[int(0.70 * len(cells)):int(0.85 * len(cells))]] = 1
        grp[r[int(0.85 * len(cells)):]] = 2
        g = grp[np.searchsorted(cells, cell)]
        tr, va, te = g == 0, g == 1, g == 2
        print("RANDOM location-grouped split of all years pooled")
    print((f"years: train<{val_y}, val={val_y}, test={test_y} | " if SPLIT != "random" else "all years pooled | ") + "sizes "
          f"{tr.sum()}/{va.sum()}/{te.sum()} | positive rate "
          f"{y[tr].mean():.3f}/{y[va].mean():.3f}/{y[te].mean():.3f}")

    # series that were missing completely: fill with the TRAIN median of each feature (no leakage)
    med = np.nanmedian(seq[tr].reshape(-1, seq.shape[2]), axis=0)
    seq = np.where(np.isnan(seq), med, seq)
    harv = harvest_features(seq)                              # from raw (un-normalised) NDVI

    # normalise with TRAIN statistics only (no leakage)
    s_mu, s_sd = seq[tr].mean((0, 1)), seq[tr].std((0, 1)) + 1e-6
    i_mu, i_sd = img[tr].mean((0, 2, 3), keepdims=True), img[tr].std((0, 2, 3), keepdims=True) + 1e-6
    if hist.shape[1]:
        hist = (hist - hist[tr].mean(0)) / (hist[tr].std(0) + 1e-6)
    harv = (harv - harv[tr].mean(0)) / (harv[tr].std(0) + 1e-6)
    seq = (seq - s_mu) / s_sd
    img = (img - i_mu) / i_sd
    parts = {}
    for name, m in [("train", tr), ("val", va), ("test", te)]:
        parts[name] = dict(seq=seq[m], img=img[m], hist=hist[m], harv=harv[m], y=y[m], lat=d["lat"][m], lon=d["lon"][m])
    parts["feature_names"] = names
    parts["hist_names"] = hist_names
    parts["harv_names"] = HARVEST_NAMES
    return parts


# ----------------------------------------------------------------------------- models
class ImageBranch(nn.Module):
    """3 conv blocks (convolution -> batchnorm -> ReLU -> max pooling), then global average pooling."""
    def __init__(self, in_ch=4, out=64):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        self.net = nn.Sequential(block(in_ch, 16), block(16, 32), block(32, out), nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, x):
        return self.net(x)


class SeqBranch(nn.Module):
    """LSTM over the 12-step feature sequence; the last hidden state summarises the trend."""
    def __init__(self, in_f=9, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(in_f, hidden, batch_first=True)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return h[-1]


class Net(nn.Module):
    """Satellite chip -> CNN; 12-step series -> LSTM; fire-history counts -> small dense layer. Joined, then a head."""
    def __init__(self, use_img=True, use_seq=True, n_feat=9, n_hist=0):
        super().__init__()
        self.use_img, self.use_seq, self.use_hist = use_img, use_seq, n_hist > 0
        self.img = ImageBranch() if use_img else None
        self.seq = SeqBranch(n_feat) if use_seq else None
        self.hist = nn.Sequential(nn.Linear(n_hist, 16), nn.ReLU()) if n_hist > 0 else None
        dim = (64 if use_img else 0) + (64 if use_seq else 0) + (16 if n_hist > 0 else 0)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(dim, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 1))  # REGULARISED

    def forward(self, img, seq, hist=None):
        feats = []
        if self.use_img:
            feats.append(self.img(img))
        if self.use_seq:
            feats.append(self.seq(seq))
        if self.use_hist:
            feats.append(self.hist(hist))
        return self.head(torch.cat(feats, 1)).squeeze(1)


# ----------------------------------------------------------------------------- training
def predict(model, part, bs=256):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(part["y"]), bs):
            img = torch.from_numpy(part["img"][i:i + bs]).to(DEVICE)
            seq = torch.from_numpy(part["seq"][i:i + bs]).to(DEVICE)
            hist = torch.from_numpy(part["hist"][i:i + bs]).to(DEVICE)
            out.append(torch.sigmoid(model(img, seq, hist)).cpu().numpy())
    return np.concatenate(out)


def augment(img):
    """Random flips / 90-degree rotations: a field looks the same from any direction."""
    if np.random.rand() < 0.5:
        img = torch.flip(img, [3])
    if np.random.rand() < 0.5:
        img = torch.flip(img, [2])
    return torch.rot90(img, np.random.randint(4), [2, 3])


def train_model(name, parts, use_img, use_seq, epochs, use_hist=False, bs=64, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_hist = parts["train"]["hist"].shape[1] if use_hist else 0
    model = Net(use_img, use_seq, parts["train"]["seq"].shape[2], n_hist).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    ytr = parts["train"]["y"]
    pos_weight = torch.tensor((ytr == 0).sum() / max((ytr == 1).sum(), 1), dtype=torch.float32, device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # handles class imbalance

    tr = parts["train"]
    img_t, seq_t, hist_t = torch.from_numpy(tr["img"]), torch.from_numpy(tr["seq"]), torch.from_numpy(tr["hist"])
    y_t = torch.from_numpy(tr["y"].astype(np.float32))
    best, best_state, wait, hist = -1, None, 0, []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(y_t))
        total = 0.0
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            img = augment(img_t[b].to(DEVICE))
            sq = seq_t[b].to(DEVICE)
            sq = sq + 0.05 * torch.randn_like(sq)
            loss = loss_fn(model(img, sq, hist_t[b].to(DEVICE)), y_t[b].to(DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        val_prob = predict(model, parts["val"])
        val_ap = average_precision_score(parts["val"]["y"], val_prob)
        hist.append(dict(epoch=ep + 1, train_loss=total / len(y_t), val_pr_auc=val_ap))
        print(f"[{name}] epoch {ep + 1:02d}  loss {total / len(y_t):.4f}  val PR-AUC {val_ap:.4f}")
        if val_ap > best:
            best, wait = val_ap, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:  # early stopping: stop when validation stops improving (overfitting guard)
                print(f"[{name}] early stop")
                break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(hist)


def rf_features(p):
    """Hand-made features for the classical baseline: last/mean/min/max/trend of each series + image band stats
    + the fire-history counts (so the baseline gets the same information as the proposed model)."""
    s, im = p["seq"], p["img"]
    return np.concatenate([s[:, -1], s.mean(1), s.min(1), s.max(1), s[:, -1] - s[:, 0],
                           im.mean((2, 3)), im.std((2, 3)), p["hist"]], axis=1)


def permutation_importance(model, part, names, hist_names=(), repeats=5, seed=0):
    """How much does test PR-AUC drop when ONE input is scrambled across samples?
    A bigger drop means the model relies on that input more. Also scores the satellite photo and fire history."""
    rng = np.random.default_rng(seed)
    base = average_precision_score(part["y"], predict(model, part))
    rows = []
    for j, n in enumerate(names):
        drops = []
        for _ in range(repeats):
            seq = part["seq"].copy()
            seq[:, :, j] = seq[rng.permutation(len(seq)), :, j]
            drops.append(base - average_precision_score(part["y"], predict(model, dict(part, seq=seq))))
        rows.append(dict(input=n, pr_auc_drop=float(np.mean(drops))))
    for j, n in enumerate(hist_names):
        drops = []
        for _ in range(repeats):
            h = part["hist"].copy()
            h[:, j] = h[rng.permutation(len(h)), j]
            drops.append(base - average_precision_score(part["y"], predict(model, dict(part, hist=h))))
        rows.append(dict(input=n, pr_auc_drop=float(np.mean(drops))))
    drops = []
    for _ in range(repeats):
        img = part["img"][rng.permutation(len(part["img"]))]
        drops.append(base - average_precision_score(part["y"], predict(model, dict(part, img=img))))
    rows.append(dict(input="satellite photo (CNN)", pr_auc_drop=float(np.mean(drops))))
    return pd.DataFrame(rows).sort_values("pr_auc_drop", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------- evaluation
def static_variant(parts, use_fire, use_harv):
    """Copy of parts whose 'hist' slot (the static-input slot the network reads) holds the chosen groups."""
    out = {}
    for k in ("train", "val", "test"):
        cols = ([parts[k]["hist"]] if use_fire else []) + ([parts[k]["harv"]] if use_harv else [])
        out[k] = dict(parts[k], hist=np.concatenate(cols, 1) if cols else np.zeros((len(parts[k]["y"]), 0), np.float32))
    out["feature_names"] = parts["feature_names"]
    out["hist_names"] = (parts["hist_names"] if use_fire else []) + (parts["harv_names"] if use_harv else [])
    out["harv_names"] = parts["harv_names"]
    return out


def best_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return float(thr[np.argmax(f1[:-1])])


def threshold_for_recall(y, p, target):
    """Highest threshold whose VALIDATION recall is still at least `target` (an early-warning style cut-off)."""
    prec, rec, thr = precision_recall_curve(y, p)
    ok = np.where(rec[:-1] >= target)[0]
    return float(thr[ok[-1]]) if len(ok) else float(thr[0])


def aggregate(rs):
    """Average the per-seed results of one model; keep the spread of the two ranking scores."""
    out = dict(rs[0])
    for k in ("threshold", "pr_auc", "roc_auc", "precision", "recall", "f1"):
        out[k] = float(np.mean([r[k] for r in rs]))
    out["pr_auc_std"] = float(np.std([r["pr_auc"] for r in rs]))
    out["roc_auc_std"] = float(np.std([r["roc_auc"] for r in rs]))
    out["runs"] = len(rs)
    return out


def evaluate(name, y_val, p_val, y_te, p_te, target_recall=None):
    # threshold picked on validation, applied to test
    thr = threshold_for_recall(y_val, p_val, target_recall) if target_recall else best_threshold(y_val, p_val)
    pred = (p_te >= thr).astype(int)
    return dict(model=name, threshold=round(thr, 3),
                pr_auc=average_precision_score(y_te, p_te), roc_auc=roc_auc_score(y_te, p_te),
                precision=precision_score(y_te, pred, zero_division=0),
                recall=recall_score(y_te, pred), f1=f1_score(y_te, pred),
                tn_fp_fn_tp=confusion_matrix(y_te, pred).ravel().tolist())


def split_table(model, P):
    """Accuracy / precision / recall / F1 / PR-AUC / ROC-AUC of the proposed model on train, validation and test.
    The cut-off is chosen on the validation set only and used unchanged for all three."""
    probs = {k: predict(model, P[k]) for k in ("train", "val", "test")}
    thr = best_threshold(P["val"]["y"], probs["val"])
    rows = []
    for k, label in (("train", "Training"), ("val", "Validation"), ("test", "Test")):
        y, p = P[k]["y"], probs[k]
        pred = (p >= thr).astype(int)
        rows.append(dict(set=label, fields=len(y), burns=int(y.sum()), accuracy=float((pred == y).mean()),
                         precision=float(precision_score(y, pred, zero_division=0)), recall=float(recall_score(y, pred)),
                         f1=float(f1_score(y, pred)), pr_auc=float(average_precision_score(y, p)),
                         roc_auc=float(roc_auc_score(y, p))))
    return pd.DataFrame(rows), thr


def write_summary(a, results, proposed, yt, started, split_df=None, split_thr=None):
    """One short, readable file per run (SUMMARY.txt) + one line per run in train_output/runs_log.csv."""
    r = next(x for x in results if x["model"] == proposed)
    tn, fp, fn, tp = r["tn_fp_fn_tp"]
    acc = (tn + tp) / (tn + fp + fn + tp)
    lines = [
        f"Run started : {started:%Y-%m-%d %H:%M:%S}   device: {DEVICE}   epochs: {a.epochs}   seeds: {a.seeds}",
        f"Split       : {SPLIT} (cell {CELL} deg)   Test fields: {len(yt)}   burns: {int(yt.sum())} (no-skill PR-AUC {yt.mean():.3f})",
        f"PROPOSED MODEL ON THE TEST SET:",
        f"  PR-AUC {r['pr_auc']:.3f}   ROC-AUC {r['roc_auc']:.3f}   Accuracy {acc:.3f}",
        f"  Precision {r['precision']:.3f}   Recall {r['recall']:.3f}   F1 {r['f1']:.3f}   (cut-off {r['threshold']})",
        "",
    ]
    if split_df is not None:
        lines += [f"PROPOSED MODEL: TRAINING vs VALIDATION vs TEST (same cut-off {split_thr:.3f}, chosen on validation):",
                  split_df.round(3).to_string(index=False),
                  "  (a big gap between Training and Test = overfitting; a small gap = the model generalises)", ""]
    lines += ["ALL MODELS (test set, best first):"]
    for x in sorted(results, key=lambda z: -z["pr_auc"]):
        lines.append(f"  {x['pr_auc']:.3f}  PR-AUC | ROC {x['roc_auc']:.3f} | P {x['precision']:.3f} R {x['recall']:.3f} | {x['model']}")
    lines += ["", "Files here: table_train_val_test.csv (train/val/test scores), comparison.csv (table of all models), pr_curves.png, confusion_matrix.png, training_curves.png,",
              "            risk_map.png, feature_importance.png, test_predictions.csv, hybrid_model.pt, metrics.json"]
    text = "\n".join(lines)
    open(os.path.join(a.out, "SUMMARY.txt"), "w", encoding="utf-8").write(text + "\n")
    log = os.path.join(HERE, "..", "train_output", "runs_log.csv")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    new = not os.path.exists(log)
    with open(log, "a", encoding="utf-8") as f:
        if new:
            f.write("started,device,epochs,seeds,pr_auc,roc_auc,accuracy,precision,recall,f1,folder\n")
        f.write(f"{started:%Y-%m-%d %H:%M:%S},{DEVICE},{a.epochs},{a.seeds},{r['pr_auc']:.3f},{r['roc_auc']:.3f},"
                f"{acc:.3f},{r['precision']:.3f},{r['recall']:.3f},{r['f1']:.3f},{os.path.basename(os.path.normpath(a.out))}\n")
    print("\n" + "=" * 70 + "\n" + text + "\n" + "=" * 70)
    print(f"Everything for this run is in: {os.path.abspath(a.out)}")
    print(f"Open SUMMARY.txt there first. All runs side by side: {os.path.abspath(log)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data", "real_dataset_matched.npz"),
                    help="dataset used for the final model (default: the one in ..\\data)")
    ap.add_argument("--out", default=None,
                    help="where tables, figures and the model are written "
                         "(default: a NEW folder ..\\train_output\\run_<date_time> for every run)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=1,
                    help="train every deep model this many times with different random starts and report mean +- spread "
                         "(use 3 or more for the final report: single runs are noisy)")
    ap.add_argument("--target_recall", type=float, default=None,
                    help="pick each model's cut-off as the strictest one that still catches this share of burns on the "
                         "validation set (e.g. 0.6). Default: the cut-off with the best F1.")
    a = ap.parse_args()
    started = datetime.datetime.now()
    if a.out is None:
        a.out = os.path.join(HERE, "..", "train_output", "run_" + started.strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(a.out, exist_ok=True)
    print("device:", DEVICE)
    parts = load_and_split(a.data)
    yv, yt = parts["val"]["y"], parts["test"]["y"]

    results, probs, hists = [], {}, {}
    has_hist = parts["train"]["hist"].shape[1] > 0
    V = {k: static_variant(parts, f, h) for k, (f, h) in dict(none=(False, False), harv=(False, True),
                                                              fire=(has_hist, False), all=(has_hist, True)).items()}
    PROPOSED = "Hybrid CNN+LSTM + harvest timing + fire history (proposed)" if has_hist \
        else "Hybrid CNN+LSTM + harvest timing (proposed)"

    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=0, n_jobs=-1)
    rf.fit(rf_features(V["all"]["train"]), V["all"]["train"]["y"])
    pv, pt = (rf.predict_proba(rf_features(V["all"][k]))[:, 1] for k in ("val", "test"))
    results.append(aggregate([evaluate("Random Forest (baseline)", yv, pv, yt, pt, a.target_recall)]))
    probs["Random Forest (baseline)"] = pt

    # (name, use image, use series, which extra inputs)
    plan = []
    if has_hist:
        plan.append(("Fire history only", False, False, "fire"))
    plan += [("Harvest timing only", False, False, "harv"), ("LSTM only", False, True, "none"),
             ("CNN only", True, False, "none"), ("Hybrid CNN+LSTM, no extra inputs", True, True, "none"),
             ("Hybrid + harvest timing", True, True, "harv")]
    if has_hist:
        plan.append(("Hybrid + fire history", True, True, "fire"))
    plan.append((PROPOSED, True, True, "all"))

    hybrid = None
    for name, ui, us, var in plan:
        P = V[var]
        runs = []
        for sd in range(a.seeds):
            model, hist = train_model(name, P, ui, us, a.epochs, use_hist=var != "none", seed=sd)
            pv, pt = predict(model, P["val"]), predict(model, P["test"])
            runs.append(evaluate(name, yv, pv, yt, pt, a.target_recall))
            if sd == 0:
                pt0, hist0, model0 = pt, hist, model          # plots, map and saved model use the first run
        results.append(aggregate(runs))
        probs[name], hists[name] = pt0, hist0
        if name == PROPOSED:
            hybrid = model0
            torch.save(model0.state_dict(), os.path.join(a.out, "hybrid_model.pt"))
            pd.DataFrame(dict(lat=parts["test"]["lat"], lon=parts["test"]["lon"], y_true=yt, risk=pt0)
                         ).to_csv(os.path.join(a.out, "test_predictions.csv"), index=False)

    # ablation: does weather help? Same hybrid model, fed ONLY the two satellite indices (NDVI, NDMI)
    names = parts["feature_names"]
    if names[:2] == ["ndvi", "ndmi"] and len(names) > 2:
        sat = {k: dict(parts[k], seq=parts[k]["seq"][:, :, :2]) for k in ("train", "val", "test")}
        name = "Hybrid, satellite only (ablation)"
        runs = []
        for sd in range(a.seeds):
            model_s, hist_s = train_model(name, sat, True, True, a.epochs, seed=sd)
            pv, pt = predict(model_s, sat["val"]), predict(model_s, sat["test"])
            runs.append(evaluate(name, yv, pv, yt, pt, a.target_recall))
            if sd == 0:
                pt0, hist0 = pt, hist_s
        results.append(aggregate(runs))
        probs[name], hists[name] = pt0, hist0

    imp = permutation_importance(hybrid, V["all"]["test"], names, V["all"]["hist_names"])
    imp.to_csv(os.path.join(a.out, "feature_importance.csv"), index=False)
    print("\nWhat the proposed model relies on (drop in test PR-AUC when the input is scrambled):")
    print(imp.round(4).to_string(index=False))

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(a.out, "comparison.csv"), index=False)
    json.dump(results, open(os.path.join(a.out, "metrics.json"), "w"), indent=2)
    print("\n", df.drop(columns=["tn_fp_fn_tp"]).round(3).to_string(index=False))
    print(f"\n(test positive rate = {yt.mean():.3f}; a useless model gets PR-AUC ~ that value)")

    # ---- plots
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for n, h in hists.items():
        ax[0].plot(h["epoch"], h["train_loss"], label=n)
        ax[1].plot(h["epoch"], h["val_pr_auc"], label=n)
    ax[0].set(title="Training loss", xlabel="epoch"); ax[1].set(title="Validation PR-AUC", xlabel="epoch")
    ax[0].legend(); fig.tight_layout(); fig.savefig(os.path.join(a.out, "training_curves.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for n, p in probs.items():
        pr, rc, _ = precision_recall_curve(yt, p)
        ax.plot(rc, pr, label=f"{n} (AP={average_precision_score(yt, p):.2f})")
    ax.axhline(yt.mean(), ls="--", c="grey", label="no-skill")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall on the test set"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "pr_curves.png"), dpi=130); plt.close(fig)

    h = next(r for r in results if r["model"] == PROPOSED)
    cm = np.array(h["tn_fp_fn_tp"]).reshape(2, 2)
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, v, ha="center", va="center", color="black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["no burn", "burn"], yticklabels=["no burn", "burn"],
           xlabel="Predicted", ylabel="Actual", title="Proposed model: confusion matrix")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "confusion_matrix.png"), dpi=130); plt.close(fig)

    tp = parts["test"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    ax[0].scatter(tp["lon"], tp["lat"], c=tp["y"], cmap="coolwarm", s=8); ax[0].set_title("Actual burn (red)")
    sc = ax[1].scatter(tp["lon"], tp["lat"], c=probs[PROPOSED], cmap="YlOrRd", s=8, vmin=0, vmax=1)
    ax[1].set_title("Predicted burn risk"); fig.colorbar(sc, ax=ax[1], label="risk")
    for x in ax: x.set(xlabel="longitude")
    ax[0].set_ylabel("latitude"); fig.tight_layout(); fig.savefig(os.path.join(a.out, "risk_map.png"), dpi=130); plt.close(fig)
    sat_names = {"ndvi", "ndmi", "satellite photo (CNN)"}
    hist_set = set(parts["hist_names"])
    harv_set = set(parts["harv_names"])
    imp_s = imp.sort_values("pr_auc_drop")
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.barh(imp_s["input"], imp_s["pr_auc_drop"],
            color=["tab:green" if n in sat_names else "tab:red" if n in hist_set else "tab:orange" if n in harv_set else "tab:blue" for n in imp_s["input"]])
    ax.axvline(0, c="grey", lw=0.8)
    ax.set(xlabel="drop in PR-AUC when this input is scrambled",
           title="What the model relies on\n(green satellite, blue weather, red fire history, orange harvest timing)")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "feature_importance.png"), dpi=130); plt.close(fig)
    split_df, split_thr = split_table(hybrid, V["all"])
    split_df.to_csv(os.path.join(a.out, "table_train_val_test.csv"), index=False)
    write_summary(a, results, PROPOSED, yt, started, split_df, split_thr)


if __name__ == "__main__":
    main()
