"""
One command that produces EVERYTHING you need for the results section of a paper.

    python paper_results.py --data data/real_dataset_matched.npz --out paper_results --seeds 3 --epochs 30

It trains every model (several random starts each), tests them on the held-out last year, and writes to --out:

  TABLES  (each as .csv for Excel and .tex for LaTeX)
    table1_dataset      samples, burns and burn rate per season, and which season is train / validation / test
    table2_models       every model: PR-AUC mean +- sd over seeds, 95% bootstrap interval, ROC-AUC, and
                        precision / recall / lift when the riskiest 20% of fields are flagged
    table3_ablation     what each input adds (weather, harvest timing, fire history, deep model vs Random Forest),
                        with a 95% bootstrap interval on the difference
    table4_topk         proposed model: precision, recall and lift when flagging the riskiest 5/10/20/30/40% of fields
    table5_importance   permutation importance of every input, mean +- sd over seeds
  FIGURES (.png at 300 dpi and .pdf for LaTeX)
    fig1_pr_curves, fig2_roc_curves, fig3_model_comparison, fig4_gains_curve, fig5_confusion_top20,
    fig6_importance, fig7_training_curves
  paper_numbers.txt    every headline number in plain sentences, ready to paste into the text
  results.json         all numbers, machine-readable

How the numbers are made (say this in the paper's method section):
  * Split by year: train < validation year < test year (the last season). The test year is used once.
  * Models with several seeds: the table shows the mean and standard deviation over seeds.
  * Bootstrap intervals: 1,000 resamples of the test set (seed-0 predictions); differences between two models use the
    SAME resamples, so they are paired.
  * Precision / recall at the "riskiest 20%" use a fixed share of fields, not a probability cut-off, because the
    cut-off drifts between years.
"""
import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

import train as T

RF = "Random Forest (baseline, all inputs)"
NONE = "Hybrid CNN+LSTM (satellite + weather)"
HARV = "Hybrid + harvest timing"
FIRE = "Hybrid + fire history"
ALL = "Hybrid + harvest timing + fire history (proposed)"
SAT = "Hybrid, satellite only (no weather)"
CNN = "CNN only (satellite photo)"
LSTM = "LSTM only (satellite + weather series)"
FONLY = "Fire history only"
HONLY = "Harvest timing only"
INK, GRID = "#222222", "#d9d9d9"
COL = {RF: "#7f7f7f", NONE: "#1f77b4", HARV: "#ff7f0e", FIRE: "#2ca02c", ALL: "#d62728", SAT: "#9467bd",
       CNN: "#8c564b", LSTM: "#17becf", FONLY: "#bcbd22", HONLY: "#e377c2"}


# --------------------------------------------------------------------------- helpers
def top_share_stats(y, p, share):
    """Flag the riskiest `share` of fields. Returns precision, recall, lift."""
    n = max(1, int(math.ceil(share * len(y))))
    idx = np.argsort(-p, kind="stable")[:n]
    hit = y[idx].sum()
    prec, rec = hit / n, hit / max(y.sum(), 1)
    return float(prec), float(rec), float(prec / max(y.mean(), 1e-9))


def boot_indices(n, reps, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n, n) for _ in range(reps)]


def boot_scores(y, p, idxs, fn):
    out = []
    for ix in idxs:
        if y[ix].min() == y[ix].max():
            continue
        out.append(fn(y[ix], p[ix]))
    return np.array(out)


def ci(a):
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def tex_table(df, caption, label, path):
    cols = list(df.columns)
    esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{" + esc(caption) + "}", r"\label{" + label + "}",
             r"\begin{tabular}{l" + "r" * (len(cols) - 1) + "}", r"\hline",
             " & ".join(esc(c) for c in cols) + r" \\", r"\hline"]
    for _, r in df.iterrows():
        lines.append(" & ".join(esc(v) for v in r.values) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    open(path, "w").write("\n".join(lines) + "\n")


def save_table(df, out, name, caption):
    df.to_csv(os.path.join(out, name + ".csv"), index=False)
    tex_table(df, caption, "tab:" + name, os.path.join(out, name + ".tex"))


def savefig(fig, out, name):
    fig.savefig(os.path.join(out, name + ".png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# --------------------------------------------------------------------------- training
def run_all(parts, seeds, epochs):
    V = {k: T.static_variant(parts, f, h) for k, (f, h) in dict(none=(False, False), harv=(False, True),
                                                                fire=(True, False), all=(True, True)).items()}
    yt = parts["test"]["y"]
    runs = {}

    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=0, n_jobs=-1)
    rf.fit(T.rf_features(V["all"]["train"]), V["all"]["train"]["y"])
    runs[RF] = dict(test=[rf.predict_proba(T.rf_features(V["all"]["test"]))[:, 1]], hist=[], models=[])
    runs[RF]["val"] = [rf.predict_proba(T.rf_features(V["all"]["val"]))[:, 1]]

    plan = [(FONLY, False, False, "fire"), (HONLY, False, False, "harv"), (LSTM, False, True, "none"),
            (CNN, True, False, "none"), (NONE, True, True, "none"), (HARV, True, True, "harv"),
            (FIRE, True, True, "fire"), (ALL, True, True, "all")]
    names = parts["feature_names"]
    for name, ui, us, var in plan:
        runs[name] = dict(test=[], hist=[], models=[])
        for sd in range(seeds):
            model, hist = T.train_model(name, V[var], ui, us, epochs, use_hist=var != "none", seed=sd)
            runs[name]["test"].append(T.predict(model, V[var]["test"]))
            runs[name].setdefault("val", []).append(T.predict(model, V[var]["val"]))
            runs[name]["hist"].append(hist)
            if name == ALL:
                runs[name]["models"].append(model)
    sat = {k: dict(parts[k], seq=parts[k]["seq"][:, :, :2], hist=np.zeros((len(parts[k]["y"]), 0), np.float32))
           for k in ("train", "val", "test")}
    runs[SAT] = dict(test=[], hist=[], models=[])
    for sd in range(seeds):
        model, hist = T.train_model(SAT, sat, True, True, epochs, use_hist=False, seed=sd)
        runs[SAT]["test"].append(T.predict(model, sat["test"]))
        runs[SAT].setdefault("val", []).append(T.predict(model, sat["val"]))
        runs[SAT]["hist"].append(hist)
    return V, runs, yt


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "real_dataset_matched.npz"))
    ap.add_argument("--out", default="paper_results")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap resamples")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print("device:", T.DEVICE)
    json.dump(dict(split=T.SPLIT, cell=T.CELL, patience=T.PATIENCE, aug=getattr(T, "AUG", "basic")),
              open(os.path.join(a.out, "config.json"), "w"), indent=2)

    parts = T.load_and_split(a.data)
    V, runs, yt = run_all(parts, a.seeds, a.epochs)
    N, base = len(yt), float(yt.mean())
    idxs = boot_indices(N, a.boot)
    P0 = {m: r["test"][0] for m, r in runs.items()}
    summary = {"test_samples": N, "test_burns": int(yt.sum()), "test_burn_rate": base, "seeds": a.seeds}

    # ---- table 1: dataset
    d = np.load(a.data)
    years = sorted(int(x) for x in np.unique(d["year"]))
    test_y, val_y = years[-1], years[-2]
    rows = []
    for yr in years:
        m = d["year"] == yr
        role = "mixed (random split)" if T.SPLIT == "random" else ("test" if yr == test_y else "validation" if yr == val_y else "train")
        rows.append({"Season": yr, "Role": role, "Samples": int(m.sum()), "Burn": int(d["y"][m].sum()),
                     "No burn": int((d["y"][m] == 0).sum()), "Burn rate": f"{d['y'][m].mean():.3f}"})
    rows.append({"Season": "All", "Role": "", "Samples": len(d["y"]), "Burn": int(d["y"].sum()),
                 "No burn": int((d["y"] == 0).sum()), "Burn rate": f"{d['y'].mean():.3f}"})
    t1 = pd.DataFrame(rows)
    save_table(t1, a.out, "table1_dataset", "Dataset: samples per season and their role in training and testing.")

    # ---- table 2: models
    rows, raw = [], {}
    for m, r in runs.items():
        prs = [average_precision_score(yt, p) for p in r["test"]]
        rocs = [roc_auc_score(yt, p) for p in r["test"]]
        t20 = np.array([top_share_stats(yt, p, 0.20) for p in r["test"]])
        lo, hi = ci(boot_scores(yt, P0[m], idxs, average_precision_score))
        raw[m] = dict(pr_auc=float(np.mean(prs)), pr_auc_sd=float(np.std(prs)), pr_ci=(lo, hi), roc_auc=float(np.mean(rocs)),
                      prec20=float(t20[:, 0].mean()), rec20=float(t20[:, 1].mean()), lift20=float(t20[:, 2].mean()),
                      runs=len(prs))
        rows.append({"Model": m, "PR-AUC (mean +- sd)": f"{np.mean(prs):.3f} +- {np.std(prs):.3f}",
                     "PR-AUC 95% CI": f"[{lo:.3f}, {hi:.3f}]", "ROC-AUC": f"{np.mean(rocs):.3f}",
                     "Precision@20%": f"{t20[:, 0].mean():.3f}", "Recall@20%": f"{t20[:, 1].mean():.3f}",
                     "Lift@20%": f"{t20[:, 2].mean():.2f}", "Runs": len(prs), "_s": np.mean(prs)})
    t2 = pd.DataFrame(rows).sort_values("_s", ascending=False).drop(columns="_s")
    save_table(t2, a.out, "table2_models",
               f"Test-year results (mean of {a.seeds} runs). No-skill PR-AUC = {base:.3f}. @20% = riskiest 20% of fields flagged.")
    summary["models"] = raw

    # ---- table 3: ablation with paired bootstrap on the difference
    def diff(a_, b_):
        d_ = np.array([average_precision_score(yt[ix], P0[a_][ix]) - average_precision_score(yt[ix], P0[b_][ix])
                       for ix in idxs if yt[ix].min() != yt[ix].max()])
        return ci(d_)
    pairs = [("Weather (satellite+weather vs satellite only)", NONE, SAT),
             ("Harvest timing (hybrid + harvest vs hybrid)", HARV, NONE),
             ("Fire history (hybrid + fire history vs hybrid)", FIRE, NONE),
             ("Harvest timing on top of fire history", ALL, FIRE),
             ("Proposed vs hybrid without extras", ALL, NONE),
             ("Deep hybrid + fire history vs Random Forest", FIRE, RF)]
    rows, ab = [], {}
    for label, x, y_ in pairs:
        g = raw[x]["pr_auc"] - raw[y_]["pr_auc"]
        lo, hi = diff(x, y_)
        verdict = "clear gain" if lo > 0 else "clear loss" if hi < 0 else "not distinguishable from zero"
        ab[label] = dict(gain=g, ci=(lo, hi), verdict=verdict)
        rows.append({"Question": label, "PR-AUC A": f"{raw[x]['pr_auc']:.3f}", "PR-AUC B": f"{raw[y_]['pr_auc']:.3f}",
                     "Gain (A-B)": f"{g:+.3f}", "95% CI of difference": f"[{lo:+.3f}, {hi:+.3f}]", "Verdict": verdict})
    t3 = pd.DataFrame(rows)
    save_table(t3, a.out, "table3_ablation", "What each input adds. The interval is a paired bootstrap on the test year.")
    summary["ablation"] = ab

    # ---- table 4: top-k for proposed
    rows, tk = [], {}
    for sh in (0.05, 0.10, 0.20, 0.30, 0.40):
        s = np.array([top_share_stats(yt, p, sh) for p in runs[ALL]["test"]])
        tk[sh] = s.mean(0).tolist()
        rows.append({"Fields flagged": f"{int(sh * 100)}%", "Precision": f"{s[:, 0].mean():.3f}",
                     "Recall (burns caught)": f"{s[:, 1].mean():.3f}", "Lift over random": f"{s[:, 2].mean():.2f}"})
    t4 = pd.DataFrame(rows)
    save_table(t4, a.out, "table4_topk", "Proposed model: precision, recall and lift when flagging the riskiest fields.")
    summary["topk"] = tk

    # ---- table 5: importance (mean +- sd over seeds)
    imps = [T.permutation_importance(m, V["all"]["test"], parts["feature_names"], V["all"]["hist_names"], repeats=10, seed=i)
            .set_index("input")["pr_auc_drop"] for i, m in enumerate(runs[ALL]["models"])]
    imp = pd.concat(imps, axis=1)
    imp_df = pd.DataFrame({"mean": imp.mean(axis=1), "sd": imp.std(axis=1, ddof=0)}).sort_values("mean", ascending=False)
    t5 = pd.DataFrame({"Input": imp_df.index, "Drop in PR-AUC when scrambled (mean +- sd)":
                       [f"{m:.4f} +- {s:.4f}" for m, s in zip(imp_df["mean"], imp_df["sd"])]})
    save_table(t5, a.out, "table5_importance", "Permutation importance of each input for the proposed model.")
    summary["importance"] = {k: [float(v["mean"]), float(v["sd"])] for k, v in imp_df.iterrows()}

    # ---- figures

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    METS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    def at_thr(y, p, t):
        pr = (p >= t).astype(int)
        return dict(accuracy=accuracy_score(y, pr), precision=precision_score(y, pr, zero_division=0),
                    recall=recall_score(y, pr), f1=f1_score(y, pr), roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p))
    yv = parts["val"]["y"]
    rows, m6 = [], {}
    for m, r in runs.items():
        df_ = pd.DataFrame([at_thr(yt, pt, T.best_threshold(yv, pv)) for pv, pt in zip(r["val"], r["test"])])
        m6[m] = {k: [float(df_[k].mean()), float(df_[k].std(ddof=0))] for k in METS}
        rows.append(dict(Model=m, **{k.replace("_", "-").title() if "auc" in k else k.title():
                         f"{df_[k].mean():.3f} +- {df_[k].std(ddof=0):.3f}" for k in METS}, _s=df_["f1"].mean()))
    t6 = pd.DataFrame(rows).sort_values("_s", ascending=False).drop(columns="_s")
    save_table(t6, a.out, "table6_metrics", "Test-year accuracy, precision, recall, F1 and AUCs (mean +- sd over runs). The cut-off was chosen on the validation year (best F1).")
    summary["test_metrics_at_val_threshold"] = m6

    sets = {"Training": "train", "Validation": "val", "Test": "test"}
    per = {n: [] for n in sets}
    cms = []
    for mdl in runs[ALL]["models"]:
        P = {n: T.predict(mdl, V["all"][k]) for n, k in sets.items()}
        t = T.best_threshold(yv, P["Validation"])
        for n, k in sets.items():
            per[n].append(at_thr(V["all"][k]["y"], P[n], t))
        pr_ = (P["Test"] >= t).astype(int)
        cms.append(np.array([[((yt == 0) & (pr_ == 0)).sum(), ((yt == 0) & (pr_ == 1)).sum()],
                             [((yt == 1) & (pr_ == 0)).sum(), ((yt == 1) & (pr_ == 1)).sum()]]))
    rows = []
    for n in sets:
        df_ = pd.DataFrame(per[n])
        ys = V["all"][sets[n]]["y"]
        rows.append(dict(Split=n, Samples=len(ys), **{k.replace("_", "-").title() if "auc" in k else k.title():
                         f"{df_[k].mean():.3f}" for k in METS}))
    t7 = pd.DataFrame(rows)
    save_table(t7, a.out, "table7_train_val_test", "Proposed model on the training, validation and test data (same cut-off). A big drop from training to test means overfitting.")
    summary["train_val_test"] = {n: {k: float(np.mean([d[k] for d in per[n]])) for k in METS} for n in sets}
    cm_t = np.mean(cms, axis=0)
    fig, ax = plt.subplots(figsize=(3.9, 3.5))
    ax.imshow(cm_t, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm_t):
        ax.text(j, i, f"{v:.0f}", ha="center", va="center", color="white" if v > cm_t.max() / 2 else INK, fontsize=12)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["no burn", "burn"], yticklabels=["no burn", "burn"],
           xlabel="Predicted (validation-chosen cut-off)", ylabel="Actual")
    savefig(fig, a.out, "fig8_confusion_threshold")
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    w = 0.26
    for i, (n, c) in enumerate(zip(sets, ["#9ecae1", "#6baed6", "#d62728"])):
        ax.bar(np.arange(4) + (i - 1) * w, [summary["train_val_test"][n][k] for k in METS[:4]], w, label=n, color=c)
    ax.set(xticks=range(4), xticklabels=["Accuracy", "Precision", "Recall", "F1"], ylabel="Score", ylim=(0, 1)); style(ax)
    ax.legend(fontsize=8); savefig(fig, a.out, "fig9_train_val_test")
    pm = summary["test_metrics_at_val_threshold"][ALL]

    # final_artifacts: the saved model, its test predictions and the cut-off, for the risk map
    import torch
    torch.save(runs[ALL]["models"][0].state_dict(), os.path.join(a.out, "proposed_model.pt"))
    p0 = T.predict(runs[ALL]["models"][0], V["all"]["test"])
    pd.DataFrame(dict(lat=parts["test"]["lat"], lon=parts["test"]["lon"], y_true=yt, risk=p0)).to_csv(
        os.path.join(a.out, "test_predictions.csv"), index=False)
    thr0 = T.best_threshold(yv, T.predict(runs[ALL]["models"][0], V["all"]["val"]))
    json.dump([dict(model="Hybrid CNN+LSTM + harvest timing + fire history (proposed)", threshold=thr0)],
              open(os.path.join(a.out, "metrics.json"), "w"), indent=2)

    show = [RF, CNN, LSTM, NONE, FIRE, ALL]
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    for m in show:
        pr, rc, _ = precision_recall_curve(yt, P0[m])
        ax.plot(rc, pr, color=COL[m], lw=2.2 if m == ALL else 1.4, label=f"{m} ({raw[m]['pr_auc']:.2f})")
    ax.axhline(base, ls="--", c="grey", lw=1, label=f"no skill ({base:.2f})")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02)); style(ax)
    ax.legend(fontsize=6.5, loc="upper right"); savefig(fig, a.out, "fig1_pr_curves")

    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    for m in show:
        fpr, tpr, _ = roc_curve(yt, P0[m])
        ax.plot(fpr, tpr, color=COL[m], lw=2.2 if m == ALL else 1.4, label=f"{m} ({raw[m]['roc_auc']:.2f})")
    ax.plot([0, 1], [0, 1], "--", c="grey", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate"); style(ax)
    ax.legend(fontsize=6.5, loc="lower right"); savefig(fig, a.out, "fig2_roc_curves")

    order = sorted(raw, key=lambda m: raw[m]["pr_auc"])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.barh(order, [raw[m]["pr_auc"] for m in order], xerr=[raw[m]["pr_auc_sd"] for m in order],
            color=[COL[m] for m in order], capsize=3, height=0.65)
    ax.axvline(base, ls="--", c="grey", lw=1); ax.text(base, -0.9, f" no skill {base:.2f}", fontsize=7, color="grey")
    ax.set(xlabel=f"PR-AUC on the test year (mean +- sd, {a.seeds} runs)", xlim=(0.3, max(0.75, max(raw[m]["pr_auc"] for m in raw) + 0.05)))
    ax.tick_params(axis="y", labelsize=7); style(ax); savefig(fig, a.out, "fig3_model_comparison")

    p = P0[ALL]
    order_i = np.argsort(-p, kind="stable")
    caught = np.cumsum(yt[order_i]) / max(yt.sum(), 1)
    frac = np.arange(1, N + 1) / N
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    ax.plot(frac, caught, color=COL[ALL], lw=2.2, label="proposed model")
    ax.plot([0, 1], [0, 1], "--", c="grey", lw=1, label="random flagging")
    ax.set(xlabel="Share of fields flagged (riskiest first)", ylabel="Share of burns caught"); style(ax)
    ax.legend(fontsize=8); savefig(fig, a.out, "fig4_gains_curve")

    n20 = int(math.ceil(0.2 * N))
    pred = np.zeros(N, int); pred[order_i[:n20]] = 1
    cm = np.array([[((yt == 0) & (pred == 0)).sum(), ((yt == 0) & (pred == 1)).sum()],
                   [((yt == 1) & (pred == 0)).sum(), ((yt == 1) & (pred == 1)).sum()]])
    fig, ax = plt.subplots(figsize=(3.8, 3.5))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, int(v), ha="center", va="center", color="white" if v > cm.max() / 2 else INK, fontsize=12)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["no burn", "burn"], yticklabels=["no burn", "burn"],
           xlabel="Predicted (riskiest 20% = burn)", ylabel="Actual")
    savefig(fig, a.out, "fig5_confusion_top20")
    summary["confusion_top20"] = cm.tolist()

    sat_set, hist_set, harv_set = {"ndvi", "ndmi", "satellite photo (CNN)"}, set(V["all"]["hist_names"][:4]), set(T.HARVEST_NAMES)
    ii = imp_df.sort_values("mean")
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.barh(ii.index, ii["mean"], xerr=ii["sd"], capsize=2,
            color=["#2ca02c" if n in sat_set else "#d62728" if n in hist_set else "#ff7f0e" if n in harv_set else "#1f77b4" for n in ii.index])
    ax.axvline(0, c="grey", lw=0.8)
    ax.set(xlabel="Drop in test PR-AUC when the input is scrambled"); ax.tick_params(axis="y", labelsize=7); style(ax)
    ax.set_title("green: satellite   blue: weather   red: fire history   orange: harvest timing", fontsize=7)
    savefig(fig, a.out, "fig6_importance")

    fig, ax = plt.subplots(1, 2, figsize=(8.6, 3.5))
    for i, h in enumerate(runs[ALL]["hist"]):
        ax[0].plot(h["epoch"], h["train_loss"], lw=1.4, label=f"run {i + 1}")
        ax[1].plot(h["epoch"], h["val_pr_auc"], lw=1.4)
    ax[0].set(xlabel="Epoch", ylabel="Training loss"); ax[1].set(xlabel="Epoch", ylabel="Validation PR-AUC")
    for x in ax: style(x)
    ax[0].legend(fontsize=7); savefig(fig, a.out, "fig7_training_curves")

    # ---- plain-language numbers
    best = max((m for m in raw if m != ALL), key=lambda m: raw[m]["pr_auc"])
    q = raw[ALL]
    txt = [
        f"Test set: {N} fields, {int(yt.sum())} burns, burn rate {base:.3f}. A model with no skill has PR-AUC {base:.3f}.",
        f"Proposed model: PR-AUC {q['pr_auc']:.3f} +- {q['pr_auc_sd']:.3f} (95% CI {q['pr_ci'][0]:.3f} to {q['pr_ci'][1]:.3f}), "
        f"ROC-AUC {q['roc_auc']:.3f}; this is {q['pr_auc'] / base:.2f} times the no-skill PR-AUC.",
        f"Flagging the riskiest 20% of fields: precision {q['prec20']:.3f}, recall {q['rec20']:.3f}, lift {q['lift20']:.2f} over random.",
        f"At the validation-chosen cut-off the proposed model scores on the test year: accuracy {pm['accuracy'][0]:.3f}, precision {pm['precision'][0]:.3f}, recall {pm['recall'][0]:.3f}, F1 {pm['f1'][0]:.3f}.",
        f"Best other model: {best} with PR-AUC {raw[best]['pr_auc']:.3f}.",
        f"Random Forest baseline: PR-AUC {raw[RF]['pr_auc']:.3f}, ROC-AUC {raw[RF]['roc_auc']:.3f}.",
    ] + [f"{k}: gain {v['gain']:+.3f} (95% CI {v['ci'][0]:+.3f} to {v['ci'][1]:+.3f}), {v['verdict']}." for k, v in ab.items()] + [
        "Most relied-on inputs (drop in PR-AUC when scrambled): " + ", ".join(f"{k} {v[0]:.3f}" for k, v in list(summary["importance"].items())[:5]) + ".",
    ]
    open(os.path.join(a.out, "paper_numbers.txt"), "w").write("\n".join(txt) + "\n")
    json.dump(summary, open(os.path.join(a.out, "results.json"), "w"), indent=2, default=float)
    print("\n".join(txt))
    print(f"\nAll tables, figures and numbers saved in {a.out}/")


if __name__ == "__main__":
    main()
