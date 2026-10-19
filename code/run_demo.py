"""
DEMO: load the saved final model and produce every result for the prototype. No training, no internet needed.

    python run_demo.py --data ..\\data\\real_dataset_matched.npz --model ..\\results\\proposed_model.pt

It writes a folder (default: demo_output) with
    demo_report.html      <- ONE page with every number, table and graph. Double-click it to show the prototype.
    risk_map.html         <- interactive map of the fields (needs internet only for the map background)
    table_*.csv           <- the tables
    fig_*.png             <- the graphs
    predictions.csv       <- risk score for every test field
It must sit next to train.py (the folder 'code' of the final project).
"""
import argparse
import base64
import io
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T  # noqa: E402

RED, BLUE, GREY = "#d62728", "#1f77b4", "#8a8a8a"


def at_threshold(y, p, t):
    pred = (p >= t).astype(int)
    return dict(Accuracy=accuracy_score(y, pred), Precision=precision_score(y, pred, zero_division=0),
                Recall=recall_score(y, pred), F1=f1_score(y, pred),
                **{"ROC-AUC": roc_auc_score(y, p), "PR-AUC": average_precision_score(y, p)})


def top_share(y, p, share):
    n = max(1, int(np.ceil(share * len(y))))
    idx = np.argsort(-p, kind="stable")[:n]
    hit = float(y[idx].sum())
    prec, rec = hit / n, hit / max(float(y.sum()), 1.0)
    return prec, rec, prec / max(float(y.mean()), 1e-9)


def style(ax):
    ax.grid(True, color="#dddddd", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, out, name, store):
    path = os.path.join(out, name + ".png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    store[name] = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "real_dataset_matched.npz"),
                    help="path to real_dataset_matched.npz")
    ap.add_argument("--model", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "proposed_model.pt"),
                    help="path to proposed_model.pt")
    ap.add_argument("--out", default="demo_output")
    ap.add_argument("--top_share", type=float, default=0.2, help="share of fields flagged on the map (0.2 = riskiest 20%%)")
    ap.add_argument("--split", choices=["year", "random"], default=None, help="only if config.json is missing")
    ap.add_argument("--cell", type=float, default=None, help="only if config.json is missing")
    a = ap.parse_args()
    for p, what in ((a.data, "dataset"), (a.model, "model")):
        if not os.path.exists(p):
            sys.exit(f"ERROR: the {what} file was not found: {p}\nCheck the path and try again.")
    os.makedirs(a.out, exist_ok=True)
    print("device:", T.DEVICE)
    cfg = os.path.join(os.path.dirname(os.path.abspath(a.model)), "config.json")
    if os.path.exists(cfg):                       # the training run wrote how the data was split
        c = json.load(open(cfg))
        T.SPLIT, T.CELL = c.get("split", "random"), float(c.get("cell", 0.1))
    if a.split:
        T.SPLIT = a.split
    if a.cell:
        T.CELL = a.cell
    pooled = T.SPLIT == "random"
    print("data split:", "all years pooled, split by location" if pooled else "by year (train 2021-2023, val 2024, test 2025)")

    print("1/6 loading the data (the same preparation that was used for training) ...")
    parts = T.load_and_split(a.data)
    V = T.static_variant(parts, True, True)
    n_feat, n_hist = V["train"]["seq"].shape[2], V["train"]["hist"].shape[1]

    print("2/6 loading the trained model ...")
    model = T.Net(True, True, n_feat, n_hist).to(T.DEVICE)
    try:
        state = torch.load(a.model, map_location=T.DEVICE, weights_only=True)
    except TypeError:
        state = torch.load(a.model, map_location=T.DEVICE)
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        sys.exit("ERROR: this model file does not match train.py in this folder.\n"
                 "Use the train.py from the SAME final_project/code folder that the model came from.\n" + str(e)[:300])
    model.eval()

    print("3/6 predicting on training, validation and test data ...")
    P = {n: T.predict(model, V[k]) for n, k in (("Training", "train"), ("Validation", "val"), ("Test", "test"))}
    Y = {n: V[k]["y"] for n, k in (("Training", "train"), ("Validation", "val"), ("Test", "test"))}
    thr = T.best_threshold(Y["Validation"], P["Validation"])
    yt, pt = Y["Test"], P["Test"]
    base = float(yt.mean())

    print("4/6 building the tables ...")
    rows = [dict(Split=n, Samples=len(Y[n]), **{k: round(v, 3) for k, v in at_threshold(Y[n], P[n], thr).items()}) for n in P]
    t_metrics = pd.DataFrame(rows)
    t_metrics.to_csv(os.path.join(a.out, "table_metrics.csv"), index=False)
    rows = []
    for sh in (0.05, 0.10, 0.20, 0.30, 0.40):
        pr, rc, lf = top_share(yt, pt, sh)
        rows.append({"Fields flagged (riskiest first)": f"{int(sh * 100)}%", "Precision": round(pr, 3),
                     "Burns caught (recall)": round(rc, 3), "Lift over random": round(lf, 2)})
    t_topk = pd.DataFrame(rows)
    t_topk.to_csv(os.path.join(a.out, "table_topk.csv"), index=False)

    pred = (pt >= thr).astype(int)
    cm = np.array([[((yt == 0) & (pred == 0)).sum(), ((yt == 0) & (pred == 1)).sum()],
                   [((yt == 1) & (pred == 0)).sum(), ((yt == 1) & (pred == 1)).sum()]])
    imp = T.permutation_importance(model, V["test"], V["feature_names"], V["hist_names"], repeats=5)
    imp = imp.rename(columns={"input": "Input", "pr_auc_drop": "Drop in PR-AUC when scrambled"})
    imp["Drop in PR-AUC when scrambled"] = imp["Drop in PR-AUC when scrambled"].round(4)
    imp.to_csv(os.path.join(a.out, "table_importance.csv"), index=False)
    pd.DataFrame(dict(lat=V["test"]["lat"], lon=V["test"]["lon"], actually_burned=yt, risk=pt.round(4),
                      flagged=pred)).to_csv(os.path.join(a.out, "predictions.csv"), index=False)
    top10 = pd.DataFrame(dict(Latitude=V["test"]["lat"].round(4), Longitude=V["test"]["lon"].round(4),
                              Risk=(pt * 100).round(0).astype(int).astype(str) + "%",
                              Actually_burned=np.where(yt == 1, "yes", "no"))).iloc[np.argsort(-pt)[:10]]

    print("5/6 drawing the graphs ...")
    figs = {}
    pr, rc, _ = precision_recall_curve(yt, pt)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.plot(rc, pr, color=RED, lw=2.2, label=f"model (PR-AUC {average_precision_score(yt, pt):.2f})")
    ax.axhline(base, ls="--", c=GREY, label=f"no skill ({base:.2f})")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02), title="Precision-recall curve (test season)")
    style(ax); ax.legend(); save(fig, a.out, "fig_precision_recall", figs)

    fpr, tpr, _ = roc_curve(yt, pt)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.plot(fpr, tpr, color=BLUE, lw=2.2, label=f"model (ROC-AUC {roc_auc_score(yt, pt):.2f})")
    ax.plot([0, 1], [0, 1], "--", c=GREY, label="random")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (test season)")
    style(ax); ax.legend(loc="lower right"); save(fig, a.out, "fig_roc", figs)

    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, int(v), ha="center", va="center", fontsize=14, color="white" if v > cm.max() / 2 else "#222")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["no burn", "burn"], yticklabels=["no burn", "burn"],
           xlabel="Predicted", ylabel="Actual", title="Confusion matrix (test season)")
    save(fig, a.out, "fig_confusion_matrix", figs)

    order = np.argsort(-pt, kind="stable")
    caught = np.cumsum(yt[order]) / max(yt.sum(), 1)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.plot(np.arange(1, len(yt) + 1) / len(yt), caught, color=RED, lw=2.2, label="model")
    ax.plot([0, 1], [0, 1], "--", c=GREY, label="random flagging")
    ax.set(xlabel="Share of fields flagged (riskiest first)", ylabel="Share of burns caught", title="How many burns are caught")
    style(ax); ax.legend(); save(fig, a.out, "fig_gains", figs)

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.hist(pt[yt == 0], bins=25, alpha=0.7, color="#2e9e4f", label="did not burn")
    ax.hist(pt[yt == 1], bins=25, alpha=0.7, color=RED, label="burned")
    ax.axvline(thr, color="#222", ls="--", lw=1.2, label=f"cut-off {thr:.2f}")
    ax.set(xlabel="Predicted burn risk", ylabel="Number of fields", title="Risk scores: burned vs not burned")
    style(ax); ax.legend(); save(fig, a.out, "fig_risk_histogram", figs)

    ii = imp.sort_values("Drop in PR-AUC when scrambled").tail(12)
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.barh(ii["Input"], ii["Drop in PR-AUC when scrambled"], color=BLUE)
    ax.set(xlabel="Drop in PR-AUC when the input is scrambled", title="What the model relies on")
    ax.tick_params(axis="y", labelsize=8); style(ax); save(fig, a.out, "fig_importance", figs)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for i, (n, c) in enumerate(zip(P, ["#9ecae1", "#6baed6", RED])):
        r = t_metrics[t_metrics.Split == n].iloc[0]
        ax.bar(np.arange(4) + (i - 1) * 0.26, [r["Accuracy"], r["Precision"], r["Recall"], r["F1"]], 0.26, label=n, color=c)
    ax.set(xticks=range(4), xticklabels=["Accuracy", "Precision", "Recall", "F1"], ylim=(0, 1), ylabel="Score",
           title="Training vs validation vs test")
    style(ax); ax.legend(fontsize=8); save(fig, a.out, "fig_train_val_test", figs)

    print("6/6 building the map and the report page ...")
    map_note = "risk_map.html was created (open it in a browser)."
    try:
        import make_map
        df = pd.read_csv(os.path.join(a.out, "predictions.csv")).rename(columns={"actually_burned": "y_true"})
        df["risk"] = df["risk"].astype(float)
        df[["lat", "lon", "y_true", "risk"]].to_csv(os.path.join(a.out, "_map_input.csv"), index=False)
        info = make_map.build_map(os.path.join(a.out, "_map_input.csv"), "", os.path.join(a.out, "risk_map.html"),
                                  top_share=a.top_share)
        os.remove(os.path.join(a.out, "_map_input.csv"))
        map_note = (f"risk_map.html: {info['fields']} fields, riskiest {int(a.top_share * 100)}% flagged "
                    f"({info['flagged']} fields), of which {info['hit']} really burned.")
    except Exception as e:  # the map needs the folium package; everything else still works
        map_note = f"Map not created ({type(e).__name__}). Install it with: pip install folium"

    m = t_metrics[t_metrics.Split == "Test"].iloc[0]
    pr20, rc20, lf20 = top_share(yt, pt, 0.20)
    where = "held-out locations never seen in training" if pooled else "the unseen test season"
    summary = (f"On {where} ({len(yt)} fields, {int(yt.sum())} burned = {base:.1%}), the model reaches "
               f"accuracy {m['Accuracy']:.3f}, precision {m['Precision']:.3f}, recall {m['Recall']:.3f}, F1 {m['F1']:.3f}, "
               f"ROC-AUC {m['ROC-AUC']:.3f} and PR-AUC {m['PR-AUC']:.3f} (a model with no skill scores a PR-AUC of {base:.3f}). "
               f"Flagging the riskiest 20% of fields catches {rc20:.0%} of all burns with {pr20:.0%} precision "
               f"({lf20:.2f}x better than random).")

    def img(name, w=360):
        return f'<img src="data:image/png;base64,{figs[name]}" style="width:{w}px;max-width:100%">'

    km = int(round(T.CELL * 111))
    if pooled:
        setup = (f"Data from all five seasons (2021-2025) is pooled. Fields are grouped into {km} km location cells: "
                 f"70% of the cells train the model, 15% validate it and 15% test it, so every test field is at a place "
                 f"the model never saw during training.")
        limits = ("Fire labels are satellite detections and are noisy. Fire history looks more useful here than it will in real "
                  "use because of how the no-burn samples were drawn. Weather is an 11 km grid. Only two districts. Because all "
                  "seasons are pooled, the test shows how well the model predicts at unseen locations in seasons it has already "
                  "learned from. It does not prove the model will forecast a completely new season.")
        split_note = "Training scores are close to test scores because all seasons share the same patterns."
    else:
        setup = "Trained on 2021-2023, validated on 2024, tested on the unseen 2025 season."
        limits = ("Fire labels are satellite detections and are noisy. Fire history looks more useful here than it will in real "
                  "use because of how the no-burn samples were drawn. Weather is an 11 km grid. Only two districts and three "
                  "training seasons. The test season is genuinely unseen, so these numbers are lower than a random split gives.")
        split_note = "Training scores are higher than test scores because the test season is new to the model."
    css = ("body{font-family:Segoe UI,Arial,sans-serif;max-width:1150px;margin:24px auto;padding:0 16px;color:#222}"
           "h1{margin-bottom:4px}h2{border-bottom:2px solid #d62728;padding-bottom:4px;margin-top:34px}"
           ".sub{color:#666}.box{background:#f6f6f6;border-left:4px solid #d62728;padding:10px 14px;margin:14px 0}"
           "table{border-collapse:collapse;margin:8px 0}th,td{border:1px solid #ccc;padding:5px 10px;text-align:right}"
           "th{background:#eee}td:first-child,th:first-child{text-align:left}.row{display:flex;flex-wrap:wrap;gap:14px}"
           ".note{font-size:13px;color:#555}")
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Stubble-burning early-warning prototype</title>
<style>{css}</style></head><body>
<h1>Stubble-burning early-warning prototype</h1>
<div class="sub">Hybrid CNN+LSTM using Sentinel-2 satellite images, ERA5 weather, past-fire history and harvest timing.
Districts: Hisar and Sirsa, Haryana. {setup}</div>
<div class="box"><b>Summary.</b> {summary}</div>
<h2>1. Performance (cut-off {thr:.2f}, chosen on the validation season)</h2>{t_metrics.to_html(index=False)}
<p class="note">Precision = of the fields flagged, the share that really burned. Recall = of the fields that burned, the share flagged.
PR-AUC is the main score because burns are the minority class. {split_note}</p>
<div class="row">{img('fig_train_val_test', 420)}{img('fig_confusion_matrix', 330)}</div>
<h2>2. Flagging the riskiest fields</h2>{t_topk.to_html(index=False)}
<div class="row">{img('fig_gains')}{img('fig_risk_histogram')}</div>
<h2>3. Curves</h2><div class="row">{img('fig_precision_recall')}{img('fig_roc')}</div>
<h2>4. What the model relies on</h2><div class="row">{img('fig_importance', 420)}</div>
<h2>5. The 10 riskiest fields in the test set</h2>{top10.to_html(index=False)}
<h2>6. Risk map</h2><p>{map_note}</p>
<h2>Limitations</h2><p class="note">{limits}</p></body></html>"""
    open(os.path.join(a.out, "demo_report.html"), "w", encoding="utf-8").write(html)

    print("\n" + summary)
    print(f"\nDONE. Open this file to show the prototype:\n  {os.path.abspath(os.path.join(a.out, 'demo_report.html'))}")
    print(map_note)


if __name__ == "__main__":
    main()
