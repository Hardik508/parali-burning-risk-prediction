"""
Build the final prototype: an interactive burn-risk map (one HTML file you can open in any browser,
email, or host for free on GitHub Pages).

Input : results/test_predictions.csv   (written by train.py: lat, lon, y_true, risk)
Output: results/risk_map.html

Each dot is one field on one observation date. Colour = predicted probability of a burn in the
next 1-5 days. Click a dot to see the risk and (on the test year, where the answer is known)
whether the field really burned.
"""
import argparse
import json
import os

import branca.colormap as cm
import folium
import pandas as pd


def build_map(pred_csv, metrics_json, out_html, threshold=None, top_share=None):
    df = pd.read_csv(pred_csv)
    if threshold is None:
        threshold = 0.5
        if os.path.exists(metrics_json):  # use the threshold chosen on the validation year
            for m in json.load(open(metrics_json)):
                if m["model"].startswith("Hybrid CNN+LSTM") and "proposed" in m["model"]:
                    threshold = float(m["threshold"])
    if top_share:
        threshold = float(df["risk"].quantile(1 - top_share))
    df["flag"] = df["risk"] >= threshold

    centre = [df.lat.mean(), df.lon.mean()]
    m = folium.Map(location=centre, zoom_start=9, tiles="OpenStreetMap", control_scale=True)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, Maxar, Earthstar Geographics", name="Satellite view").add_to(m)

    cmap = cm.LinearColormap(["#2e9e4f", "#f2d338", "#d93025"], vmin=0, vmax=1,
                             caption="Predicted burn risk (next 1-5 days)")
    cmap.add_to(m)

    groups = {"High risk (flagged)": folium.FeatureGroup("High risk (flagged)", show=True),
              "Lower risk": folium.FeatureGroup("Lower risk", show=True)}
    for r in df.itertuples():
        txt = f"<b>Risk: {r.risk * 100:.0f}%</b><br>Flagged: {'yes' if r.flag else 'no'}"
        if hasattr(r, "y_true"):
            txt += f"<br>Actually burned: {'yes' if r.y_true == 1 else 'no'}"
        folium.CircleMarker(
            [r.lat, r.lon], radius=6 if r.flag else 4, weight=1, color="#333333",
            fill=True, fill_color=cmap(r.risk), fill_opacity=0.85 if r.flag else 0.55,
            popup=folium.Popup(txt, max_width=200),
        ).add_to(groups["High risk (flagged)" if r.flag else "Lower risk"])
    for g in groups.values():
        g.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    n_flag = int(df.flag.sum())
    hit = int(((df.flag) & (df.y_true == 1)).sum()) if "y_true" in df else None
    total_burn = int((df.y_true == 1).sum()) if "y_true" in df else None
    lines = [f"<b>Stubble-burning early-warning prototype</b>",
             f"{len(df)} fields shown, {n_flag} flagged high risk (threshold {threshold:.2f})"]
    if hit is not None:
        lines.append(f"Of {total_burn} fields that really burned, {hit} were flagged "
                     f"(recall {hit / max(total_burn, 1) * 100:.0f}%); "
                     f"of {n_flag} flagged, {hit} burned (precision {hit / max(n_flag, 1) * 100:.0f}%)")
    box = ('<div style="position:fixed;top:12px;left:60px;z-index:9999;background:white;padding:8px 12px;'
           'border:1px solid #888;border-radius:6px;font:13px/1.4 sans-serif;max-width:430px">'
           + "<br>".join(lines) + "</div>")
    m.get_root().html.add_child(folium.Element(box))

    os.makedirs(os.path.dirname(out_html) or ".", exist_ok=True)
    m.save(out_html)
    return dict(fields=len(df), flagged=n_flag, hit=hit, burned=total_burn, threshold=threshold)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results", help="folder written by train.py")
    ap.add_argument("--out", default=None)
    ap.add_argument("--threshold", type=float, default=None, help="override the validation-chosen threshold")
    ap.add_argument("--top_share", type=float, default=None, help="flag the top X share instead, e.g. 0.1 = top 10%%")
    a = ap.parse_args()
    out = a.out or os.path.join(a.results, "risk_map.html")
    info = build_map(os.path.join(a.results, "test_predictions.csv"), os.path.join(a.results, "metrics.json"),
                     out, a.threshold, a.top_share)
    print(f"saved {out}: {info}")
