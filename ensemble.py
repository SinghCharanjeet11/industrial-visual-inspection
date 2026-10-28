"""Combine pc3 outputs into a submission.

Per (category, bank): each chosen backbone's scores are standardised with the mean/std of its leave-one-out good
scores, then averaged. Threshold:
  - default: percentile `pct` of the ensembled leave-one-out good scores;
  - --gap: if the sorted test scores have an empty stretch (>= --min-gap wide) between the good-score percentiles
    p90 and p100 + 1, the threshold moves to the middle of the widest such stretch (low-density separation;
    uses no labels), otherwise it stays at the percentile.
Usage:
  python ensemble.py <out_dir> <data_dir> [--use cable=dn201,screw=wrn50+dn201] [--pct 97.5] [--gap] [--name v3]
"""
import argparse, os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CATS = ["bottle", "cable", "hazelnut", "screw"]
ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("data")
ap.add_argument("--default", default="wrn50+dn201", help="backbones used where --use gives nothing")
ap.add_argument("--use", default="", help="per-category backbones, e.g. cable=dn201,screw=wrn50+dn201_hr")
ap.add_argument("--pct", type=float, default=97.5)
ap.add_argument("--gap", action="store_true")
ap.add_argument("--min-gap", type=float, default=0.5)
ap.add_argument("--override", default="", help="per (cat_bank) z threshold, e.g. cable_gray=1.0")
ap.add_argument("--name", default="v3")
a = ap.parse_args()
use = {c: a.default.split("+") for c in CATS}
for kv in filter(None, a.use.split(",")):
    k, v = kv.split("="); use[k] = v.split("+")
override = {k: float(v) for k, v in (kv.split("=") for kv in filter(None, a.override.split(",")))}
te = pd.read_csv(os.path.join(a.data, "test.csv"))

rows = []
for cat in CATS:
    for b in ["color", "gray"]:
        fs = {m: os.path.join(a.out, f"{cat}_{b}_{m}.npz") for m in use[cat]}
        if not any(os.path.exists(f) for f in fs.values()):
            continue
        miss = [m for m, f in fs.items() if not os.path.exists(f)]
        assert not miss, f"{cat} {b}: missing {miss}"
        d = {m: np.load(f, allow_pickle=True) for m, f in fs.items()}
        ids = d[use[cat][0]]["test_ids"]
        z = {}
        for m in d:
            assert (d[m]["test_ids"] == ids).all()
            g = d[m]["good"]; mu, sd = g.mean(), g.std() + 1e-9
            z[m] = {k: (d[m][k] - mu) / sd for k in ("test", "good", "syn")}
        e = {k: np.mean([z[m][k] for m in z], 0) for k in ("test", "good", "syn")}
        y = np.r_[np.zeros(len(e["good"])), np.ones(len(e["syn"]))]
        auc = roc_auc_score(y, np.r_[e["good"], e["syn"]])
        thr, how = np.percentile(e["good"], a.pct), f"p{a.pct:g}"
        if a.gap:
            lo, hi = np.percentile(e["good"], 90), e["good"].max() + 1.0
            s = np.sort(e["test"]); m = (s[:-1] >= lo) & (s[1:] <= hi)
            if m.any():
                gaps = np.diff(s)[m]; k = int(np.argmax(gaps))
                if gaps[k] >= a.min_gap:
                    thr, how = (s[:-1][m][k] + s[1:][m][k]) / 2, f"gap {s[:-1][m][k]:.2f}-{s[1:][m][k]:.2f}"
        if f"{cat}_{b}" in override:
            thr, how = override[f"{cat}_{b}"], "override"
        lab = (e["test"] > thr).astype(int)
        print(f"{cat:9s}{b:6s} {'+'.join(use[cat]):18s} n={len(ids):3d} synAUROC={auc:.4f} thr={thr:.2f} ({how}) "
              f"flag={lab.mean():.2f} goodFPR={(e['good'] > thr).mean():.3f}")
        rows.append(pd.DataFrame({"id": ids, "score": e["test"], "thr": thr, "label": lab}))

sc = pd.concat(rows)
sub = te[["id", "category"]].merge(sc, on="id", how="left")
if sub.label.isna().any():
    raise SystemExit(f"{sub.label.isna().sum()} test images have no score yet; no submission written")
sub["label"] = sub.label.astype(int)
f = os.path.join(a.out, f"submission_{a.name}.csv")
sub[["id", "label"]].to_csv(f, index=False)
print(f"defect frac {sub.label.mean():.3f} {sub.groupby('category').label.mean().round(3).to_dict()} -> {f}")
