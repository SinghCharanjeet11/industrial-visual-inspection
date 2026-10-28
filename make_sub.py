"""Build submission.csv from per-category test scores + CV scores.

Threshold per (category, bank) = percentile `pct` of out-of-fold good scores.
Prints proxy F1 per category (synthetic defects : goods = 2 : 1) at several percentiles.
Usage: python make_sub.py <out_dir> <data_dir> [pct]
"""
import sys, os
import numpy as np, pandas as pd

out, data = sys.argv[1], sys.argv[2]
pct = float(sys.argv[3]) if len(sys.argv) > 3 else 99
te = pd.read_csv(os.path.join(data, "test.csv"))
rows = []
tot = {p: [0, 0, 0] for p in [90, 95, 97.5, 99, 100]}
for cat in ["bottle", "cable", "hazelnut", "screw"]:
    f = os.path.join(out, f"test_{cat}.csv")
    if not os.path.exists(f):
        print(cat, "missing"); continue
    t = pd.read_csv(f)
    cv = np.load(os.path.join(out, f"cv_{cat}.npz"))
    for b in t.bank.unique():
        g = cv[f"{b}_good"]; g = g[~np.isnan(g)]
        syn = np.concatenate([cv[k] for k in cv.files if k.startswith(b + "_") and k != f"{b}_good" and cv[k].size])
        line = []
        for p in tot:
            th = np.percentile(g, p)
            tp = (syn > th).sum(); fn = len(syn) - tp; fp = 2 * (g > th).sum()
            # expected counts if this bank's test images were 2/3 defective (test is defect-heavy)
            w = (t.bank == b).sum()
            rec, fpr = tp / len(syn), (g > th).mean()
            tot[p][0] += w * 2 / 3 * rec; tot[p][1] += w * 2 / 3 * (1 - rec); tot[p][2] += w / 3 * fpr
            line.append(f"p{p}:F1={2 * tp / (2 * tp + fp + fn):.3f}")
        th = np.percentile(g, pct)
        m = t.bank == b
        t.loc[m, "thr"] = th
        print(f"{cat:9s}{b:6s} n_test={m.sum():3d} flagged={(t[m].score > th).mean():.2f}  " + "  ".join(line))
    rows.append(t)
sc = pd.concat(rows)
sub = te[["id"]].merge(sc, on="id", how="left")
assert sub.score.notna().all(), "missing test scores"
sub["label"] = (sub.score > sub.thr).astype(int)
sub[["id", "label"]].to_csv(os.path.join(out, "submission.csv"), index=False)
print(f"\nsubmission @p{pct}: {len(sub)} rows, defective frac {sub.label.mean():.3f}")
print(sub.groupby("category").label.mean().round(3).to_dict())
for p, (tp, fn, fp) in tot.items():
    print(f"overall proxy F1 @p{p}: {2 * tp / (2 * tp + fp + fn):.4f}")
